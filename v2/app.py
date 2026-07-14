#!/usr/bin/env python3
"""
Person Detector v2 — YOLO + Gemini Live (WebSocket) with Web UI.
Run: python app.py [--path VIDEO.mp4] [--port 8080]
UI: http://localhost:8080  (8000 is used by the opencode CLI)
"""
import argparse
import asyncio
import base64
import cv2
import os
import re
import numpy as np
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Tuple

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
from google import genai
from google.genai import types
from ultralytics import YOLO

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY not found in .env")

YOLO_MODEL_PATH = "../v1/yolo11n.pt"
YOLO_IMGSZ = 320
YOLO_CONF = 0.4
YOLO_PAD = 0.1

MAX_LOST_TIME = 5.0      # seconds a track survives without detection
REID_MEMORY_TIME = 30.0  # seconds we remember appearance for leave-and-return
REFRESH_INTERVAL = 8.0   # seconds between re-describing the same person
MAX_CROPS_PER_CYCLE = 3  # max persons described per Gemini cycle (spreads load)

# ---------------------------------------------------------------------------
# Globals / shared state
# ---------------------------------------------------------------------------
client = genai.Client(api_key=API_KEY)
model = YOLO(YOLO_MODEL_PATH)

latest_frame: Optional[np.ndarray] = None
frame_ready = threading.Event()
frame_lock = threading.Lock()

people_boxes: List[Tuple[int, int, int, int, int, float]] = []
people_lock = threading.Lock()

descriptions: dict[int, str] = {}
descriptions_ts: dict[int, float] = {}

# Tracker state
tracks: dict[int, dict] = {}      # id -> {cx, cy, vx, vy, feat, last}
memory: dict[int, dict] = {}      # id -> {feat, last}  (leave-and-return)
next_id: List[int] = [1]

running = True

# ---------------------------------------------------------------------------
# Feature / tracker
# ---------------------------------------------------------------------------
def _feature(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 18], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _feat_dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 1.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def track_persons(raw: List[Tuple[int, int, int, int, float]],
                  feats: List[Optional[np.ndarray]],
                  now: float) -> List[Tuple[int, int, int, int, int, float]]:
    """Assign stable person IDs via motion + appearance (+ long-term memory)."""
    global tracks, memory, next_id
    updated = set()
    result = []

    for i, (x1, y1, x2, y2, conf) in enumerate(raw):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        feat = feats[i]
        best_id, best_score = None, -1.0

        # 1) match against live/lost tracks (within MAX_LOST_TIME)
        for tid, t in tracks.items():
            if tid in updated:
                continue
            pcx = t['cx'] + t.get('vx', 0.0)
            pcy = t['cy'] + t.get('vy', 0.0)
            d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            prox = max(0.0, 1.0 - d / 200.0)
            appr = 1.0 - _feat_dist(t.get('feat'), feats[i])
            score = 0.5 * prox + 0.5 * appr
            if score > best_score and score > 0.45:
                best_score, best_id = score, tid

        # 2) long-term memory (leave-and-return) — require strong appearance match
        if best_id is None:
            for mid, m in memory.items():
                if mid in updated:
                    continue
                appr = 1.0 - _feat_dist(m['feat'], feats[i])
                if appr > 0.75 and appr > best_score:
                    best_score, best_id = appr, mid

        if best_id is None:
            best_id = next_id[0]
            next_id[0] += 1

        t = tracks.get(best_id)
        if t is None:
            t = {'cx': cx, 'cy': cy, 'vx': 0.0, 'vy': 0.0}
        else:
            t['vx'] = cx - t['cx']
            t['vy'] = cy - t['cy']
        t['cx'], t['cy'] = cx, cy
        t['feat'] = feats[i]
        t['last'] = time.time()
        tracks[best_id] = t
        memory.pop(best_id, None)
        updated.add(best_id)
        result.append((best_id, x1, y1, x2, y2, conf))

    # demote stale tracks -> memory
    for tid in list(tracks):
        if tid not in updated and time.time() - tracks[tid]['last'] > 5.0:
            memory[tid] = {'feat': tracks[tid]['feat'], 'last': time.time()}
            del tracks[tid]
    for mid in list(memory):
        if time.time() - memory[mid]['last'] > 30.0:
            del memory[mid]

    return result


# ---------------------------------------------------------------------------
# Gemini worker
# ---------------------------------------------------------------------------
async def gemini_worker():
    global running

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(
            parts=[types.Part(text=(
                "You are shown cropped images of different people. "
                "For EACH, describe what the person is DOING in 3-6 words only "
                "(action, not appearance). No thinking, no preamble, no explanation. "
                "Examples: 'looking at phone', 'drinking coffee', 'walking', "
                "'sitting reading', 'looking at camera', 'talking to someone'."
            ))]
        ),
    )

    async with client.aio.live.connect(
        model="gemini-2.5-flash-native-audio-latest",
        config=config,
    ) as session:
        while running:
            # snapshot current people
            with people_lock:
                targets = list(people_boxes)
            if not targets:
                await asyncio.sleep(2)
                continue

            # build all crops in one message (keeps crop-level action quality)
            crops, ids_order = [], []
            with frame_lock:
                if latest_frame is None:
                    await asyncio.sleep(1)
                    continue
                for pid, x1, y1, x2, y2, conf in targets:
                    crop = latest_frame[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        continue
                    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    crops.append(buf.tobytes())
                    ids_order.append(pid)
            if not crops:
                await asyncio.sleep(1)
                continue

            parts = [types.Part(inline_data=types.Blob(data=d, mime_type="image/jpeg"))
                     for d in crops]
            parts.append(types.Part(text=(
                f"These are {len(crops)} different people, in order 1..{len(crops)}. "
                "For EACH, output exactly one line 'i: 3-6 word action description'. "
                "No preamble, no extra text."
            )))

            try:
                await session.send_client_content(
                    turns=types.Content(role="user", parts=parts),
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Send error] {e}")
                await asyncio.sleep(2)
                continue

            turn_text = ""
            try:
                async for msg in session.receive():
                    if not running:
                        break
                    if msg.server_content:
                        chunk = ""
                        if msg.server_content.output_transcription:
                            chunk = msg.server_content.output_transcription.text.strip()
                        if chunk and "**" not in chunk:
                            turn_text = (turn_text + " " + chunk).strip()
                        if msg.server_content.turn_complete:
                            # audio transcription arrives word-by-word; parse by number markers
                            for m in re.finditer(r"(\d+)\s*[:\-]\s*(.+?)(?=\s+\d+\s*[:\-]|\Z)", turn_text, re.S):
                                idx = int(m.group(1)) - 1
                                desc = m.group(2).strip()
                                if 0 <= idx < len(ids_order) and desc:
                                    pid = ids_order[idx]
                                    descriptions[pid] = desc
                                    descriptions_ts[pid] = time.time()
                                    print(f"  [id {pid}] {desc}", flush=True)
                            break
            except Exception as e:
                if running:
                    print(f"[Receive error] {e}")

            await asyncio.sleep(REFRESH_INTERVAL)


# ---------------------------------------------------------------------------
# Detection loop
# ---------------------------------------------------------------------------
def detection_loop(source: str):
    global latest_frame, people_boxes, running

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")

    video_mode = not isinstance(source, int)
    print(f"Detection loop started on {source}")

    DETECT_INTERVAL = 0.05  # run YOLO ~20 fps (tracker tolerates gaps)
    last_det = 0.0

    while running:
        ret, frame = cap.read()
        if not ret:
            if video_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # keep latest frame fresh every read for smooth UI
        with frame_lock:
            latest_frame = frame.copy()
        frame_ready.set()

        now = time.time()
        if now - last_det < DETECT_INTERVAL:
            time.sleep(0.005)
            continue
        last_det = now

        # YOLO
        results = model(frame, imgsz=320, conf=0.4, verbose=False)
        raw = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw.append((x1, y1, x2, y2, float(box.conf[0])))

        # features
        feats = []
        for (x1, y1, x2, y2, conf) in raw:
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
            cx2 = min(frame.shape[1], x2 + pad_x)
            cy2 = min(frame.shape[0], y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            feats.append(_feature(crop))

        # track
        tracked = track_persons(raw, feats, time.time())
        with people_lock:
            globals()['people_boxes'] = tracked

        time.sleep(0.005)

    cap.release()
    print("Detection loop stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global running
    running = True

    det_thread = threading.Thread(target=detection_loop, args=(VIDEO_SOURCE,), daemon=True)
    det_thread.start()

    gem_thread = threading.Thread(target=lambda: asyncio.run(gemini_worker()), daemon=True)
    gem_thread.start()

    yield

    running = False
    det_thread.join(timeout=2)
    print("Shutdown complete")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        last_sent = 0
        while True:
            now = time.time()
            if now - last_sent >= 0.1:  # ~10 fps
                with frame_lock:
                    if latest_frame is not None:
                        _, buf = cv2.imencode(".jpg", latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        b64 = base64.b64encode(buf).decode()
                        h, w = latest_frame.shape[:2]

                    with people_lock:
                        people_data = [
                            {"id": pid, "box": [x1, y1, x2, y2], "desc": descriptions.get(pid, "")}
                            for (pid, x1, y1, x2, y2, conf) in people_boxes
                        ]

                    await ws.send_json({
                        "type": "frame",
                        "image": b64,
                        "width": w,
                        "height": h,
                        "people": people_data
                    })
                    last_sent = now
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
VIDEO_SOURCE = 0

def main():
    global VIDEO_SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to video file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP port (8000 is taken by the opencode CLI)")
    args = parser.parse_args()
    if args.path:
        VIDEO_SOURCE = args.path

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()