#!/usr/bin/env python3
"""
Person Detector v2 — YOLO + Gemini Live API (WebSocket).

Opens webcam (or a video file via --path), runs YOLO person detection,
sends person crops to Gemini Live API via WebSocket for real-time description.
Press Q to quit.
"""

import argparse
import asyncio
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY not found in .env")

YOLO_MODEL_PATH = "../v1/yolo11n.pt"
YOLO_IMGSZ = 320
YOLO_CONF = 0.4
YOLO_PAD = 0.1  # padding ratio around each person crop (context for Gemini)
FRAME_INTERVAL = 2.0  # fallback wait on send error
POST_TURN_GAP = 3.0  # wait after a response before next crop batch
REFRESH_INTERVAL = 8.0     # seconds between re-describing the same person
MAX_CROPS_PER_CYCLE = 3    # max persons described per Gemini cycle (spreads load)

# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------
latest_frame: Optional[np.ndarray] = None
running: bool = True
descriptions: dict = {}          # person_id -> latest description text
desc_ts: dict = {}              # person_id -> timestamp of last description
pending_crops: list = []         # [(person_id, crop_bgr), ...] crops needing refresh
tracks: dict = {}               # id -> {cx, cy, vx, vy, feat, last}
memory: dict = {}               # id -> {feat, last}  (leave-and-return re-ID)
next_id: list = [1]             # monotonic id counter
MAX_LOST_TIME = 5.0             # s a track survives without detection
REID_MEMORY_TIME = 30.0         # s we remember appearance for return

# ---------------------------------------------------------------------------
# Person identity tracker (motion prediction + appearance re-ID)
# ---------------------------------------------------------------------------
def _feature(crop):
    """Lightweight appearance descriptor: normalized HSV color histogram."""
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 18], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _feat_dist(a, b) -> float:
    """Bhattacharyya distance between descriptors (0 = identical)."""
    if a is None or b is None:
        return 1.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def track_persons(raw, feats, now) -> list:
    """Assign stable person IDs across frames.

    Matches detections to live tracks by predicted position + appearance, and
    to recently-left people by strong appearance (re-ID). A track that vanishes
    for MAX_LOST_TIME is kept in memory for REID_MEMORY_TIME, so a person who
    leaves and returns keeps the same id. A one-frame YOLO miss does not reset
    the id because the track survives MAX_LOST_TIME without detection.
    """
    global tracks, memory, next_id
    updated = set()
    result = []
    for i, (x1, y1, x2, y2, conf) in enumerate(raw):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        feat = feats[i]
        best_id, best_score = None, -1.0

        # 1) live/lost tracks (within MAX_LOST_TIME)
        for tid, t in tracks.items():
            if tid in updated:
                continue
            pcx = t['cx'] + t.get('vx', 0.0)
            pcy = t['cy'] + t.get('vy', 0.0)
            d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            prox = max(0.0, 1.0 - d / 200.0)
            appr = 1.0 - _feat_dist(t['feat'], feat)
            score = 0.5 * prox + 0.5 * appr
            if score > best_score and score > 0.45:
                best_score, best_id = score, tid

        # 2) long-term memory: leave-and-return, require strong appearance
        if best_id is None:
            for mid, m in memory.items():
                if mid in updated:
                    continue
                appr = 1.0 - _feat_dist(m['feat'], feat)
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
        t['feat'] = feat
        t['last'] = now
        tracks[best_id] = t
        memory.pop(best_id, None)
        updated.add(best_id)
        result.append((best_id, x1, y1, x2, y2, conf))

    # demote stale tracks to memory
    for tid in list(tracks):
        if tid not in updated and now - tracks[tid]['last'] > MAX_LOST_TIME:
            memory[tid] = {'feat': tracks[tid]['feat'], 'last': now}
            del tracks[tid]
    # forget old memories
    for mid in list(memory):
        if now - memory[mid]['last'] > REID_MEMORY_TIME:
            del memory[mid]
    return result


# ---------------------------------------------------------------------------
# Gemini Live WebSocket session
# ---------------------------------------------------------------------------
async def gemini_session() -> None:
    """Connect to Gemini Live API, send person crops, receive descriptions."""
    global descriptions, desc_ts, running

    client = genai.Client(api_key=API_KEY)

    # Use Text modality (no audio) — cheaper and simpler
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(
            parts=[
                types.Part(
                    text=(
                        "You are watching a live camera feed. "
                        "Describe what THIS person is doing in 3-6 words only. "
                        "No thinking, no explanation, no preamble. "
                        "Examples: 'looking at phone', 'drinking coffee', "
                        "'sitting reading', 'walking', 'looking at camera', "
                        "'writing'"
                    )
                )
            ]
        ),
    )

    async with client.aio.live.connect(
        model="gemini-2.5-flash-native-audio-latest",
        config=config,
    ) as session:

        # Single sequential loop: send frame -> receive until turn_complete
        # -> wait POST_TURN_GAP -> repeat. Avoids the race between concurrent
        # send/receive tasks that froze the session after the first turn.
        try:
            while running:
                crops = list(pending_crops)
                if crops:
                    for pid, crop in crops:
                        if crop is None or crop.size == 0:
                            continue
                        # skip persons whose description is still fresh
                        if pid in descriptions and (time.time() - desc_ts.get(pid, 0.0)) < REFRESH_INTERVAL:
                            continue
                        _, buffer = cv2.imencode(
                            ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )
                        try:
                            await session.send_client_content(
                                turns=types.Content(
                                    role="user",
                                    parts=[
                                        types.Part(
                                            inline_data=types.Blob(
                                                data=buffer.tobytes(),
                                                mime_type="image/jpeg",
                                            )
                                        ),
                                        types.Part(
                                            text="Describe what THIS person is doing in 3-6 words."
                                        ),
                                    ],
                                ),
                                turn_complete=True,
                            )
                        except Exception as e:
                            print(f"  [Send error] {e}")
                            await asyncio.sleep(FRAME_INTERVAL)
                            continue

                        # Receive this turn until the model signals completion.
                        # Accumulate streamed transcription so the box shows the
                        # full phrase, not just the last fragment. Skip "**thinking**".
                        turn_text = ""
                        try:
                            async for msg in session.receive():
                                if not running:
                                    break
                                if msg.server_content:
                                    chunk = ""
                                    if msg.server_content.model_turn:
                                        for part in msg.server_content.model_turn.parts:
                                            if part.text:
                                                chunk = part.text.strip()
                                    if msg.server_content.output_transcription:
                                        chunk = msg.server_content.output_transcription.text.strip()
                                    if chunk and "**" not in chunk:
                                        turn_text = (turn_text + " " + chunk).strip()
                                    if msg.server_content.turn_complete:
                                        text = turn_text.strip() or descriptions.get(pid, "")
                                        descriptions[pid] = text
                                        desc_ts[pid] = time.time()
                                        print(f"  [id {pid}] [{time.strftime('%H:%M:%S')}] {text}")
                                        break
                        except Exception as exc:
                            if running:
                                print(f"  [Receive error] {exc}")

                # Wait the post-turn gap before the next crop batch.
                await asyncio.sleep(POST_TURN_GAP)
        finally:
            pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    """Open webcam, run detection, display results."""
    global latest_frame, running

    print("Loading YOLO...")
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        raise SystemExit(f"Failed to load YOLO: {e}")

    parser = argparse.ArgumentParser(description="Person Detector v2 - YOLO + Gemini Live")
    parser.add_argument("--path", help="Path to a video file instead of webcam")
    args = parser.parse_args()

    if args.path:
        cap = cv2.VideoCapture(args.path)
        src_label = args.path
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {args.path}")
    else:
        cap = cv2.VideoCapture(0)
        src_label = "webcam /dev/video0"
        if not cap.isOpened():
            raise SystemExit("Cannot open /dev/video0")
    video_mode = args.path is not None

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Start Gemini WebSocket in background thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws_thread = threading.Thread(
        target=loop.run_until_complete, args=(gemini_session(),), daemon=True
    )
    ws_thread.start()

    print(f"Source: {src_label} {w}x{h} + YOLO + Gemini Live (WebSocket)")
    print(f"Describe each person every ~{REFRESH_INTERVAL}s (max {MAX_CROPS_PER_CYCLE}/cycle) | Q to quit")

    while running:
        ret, frame = cap.read()
        if not ret:
            if video_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        latest_frame = frame.copy()
        display = frame.copy()

        # --- YOLO detection ---
        results = model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
        raw = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw.append((x1, y1, x2, y2, float(box.conf[0])))

        now = time.time()
        # Build padded crops + appearance features once per person
        person_crops = []
        feats = []
        for (x1, y1, x2, y2, conf) in raw:
            pad_x = int((x2 - x1) * YOLO_PAD)
            pad_y = int((y2 - y1) * YOLO_PAD)
            cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
            cx2 = min(frame.shape[1], x2 + pad_x); cy2 = min(frame.shape[0], y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            person_crops.append(crop)
            feats.append(_feature(crop))

        # Stable IDs + identity across frames
        person_boxes = track_persons(raw, feats, now)

        # Hand only the stalest persons' crops to Gemini (throttle calls)
        need = []
        for (pid, x1, y1, x2, y2, conf), crop in zip(person_boxes, person_crops):
            if crop is None or crop.size == 0:
                continue
            age = now - desc_ts.get(pid, 0.0)
            if pid not in descriptions or age > REFRESH_INTERVAL:
                need.append((age, pid, crop))
        need.sort(reverse=True)  # oldest first
        pending_crops[:] = [(pid, crop) for (_, pid, crop) in need[:MAX_CROPS_PER_CYCLE]]

        # --- Draw person boxes + their own descriptions ---
        for (pid, x1, y1, x2, y2, conf) in person_boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            desc = descriptions.get(pid, "")
            if desc:
                text_size = cv2.getTextSize(
                    desc, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )[0]
                cv2.rectangle(
                    display,
                    (x1, y1 - 30),
                    (x1 + text_size[0] + 10, y1 - 5),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    display, desc, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                )

        # --- Status overlay ---
        status = f"Gemini Live (WS) | ~{POST_TURN_GAP}s/turn | {len(person_boxes)} people"
        cv2.putText(
            display, status, (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
        )

        cv2.imshow("Person Detector v2 - Gemini Live", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
