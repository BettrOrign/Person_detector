import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket, WebSocketDisconnect

from pipeline import Pipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

pipeline: Pipeline | None = None
_thread: threading.Thread | None = None
app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(static_dir, "index.html")) as f:
        return f.read()


@app.get("/gallery")
def gallery_list():
    if pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)
    return pipeline.gallery.list_all()


@app.get("/gallery/{pid}")
def gallery_get(pid: int):
    if pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)
    cur = pipeline.gallery._conn.execute(
        "SELECT id, plate_text, first_seen, last_seen FROM plates WHERE id=?", (pid,)
    )
    row = cur.fetchone()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"id": row[0], "plate": row[1], "first_seen": row[2], "last_seen": row[3]}


@app.delete("/gallery/{pid}")
def gallery_delete(pid: int):
    if pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)
    ok = pipeline.gallery.delete(pid)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"ok": True}


RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")
VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".ts": "video/mp2t",
    ".flv": "video/x-flv",
}


@app.get("/video")
async def video_stream(request: Request):
    if pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)
    path = pipeline.source
    if not path or not os.path.isfile(path):
        return JSONResponse({"error": "no video"}, status_code=404)

    total = os.path.getsize(path)
    mime = VIDEO_MIME.get(os.path.splitext(path)[1].lower(), "video/mp4")
    range_header = request.headers.get("range", "")
    m = RANGE_RE.match(range_header)
    if m:
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else total - 1
        length = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(length)
        return Response(
            body,
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Accept-Ranges": "bytes",
                "Content-Type": mime,
                "Content-Length": str(length),
            },
        )

    with open(path, "rb") as f:
        body = f.read()
    return Response(
        body,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Type": mime,
            "Content-Length": str(total),
        },
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    if pipeline is None:
        await websocket.close(1011, "pipeline not ready")
        return
    await websocket.accept()
    try:
        while pipeline.running:
            tracks = pipeline._latest_tracks
            data = [
                {
                    "id": t.track_id,
                    "bbox": t.bbox,
                    "plate": t.plate_text,
                    "plate_conf": round(t.plate_conf, 3),
                }
                for t in tracks
            ]
            await websocket.send_text(json.dumps({
                "cars": data,
                "gallery_count": pipeline.gallery.count(),
            }))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def detection_loop(p: Pipeline):
    if _is_image(p.source):
        _process_image(p, p.source)
        p.running = False
        return

    cap = cv2.VideoCapture(p.source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {p.source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1.0 / fps
    logger.info(f"Playing: {p.source} ({fps:.1f} fps)")

    while p.running:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        p.process_frame(frame)
        dt = time.time() - t0
        logger.info(f"Cars={len(p._latest_tracks)} gallery={p.gallery.count()} fps={1/max(dt, 0.01):.1f}")
        remaining = delay - dt
        if remaining > 0:
            time.sleep(remaining)
    cap.release()
    logger.info("Detection loop stopped")


def _process_image(p: Pipeline, path: str):
    frame = cv2.imread(path)
    if frame is None:
        logger.error(f"Cannot read image: {path}")
        return
    logger.info(f"Processing image: {path} ({frame.shape[1]}x{frame.shape[0]})")

    p.process_frame(frame)

    print("=" * 50)
    print(f"Image: {path}")
    if not p._latest_tracks:
        print("No cars detected.")
    else:
        print(f"Cars detected: {len(p._latest_tracks)}")
    for t in p._latest_tracks:
        plate = t.plate_text or "(not recognized)"
        print(f"  Car #{t.track_id}: bbox={t.bbox}, plate={plate}, conf={t.plate_conf}")
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = t.plate_text if t.plate_text else f"Car #{t.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bg_y1 = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    print(f"Gallery entries: {p.gallery.count()}")
    print("=" * 50)

    base, ext = os.path.splitext(path)
    out_path = f"{base}_annotated{ext}"
    cv2.imwrite(out_path, frame)
    logger.info(f"Saved annotated: {out_path}")

    try:
        cv2.imshow("Cars Detector", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def _start_pipeline(source: str, gallery_path: str):
    global pipeline, _thread
    pipeline = Pipeline(source=source, gallery_path=gallery_path)
    _thread = threading.Thread(target=detection_loop, args=(pipeline,), daemon=True)
    _thread.start()


def _stop_pipeline():
    global pipeline, _thread
    if pipeline:
        pipeline.running = False
        pipeline.gallery.close()
        pipeline = None
    _thread = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video/camera source or image path")
    parser.add_argument("--gallery", default="gallery")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if _is_image(args.source):
        p = Pipeline(source=args.source, gallery_path=args.gallery)
        _process_image(p, args.source)
        p.gallery.close()
        return

    _start_pipeline(args.source, args.gallery)

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
    finally:
        _stop_pipeline()


if __name__ == "__main__":
    main()
