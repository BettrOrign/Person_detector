import argparse
import asyncio
import json
import logging
import os
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket, WebSocketDisconnect

from pipeline import Pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

pipeline: Pipeline | None = None
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


def detection_loop(p: Pipeline):
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


@app.on_event("startup")
def startup():
    global pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--gallery", default="gallery")
    parser.add_argument("--port", type=int, default=8080)
    args, _ = parser.parse_known_args()

    pipeline = Pipeline(source=args.source, gallery_path=args.gallery)

    import threading
    t = threading.Thread(target=detection_loop, args=(pipeline,), daemon=True)
    t.start()


@app.on_event("shutdown")
def shutdown():
    if pipeline:
        pipeline.running = False
        pipeline.gallery.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video source path or camera index")
    parser.add_argument("--gallery", default="gallery")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    os.environ.setdefault("CARSDETECTOR_SOURCE", args.source)
    os.environ.setdefault("CARSDETECTOR_GALLERY", args.gallery)
    os.environ.setdefault("CARSDETECTOR_PORT", str(args.port))

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
