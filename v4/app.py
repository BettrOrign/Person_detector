import argparse
import asyncio
import base64
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from pipeline import Pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HERE = os.path.dirname(os.path.abspath(__file__))

VIDEO_SOURCE: int | str = 0
running = True
latest_frame: np.ndarray | None = None
latest_people: list = []
frame_lock = threading.Lock()
people_lock = threading.Lock()
fps_value = [0.0]
fps_lock = threading.Lock()
fps_count = [0]
fps_stamp = [time.time()]


def detection_loop(source, pipeline: Pipeline):
    global latest_frame, latest_people, running

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return

    video_mode = not isinstance(source, int)
    if video_mode:
        logger.info(f"Playing: {source}")

    while running:
        ret, frame = cap.read()
        if not ret:
            if video_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        tracks = pipeline.process_frame(frame)

        # Annotate frame
        display = frame.copy()
        h, w = frame.shape[:2]
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            color = (0, 255, 0) if t.gallery_id else (255, 191, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            label = t.name or f"ID:{t.track_id}"
            if t.action:
                label += f" | {t.action}"

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display, (x1, y1 - 30), (x1 + text_size[0] + 10, y1 - 5), color, -1)
            cv2.putText(display, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        people_out = [
            {
                "id": t.track_id,
                "bbox": list(t.bbox),
                "name": t.name or f"ID:{t.track_id}",
                "action": t.action,
                "known": t.gallery_id is not None,
            }
            for t in tracks
        ]

        with frame_lock:
            latest_frame = display
        with people_lock:
            latest_people = people_out

        with fps_lock:
            fps_count[0] += 1
            now = time.time()
            if now - fps_stamp[0] >= 1.0:
                fps_value[0] = fps_count[0] / (now - fps_stamp[0])
                fps_count[0] = 0
                fps_stamp[0] = now

        if int(time.time()) % 5 == 0:
            logger.info(f"people={len(tracks)} gallery={pipeline.gallery.count()} fps={fps_value[0]:.1f}")

        time.sleep(0.001)

    cap.release()
    logger.info("Detection loop stopped")


app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/gallery")
async def gallery_list():
    return pipeline.gallery.list_persons()


@app.get("/gallery/{pid}/image")
async def gallery_image(pid: int):
    from fastapi.responses import Response
    from fastapi import HTTPException
    person = pipeline.gallery.get_person(pid)
    if person is None or person.get("image") is None:
        raise HTTPException(status_code=404)
    return Response(content=person["image"], media_type="image/jpeg")


@app.delete("/gallery/{pid}")
async def gallery_delete(pid: int):
    ok = pipeline.gallery.delete(pid)
    return {"ok": ok}


@app.post("/gallery/{pid}/rename")
async def gallery_rename(pid: int, name: str):
    ok = pipeline.gallery.rename(pid, name)
    return {"ok": ok}


@app.post("/enroll")
async def enroll(name: str = "", person_id: int = -1):
    from fastapi import HTTPException
    with people_lock:
        p = [x for x in latest_people if x["id"] == person_id]
        if not p:
            raise HTTPException(status_code=400, detail="Person not found")
        # Get the PersonState from pipeline
        pp = pipeline.persons.get(person_id)
        if pp is None or pp.face_embedding is None:
            raise HTTPException(status_code=400, detail="No face data for this person")

        pid = pipeline.gallery.add(pp.face_embedding, name or f"person_{person_id}")
        pp.gallery_id = pid
        pp.name = name or f"person_{person_id}"
        for t in latest_people:
            if t["id"] == person_id:
                t["name"] = pp.name
                t["known"] = True

    logger.info(f"Enrolled person {person_id} as {name} (gallery_id={pid})")
    return {"pid": pid}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        last_sent = 0
        while True:
            now = time.time()
            if now - last_sent >= 0.1:
                with frame_lock:
                    if latest_frame is not None:
                        _, buf = cv2.imencode(
                            ".jpg", latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                        )
                        b64 = base64.b64encode(buf).decode()
                        h, w = latest_frame.shape[:2]

                with people_lock:
                    people = list(latest_people)

                fps = fps_value[0]

                await ws.send_json({
                    "type": "frame",
                    "image": b64,
                    "width": w,
                    "height": h,
                    "people": people,
                    "gallery": pipeline.gallery.count(),
                    "fps": round(fps, 1),
                })
                last_sent = now

            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass


pipeline: Pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global running, pipeline
    running = True
    pipeline = Pipeline(source=VIDEO_SOURCE, gallery_path=os.path.join(HERE, "gallery"))

    def start_gemini():
        asyncio.run(pipeline._gemini_loop())

    det_thread = threading.Thread(
        target=detection_loop, args=(VIDEO_SOURCE, pipeline), daemon=True
    )
    det_thread.start()

    gem_thread = threading.Thread(target=start_gemini, daemon=True)
    gem_thread.start()

    yield
    running = False
    pipeline.running = False
    det_thread.join(timeout=2)
    pipeline.gallery.close()
    logger.info("Shutdown complete")


app.router.lifespan_context = lifespan


def main():
    global VIDEO_SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Video file path or camera index")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.source:
        VIDEO_SOURCE = int(args.source) if args.source.isdigit() else args.source

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
