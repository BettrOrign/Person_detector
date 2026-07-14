import argparse
import asyncio
import base64
import os
import cv2
import sys
import threading
import time
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from face_reco import FaceReco
from gallery import Gallery
from tracker import FaceTracker

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE: int | str = 0
MATCH_THR = 0.35
DET_INTERVAL = 0.0

running = True
latest_frame = None
frame_lock = threading.Lock()
people_out = []
people_lock = threading.Lock()
dbg_info = {}
fps_count = [0]
fps_value = [0.0]
fps_stamp = [time.time()]
fps_lock = threading.Lock()


def detection_loop(source, face_reco, gallery, tracker):
    global latest_frame, people_out, dbg_info, running
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {source}")
    video_mode = not isinstance(source, int)
    print(f"[v3] face-reco loop started on {source}")

    while running:
        ret, frame = cap.read()
        if not ret:
            if video_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        with frame_lock:
            latest_frame = frame.copy()

        faces = face_reco.detect(frame)

        dets = []
        for f in faces:
            emb = f.get("embedding")
            pid = None
            if emb is not None:
                matched_pid, sim = gallery.match(emb, MATCH_THR)
                if matched_pid is not None:
                    gallery.update(matched_pid)
                    pid = matched_pid
            dets.append({
                "bbox": f["bbox"],
                "embedding": emb,
                "person_id": pid,
                "kps": f.get("kps"),
            })

        tracks = tracker.update(dets)

        for t in tracks:
            pid = t.get("person_id")
            emb = t.get("embedding")
            if pid is None and emb is not None:
                matched_pid, sim = gallery.match(emb, MATCH_THR)
                if matched_pid is not None:
                    gallery.update(matched_pid)
                    t["person_id"] = matched_pid
                else:
                    crop = face_reco.crop_face(frame, t["bbox"])
                    image_bytes = None
                    if crop is not None and crop.size > 0:
                        ok, buf = cv2.imencode(".jpg", crop)
                        if ok:
                            image_bytes = buf.tobytes()
                    new_pid = gallery.add(emb, image=image_bytes)
                    t["person_id"] = new_pid

        with people_lock:
            people_out = [
                {
                    "id": t.get("person_id", -1),
                    "track": t["track_id"],
                    "bbox": list(t["bbox"]),
                    "known": t.get("person_id") is not None,
                    "frames": t["frames"],
                }
                for t in tracks
            ]

        with fps_lock:
            fps_count[0] += 1
            now = time.time()
            if now - fps_stamp[0] >= 1.0:
                fps_value[0] = fps_count[0] / (now - fps_stamp[0])
                fps_count[0] = 0
                fps_stamp[0] = now

        if int(time.time()) != getattr(detection_loop, "_dbg_sec", -1):
            detection_loop._dbg_sec = int(time.time())
            print(f"[v3] faces={len(faces)} tracks={len(tracks)} "
                  f"gallery={gallery.count()} fps={fps_value[0]:.1f}", flush=True)

        time.sleep(0.005)

    cap.release()
    print("[v3] detection loop stopped")


app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/gallery")
async def gallery_list():
    return gallery.list_persons()


@app.get("/gallery/{pid}/image")
async def gallery_image(pid: int):
    from fastapi.responses import Response
    from fastapi import HTTPException
    data = gallery.get_image(pid)
    if data is None:
        raise HTTPException(status_code=404)
    return Response(content=data, media_type="image/jpeg")


@app.delete("/gallery/{pid}")
async def gallery_delete(pid: int):
    gallery.delete(pid)
    return {"ok": True}


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
                    people = list(people_out)
                with fps_lock:
                    fps = fps_value[0]
                await ws.send_json({
                    "type": "frame",
                    "image": b64,
                    "width": w,
                    "height": h,
                    "people": people,
                    "gallery": gallery.count(),
                    "fps": round(fps, 1),
                })
                last_sent = now
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass


face_reco: FaceReco = None
gallery: Gallery = None
tracker: FaceTracker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global running, face_reco, gallery, tracker
    running = True
    face_reco = FaceReco(ctx_id=-1, det_size=(640, 640))
    gallery = Gallery(os.path.join(HERE, "v3_faces.db"))
    tracker = FaceTracker(iou_thresh=0.3, max_age=2.0)
    det_thread = threading.Thread(
        target=detection_loop, args=(VIDEO_SOURCE, face_reco, gallery, tracker),
        daemon=True
    )
    det_thread.start()
    yield
    running = False
    det_thread.join(timeout=2)
    print("[v3] shutdown complete")


app.router.lifespan_context = lifespan


def main():
    global VIDEO_SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Video file path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.path:
        VIDEO_SOURCE = args.path
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
