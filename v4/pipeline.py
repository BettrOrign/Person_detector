import asyncio
import logging
import os
import re
import time
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ultralytics import YOLO

from face import SCRFD, ArcFace
from gallery import Gallery
from tracker import Tracker

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY not found in .env")

YOLO_PATH = "../v1/yolo11n.pt"
YOLO_IMGSZ = 320
YOLO_CONF = 0.4
YOLO_PAD = 0.1

FACE_MATCH_THRESH = 0.35
GEMINI_REFRESH = 8.0
GEMINI_POST_GAP = 3.0


class PersonState:
    def __init__(self, track_id: int, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.name: Optional[str] = None
        self.gallery_id: Optional[int] = None
        self.action: str = ""
        self.last_action_update: float = 0.0
        self.face_embedding: Optional[np.ndarray] = None


class Pipeline:
    def __init__(self, source: str, gallery_path: str = "gallery",
                 det_weight: str = "../yakhyo_face_reid/weights/det_500m.onnx",
                 rec_weight: str = "../yakhyo_face_reid/weights/w600k_mbf.onnx"):
        self.source = source
        self.running = True

        logger.info("Loading YOLO...")
        self.yolo = YOLO(YOLO_PATH)

        logger.info("Loading face models...")
        self.detector = SCRFD(det_weight)
        self.recognizer = ArcFace(rec_weight)
        self.gallery = Gallery(gallery_path, dim=self.recognizer.embedding_size)
        self.tracker = Tracker()

        self.pending_gemini: list[tuple[int, np.ndarray]] = []
        self.persons: dict[int, PersonState] = {}

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_tracks: list = []

        self.client = genai.Client(api_key=API_KEY)

    def _associate_faces_to_persons(self, face_boxes, face_kpss, person_boxes):
        kps_arr = face_kpss if face_kpss is not None else np.empty((0, 5, 2))
        for fbox, kps in zip(face_boxes, kps_arr):
            fx1, fy1, fx2, fy2 = map(int, fbox[:4])
            fcy = (fy1 + fy2) / 2
            best_pid, best_dist = None, float("inf")
            for pid, p in self.persons.items():
                px1, py1, px2, py2 = p.bbox
                if fx1 >= px2 or fx2 <= px1:
                    continue
                pcy = (py1 + py2) / 2
                dist = abs(fcy - pcy)
                if dist < best_dist:
                    best_dist, best_pid = dist, pid
            if best_pid is not None and best_dist < (py2 - py1) * 0.5:
                person = self.persons.get(best_pid)
                if person is None:
                    continue
                try:
                    emb = self.recognizer.get_embedding(self._latest_frame, kps)
                    emb = emb.astype(np.float32)
                    pid, sim, name = self.gallery.match(emb, FACE_MATCH_THRESH)
                    if pid is not None:
                        person.name = name
                        person.gallery_id = pid
                    person.face_embedding = emb
                except Exception as e:
                    logger.warning(f"Face embedding error: {e}")

    async def _gemini_loop(self):
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(
                parts=[types.Part(text=(
                    "You are shown a person from a live camera. "
                    "Describe what they are DOING in 3-6 words only. "
                    "No thinking, no preamble. Examples: 'looking at phone', "
                    "'drinking coffee', 'walking', 'sitting reading', "
                    "'looking at camera', 'writing', 'talking to someone'."
                ))]
            ),
        )
        async with self.client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-latest",
            config=config,
        ) as session:
            while self.running:
                crops = list(self.pending_gemini)
                if crops:
                    for pid, crop in crops:
                        if crop is None or crop.size == 0:
                            continue
                        person = self.persons.get(pid)
                        if person and time.time() - person.last_action_update < GEMINI_REFRESH:
                            continue
                        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        try:
                            await session.send_client_content(
                                turns=types.Content(
                                    role="user",
                                    parts=[
                                        types.Part(inline_data=types.Blob(
                                            data=buf.tobytes(), mime_type="image/jpeg")),
                                        types.Part(text="Describe what THIS person is doing in 3-6 words."),
                                    ],
                                ),
                                turn_complete=True,
                            )
                        except Exception as e:
                            logger.warning(f"Gemini send error: {e}")
                            await asyncio.sleep(2)
                            continue

                        turn_text = ""
                        try:
                            async for msg in session.receive():
                                if not self.running:
                                    break
                                if msg.server_content:
                                    chunk = ""
                                    if msg.server_content.output_transcription:
                                        chunk = msg.server_content.output_transcription.text.strip()
                                    if chunk and "**" not in chunk:
                                        turn_text = (turn_text + " " + chunk).strip()
                                    if msg.server_content.turn_complete:
                                        if person:
                                            person.action = turn_text.strip() or person.action
                                            person.last_action_update = time.time()
                                            logger.info(f"[person {pid}] {person.action}")
                                        break
                        except Exception as e:
                            if self.running:
                                logger.warning(f"Gemini receive error: {e}")

                await asyncio.sleep(GEMINI_POST_GAP)

    def process_frame(self, frame: np.ndarray):
        self._latest_frame = frame
        h, w = frame.shape[:2]
        now = time.time()

        # 1. YOLO person detection
        results = self.yolo(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
        person_boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2, float(box.conf[0])))

        # 2. Ensure PersonState for each YOLO detection
        dets = []
        fresh_pids = set()
        for (x1, y1, x2, y2, conf) in person_boxes:
            # Find best matching existing person by IOU
            best_pid, best_iou = None, 0.3
            for pid, p in self.persons.items():
                iou = self._bbox_iou((x1, y1, x2, y2), p.bbox)
                if iou > best_iou:
                    best_iou, best_pid = iou, pid
            if best_pid is None:
                pid = max(self.persons.keys()) + 1 if self.persons else 1
                self.persons[pid] = PersonState(pid, (x1, y1, x2, y2))
            else:
                self.persons[best_pid].bbox = (x1, y1, x2, y2)
                pid = best_pid
            fresh_pids.add(pid)
            dets.append({"pid": pid, "bbox": (x1, y1, x2, y2), "conf": conf})

        # 3. SCRFD face detection
        face_boxes, face_kpss = self.detector.detect(frame)

        # 4. Associate faces to persons
        if face_kpss is not None and len(face_kpss) > 0:
            self._associate_faces_to_persons(face_boxes, face_kpss, person_boxes)

        # 5. Build Gemini pending crops (throttled)
        self.pending_gemini.clear()
        for pid in fresh_pids:
            p = self.persons[pid]
            if now - p.last_action_update < GEMINI_REFRESH:
                continue
            x1, y1, x2, y2 = p.bbox
            pad_x = int((x2 - x1) * YOLO_PAD)
            pad_y = int((y2 - y1) * YOLO_PAD)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                self.pending_gemini.append((pid, crop))

        # 6. Clean dead persons
        dead = [pid for pid, p in self.persons.items()
                if pid not in fresh_pids and now - p.last_action_update > 2.0]
        for pid in dead:
            del self.persons[pid]

        # 7. Build track data for output
        tracks = []
        for pid in sorted(fresh_pids):
            p = self.persons[pid]
            tracks.append(p)

        return tracks

    def _bbox_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        ab = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / (aa + ab - inter)
