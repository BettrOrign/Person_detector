import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

from gallery import PlateGallery
from tracker import Tracker

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PLATE_MODEL_PATH = os.path.join(MODEL_DIR, "plate_detector.pt")
CAR_CLASSES = {2, 5, 7}
YOLO_IMGSZ = 640
YOLO_CONF = 0.25
PLATE_CONF = 0.25
OCR_MAX_SIZE = 320


class CarState:
    def __init__(self, track_id: int, bbox: tuple):
        self.track_id = track_id
        self.bbox = bbox
        self.plate_text: str = ""
        self.plate_conf: float = 0.0
        self.gallery_id: Optional[int] = None
        self.last_seen: float = time.time()


class Pipeline:
    def __init__(self, source: str, gallery_path: str = "gallery"):
        self.source = source
        self.running = True

        logger.info("Loading YOLO for car detection...")
        self.yolo = YOLO("/home/sirius/Projects/person_detector/v1/yolo11n.pt")

        logger.info("Loading plate detection model...")
        self.plate_model = YOLO(PLATE_MODEL_PATH)

        self.gallery = PlateGallery(gallery_path)
        self.tracker = Tracker()

        self.cars: dict[int, CarState] = {}
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_tracks: list = []

    def _ocr(self, crop: np.ndarray) -> str:
        h, w = crop.shape[:2]
        if max(h, w) > OCR_MAX_SIZE:
            scale = OCR_MAX_SIZE / max(h, w)
            crop = cv2.resize(crop, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(
            thresh,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        return text.strip().upper()

    def process_frame(self, frame: np.ndarray):
        self._latest_frame = frame
        h, w = frame.shape[:2]
        now = time.time()

        results = self.yolo(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False, classes=list(CAR_CLASSES))
        car_boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_boxes.append((x1, y1, x2, y2, float(box.conf[0])))

        fresh_ids = set()
        for x1, y1, x2, y2, conf in car_boxes:
            best_id, best_iou = None, 0.3
            for cid, c in self.cars.items():
                iou = self._bbox_iou((x1, y1, x2, y2), c.bbox)
                if iou > best_iou:
                    best_iou, best_id = iou, cid
            if best_id is None:
                cid = self.tracker.next_id()
                self.cars[cid] = CarState(cid, (x1, y1, x2, y2))
            else:
                self.cars[best_id].bbox = (x1, y1, x2, y2)
                cid = best_id
            fresh_ids.add(cid)

        for cid in fresh_ids:
            car = self.cars[cid]
            x1, y1, x2, y2 = car.bbox
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.15)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            car_crop = frame[cy1:cy2, cx1:cx2]
            if car_crop.size == 0:
                continue

            plates = self.plate_model(car_crop, imgsz=640, conf=PLATE_CONF, verbose=False)
            ndet = len(plates[0].boxes) if (plates and plates[0].boxes is not None) else 0
            if ndet == 0:
                logger.debug(f"Car {cid}: no plate detected in crop ({car_crop.shape})")
            for pbox in (plates[0].boxes if ndet > 0 else []):
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                gx1 = max(0, cx1 + px1 - 4)
                gy1 = max(0, cy1 + py1 - 4)
                gx2 = min(w, cx1 + px2 + 4)
                gy2 = min(h, cy1 + py2 + 4)
                plate_crop = frame[gy1:gy2, gx1:gx2]
                if plate_crop.size == 0:
                    continue
                try:
                    text = self._ocr(plate_crop)
                    logger.debug(f"Car {cid} OCR raw: '{text}'")
                    if len(text) >= 3:
                        if text != car.plate_text:
                            car.plate_text = text
                            car.plate_conf = float(pbox.conf[0])
                            car.gallery_id = self.gallery.get_or_create(
                                text,
                                cv2.imencode(".jpg", plate_crop,
                                              [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(),
                            )
                            logger.info(f"Car {cid} plate: {text}")
                        break
                except Exception as e:
                    logger.warning(f"OCR error for car {cid}: {e}")

        dead = [cid for cid, c in self.cars.items()
                if cid not in fresh_ids and now - c.last_seen > 3.0]
        for cid in dead:
            del self.cars[cid]

        for cid in fresh_ids:
            self.cars[cid].last_seen = now

        tracks = [self.cars[cid] for cid in sorted(fresh_ids)]
        self._latest_tracks = tracks
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
