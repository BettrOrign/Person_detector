import time
import numpy as np


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


class FaceTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: float = 2.0):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, dets: list[dict]) -> list[dict]:
        now = time.time()
        for d in dets:
            best_id, best_iou = None, self.iou_thresh
            for tid, t in self.tracks.items():
                iou = _iou(d["bbox"], t["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.tracks[best_id] = {
                    "bbox": d["bbox"],
                    "embedding": d.get("embedding"),
                    "person_id": d.get("person_id"),
                    "kps": d.get("kps"),
                    "frames": 0,
                    "last_seen": now,
                }
            else:
                t = self.tracks[best_id]
                t["bbox"] = d["bbox"]
                emb = d.get("embedding")
                if emb is not None:
                    t["embedding"] = emb
                pid = d.get("person_id")
                if pid is not None:
                    t["person_id"] = pid
                t["kps"] = d.get("kps")
                t["last_seen"] = now
            self.tracks[best_id]["frames"] += 1

        dead = [tid for tid, t in self.tracks.items()
                if now - t["last_seen"] > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        return [
            {
                "track_id": tid,
                "bbox": t["bbox"],
                "embedding": t.get("embedding"),
                "person_id": t.get("person_id"),
                "frames": t["frames"],
            }
            for tid, t in sorted(self.tracks.items())
        ]
