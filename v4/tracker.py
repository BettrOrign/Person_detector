import time


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


class Track:
    __slots__ = ("track_id", "bbox", "name", "action", "gallery_id",
                 "frames", "last_seen", "age", "color")

    def __init__(self, track_id: int, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.name = None
        self.action = ""
        self.gallery_id = None
        self.frames = 1
        self.last_seen = time.time()
        self.age = 0.0
        self.color = None


class Tracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: float = 2.0):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, dets: list[dict]) -> list[Track]:
        now = time.time()

        for d in dets:
            best_id, best_iou = None, self.iou_thresh
            for tid, t in self.tracks.items():
                iou = _iou(d["bbox"], t.bbox)
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is None:
                track = Track(self._next_id, d["bbox"])
                self._next_id += 1
                self._assign(track, d)
                self.tracks[track.track_id] = track
            else:
                t = self.tracks[best_id]
                t.bbox = d["bbox"]
                self._assign(t, d)
                t.last_seen = now
                t.frames += 1

        dead = [tid for tid, t in self.tracks.items()
                if now - t.last_seen > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        for t in self.tracks.values():
            t.age = now - t.last_seen

        return [t for _, t in sorted(self.tracks.items())]

    def _assign(self, track: Track, det: dict):
        if det.get("name") is not None:
            track.name = det["name"]
        if det.get("gallery_id") is not None:
            track.gallery_id = det["gallery_id"]
        if det.get("action") is not None:
            track.action = det["action"]
        if det.get("color") is not None:
            track.color = det["color"]
