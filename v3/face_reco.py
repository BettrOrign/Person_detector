import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

PAD = 0.3


class FaceReco:
    def __init__(self, ctx_id: int = -1, det_size: tuple = (640, 640)):
        self.app = FaceAnalysis(name="buffalo_s", root="~/.insightface")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, img: np.ndarray) -> list[dict]:
        raw = self.app.get(img)
        out = []
        h, w = img.shape[:2]
        for f in raw:
            x1, y1, x2, y2 = map(int, f.bbox)
            pad_x = int((x2 - x1) * PAD)
            pad_y = int((y2 - y1) * PAD)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            emb = None
            if hasattr(f, "embedding") and f.embedding is not None:
                emb = f.embedding.astype(np.float32)
                n = np.linalg.norm(emb)
                if n > 1e-6:
                    emb /= n
            kps = np.array(f.kps, dtype=np.float32) if hasattr(f, "kps") else None
            out.append({
                "bbox": (x1, y1, x2, y2),
                "kps": kps,
                "score": float(f.det_score),
                "embedding": emb,
            })
        return out

    def crop_face(self, img: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
