import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


YOLO_MODEL_PATH = "yolo26n.pt"
OSNET_MODEL_PATH = "models/osnet_x1_0.pt"
CONFIDENCE_THRESHOLD = 0.5
OSNET_INPUT_SIZE = (128, 256)
# CAMERA_ID = "./test/test-video.mp4"
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PersonDetector:
    def __init__(self, model_path: str = YOLO_MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list:
        results = self.model.predict(frame, classes=[0], verbose=False)
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, "person"))
        return detections


class OSNetFeatureExtractor:
    def __init__(self, model_path: str = OSNET_MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((3, OSNET_INPUT_SIZE[1], OSNET_INPUT_SIZE[0]), dtype=np.float32)
        img = cv2.resize(crop, OSNET_INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)
        return img

    def extract(self, crops: list[np.ndarray]) -> list:
        if not crops:
            return []
        batch = np.stack([self.preprocess(c) for c in crops])
        batch_tensor = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            features = self.model(batch_tensor).cpu().numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
        return features.tolist()


class PersonReIDApp:
    def __init__(self):
        self.detector = PersonDetector()
        self.extractor = OSNetFeatureExtractor()
        self.tracker = DeepSort(
            max_age=120,
            n_init=3,
            nn_budget=100,
            embedder=None,
            max_cosine_distance=0.3,
        )
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def crop_person(self, frame: np.ndarray, bbox_ltwh: list) -> np.ndarray:
        x, y, w, h = [int(v) for v in bbox_ltwh]
        x = max(0, x)
        y = max(0, y)
        return frame[y : y + h, x : x + w]

    def run(self):
        if not self.cap.isOpened():
            print("Camera not available")
            return

        cv2.namedWindow("Person Re-ID (OSNet)", cv2.WINDOW_NORMAL)

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            detections = self.detector.detect(frame)

            embeds = []
            valid_detections = []
            for det in detections:
                bbox_ltwh = det[0]
                if bbox_ltwh[2] <= 0 or bbox_ltwh[3] <= 0:
                    continue
                crop = self.crop_person(frame, bbox_ltwh)
                embeds.append(crop)
                valid_detections.append(det)

            features = self.extractor.extract(embeds)

            tracks = self.tracker.update_tracks(valid_detections, embeds=features, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = [int(v) for v in ltrb]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                id_text = f"ID {track_id}"
                (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 255, 255), -1)
                cv2.putText(frame, id_text, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

            cv2.imshow("Person Re-ID (OSNet)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = PersonReIDApp()
    app.run()
