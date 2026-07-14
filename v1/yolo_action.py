"""
Multi-person action recognition.
Pipeline: YOLO detects → per-person crop → MediaPipe Holistic → LSTM with 15-frame buffer
Processes one person at a time (sequential), classifies after 15 frames, moves to next.
Uses a sliding-window buffer — once classified, a person's label stays visible forever
with periodic re-classification (round-robin) to keep predictions fresh.
"""

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# MediaPipe aliases (same pattern as skeleton_trainer.py)
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Constants ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-detect YOLO model: prefer yolo11n (faster), fallback to yolo26n
_YOLO_FAST = os.path.join(MODEL_DIR, "yolo11n.pt")
_YOLO_ACCURATE = os.path.join(MODEL_DIR, "yolo26n.pt")
YOLO_MODEL_PATH = _YOLO_FAST if os.path.exists(_YOLO_FAST) else _YOLO_ACCURATE

HOLISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "holistic_landmarker.task")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "skeleton_model.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "skeleton_config.json")

BBOX_PAD_FACTOR = 0.2  # 20% padding on each side of bbox
FIXED_CROP_SIZE = 256  # Resize crop to this size before MediaPipe to avoid crash

# COCO class IDs → action class indices
# See: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
# Action classes: 0=drinking, 1=looking_at_camera, 2=playing_phone, 3=reading, 4=walking, 5=writing
COCO_TO_ACTION = {
    67: 2,   # cell phone → playing_phone
    73: 3,   # book → reading
    41: 0,   # cup → drinking
    39: 0,   # bottle → drinking
    40: 0,   # wine glass → drinking
    63: 3,   # laptop → reading
}

# MediaPipe skeleton connections for drawing (same as skeleton_trainer.py)
_POSE_CONNS = frozenset([
    (0,1),(0,4),(1,2),(2,3),(3,7),(4,5),(5,6),(6,8),(9,10),(11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),(16,18),
    (16,20),(16,22),(17,19),(18,20),(23,24),(23,25),(24,26),(25,27),
    (26,28),(27,29),(27,31),(28,30),(28,32),(29,31),(30,32),
])
_HAND_CONNS = frozenset([
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
    (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),
])


def draw_skeleton_on_crop(crop_rgb: np.ndarray, landmarks_225: np.ndarray) -> np.ndarray:
    """Draw MediaPipe skeleton on a crop image given 225-dim landmarks.

    Args:
        crop_rgb: RGB crop image (H, W, 3)
        landmarks_225: 225-dim landmark vector (pose 99 + lh 63 + rh 63)

    Returns:
        BGR image with skeleton drawn
    """
    img = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = img.shape[:2]

    # Parse landmarks
    pose = landmarks_225[:99].reshape(33, 3)
    lh = landmarks_225[99:162].reshape(21, 3)
    rh = landmarks_225[162:].reshape(21, 3)

    def draw_pts(pts: np.ndarray, conns: frozenset,
                 color: tuple[int, int, int]) -> None:
        """Draw landmarks and connections on image."""
        coords: list[tuple[int, int]] = []
        for p in pts:
            x = int(p[0] * w)
            y = int(p[1] * h)
            coords.append((x, y))
            cv2.circle(img, (x, y), 2, color, -1)
        for i, j in conns:
            if i < len(coords) and j < len(coords):
                xi, yi = coords[i]
                xj, yj = coords[j]
                if (xi > 0 or yi > 0) and (xj > 0 or yj > 0):
                    cv2.line(img, coords[i], coords[j], color, 1)

    draw_pts(pose, _POSE_CONNS, (0, 255, 0))
    draw_pts(lh, _HAND_CONNS, (255, 0, 0))
    draw_pts(rh, _HAND_CONNS, (0, 0, 255))

    return img


@dataclass
class Track:
    """Tracks a single person across frames with landmark buffer."""
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (absolute, padded)
    buf: deque = field(default_factory=lambda: deque(maxlen=30))  # landmark buffer
    last_prediction: str = "..."
    last_probability: float = 0.0
    classified: bool = False
    age: int = 0        # frames since creation
    missed: int = 0     # frames since last YOLO match
    buffer_count: int = 0  # how many frames added to buffer


class SkeletonLSTM(nn.Module):
    """Same architecture as in skeleton_trainer.py."""
    def __init__(self, input_dim: int = 225, hidden: int = 128,
                 layers: int = 2, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM and classifier.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out)


def load_lstm() -> tuple[nn.Module, list[str], int]:
    """Load trained LSTM model from checkpoint.

    Returns:
        Tuple of (model, classes_list, seq_len)

    Raises:
        FileNotFoundError: If model checkpoint does not exist.
    """
    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(
            f"Модель не найдена: {LSTM_MODEL_PATH}. Сначала выполните train."
        )

    ckpt = torch.load(LSTM_MODEL_PATH, map_location='cpu')
    classes: list[str] = ckpt['classes']
    seq_len: int = ckpt.get('seq_len', 15)
    input_dim: int = ckpt.get('input_dim', 225)

    model = SkeletonLSTM(input_dim, 128, 2, len(classes), 0.5)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, classes, seq_len


class HolisticProcessor:
    """Manages a single MediaPipe Holistic landmarker instance."""

    def __init__(self) -> None:
        self.model_path = HOLISTIC_MODEL_PATH
        self.landmarker = None

    def __enter__(self) -> 'HolisticProcessor':
        self.landmarker = HolisticLandmarker.create_from_options(
            HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.IMAGE,
            )
        )
        return self

    def __exit__(self, *args: object) -> None:
        if self.landmarker:
            self.landmarker.close()

    def extract(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Extract 225-dim landmarks from a full-frame or cropped RGB image.

        Args:
            frame_rgb: RGB image (H, W, 3)

        Returns:
            Concatenated 225-dim array (pose 99 + left hand 63 + right hand 63)
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)

        pose = np.zeros(33 * 3)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks):
                pose[i*3:i*3+3] = [lm.x, lm.y, lm.z]

        lh = np.zeros(21 * 3)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks):
                lh[i*3:i*3+3] = [lm.x, lm.y, lm.z]

        rh = np.zeros(21 * 3)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks):
                rh[i*3:i*3+3] = [lm.x, lm.y, lm.z]

        return np.concatenate([pose, lh, rh])


def crop_and_pad(frame: np.ndarray, bbox: tuple[int, int, int, int],
                 pad_factor: float = 0.2) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop bbox from frame with padding, clipped to frame boundaries.

    Args:
        frame: Input BGR frame (H, W, 3)
        bbox: (x1, y1, x2, y2) absolute coordinates
        pad_factor: Fraction of bbox size to add as padding on each side

    Returns:
        Tuple of (cropped_image, padded_bbox)
        - cropped_image: The cropped (and padded) image region
        - padded_bbox: (px1, py1, px2, py2) absolute coordinates of padded bbox
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * pad_factor), int(bh * pad_factor)

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)

    return frame[py1:py2, px1:px2], (px1, py1, px2, py2)


def convert_crop_to_fullframe(landmarks_225: np.ndarray,
                               crop_w: int, crop_h: int,
                               bbox_x1: int, bbox_y1: int,
                               frame_w: int, frame_h: int) -> np.ndarray:
    """Convert a 225-dim landmark vector from crop-relative to full-frame-relative.

    MediaPipe returns landmarks in [0,1] of the resized crop. The LSTM model was
    trained on landmarks in [0,1] of the full frame. This function converts x,y
    coordinates back to full-frame normalized [0,1] space, and scales z
    proportionally.

    Args:
        landmarks_225: 225-dim array [pose(99), lh(63), rh(63)] where x,y are in
                       [0,1] of the crop
        crop_w: Width of the crop BEFORE resize to 256x256 (pixels)
        crop_h: Height of the crop BEFORE resize to 256x256 (pixels)
        bbox_x1: Top-left x of padded bbox in full frame pixels
        bbox_y1: Top-left y of padded bbox in full frame pixels
        frame_w: Full frame width (pixels)
        frame_h: Full frame height (pixels)

    Returns:
        Updated 225-dim array with coordinates in full-frame [0,1] space
    """
    result = landmarks_225.copy()
    n_pose = 33
    n_lhand = 21
    n_rhand = 21

    # Pose landmarks: 33 points × 3 = 99
    for i in range(n_pose):
        idx = i * 3
        result[idx] = (landmarks_225[idx] * crop_w + bbox_x1) / frame_w
        result[idx + 1] = (landmarks_225[idx + 1] * crop_h + bbox_y1) / frame_h
        result[idx + 2] = landmarks_225[idx + 2] * (crop_w / frame_w)

    # Left hand landmarks: 21 points × 3 = 63
    offset = n_pose * 3
    for i in range(n_lhand):
        idx = offset + i * 3
        result[idx] = (landmarks_225[idx] * crop_w + bbox_x1) / frame_w
        result[idx + 1] = (landmarks_225[idx + 1] * crop_h + bbox_y1) / frame_h
        result[idx + 2] = landmarks_225[idx + 2] * (crop_w / frame_w)

    # Right hand landmarks: 21 points × 3 = 63
    offset = n_pose * 3 + n_lhand * 3
    for i in range(n_rhand):
        idx = offset + i * 3
        result[idx] = (landmarks_225[idx] * crop_w + bbox_x1) / frame_w
        result[idx + 1] = (landmarks_225[idx + 1] * crop_h + bbox_y1) / frame_h
        result[idx + 2] = landmarks_225[idx + 2] * (crop_w / frame_w)

    return result


def compute_iou(a: tuple[int, int, int, int],
                b: tuple[int, int, int, int]) -> float:
    """Intersection-over-Union of two bounding boxes.

    Args:
        a: First bbox (x1, y1, x2, y2)
        b: Second bbox (x1, y1, x2, y2)

    Returns:
        IoU value in [0.0, 1.0]
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_object_boost(person_bbox: tuple[int, int, int, int],
                         object_detections: list[tuple[int, int, int, int, int]],
                         coco_to_action: dict[int, int],
                         num_classes: int) -> np.ndarray:
    """Check which objects overlap with person bbox, return boost vector.

    For each object whose center falls inside the person bounding box,
    computes an overlap-ratio-based boost for the corresponding action class.

    Args:
        person_bbox: (x1, y1, x2, y2) of the person in absolute pixel coords
        object_detections: List of (x1, y1, x2, y2, cls_id) for detected objects
        coco_to_action: Mapping from COCO class ID to action class index
        num_classes: Total number of action classes

    Returns:
        ndarray of shape (num_classes,) with boost values in [0.0, 2.0]
    """
    boost = np.zeros(num_classes)
    px1, py1, px2, py2 = person_bbox

    for ox1, oy1, ox2, oy2, cls_id in object_detections:
        # Check if object center is inside person bbox
        cx = (ox1 + ox2) / 2
        cy = (oy1 + oy2) / 2
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            action_idx = coco_to_action[cls_id]
            # Compute intersection between object and person bbox
            ix1 = max(px1, ox1)
            iy1 = max(py1, oy1)
            ix2 = min(px2, ox2)
            iy2 = min(py2, oy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            person_area = (px2 - px1) * (py2 - py1)
            if person_area > 0:
                overlap_ratio = inter / person_area
                # Boost based on how much of the person's bbox the object occupies
                boost[action_idx] = max(boost[action_idx], overlap_ratio * 2.0)

    return boost


def run_pipeline(video_path: Optional[str] = None,
                 camera_id: int = 0,
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.3,
                 max_missed: int = 30,
                 display_scale: float = 1.0) -> None:
    """Main pipeline: sequential per-person action recognition (sliding window).

    Process flow:
      1. YOLO detects all people → assigns/updates track IDs
      2. Picks an unclassified person (by track ID, lowest first)
      3. For that person: crop → MediaPipe Holistic → append 1 frame to sliding buffer
      4. After 15 frames → LSTM predicts → mark as classified (label stays forever)
      5. Move to next unclassified person
      6. When all classified → rotate round-robin through all people, re-classifying
         every 15 frames via sliding window (label never disappears)

    Args:
        video_path: Path to video file, or None for webcam
        camera_id: Camera device ID (used when video_path is None)
        conf_threshold: YOLO confidence threshold
        iou_threshold: IoU threshold for track association
        max_missed: Max frames a track can be invisible before removal
        display_scale: Window scale factor (0.5 = half size)
    """
    # --- Load models ---
    print("Загрузка YOLO...")
    yolo = YOLO(YOLO_MODEL_PATH)

    print("Загрузка LSTM...")
    lstm_model, classes, seq_len = load_lstm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.to(device)
    print(f"  Классы: {classes}, seq_len={seq_len}")

    # --- Open video source ---
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    if not cap.isOpened():
        print("Ошибка открытия камеры/видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- State ---
    tracks: dict[int, Track] = {}
    next_id = 1
    current_target_id: Optional[int] = None
    frame_count = 0
    display_mode: int = 1  # 1=full pipeline, 2=YOLO only, 3=YOLO+MP skeleton

    print(f"Запуск. Кадры: {frame_w}x{frame_h} @ {fps}fps")
    print("Controls: Tab=режим, Q=выход, R=сброс")

    holistic_processor = HolisticProcessor()

    with holistic_processor as hp:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # --- Step 1: YOLO detection (always runs for all modes) ---
            results = yolo(frame, imgsz=320, conf=conf_threshold,
                          verbose=False)

            person_detections: list[tuple[int, int, int, int, float]] = []
            object_detections: list[tuple[int, int, int, int, int]] = []
            all_objects: list[tuple[int, int, int, int, int, float]] = []
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    if cls_id == 0 and conf >= conf_threshold:
                        person_detections.append((x1, y1, x2, y2, conf))
                    elif conf >= conf_threshold:
                        all_objects.append((x1, y1, x2, y2, cls_id, conf))
                        if cls_id in COCO_TO_ACTION:
                            object_detections.append((x1, y1, x2, y2, cls_id))

            # detections alias for backward-compat with track matching
            detections = person_detections
            skel_img: Optional[np.ndarray] = None

            # --- Mode-specific processing ---
            if display_mode in (1, 3):

                # --- Step 2: Update tracks (IoU matching) ---
                matched_det: set[int] = set()
                det_to_track: dict[int, int] = {}

                for tid in list(tracks.keys()):
                    track = tracks[tid]
                    best_iou = iou_threshold
                    best_det = -1
                    for di, det in enumerate(detections):
                        if di in matched_det:
                            continue
                        iou = compute_iou(track.bbox, det[:4])
                        if iou > best_iou:
                            best_iou = iou
                            best_det = di

                    if best_det >= 0:
                        matched_det.add(best_det)
                        det_to_track[best_det] = tid
                        track.bbox = detections[best_det][:4]
                        track.missed = 0
                        track.age += 1
                    else:
                        track.missed += 1
                        track.age += 1

                # Create new tracks for unmatched detections
                for di, det in enumerate(detections):
                    if di not in matched_det:
                        track = Track(
                            track_id=next_id,
                            bbox=det[:4],
                            buf=deque(maxlen=seq_len),
                        )
                        tracks[next_id] = track
                        det_to_track[di] = next_id
                        next_id += 1

                # Remove stale tracks
                for tid in list(tracks.keys()):
                    if tracks[tid].missed > max_missed:
                        del tracks[tid]

                # --- Step 3: Pick target person (prefer unclassified, else rotate) ---
                unclassified = [tid for tid in sorted(tracks.keys())
                                if not tracks[tid].classified]
                if unclassified:
                    new_target = unclassified[0]
                    if current_target_id != new_target:
                        current_target_id = new_target
                        if display_mode == 1:
                            print(f"  → Начинаю обработку Person {current_target_id}")
                elif tracks:
                    sorted_ids = sorted(tracks.keys())
                    if current_target_id is None or current_target_id not in tracks:
                        current_target_id = sorted_ids[0]
                    else:
                        idx = (sorted_ids.index(current_target_id) + 1) % len(sorted_ids)
                        current_target_id = sorted_ids[idx]
                    tracks[current_target_id].buf.clear()
                    tracks[current_target_id].buffer_count = 0

                # --- Step 4: Process current target (sliding window) ---
                if current_target_id is not None and current_target_id in tracks:
                    target = tracks[current_target_id]
                    crop, padded_bbox = crop_and_pad(frame, target.bbox, BBOX_PAD_FACTOR)

                    if crop.size > 0 and min(crop.shape[:2]) >= 30:
                        # Save original crop dimensions before resize (needed for
                        # converting landmark coordinates from crop-space to full-frame-space)
                        orig_crop_h, orig_crop_w = crop.shape[:2]

                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        # Resize to fixed size to avoid MediaPipe SegmentationSmoothingCalculator crash
                        # (crop size varies with bbox, causing inconsistent internal state dimensions)
                        crop_rgb_resized = cv2.resize(crop_rgb, (FIXED_CROP_SIZE, FIXED_CROP_SIZE))
                        landmarks = hp.extract(crop_rgb_resized)

                        # Convert coordinates from crop-[0,1] space to full-frame-[0,1] space,
                        # because the LSTM was trained on full-frame-normalized landmarks
                        px1, py1, _, _ = padded_bbox
                        frame_h, frame_w = frame.shape[:2]
                        landmarks = convert_crop_to_fullframe(
                            landmarks, orig_crop_w, orig_crop_h,
                            px1, py1, frame_w, frame_h,
                        )

                        # Mode 1: LSTM prediction
                        if display_mode == 1:
                            target.buf.append(landmarks)
                            target.buffer_count += 1

                            if len(target.buf) == seq_len:
                                inp = torch.FloatTensor(
                                    np.array(target.buf)
                                ).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    out = lstm_model(inp)
                                probs = torch.softmax(out, dim=1)[0].cpu().numpy()

                                # Apply object-detection-based action boosting
                                object_boost = compute_object_boost(
                                    target.bbox, object_detections,
                                    COCO_TO_ACTION, len(classes),
                                )
                                boosted = probs * (1.0 + object_boost)
                                boosted = boosted / boosted.sum()

                                pred = int(boosted.argmax().item())
                                prob = float(boosted[pred].item())

                                new_label = f"{classes[pred]} ({prob*100:.0f}%)"

                                if target.last_prediction != new_label:
                                    print(f"  ↻ Person {current_target_id}: "
                                          f"{target.last_prediction} → {new_label}")

                                target.last_prediction = new_label
                                target.last_probability = prob

                                if not target.classified:
                                    target.classified = True
                                    print(f"  ✓ Person {current_target_id}: "
                                          f"{target.last_prediction}")
                                    current_target_id = None

                        # Mode 3: skeleton display
                        if display_mode == 3:
                            skel_img = draw_skeleton_on_crop(crop_rgb, landmarks)

            # --- Step 5: Draw everything (mode-specific) ---
            display = frame.copy()

            if display_mode == 1:
                # Full pipeline: bboxes + labels + queue info
                for tid, track in sorted(tracks.items()):
                    x1, y1, x2, y2 = track.bbox

                    if track.classified:
                        color = (0, 255, 0)      # green — classified
                    elif tid == current_target_id:
                        color = (0, 255, 255)    # yellow — being processed
                    else:
                        color = (128, 128, 128)  # gray — waiting

                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                    if track.classified:
                        cv2.putText(display, track.last_prediction,
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 2)
                        cv2.putText(display, f"ID:{tid}",
                                    (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 1)
                    elif tid == current_target_id:
                        progress = f"ID:{tid} {track.buffer_count}/{seq_len}"
                        cv2.putText(display, progress,
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 1)
                    else:
                        cv2.putText(display, f"ID:{tid} в очереди",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 1)

            elif display_mode == 2:
                # YOLO only: draw all person bboxes
                for x1, y1, x2, y2, conf in person_detections:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, f"person {conf:.0%}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)

                # Draw all object bboxes with COCO class names
                coco_names = results[0].names if len(results) > 0 else {}
                for ox1, oy1, ox2, oy2, cls_id, conf in all_objects:
                    name = coco_names.get(cls_id, f"cls_{cls_id}")
                    cv2.rectangle(display, (ox1, oy1), (ox2, oy2), (255, 255, 0), 2)
                    cv2.putText(display, f"{name} {conf:.0%}",
                                (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 0), 1)

            elif display_mode == 3:
                # YOLO + skeleton: bboxes + skeleton corner overlay
                for tid, track in sorted(tracks.items()):
                    x1, y1, x2, y2 = track.bbox
                    color = (0, 255, 0) if tid == current_target_id else (255, 255, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{tid}" + (" skeleton" if tid == current_target_id else "")
                    cv2.putText(display, label,
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1)

                if skel_img is not None:
                    skel_resized = cv2.resize(skel_img, (256, 256))
                    x_offset = frame_w - 266
                    y_offset = 10
                    if x_offset > 0 and y_offset + 256 <= frame_h:
                        display[y_offset:y_offset+256, x_offset:x_offset+256] = skel_resized
                        cv2.rectangle(display, (x_offset, y_offset),
                                      (x_offset+256, y_offset+256), (0, 255, 255), 1)
                        cv2.putText(display, "skeleton",
                                    (x_offset, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 255), 1)

            # Info overlay
            mode_names = {1: "YOLO+MP+LSTM", 2: "YOLO only", 3: "YOLO+MP skeleton"}
            info = f"Mode {display_mode}: {mode_names[display_mode]} | People: {len(tracks)}"
            if display_mode == 1:
                info += f" | Processing: ID {current_target_id}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.putText(display, "Tab: режим | Q: выход | R: сброс",
                        (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100, 100, 100), 1)

            # Show frame
            cv2.namedWindow("Multi-Person Action", cv2.WINDOW_NORMAL)
            if display_scale != 1.0:
                dw = int(frame_w * display_scale)
                dh = int(frame_h * display_scale)
                cv2.resizeWindow("Multi-Person Action", dw, dh)
            cv2.imshow("Multi-Person Action", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 9:  # Tab key
                display_mode = (display_mode % 3) + 1
                print(f"  Режим: {mode_names[display_mode]}")
            elif key == ord('r'):
                tracks.clear()
                next_id = 1
                current_target_id = None
                if display_mode == 1:
                    print("  ↺ Сброс")

    cap.release()
    cv2.destroyAllWindows()


# ====== MAIN ======
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Multi-person action recognition pipeline"
    )
    parser.add_argument('--video', type=str, default=None,
                        help='Путь к видеофайлу (по умолчанию камера)')
    parser.add_argument('--camera', type=int, default=0,
                        help='ID камеры')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Порог уверенности YOLO (default: 0.5)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Масштаб окна (0.5 = половина)')
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        camera_id=args.camera,
        conf_threshold=args.conf,
        display_scale=args.scale,
    )
