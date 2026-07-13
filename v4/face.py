import logging
from typing import Tuple

import cv2
import numpy as np
import onnxruntime
from skimage.transform import SimilarityTransform

logger = logging.getLogger(__name__)

REFERENCE_ALIGNMENT = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32
)


def _align_face(image: np.ndarray, landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    alignment = REFERENCE_ALIGNMENT * ratio
    alignment[:, 0] += diff_x
    M = SimilarityTransform.from_estimate(landmarks, alignment).params[0:2, :]
    return cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)


def _distance2bbox(points: np.ndarray, distance: np.ndarray, max_shape: Tuple[int, int] | None = None) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray, max_shape: Tuple[int, int] | None = None) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640),
                 conf_thres: float = 0.5, iou_thres: float = 0.4):
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        self.mean = 127.5
        self.std = 128.0
        self.center_cache = {}
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.input_names = [x.name for x in self.session.get_inputs()]
        logger.info(f"Loaded SCRFD from {model_path}")

    def detect(self, image: np.ndarray, max_num: int = 0) -> Tuple[np.ndarray, np.ndarray | None]:
        width, height = self.input_size
        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = height / width
        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((height, width, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized

        blob = cv2.dnn.blobFromImage(det_img, 1.0 / self.std, (width, height),
                                     (self.mean, self.mean, self.mean), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_names[0]: blob})

        scores_list, bboxes_list, kpss_list = [], [], []
        input_h, input_w = blob.shape[2], blob.shape[3]
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.fmc] * stride
            kps_preds = outputs[idx + self.fmc * 2] * stride if self.use_kps else None

            h, w = input_h // stride, input_w // stride
            key = (h, w, stride)
            if key in self.center_cache:
                centers = self.center_cache[key]
            else:
                centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    centers = np.stack([centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = centers

            pos = np.where(scores >= self.conf_thres)[0]
            bboxes = _distance2bbox(centers, bbox_preds)
            scores_list.append(scores[pos])
            bboxes_list.append(bboxes[pos])
            if self.use_kps:
                kpss = _distance2kps(centers, kps_preds).reshape((-1, 5, 2))
                kpss_list.append(kpss[pos])

        scores = np.vstack(scores_list).ravel()
        order = scores.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale if self.use_kps else None

        det = np.hstack((bboxes, scores.reshape(-1, 1))).astype(np.float32)
        det = det[order]
        keep = self._nms(det)
        det = det[keep]
        if kpss is not None:
            kpss = kpss[order][keep]

        if 0 < max_num < len(det):
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            cx, cy = image.shape[1] / 2, image.shape[0] / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - cx, (det[:, 1] + det[:, 3]) / 2 - cy])
            values = area - np.sum(np.power(offsets, 2.0), 0) * 2.0
            det = det[np.argsort(values)[::-1][:max_num]]
            if kpss is not None:
                kpss = kpss[np.argsort(values)[::-1][:max_num]]

        return det, kpss

    def _nms(self, dets: np.ndarray) -> list[int]:
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= self.iou_thres)[0] + 1]
        return keep


class ArcFace:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.input_size = (112, 112)
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.embedding_size = self.session.get_outputs()[0].shape[1]
        logger.info(f"Loaded ArcFace from {model_path} (embedding_size={self.embedding_size})")

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        aligned = _align_face(image, landmarks, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, self.input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        embedding = self.session.run(self.output_names, {self.input_name: blob})[0]
        return embedding.flatten()
