"""
Реалтайм-тест TimeSformer (предобучен на Kinetics-400) через веб-камеру.

Установка зависимостей:
    pip install transformers torch torchvision opencv-python pillow --break-system-packages

Запуск:
    python realtime_timesformer.py

Нажми 'q' чтобы выйти.
"""

import cv2
import torch
import numpy as np
from collections import deque
from transformers import AutoImageProcessor, TimesformerForVideoClassification

# -------------------------------
# Настройки
# -------------------------------
MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
NUM_FRAMES = 8          # TimeSformer base ожидает окно из 8 кадров
INFER_EVERY_N = 5       # запускать инференс каждые N новых кадров (для скорости)
TOP_K = 3               # сколько топ-предсказаний показывать

# -------------------------------
# Загрузка модели
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

print("Загрузка модели (может занять время при первом запуске)...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = TimesformerForVideoClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

id2label = model.config.id2label
print(f"Модель загружена. Всего классов: {len(id2label)}")

# -------------------------------
# Буфер кадров (скользящее окно)
# -------------------------------
frame_buffer = deque(maxlen=NUM_FRAMES)
frame_counter = 0
last_predictions = []  # список (label, prob) для отображения между инференсами


def predict(frames_rgb_list):
    """Принимает список из NUM_FRAMES кадров в формате RGB (numpy), возвращает топ предсказания."""
    inputs = processor(images=frames_rgb_list, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # shape: (1, T, C, H, W)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)

    top_probs, top_idxs = torch.topk(probs, TOP_K)
    results = [(id2label[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_idxs)]
    return results


# -------------------------------
# Основной цикл с веб-камерой
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру. Проверь, что она подключена и доступна.")

print("Запуск. Нажми 'q' в окне видео, чтобы выйти.")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры.")
        break

    # Конвертируем BGR (OpenCV) -> RGB (нужно для модели)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_buffer.append(frame_rgb)
    frame_counter += 1

    # Запускаем инференс, когда буфер заполнен и прошло INFER_EVERY_N новых кадров
    if len(frame_buffer) == NUM_FRAMES and frame_counter % INFER_EVERY_N == 0:
        try:
            last_predictions = predict(list(frame_buffer))
        except Exception as e:
            print(f"Ошибка инференса: {e}")

    # Отрисовка предсказаний на кадре
    display_frame = frame_bgr.copy()
    y_offset = 30
    if last_predictions:
        for label, prob in last_predictions:
            text = f"{label}: {prob*100:.1f}%"
            cv2.putText(display_frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
    else:
        cv2.putText(display_frame, "Buffering...", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("TimeSformer Realtime Action Recognition (q - exit)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()