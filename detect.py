import cv2
import time
from ultralytics import YOLO

# Загружаем модель
model = YOLO("yolo26n.pt")

# Открываем камеру (0 - стандартная вебка)
cap = cv2.VideoCapture(0)

# Устанавливаем разрешение (можно уменьшить для еще большей скорости)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

print("Нажми 'q' для выхода.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Замеряем время работы нейросети
    start_infer = time.time()
    
    # Запускаем детекцию
    results = model(frame, verbose=False)
    
    infer_time = (time.time() - start_infer) * 1000  # в миллисекундах

    # Рисуем квадраты
    annotated_frame = results[0].plot()

    # Считаем общий FPS (с учетом захвата камеры и отрисовки)
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Выводим статистику прямо на картинку
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Inference: {infer_time:.1f} ms", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLO26n Real-Time", annotated_frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()