import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

print("▶️ Запуск BoT-SORT со встроенным Re-ID...")
print("Проверяем: выйди из кадра на пару секунд и вернись. Нажми 'q' для выхода.")

cv2.namedWindow("Smart Re-ID", cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Указываем наш конфиг, где включен reid: true
    results = model.track(frame, persist=True, tracker="tracker.yaml", classes=[0], verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2

            id_text = f"ID {track_id}"
            
            # Черный контур для читаемости
            cv2.putText(frame, id_text, (center_x - 30, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
            cv2.putText(frame, id_text, (center_x - 30, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

    cv2.imshow("Smart Re-ID", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()