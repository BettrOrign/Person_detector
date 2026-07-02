import cv2
from ultralytics import YOLO

# --- НАСТРОЙКИ ---
model = YOLO("yolo26n-pose.pt")
input_video_path = "./input_video.mp4" # Укажи путь к своему видео

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(f"Ошибка: Не удалось открыть видео {input_video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Защита от нулевого FPS
if fps <= 0 or fps > 120: fps = 30.0 

# Задержка для cv2.waitKey в миллисекундах (чтобы видео шло в реальном времени)
delay_ms = int(1000 / fps)

print(f"▶️ Запуск YOLO26n-Pose на видео {width}x{height} @ {fps:.1f} FPS...")
print("Нажми клавишу 'q' в окне с видео, чтобы закрыть его.")

# Создаем обычное системное окно (можно растягивать мышкой)
cv2.namedWindow("YOLO26n Pose", cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Инференс модели ПОЗЫ
    results = model(frame, verbose=False)
    
    # results[0].plot() автоматически нарисует красивый скелет и точки
    annotated_frame = results[0].plot()

    # ---------------------------------------------------------
    # ЗАГОВОР НА БУДУЩЕЕ: Как достать координаты суставов.
    # Когда будем делать анализ действий, эти данные пойдут в другую нейросеть.
    # ---------------------------------------------------------
    if results[0].keypoints is not None:
        # shape: [кол-во_людей, 17_точек_скелета, x/y]
        keypoints = results[0].keypoints.xy.cpu().numpy() 
        
        for person_id, kps in enumerate(keypoints):
            # kps[0] - нос, kps[5] - левое плечо, kps[11] - левое бедро и т.д.
            # Мы пока ничего с ними не делаем, просто собираем.
            pass
    # ---------------------------------------------------------

    # Показываем кадр в окне (OpenCV сам рендерит матрицу пикселей на экран)
    cv2.imshow("YOLO26n Pose", annotated_frame)
    
    # Ждем нажатия кнопки. Задержка = delay_ms (соответствует FPS видео)
    # Если нажата клавиша 'q' - выходим из цикла
    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        break

# Обязательно освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
print("✅ Готово! Скелеты построены.")