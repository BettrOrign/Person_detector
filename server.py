import asyncio
import websockets
import json
from datetime import datetime

async def listen():
    uri = "ws://100.79.205.49:8765"
    async with websockets.connect(uri) as websocket:
        print(f"Подключено к {uri}")
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Превращаем timestamp в читаемое время
                dt = datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S.%f')[:-3]
                
                print(f"\n[ {dt} ] Кадр: {data['frame_id']} | Объектов: {len(data['detections'])}")
                
                for det in data['detections']:
                    cls = det['class']
                    conf = det['conf']
                    # Координаты: [x1, y1, x2, y2]
                    bbox = [round(x, 1) for x in det['bbox']]
                    
                    print(f"  - {cls} ({conf:.2f}) Позиция: {bbox}")
                    
            except websockets.ConnectionClosed:
                print("Соединение закрыто сервером")
                break
            except Exception as e:
                print(f"Ошибка: {e}")
                break

if __name__ == '__main__':
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("\nОстановка клиента...")