import argparse
import json
import os
import time
from collections import deque
from glob import glob

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

_POSE_CONNS = frozenset([
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)])
_HAND_CONNS = frozenset([
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)])

def _draw_landmarks(img, landmarks, connections, color=(0,255,0), mirror=False):
    h, w, _ = img.shape
    pts = [None]*len(landmarks)
    for i, lm in enumerate(landmarks):
        x = int((1 - lm.x)*w) if mirror else int(lm.x*w)
        y = int(lm.y*h)
        pts[i] = (x, y)
        cv2.circle(img, (x, y), 2, color, -1)
    for i, j in connections:
        if pts[i] and pts[j]:
            cv2.line(img, pts[i], pts[j], color, 1)

SEQUENCE_LENGTH = 30
LANDMARK_DIM = 225
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MODEL_DIR, 'skeleton_data')
CONFIG_PATH = os.path.join(MODEL_DIR, 'skeleton_config.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'skeleton_model.pth')

HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
MODEL_FILE = os.path.join(MODEL_DIR, 'holistic_landmarker.task')


def init_holistic():
    return HolisticLandmarker.create_from_options(HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=VisionRunningMode.IMAGE,
    ))


def extract_landmarks(frame_rgb, landmarker):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = landmarker.detect(mp_image)
    pose = np.zeros(33*3)
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks]).flatten()
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks]).flatten()
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks]).flatten()
    return np.concatenate([pose, lh, rh])


def extract_skeleton_from_video(video_path, landmarker):
    cap = cv2.VideoCapture(video_path)
    seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = extract_landmarks(frame_rgb, landmarker)
        seq.append(lm)
    cap.release()
    return np.array(seq) if seq else np.zeros((1, 225))


class SkeletonLSTM(nn.Module):
    def __init__(self, input_dim=225, hidden=128, layers=2, num_classes=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out)


class SkeletonDataset(Dataset):
    def __init__(self, data_root, classes, seq_len=30, augment=False):
        self.seq_len = seq_len
        self.augment = augment
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            d = os.path.join(data_root, c)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith('.npy'):
                    self.samples.append((os.path.join(d, f), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path)
        T, D = seq.shape
        if T >= self.seq_len:
            start = np.random.randint(0, T - self.seq_len + 1)
            seq = seq[start:start+self.seq_len]
        else:
            seq = np.pad(seq, ((0, self.seq_len-T), (0, 0)), 'constant')
        if self.augment:
            seq += np.random.normal(0, 0.005, seq.shape)
        return torch.FloatTensor(seq), label


# ====== PROCESS ======
def cmd_process(data_root):
    if os.path.isfile(data_root):
        cls = os.path.basename(os.path.dirname(os.path.abspath(data_root)))
        videos = [data_root]
    else:
        classes = sorted([d for d in os.listdir(data_root)
                          if os.path.isdir(os.path.join(data_root, d))])
        videos = []
        cls_map = {}
        for cls in classes:
            cls_dir = os.path.join(data_root, cls)
            found = glob(os.path.join(cls_dir, '*.mp4')) + \
                    glob(os.path.join(cls_dir, '*.avi')) + \
                    glob(os.path.join(cls_dir, '*.mov')) + \
                    glob(os.path.join(cls_dir, '*.mkv'))
            for v in found:
                cls_map[v] = cls
            videos.extend(found)

    if not videos:
        print("Видео не найдены")
        return

    # Фильтруем только новые видео
    pending = []
    for v in videos:
        cls = cls_map.get(v, os.path.basename(os.path.dirname(os.path.abspath(v))))
        save_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(v))[0]
        out_path = os.path.join(save_dir, f"{base}.npy")
        if os.path.exists(out_path):
            print(f"  {base}.npy уже есть, пропускаю")
            continue
        pending.append((v, cls, base, out_path))

    if not pending:
        print("Новых видео нет")
        return

    ok = 0
    fail = 0
    with init_holistic() as lm:
        for v, cls, base, out_path in pending:
            print(f"  {cls}/{base}...", end=' ', flush=True)
            try:
                seq = extract_skeleton_from_video(v, lm)
                np.save(out_path, seq)
                print(f"{len(seq)} кадров")
                ok += 1
            except Exception as e:
                print(f"ОШИБКА: {e}")
                fail += 1
    print(f"Обработано: {ok} видео, ошибок: {fail}")

    # Обновляем конфиг
    dirs = sorted([d for d in os.listdir(DATA_DIR)
                   if os.path.isdir(os.path.join(DATA_DIR, d))])
    with open(CONFIG_PATH, 'w') as f:
        json.dump({'classes': dirs}, f)
    print(f"Классы в конфиге: {dirs}")


# ====== TRAIN ======
def cmd_train():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            classes = json.load(f).get('classes', [])
    else:
        classes = sorted([d for d in os.listdir(DATA_DIR)
                          if os.path.isdir(os.path.join(DATA_DIR, d))])
    classes = [c for c in classes if os.path.isdir(os.path.join(DATA_DIR, c))]

    if len(classes) < 1:
        print("Нет данных.")
        return

    print(f"Классы: {classes}")
    dataset = SkeletonDataset(DATA_DIR, classes, SEQUENCE_LENGTH, augment=True)
    print(f"Всего примеров: {len(dataset)}")

    if len(dataset) < 2:
        print("Слишком мало данных.")
        return

    train_len = max(1, int(0.8 * len(dataset)))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkeletonLSTM(LANDMARK_DIM, HIDDEN_DIM, NUM_LAYERS, len(classes), DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | TrLoss: {tr_loss/len(train_loader):.4f} "
                  f"| ValLoss: {val_loss/len(val_loader):.4f} | ValAcc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'classes': classes,
                'seq_len': SEQUENCE_LENGTH,
                'input_dim': LANDMARK_DIM,
            }, MODEL_PATH)
            print(f"  >> Сохранено (acc={val_acc:.4f})")

        if val_acc >= 0.99 and epoch >= 20:
            break

    print(f"\nЛучшая точность: {best_acc:.4f}")
    print(f"Модель: {MODEL_PATH}")


# ====== INFER ======
def cmd_infer(path=None):
    if not os.path.exists(MODEL_PATH):
        print("Модель не найдена. Сначала train.")
        return

    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    classes = ckpt['classes']
    seq_len = ckpt.get('seq_len', SEQUENCE_LENGTH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkeletonLSTM(LANDMARK_DIM, HIDDEN_DIM, NUM_LAYERS, len(classes), DROPOUT)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    print(f"Классы: {classes}")
    print("Tab - переключение режимов (pred/skeleton/both), q - выход")
    cap = cv2.VideoCapture(0 if path is None else path)
    if not cap.isOpened():
        print("Камера не открылась" if path is None else f"Видео не открылось: {path}")
        return

    buf = deque(maxlen=seq_len)
    last_pred = "..."
    show_skel = False

    with init_holistic() as lm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = lm.detect(mp_image)

            pose = np.zeros(33*3)
            lh = np.zeros(21*3)
            rh = np.zeros(21*3)
            if results.pose_landmarks:
                pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks]).flatten()
            if results.left_hand_landmarks:
                lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks]).flatten()
            if results.right_hand_landmarks:
                rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks]).flatten()
            buf.append(np.concatenate([pose, lh, rh]))

            if len(buf) == seq_len:
                with torch.no_grad():
                    inp = torch.FloatTensor(np.array(buf)).unsqueeze(0).to(device)
                    out = model(inp)
                    pred = out.argmax(1).item()
                    prob = torch.softmax(out, dim=1)[0, pred].item()
                    last_pred = f"{classes[pred]} ({prob*100:.0f}%)"

            display = frame if path else cv2.flip(frame, 1)
            mirror_draw = path is None
            if show_skel:
                if results.pose_landmarks:
                    _draw_landmarks(display, results.pose_landmarks, _POSE_CONNS, (0,255,0), mirror_draw)
                if results.left_hand_landmarks:
                    _draw_landmarks(display, results.left_hand_landmarks, _HAND_CONNS, (255,0,0), mirror_draw)
                if results.right_hand_landmarks:
                    _draw_landmarks(display, results.right_hand_landmarks, _HAND_CONNS, (0,0,255), mirror_draw)
                cv2.putText(display, "[Tab] skeleton",
                            (30, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(display, f"Action: {last_pred}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display, "[Tab] prediction",
                            (30, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Infer (q - exit)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 9:
                show_skel = not show_skel
    cap.release()
    cv2.destroyAllWindows()


# ====== MAIN ======
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Skeleton trainer")
    parser.add_argument('mode', choices=['process', 'train', 'infer'])
    parser.add_argument('--path', type=str, default=None,
                        help='Путь к папке с видео (классы подпапками) или к одному видео')
    args = parser.parse_args()

    if args.mode == 'process':
        if not args.path:
            print("Укажи --path с видео")
        else:
            cmd_process(args.path)
    elif args.mode == 'train':
        cmd_train()
    elif args.mode == 'infer':
        cmd_infer(args.path)
