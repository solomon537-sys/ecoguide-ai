import cv2
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model
from pathlib import Path

# ---------- Настройки ----------
CONFIDENCE_THRESHOLD = 0.80   # минимальная уверенность
SMOOTHING_FRAMES = 10         # сколько кадров усредняем
PREDICT_EVERY = 2             # предсказывать каждый N-й кадр
MODEL_SIZE = 224

# Соответствие категорий и контейнеров: (текст, цвет BGR)
BINS = {
    "plastic": ("PLASTIC BIN", (0, 200, 255)),
    "paper":   ("PAPER BIN",   (255, 150, 0)),
    "metal":   ("METAL BIN",   (200, 200, 200)),
}
IGNORED = {"background", "empty", "nothing", "фон"}  # классы «ничего нет»

# ---------- Загрузка ----------
folder = Path(__file__).resolve().parent
model = load_model(str(folder / "keras_model.h5"), compile=False)

with open(folder / "labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip().split(" ", 1)[-1] for line in f if line.strip()]

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Camera not found!")

print("EcoSort AI started! Press Q to exit")


def center_square(frame):
    """Вырезает центральный квадрат, чтобы не искажать пропорции."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side], (x0, y0, side)


def preprocess(square):
    img = cv2.resize(square, (MODEL_SIZE, MODEL_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1
    return np.expand_dims(img, axis=0)


history = deque(maxlen=SMOOTHING_FRAMES)
frame_count = 0
prev_time = time.time()
fps = 0.0
avg_prediction = None

while True:
    success, frame = camera.read()
    if not success:
        break

    square, (x0, y0, side) = center_square(frame)

    # Предсказание не на каждом кадре — быстрее
    if frame_count % PREDICT_EVERY == 0:
        pred = model(preprocess(square), training=False).numpy()[0]
        history.append(pred)
        avg_prediction = np.mean(history, axis=0)
    frame_count += 1

    # Результат
    if avg_prediction is not None and len(avg_prediction) == len(labels):
        index = int(np.argmax(avg_prediction))
        confidence = float(avg_prediction[index])
        category = labels[index].lower()

        if category in IGNORED:
            result, color = "SHOW AN OBJECT", (180, 180, 180)
        elif confidence < CONFIDENCE_THRESHOLD:
            result, color = "NOT SURE - CHECK OBJECT", (0, 0, 255)
        elif category in BINS:
            result, color = BINS[category]
        else:
            result, color = "OTHER OBJECT", (0, 165, 255)
    else:
        result, color, confidence = "LOADING...", (180, 180, 180), 0.0

    # Рамка области, которую видит модель
    cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (255, 255, 255), 1)

    # Зеркальное отображение (только для показа)
    frame = cv2.flip(frame, 1)

    # FPS
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1 / max(now - prev_time, 1e-6))
    prev_time = now

    # Текст с подложкой
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
    cv2.putText(frame, result, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 120, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("EcoSort AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
