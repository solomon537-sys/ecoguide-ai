"""EcoGuide AI: FastAPI-бэкенд.

POST /predict  - принимает кадр, возвращает категорию и подсказку
GET  /rules    - правила сортировки (rules.json)
GET  /stats    - простая статистика событий
GET  /         - страница с камерой (frontend/index.html)
Кадры нигде не сохраняются, в базу пишутся только обезличенные события.
"""
import io
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "keras_model.h5"
LABELS_PATH = BASE_DIR / "model" / "labels.txt"
RULES_PATH = BASE_DIR / "rules.json"
DB_PATH = Path(os.environ.get("DB_PATH", BASE_DIR / "events.db"))
MODEL_SIZE = 224

RULES = json.loads(RULES_PATH.read_text(encoding="utf-8"))
CONFIDENT = RULES["thresholds"]["confident"]
UNSURE = RULES["thresholds"]["unsure"]

app = FastAPI(title="EcoGuide AI")

model = None
labels = []


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, label TEXT, confidence REAL,
                status TEXT, fraction TEXT, latency_ms REAL)"""
        )


def log_event(label, confidence, status, fraction, latency_ms):
    try:
        with sqlite3.connect(DB_PATH) as db:
            db.execute(
                "INSERT INTO events (ts, label, confidence, status, fraction, latency_ms)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), label, confidence, status, fraction, latency_ms),
            )
    except sqlite3.Error:
        pass  # статистика не должна ломать распознавание


@app.on_event("startup")
def startup():
    global model, labels
    from tensorflow.keras.models import load_model  # тяжёлый импорт, грузим при старте

    model = load_model(str(MODEL_PATH), compile=False)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        # формат Teachable Machine: "0 Paper"
        labels = [line.strip().split(" ", 1)[-1] for line in f if line.strip()]
    init_db()


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    side = min(w, h)  # центральный квадрат, как в EcoGuide.py
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((MODEL_SIZE, MODEL_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1
    return np.expand_dims(arr, axis=0)


def build_answer(label: str, confidence: float) -> dict:
    rule = RULES["classes"].get(label)
    if rule is None:
        return {"status": "unsure", "message": "Не удалось определить предмет",
                "fraction": None, "color_hex": None, "tip": ""}

    fraction = RULES["fractions"][rule["fraction"]]
    base = {
        "item": rule["name"],
        "fraction": rule["fraction"],
        "fraction_name": fraction["name"],
        "color": fraction["color"],
        "color_hex": fraction["hex"],
        "tip": rule["tip"],
    }
    if confidence >= CONFIDENT:
        base["status"] = "ok"
        base["message"] = (f"{rule['name']} → {fraction['name']} ({rule['fraction']}). "
                           f"Бросьте в {fraction['color']} контейнер")
    elif confidence >= UNSURE:
        base["status"] = "confirm"
        base["message"] = (f"Похоже на: {rule['name']} ({confidence * 100:.0f}%). "
                           "Подтвердите или выберите вручную")
    else:
        base["status"] = "unsure"
        base["message"] = "Не удалось точно определить материал. Выберите вручную"
        base["fraction"] = None
        base["color_hex"] = None
    return base


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель ещё загружается")
    data = await file.read()
    try:
        x = preprocess(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    t0 = time.perf_counter()
    probs = model(x, training=False).numpy()[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    label = labels[idx].lower() if idx < len(labels) else "unknown"

    answer = build_answer(label, confidence)
    answer.update({"label": label, "confidence": round(confidence, 3),
                   "latency_ms": round(latency_ms, 1)})
    log_event(label, confidence, answer["status"], answer.get("fraction"), latency_ms)
    return answer


@app.get("/rules")
def get_rules():
    return RULES


@app.get("/stats")
def stats():
    with sqlite3.connect(DB_PATH) as db:
        total = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        by_fraction = db.execute(
            "SELECT COALESCE(fraction, 'нет'), COUNT(*) FROM events GROUP BY 1").fetchall()
        by_status = db.execute(
            "SELECT status, COUNT(*) FROM events GROUP BY 1").fetchall()
        avg_latency = db.execute("SELECT AVG(latency_ms) FROM events").fetchone()[0]
    return {
        "total": total,
        "by_fraction": dict(by_fraction),
        "by_status": dict(by_status),
        "avg_model_latency_ms": round(avg_latency, 1) if avg_latency else None,
    }


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": model is not None}


# Фронтенд отдаём с того же сервера: одна ссылка на всё приложение
app.mount("/", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")
