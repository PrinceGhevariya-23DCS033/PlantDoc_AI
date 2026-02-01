from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import logging
import os

from config import config

# ================= LOGGING =================
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# ================= GLOBAL MODEL =================
model = None

# ================= APP LIFESPAN =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("🚀 Starting application...")

    model = load_model()

    if model is None:
        logger.warning(" Model not loaded. Prediction endpoints will not work.")
    else:
        logger.info(" Model loaded successfully")

    yield
    logger.info(" Application shutting down")

# ================= FASTAPI APP =================
app = FastAPI(
    title="Plant Disease Classification API",
    version="1.0.0",
    lifespan=lifespan
)

# ================= STATIC FILES =================
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ================= MODEL LOADING =================
def load_model():
    try:
        if os.path.exists(config.MODEL_PATH):
            logger.info(f"Loading SavedModel from {config.MODEL_PATH}")
            return tf.keras.models.load_model(config.MODEL_PATH)

        if os.path.exists(config.H5_MODEL_PATH):
            logger.info(f"Loading H5 model from {config.H5_MODEL_PATH}")
            return tf.keras.models.load_model(config.H5_MODEL_PATH)

        logger.error(" No valid model path found")
        return None

    except Exception as e:
        logger.error(f" Model loading failed: {e}")
        return None

# ================= IMAGE PREPROCESS =================
def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(config.IMAGE_SIZE)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ================= HEALTH =================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": len(config.CLASS_NAMES)
    }

# ================= ROOT =================
@app.get("/")
def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Plant Disease Classification API is running"}

# ================= PREDICT =================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    processed = preprocess_image(image)
    predictions = model.predict(processed)

    idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][idx])
    class_name = config.CLASS_NAMES[idx]

    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
    else:
        plant, disease = class_name, "Unknown"

    return {
        "prediction": {
            "class": class_name,
            "plant": plant,
            "disease": disease,
            "confidence": confidence
        }
    }

# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
