from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
import sqlite3
import time
from datetime import datetime, timezone

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(backend_dir, ".."))

@register_keras_serializable()
class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.75, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow((1 - p_t), self.gamma)

        loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "from_logits": self.from_logits
        })
        return config

_STRIP_KWARGS = {'renorm', 'renorm_clipping', 'renorm_momentum', 'quantization_config'}
def _make_patched_init(orig_init):
    def _patched(self, *args, **kwargs):
        for k in _STRIP_KWARGS:
            kwargs.pop(k, None)
        orig_init(self, *args, **kwargs)
    return _patched
for _cls in [tf.keras.layers.Dense, tf.keras.layers.Conv2D,
             tf.keras.layers.DepthwiseConv2D, tf.keras.layers.BatchNormalization]:
    _cls.__init__ = _make_patched_init(_cls.__init__)

model_path = os.path.join(backend_dir, "breakhis_mobilenet_improved_model.keras")
try:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"FocalLoss": FocalLoss},
        compile=False
    )
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

class_names = ["benign", "malignant"]
holdout_dir = os.path.join(project_root, "holdout_test_set")

# ===================== PRODUCTION MONITORING =====================
# Process start time, used by /health to report uptime.
START_TIME = time.time()

# SQLite is used because there is no existing database and it needs zero
# setup (single file, ships with Python). The file lives next to this script.
DB_PATH = os.path.join(backend_dir, "monitoring.db")


def init_db():
    """Create the predictions log table once at startup if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_class TEXT,
            confidence REAL,
            img_width INTEGER,
            img_height INTEGER,
            mean_intensity REAL,
            std_intensity REAL,
            input_flagged_as_outlier INTEGER,
            inference_ms REAL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()

# --- Input-drift reference distribution ---
# These four numbers describe the TRAINING set's pixel-intensity distribution.
# For every training image (resized to 224x224, the same size the model sees)
# we computed the per-image mean and std of raw pixel intensity in [0, 255]
# averaged over RGB. We then took the mean and std of those per-image values
# across all 7,859 training images. Generated once by a one-off script over
# BreaKHis_v1 (the holdout set was already removed, so there is no leakage).
TRAIN_MEAN_INTENSITY_MEAN = 185.0371  # average per-image mean intensity
TRAIN_MEAN_INTENSITY_STD = 15.6567    # spread of per-image mean intensity
TRAIN_STD_INTENSITY_MEAN = 35.2255    # average per-image std intensity
TRAIN_STD_INTENSITY_STD = 11.7824     # spread of per-image std intensity

# A new image is flagged as an outlier if its mean OR std intensity falls more
# than this many standard deviations from the training mean. 2.5 sits in the
# requested 2-3 range: under a roughly normal distribution ~99% of in-domain
# images stay inside +/-2.5 sigma, so anything outside is genuinely unusual
# (e.g. a non-histology image or a very different stain/scanner) while keeping
# false alarms low. The flag is advisory only and never blocks a prediction.
DRIFT_SIGMA = 2.5


def compute_image_stats(resized_array):
    """Mean and std of raw pixel intensity (0-255, averaged over RGB) for the
    224x224 image, matching how the training distribution was computed."""
    return float(resized_array.mean()), float(resized_array.std())


def is_outlier(mean_intensity, std_intensity):
    """True if the image's mean or std intensity is outside DRIFT_SIGMA std
    devs of the training distribution for that statistic."""
    mean_off = abs(mean_intensity - TRAIN_MEAN_INTENSITY_MEAN) > DRIFT_SIGMA * TRAIN_MEAN_INTENSITY_STD
    std_off = abs(std_intensity - TRAIN_STD_INTENSITY_MEAN) > DRIFT_SIGMA * TRAIN_STD_INTENSITY_STD
    return bool(mean_off or std_off)


def log_prediction(predicted_class, confidence, width, height,
                   mean_intensity, std_intensity, flagged, inference_ms):
    """Insert one row into the predictions table. Logging failures are swallowed
    so monitoring can never break the prediction response."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp, predicted_class, confidence, img_width, img_height,
                mean_intensity, std_intensity, input_flagged_as_outlier, inference_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                predicted_class, confidence, width, height,
                mean_intensity, std_intensity, int(flagged), inference_ms,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to log prediction: {e}")
# =================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/images")
def list_images():
    result = []
    for cat in os.listdir(holdout_dir):
        cat_path = os.path.join(holdout_dir, cat)
        if os.path.isdir(cat_path):
            for file in os.listdir(cat_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    result.append({
                        "filename": file,
                        "category": cat
                    })
    return result

@app.post("/api/predict")
async def predict(
    image: UploadFile = File(None),
    filename: str = Form(None),
    category: str = Form(None)
):
    if model is None:
        return {"error": "Model not loaded."}

    try:
        if image and image.filename:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            actual_label = "uploaded"
        elif filename and category:
            image_path = os.path.join(holdout_dir, category, filename)
            pil_image = Image.open(image_path).convert("RGB")
            actual_label = category.lower()
        else:
            return {"error": "No image provided."}

        # Original dimensions logged for reference; stats below are computed on
        # the 224x224 image because that is what the model and the training
        # drift distribution are based on.
        orig_width, orig_height = pil_image.size

        pil_image = pil_image.resize((224, 224))
        image_array = np.array(pil_image, dtype=np.float32)
        mean_intensity, std_intensity = compute_image_stats(image_array)
        flagged = is_outlier(mean_intensity, std_intensity)

        model_input = preprocess_input(np.expand_dims(image_array, axis=0))

        inference_start = time.time()
        prediction = model.predict(model_input)
        inference_ms = (time.time() - inference_start) * 1000.0

        confidence_score = float(prediction[0][0])
        predicted_class = int(round(confidence_score))
        predicted_label = class_names[predicted_class]

        # Log this request for monitoring. Done after inference and before the
        # response is built; it does not change what the frontend receives
        # except for the added advisory outlier flag.
        log_prediction(
            predicted_label, confidence_score, orig_width, orig_height,
            mean_intensity, std_intensity, flagged, inference_ms,
        )

        return {
            "predictedLabel": predicted_label,
            "confidence": confidence_score,
            "actualLabel": actual_label,
            "input_flagged_as_outlier": flagged
        }

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

@app.get("/api/plots/{plot_name}")
def get_plot(plot_name: str):
    allowed = {"training_history.png", "probability_distribution.png"}
    if plot_name not in allowed:
        return {"error": "Plot not found."}
    plot_path = os.path.join(backend_dir, plot_name)
    if not os.path.exists(plot_path):
        return {"error": f"{plot_name} not found on disk."}
    return FileResponse(plot_path, media_type="image/png")


@app.get("/health")
def health():
    """Liveness check: how long the process has been up and how many
    predictions it has served (read from the log table)."""
    conn = sqlite3.connect(DB_PATH)
    total_requests = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "total_requests": total_requests,
    }


@app.get("/stats")
def stats():
    """Aggregate stats over all logged predictions: total count, predicted
    class distribution, average confidence, average inference latency, and
    how many inputs were flagged as outliers."""
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    if total == 0:
        conn.close()
        return {
            "total_requests": 0,
            "predicted_class_distribution": {},
            "average_confidence": None,
            "average_inference_ms": None,
            "outliers_flagged": 0,
        }

    class_rows = conn.execute(
        "SELECT predicted_class, COUNT(*) FROM predictions GROUP BY predicted_class"
    ).fetchall()
    avg_confidence = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
    avg_inference_ms = conn.execute("SELECT AVG(inference_ms) FROM predictions").fetchone()[0]
    outliers = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE input_flagged_as_outlier = 1"
    ).fetchone()[0]
    conn.close()

    return {
        "total_requests": total,
        "predicted_class_distribution": {cls: count for cls, count in class_rows},
        "average_confidence": round(avg_confidence, 4),
        "average_inference_ms": round(avg_inference_ms, 2),
        "outliers_flagged": outliers,
    }
