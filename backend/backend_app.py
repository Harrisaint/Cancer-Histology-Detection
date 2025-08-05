from fastapi import FastAPI, Body, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.metrics import Precision
import io

# --- Register Custom Loss: FocalLoss ---
@register_keras_serializable()
class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, **kwargs):
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

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma) * y_true + \
                 (1 - self.alpha) * tf.pow(y_pred, self.gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)

# --- Register Custom Metric ---
@register_keras_serializable()
def precision_at_recall_fn(y_true, y_pred):
    metric = Precision()
    metric.update_state(y_true, y_pred)
    return metric.result()

# --- Load Model ---
try:
    model = tf.keras.models.load_model(
        "breakhis_mobilenet_improved_model.keras",
        custom_objects={
            "FocalLoss": FocalLoss,
            "precision_at_recall_fn": precision_at_recall_fn
        }
    )
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# --- Class Index Mapping ---
class_names = ["benign", "malignant"]

# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoint: List Holdout Test Images ---
@app.get("/api/images")
def list_images():
    base = "holdout_test_set"
    result = []
    for cat in os.listdir(base):
        cat_path = os.path.join(base, cat)
        if os.path.isdir(cat_path):
            for file in os.listdir(cat_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    result.append({
                        "filename": file,
                        "category": cat
                    })
    return result

# --- Endpoint: Predict from Uploaded or Predefined Image ---
@app.post("/api/predict")
async def predict(
    image: UploadFile = File(None),
    filename: str = Form(None),
    category: str = Form(None)
):
    try:
        # Handle uploaded image
        if image:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            actual_label = "uploaded"

        # Handle test set image
        elif filename and category:
            image_path = os.path.join("holdout_test_set", category, filename)
            pil_image = Image.open(image_path).convert("RGB")
            actual_label = category.lower()

        else:
            return {"error": "No image provided."}

        # Preprocess image
        pil_image = pil_image.resize((224, 224))
        image_array = np.array(pil_image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Model prediction
        prediction = model.predict(image_array)
        confidence_score = float(prediction[0][0])  # between 0 and 1
        predicted_class = int(round(confidence_score))
        predicted_label = class_names[predicted_class]

        return {
            "predictedLabel": predicted_label,
            "confidence": confidence_score,
            "actualLabel": actual_label
        }

    except Exception as e:
        return {"error": f"❌ Failed to process image: {str(e)}"}
