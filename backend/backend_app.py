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

        pil_image = pil_image.resize((224, 224))
        image_array = np.array(pil_image, dtype=np.float32)
        image_array = preprocess_input(np.expand_dims(image_array, axis=0))

        prediction = model.predict(image_array)
        confidence_score = float(prediction[0][0])
        predicted_class = int(round(confidence_score))
        predicted_label = class_names[predicted_class]

        return {
            "predictedLabel": predicted_label,
            "confidence": confidence_score,
            "actualLabel": actual_label
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
