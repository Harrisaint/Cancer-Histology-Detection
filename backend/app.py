import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import register_keras_serializable

# === CLEAR STATE AT APP START ===
for key in st.session_state.keys():
    del st.session_state[key]

# === CUSTOM OBJECTS ===
@register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow((1 - p_t), self.gamma)

        loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "from_logits": self.from_logits
        }

@register_keras_serializable()
def precision_at_recall_fn(y_true, y_pred):
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred_binary)
    fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
    fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    return precision

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "breakhis_mobilenet_improved_model.keras",
        custom_objects={
            "FocalLoss": FocalLoss,
            "precision_at_recall_fn": precision_at_recall_fn
        }
    )

model = load_model()

# === UI SETUP ===
st.set_page_config(page_title="Breast Tumor Classifier", layout="wide")
st.title("🔬 Breast Tumor Cell Classification Dashboard")
st.markdown("Upload a histology image from the **holdout_test_set** below to classify as **benign** or **malignant**.")

# === HOLDOUT TEST SET BROWSER ===
holdout_path = os.path.abspath("../holdout_test_set")
options = []
for category in ["benign", "malignant"]:
    folder = os.path.join(holdout_path, category)
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                options.append((file, os.path.join(folder, file), category))

# === IMAGE SELECTION ===
st.subheader("Choose an image from the holdout test set")
selected = st.selectbox("Select image:", options, format_func=lambda x: f"{x[0]} (Actual: {x[2]})")

if selected:
    filename, filepath, actual_label = selected
    image = Image.open(filepath).convert("RGB")
    st.image(image, caption=f"Selected Image: {filename}", width=500)

    # === PREDICT ===
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    predicted_label = "malignant" if prediction > 0.5 else "benign"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # === DISPLAY RESULTS ===
    st.markdown("### 🧠 Prediction Result")
    st.write(f"**Predicted Label:** `{predicted_label.title()}`")
    st.write(f"**Model Confidence:** `{confidence * 100:.2f}%`")
    st.write(f"**Actual Label:** `{actual_label.title()}`")

    if predicted_label == actual_label:
        st.success("✅ The model predicted correctly.")
    else:
        st.error("❌ The model predicted incorrectly.")
