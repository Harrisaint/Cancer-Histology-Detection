# 🧬 Cancer Histology Detection with MobileNetV2 + Streamlit

A deep learning project for binary classification of breast cancer histology images (benign vs. malignant) using **MobileNetV2**, with a lightweight **Streamlit frontend** for image testing. Trained on the [BreaKHis](https://www.kaggle.com/datasets/ambarish/breakhis) dataset with **custom focal loss**, precision-recall tuning, and advanced metrics tracking.

---

## 🔍 Project Overview

- **Goal**: Classify histopathological images into benign or malignant classes.
- **Model**: Transfer learning using `MobileNetV2`, fine-tuned with data augmentation.
- **Loss**: Custom `FocalLoss` to handle class imbalance.
- **Frontend**: Minimal Streamlit app to test model predictions on unseen images.
- **Dataset**: BreaKHis v1 (not fully included due to size — see instructions below).

---

## 📁 Repository Structure

```bash
.
├── backend/
│   ├── app.py                    # Streamlit frontend
│   ├── model_with_finetuning_dataaugmentation.py  # Full training script
│   ├── checkclassnames.py        # Class mapping helper
│   ├── sanitycheck.py            # Dev debug tools
│   └── breakhis_mobilenet_improved_model.keras  # Final trained model
├── extract_holdout_set.py        # Helper to extract test set from BreaKHis
├── holdout_test_set/             # Small sample for frontend testing (included)
├── .gitignore
└── README.md                     # This file
```

---

## ⚙️ Getting Started

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/Harrisaint/Cancer-Histology-Detection.git
cd Cancer-Histology-Detection
```

---

## 🧪 Option 1: Run Streamlit Frontend (Test-Only)

This uses the **included `holdout_test_set/`** to try the model right away.

### ✅ Requirements

Install dependencies:

```bash
cd backend
pip install -r requirements.txt  # Or manually: streamlit, tensorflow, pillow, numpy
```

### ▶️ Launch the app

```bash
streamlit run app.py
```

You’ll get a web interface to pick and classify holdout histology images.

---

## 🏋️ Option 2: Retrain Model with BreaKHis Dataset

> The full BreaKHis dataset is **NOT included** here due to size limits.

### 📦 Step 1: Download Dataset

Download the full dataset (v1) from Kaggle:
https://www.kaggle.com/datasets/ambarish/breakhis

Extract it to a folder named:

```bash
BreakHis_v1/
```

Place it **one level above** the project folder (i.e., `../BreakHis_v1/`).

---

### 🧠 Step 2: Train the Model

```bash
cd backend
python model_with_finetuning_dataaugmentation.py
```

The script will:
- Perform data augmentation
- Train MobileNetV2 with early stopping + learning rate decay
- Evaluate performance
- Save the model as: `breakhis_mobilenet_improved_model.keras`

---

### 📦 Step 3 (Optional): Extract Holdout Set

To extract a balanced test set:

```bash
python ../extract_holdout_set.py
```

This will create a `holdout_test_set/` directory used by the Streamlit app.

---

## 📊 Results Summary

| Metric              | Value (Optimal Threshold) |
|---------------------|---------------------------|
| Accuracy            | 94%                        |
| Precision (Malignant) | 92%                     |
| Recall (Benign)     | 80%                        |
| F1 Score            | 0.89 – 0.96                |
| AUC (Val)           | 0.995+                     |

---

## 🧠 Custom Loss Function

We use `FocalLoss` to improve sensitivity for the minority class (benign):

```python
@register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    ...
```

This helps shift the decision boundary in highly imbalanced datasets.

---

## 📁 .gitignore (Important!)

The repository uses a `.gitignore` file to:

- Exclude the full BreaKHis dataset
- Avoid committing model checkpoints, `.h5`, `.keras`, `.log`, etc.
- Skip virtual environments

---

## 📌 Credits

- BreaKHis Dataset © [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)
- Streamlit, TensorFlow, and Keras open-source tools

---

## 📬 Contact

Made by [Harrisaint](https://github.com/Harrisaint)
