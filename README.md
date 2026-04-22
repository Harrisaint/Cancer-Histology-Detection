# Cancer Histology Detection

A full-stack deep learning application that classifies breast cancer histopathology images as **benign** or **malignant**. Built with a MobileNetV2 transfer learning model trained on the BreaKHis dataset, a FastAPI backend for inference, and a React frontend for interactive analysis.

---

## Model Performance

Evaluated on a held-out validation set with an F1-optimized threshold of 0.55:

| Metric | Benign | Malignant |
|---|---|---|
| Precision | 0.83 | 0.87 |
| Recall | 0.66 | **0.94** |
| F1 Score | 0.73 | **0.91** |

- **Overall Accuracy**: 86%
- **Validation AUC**: 0.935
- **Malignant Recall**: 94% — catches the vast majority of cancerous samples
- **Weighted F1**: 0.86

---

## Tech Stack

- **Model**: TensorFlow/Keras 2.19, MobileNetV2 (ImageNet pretrained), scikit-learn
- **Backend**: FastAPI, Uvicorn, Pillow, NumPy, Matplotlib
- **Frontend**: React 19, Material UI v5, Framer Motion, Emotion
- **Dataset**: [BreaKHis](https://www.kaggle.com/datasets/ambarish/breakhis) — Breast Cancer Histopathological Database

---

## Features

- Upload your own histology image or select from the holdout test set
- Real-time benign/malignant classification via the trained model
- Confidence score display with ground truth comparison for holdout images
- Training history and probability distribution charts on the app page
- Responsive design for desktop and mobile

---

## Architecture

```
Cancer-Histology-Detection/
├── backend/
│   ├── backend_app.py              # FastAPI server (inference + image API)
│   ├── app.py                      # Streamlit demo (alternative frontend)
│   ├── train.py                    # Full training pipeline
│   ├── breakhis_mobilenet_improved_model.keras  # Trained model
│   ├── optimal_threshold.txt       # F1-optimized threshold (0.55)
│   ├── training_history.png        # Loss/accuracy/AUC/recall curves
│   ├── probability_distribution.png # Prediction distributions by class
│   ├── requirements.txt            # Python dependencies
│   └── runtime.txt                 # Python version for deployment
├── cancer-histology-frontend/      # React frontend
│   ├── src/
│   │   ├── App.js                  # Main app with prediction UI + charts
│   │   └── components/
│   │       ├── Header.js
│   │       └── ImageSelector.js    # Image upload + holdout selector
│   └── public/
├── extract_holdout_set.py          # Script to create holdout test set
├── holdout_test_set/               # Held-out images (benign/ + malignant/)
├── .gitignore
└── README.md
```

**Flow**: Frontend sends image to `POST /api/predict` → FastAPI preprocesses with MobileNetV2's `preprocess_input` → model outputs probability → threshold applied → result returned as JSON.

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn backend_app:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend

```bash
cd cancer-histology-frontend
npm install
npm start
```

Opens at `http://localhost:3000`. Set `REACT_APP_API_URL` to point to a different backend if needed.

### Training (optional)

To retrain the model from scratch:

1. Download the [BreaKHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis) and place it in `BreaKHis_v1/` at the project root with `benign/` and `malignant/` subdirectories containing images directly (no nested subtype folders).
2. Run the holdout extraction script:
   ```bash
   python extract_holdout_set.py
   ```
3. Run training:
   ```bash
   cd backend
   python train.py
   ```

Training outputs the model (`.keras`), optimal threshold (`.txt`), and performance plots (`.png`) to the `backend/` directory.

---

## API Endpoints

### `POST /api/predict`
Classify a histology image.

**Input** (multipart form):
- `image` (file) — uploaded image, OR
- `filename` + `category` (strings) — to select from the holdout set

**Response**:
```json
{
  "predictedLabel": "malignant",
  "confidence": 0.87,
  "actualLabel": "malignant"
}
```

### `GET /api/images`
List all images in the holdout test set.

**Response**: Array of `{ "filename": "...", "category": "benign" | "malignant" }`

### `GET /api/plots/{plot_name}`
Serve training plots. Allowed values: `training_history.png`, `probability_distribution.png`.

---

## Training Details

- **Architecture**: MobileNetV2 base (ImageNet weights) + GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)
- **Two-phase training**: Frozen base for 10 epochs, then fine-tune top layers at 1e-5 learning rate
- **Loss**: Focal Loss (alpha=0.75, gamma=2.0) to handle class imbalance
- **Class weights**: 1.5x boost for malignant class
- **Augmentation**: Horizontal + vertical flips, random rotation, zoom, contrast, brightness
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (monitoring val_recall)
- **Threshold optimization**: Grid search over [0.1, 0.9] to maximize F1 score

---

## Deployment

- **Backend**: Deployed on [Render](https://render.com) as a web service
- **Frontend**: Deployed on [Vercel](https://vercel.com)

---

## License

MIT License

---

## Acknowledgments

- [BreaKHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis) — Spanhol et al.
- TensorFlow, Keras, FastAPI, React, Material UI, and the open-source community
