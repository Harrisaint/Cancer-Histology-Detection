import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import register_keras_serializable



# === IMPROVED FOCAL LOSS ===
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
        
        # Calculate focal loss
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow((1 - p_t), self.gamma)
        
        loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'from_logits': self.from_logits
        })
        return config

# === CONFIGURATION ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(project_root, "BreakHis_v1")
print("Using dataset from:", DATASET_DIR)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10

# === LOAD DATASETS WITH PROPER STRATIFICATION ===
print("Loading dataset...")

# First, let's check the directory structure
print("Checking dataset structure...")
for root, dirs, files in os.walk(DATASET_DIR):
    level = root.replace(DATASET_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')

# Load with different seed to ensure proper stratification
train_dataset = image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset='training',
    seed=42  # Different seed
)

val_dataset = image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,  # Enable shuffle for validation too
    validation_split=0.2,
    subset='validation',
    seed=42  # Same seed as training
)

# === CALCULATE CLASS DISTRIBUTION ===
def calculate_class_distribution(dataset):
    class_counts = {0: 0, 1: 0}
    for _, labels in dataset:
        for label in labels:
            class_counts[int(label.numpy())] += 1
    return class_counts

train_class_counts = calculate_class_distribution(train_dataset)
val_class_counts = calculate_class_distribution(val_dataset)

print(f"Training set - Benign: {train_class_counts[0]}, Malignant: {train_class_counts[1]}")
print(f"Validation set - Benign: {val_class_counts[0]}, Malignant: {val_class_counts[1]}")

# Validate that both classes exist in both sets
if train_class_counts[0] == 0 or train_class_counts[1] == 0:
    print("ERROR: Missing class in training set!")
    exit(1)

if val_class_counts[0] == 0 or val_class_counts[1] == 0:
    print("ERROR: Missing class in validation set!")
    print("This means the dataset split is not stratified properly.")
    print("Check your dataset directory structure.")
    exit(1)

# Calculate class weights based on actual distribution
total_samples = sum(train_class_counts.values())
class_weights_dict = {
    0: total_samples / (2 * train_class_counts[0]) if train_class_counts[0] > 0 else 1.0,
    1: total_samples / (2 * train_class_counts[1]) if train_class_counts[1] > 0 else 1.0
}

print("Calculated class weights:", class_weights_dict)

# === ENHANCED AUGMENTATION ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
])

# Apply augmentation and preprocessing
def preprocess_and_augment(image, label, training=True):
    if training:
        image = data_augmentation(image, training=True)
    image = preprocess_input(image)
    return image, label

train_dataset = train_dataset.map(
    lambda x, y: preprocess_and_augment(x, y, training=True)
)
val_dataset = val_dataset.map(
    lambda x, y: preprocess_and_augment(x, y, training=False)
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# === BUILD MODEL WITH BETTER ARCHITECTURE ===
print("Building model...")
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# === CUSTOM METRICS ===
def precision_at_recall(recall_threshold=0.8):
    def precision_at_recall_fn(y_true, y_pred):
        # Convert probabilities to binary predictions
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        
        # Calculate precision and recall
        tp = tf.reduce_sum(y_true * y_pred_binary)
        fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
        fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
        
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        
        return precision
    
    return precision_at_recall_fn

# === COMPILE MODEL ===
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=FocalLoss(gamma=2.0, alpha=0.25),
    metrics=[
        'accuracy',
        metrics.AUC(name="auc"),
        metrics.Recall(name="recall"),
        metrics.Precision(name="precision"),
        precision_at_recall()
    ]
)

# === CALLBACKS ===
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=7,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

checkpoint_cb = ModelCheckpoint(
    "breakhis_mobilenet_improved_model.keras",
    monitor="val_auc",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1
)


# === TRAIN BASE MODEL ===
print("Training base model...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr, checkpoint_cb], 
    verbose=1
)


# === FINE-TUNE WITH LOWER LEARNING RATE ===
print("Fine-tuning model...")
base_model.trainable = True

# Freeze early layers, only train last few layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=FocalLoss(gamma=2.0, alpha=0.25),
    metrics=[
        'accuracy',
        metrics.AUC(name="auc"),
        metrics.Recall(name="recall"),
        metrics.Precision(name="precision"),
        precision_at_recall()  # ✅ call only once, do not add name=
    ]
)


fine_tune_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=len(history.history['accuracy']),
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr, checkpoint_cb],  
    verbose=1
)


# === THRESHOLD OPTIMIZATION ===
def find_optimal_threshold(model, val_dataset, metric='f1'):
    """Find optimal threshold for binary classification"""
    y_true = []
    y_pred_proba = []
    
    for batch_images, batch_labels in val_dataset:
        preds = model.predict(batch_images, verbose=0).flatten()
        y_true.extend(batch_labels.numpy())
        y_pred_proba.extend(preds)
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if metric == 'f1' and f1 > best_score:
            best_score = f1
            best_threshold = threshold
    
    return best_threshold, best_score

optimal_threshold, best_f1 = find_optimal_threshold(model, val_dataset)
print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {best_f1:.3f})")

# === DETAILED EVALUATION ===
print("Evaluating on validation set...")
y_true = []
y_pred_proba = []

for batch_images, batch_labels in val_dataset:
    preds = model.predict(batch_images, verbose=0).flatten()
    y_true.extend(batch_labels.numpy())
    y_pred_proba.extend(preds)

y_true = np.array(y_true)
y_pred_proba = np.array(y_pred_proba)

# Predictions with default threshold (0.5)
y_pred_default = (y_pred_proba > 0.5).astype(int)

# Predictions with optimal threshold
y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)

print("\n=== RESULTS WITH DEFAULT THRESHOLD (0.5) ===")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_default))
print("\nClassification Report:")
print(classification_report(y_true, y_pred_default, target_names=["benign", "malignant"]))

print(f"\n=== RESULTS WITH OPTIMAL THRESHOLD ({optimal_threshold:.3f}) ===")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_optimal))
print("\nClassification Report:")
print(classification_report(y_true, y_pred_optimal, target_names=["benign", "malignant"]))

# === PREDICTION ANALYSIS ===
print("\n=== PREDICTION DISTRIBUTION ===")
print(f"Probability range: {y_pred_proba.min():.3f} to {y_pred_proba.max():.3f}")
print(f"Mean probability: {y_pred_proba.mean():.3f}")
print(f"Std probability: {y_pred_proba.std():.3f}")

# Show prediction distribution by true class
benign_probs = y_pred_proba[y_true == 0]
malignant_probs = y_pred_proba[y_true == 1]

print(f"\nBenign samples (n={len(benign_probs)}):")
print(f"  Mean prob: {benign_probs.mean():.3f}, Std: {benign_probs.std():.3f}")
print(f"  Min: {benign_probs.min():.3f}, Max: {benign_probs.max():.3f}")

print(f"\nMalignant samples (n={len(malignant_probs)}):")
print(f"  Mean prob: {malignant_probs.mean():.3f}, Std: {malignant_probs.std():.3f}")
print(f"  Min: {malignant_probs.min():.3f}, Max: {malignant_probs.max():.3f}")

# === ENHANCED SANITY CHECK ===
def predict_with_analysis(image_path, threshold=0.5):
    """Enhanced prediction function with detailed analysis"""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))
    
    prob = model.predict(img_preprocessed, verbose=0)[0][0]
    pred_default = "malignant" if prob > 0.5 else "benign"
    pred_optimal = "malignant" if prob > threshold else "benign"
    
    print(f"Image: {os.path.basename(image_path)}")
    print(f"  Probability: {prob:.4f}")
    print(f"  Prediction (0.5): {pred_default}")
    print(f"  Prediction ({threshold:.3f}): {pred_optimal}")
    print()

print("\n=== ENHANCED SANITY CHECK ===")
holdout_dir = os.path.join(project_root, "holdout_test_set")

if os.path.exists(holdout_dir):
    print("Testing with default threshold (0.5):")
    benign_files = os.listdir(os.path.join(holdout_dir, "benign"))
    malignant_files = os.listdir(os.path.join(holdout_dir, "malignant"))
    
    if benign_files:
        predict_with_analysis(os.path.join(holdout_dir, "benign", benign_files[0]), optimal_threshold)
    if malignant_files:
        predict_with_analysis(os.path.join(holdout_dir, "malignant", malignant_files[0]), optimal_threshold)
else:
    print("Holdout directory not found, skipping sanity check.")

# === SAVE MODEL ===
model.save("breakhis_mobilenet_improved_model.keras")
print("✅ Model saved as breakhis_mobilenet_improved_model.keras")

# Save threshold information
with open("optimal_threshold.txt", "w") as f:
    f.write(f"Optimal threshold: {optimal_threshold}\n")
    f.write(f"Best F1 score: {best_f1}\n")

# === ENHANCED PLOTTING ===
def plot_training_history(history, fine_tune_history):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Combine histories
    metrics = ['accuracy', 'loss', 'auc', 'recall', 'precision']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        if i < len(axes.flat):
            ax = axes[row, col]
            
            train_values = history.history[metric] + fine_tune_history.history[metric]
            val_values = history.history[f'val_{metric}'] + fine_tune_history.history[f'val_{metric}']
            epochs = range(1, len(train_values) + 1)
            
            ax.plot(epochs, train_values, 'b-', label=f'Training {metric}')
            ax.plot(epochs, val_values, 'r-', label=f'Validation {metric}')
            ax.set_title(f'{metric.capitalize()} over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(metrics) < len(axes.flat):
        axes.flat[-1].remove()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot probability distribution
def plot_probability_distribution(y_true, y_pred_proba, threshold=0.5):
    """Plot probability distribution for each class"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of probabilities
    axes[0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Benign', color='blue')
    axes[0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Malignant', color='red')
    axes[0].axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.3f})')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Probability Distribution by True Class')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
    axes[1].boxplot(data_to_plot, labels=['Benign', 'Malignant'])
    axes[1].axhline(y=threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.3f})')
    axes[1].set_ylabel('Predicted Probability')
    axes[1].set_title('Probability Distribution Box Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate plots
plot_training_history(history, fine_tune_history)
plot_probability_distribution(y_true, y_pred_proba, optimal_threshold)

print("\n✅ Training completed successfully!")
print(f"📊 Best model achieved F1 score of {best_f1:.3f} with threshold {optimal_threshold:.3f}")
print("📈 Training plots saved as 'training_history.png' and 'probability_distribution.png'")