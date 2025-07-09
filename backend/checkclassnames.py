import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Update this path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(project_root, "BreakHis_v1")

dataset = image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(224, 224),
    batch_size=32
)

print("Class names:", dataset.class_names)
