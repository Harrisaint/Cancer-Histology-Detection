import os
import random
import shutil

# === CONFIG ===
SOURCE_DIR = "BreaKHis_v1"
DEST_DIR = "holdout_test_set"
CLASSES = ["benign", "malignant"]
NUM_IMAGES_PER_CLASS = 25
SEED = 42  # for reproducibility

random.seed(SEED)

# === Ensure destination subfolders exist ===
for cls in CLASSES:
    os.makedirs(os.path.join(DEST_DIR, cls), exist_ok=True)

# === Copy random images to holdout folder ===
for cls in CLASSES:
    src_folder = os.path.join(SOURCE_DIR, cls)
    dst_folder = os.path.join(DEST_DIR, cls)

    # List all images
    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(images) < NUM_IMAGES_PER_CLASS:
        raise ValueError(f"Not enough images in {src_folder} to sample {NUM_IMAGES_PER_CLASS}.")

    # Pick random images
    selected = random.sample(images, NUM_IMAGES_PER_CLASS)

    # Move images to holdout folder
    for img_name in selected:
        src_path = os.path.join(src_folder, img_name)
        dst_path = os.path.join(dst_folder, img_name)

        shutil.move(src_path, dst_path)

    print(f"Moved {NUM_IMAGES_PER_CLASS} {cls} images to {dst_folder}")
