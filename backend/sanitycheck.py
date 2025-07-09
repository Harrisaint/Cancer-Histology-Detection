import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_dir = os.path.join(project_root, "BreakHis_v1")

for label in ["benign", "malignant"]:
    label_dir = os.path.join(dataset_dir, label)
    if os.path.exists(label_dir):
        count = len(os.listdir(label_dir))
        print(f"{label}: {count} images")
    else:
        print(f"Directory not found: {label_dir}")
