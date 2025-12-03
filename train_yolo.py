"""
train_yolo.py

Train a YOLOv8 detector on the 6-class OfficeItems dataset and
export detailed evaluation metrics (overall + per-class), similar
to the MobilenetV2 training pipeline.

Requires:
- ultralytics
- torch (PyTorch 2.6+ compatible; we patch safe loading below)
- data/yolo_office.yaml pointing to your Roboflow dataset
"""

import os
from pathlib import Path
import json

import torch
from torch.serialization import add_safe_globals

# ---- Allow PyTorch to unpickle Ultralytics YOLO weights (PyTorch 2.6+ change) ----
# We trust Ultralytics classes because weights are from official repo.
from ultralytics import nn as yolo_nn
from ultralytics.nn import tasks as yolo_tasks, modules as yolo_modules
import torch.nn as nn

allowed = []

# Collect all class objects from these modules and allowlist them
for module in (yolo_tasks, yolo_modules, nn):
    for attr in vars(module).values():
        if isinstance(attr, type):
            allowed.append(attr)

add_safe_globals(allowed)

# Also force old behaviour (weights_only = False)
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
# --------------------------------------------------------------------------


from ultralytics import YOLO  # import AFTER safe_globals patch


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Path to dataset YAML (relative to this script)
DATA_CFG = "data/yolo_office.yaml"

# Base model to start from (tiny + fast, good for Pi)
BASE_MODEL = "yolov8n.pt"

# Training hyperparameters
EPOCHS = 30
IMG_SIZE = 640
BATCH_SIZE = 16         # drop to 8 / 4 if you get OOM
DEVICE = "cpu"          # "0" for first GPU, "cpu" for CPU
RUN_NAME = "office_yolo"

# Where to dump numeric metrics (in addition to YOLO's own plots)
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def train_and_evaluate():
    print(f"[INFO] Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # ------------------- TRAIN -------------------
    print("[INFO] Starting YOLO training...")
    model.train(
        data=DATA_CFG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=4,
        name=RUN_NAME,
        verbose=True,
    )

    # After training, load the best weights
    run_dir = Path("runs") / "detect" / RUN_NAME
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(
            f"[ERROR] Could not find best weights at {best_weights}. "
            "Check that training completed successfully."
        )

    print(f"[INFO] Loading best model from: {best_weights}")
    best_model = YOLO(str(best_weights))

    # ------------------- VALIDATE -------------------
    print("[INFO] Running validation with detailed metrics and plots...")
    metrics = best_model.val(
        data=DATA_CFG,
        imgsz=IMG_SIZE,
        split="val",
        plots=True,
        save_hybrid=False,
    )

    # ------------------- OVERALL METRICS -------------------
    overall = metrics.results_dict  # DetMetrics -> dict

    print("\n[INFO] Overall detection metrics (validation split):")
    for k, v in overall.items():
        try:
            print(f"  {k}: {v:.4f}")
        except TypeError:
            print(f"  {k}: {v}")

    overall_path = DOCS_DIR / "yolo_overall_metrics.json"
    with overall_path.open("w") as f:
        json.dump(overall, f, indent=2)
    print(f"[INFO] Saved overall metrics to {overall_path}")

    # ------------------- PER-CLASS METRICS -------------------
    try:
        per_class = metrics.summary()  # list[dict] with per-class rows
    except AttributeError:
        per_class = []

    per_class_path = DOCS_DIR / "yolo_per_class_metrics.csv"
    if per_class:
        keys = list(per_class[0].keys())
        with per_class_path.open("w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in per_class:
                f.write(",".join(str(row[k]) for k in keys) + "\n")

        print(f"[INFO] Saved per-class metrics to {per_class_path}")
    else:
        print("[WARN] No per-class metrics were returned (empty dataset or API changed).")

    # ------------------- SUMMARY -------------------
    print("\n[SUMMARY]")
    print(f"  Best weights:    {best_weights}")
    print(f"  Overall metrics: {overall_path}")
    print(f"  Per-class CSV:   {per_class_path}")
    print(f"  YOLO plots dir:  {run_dir}")
    print("    - confusion_matrix.png")
    print("    - results.png")
    print("    - PR_curve.png")
    print("    - F1_curve.png")
    print("Done.")


if __name__ == "__main__":
    train_and_evaluate()
