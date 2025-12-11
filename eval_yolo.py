#!/usr/bin/env python3
"""
eval_yolo.py

Small helper script to evaluate a trained YOLO model on the
validation split and save the numeric metrics to disk.

This is used for:
- reporting mAP / precision / recall in the coursework report
- keeping a JSON copy of the results for later analysis
"""

from ultralytics import YOLO
from pathlib import Path
import json

# Path to the trained weights we want to evaluate
MODEL_PATH = "models/office_yolo.pt"

# YOLO data config (points to train/val images + labels)
DATA_CFG = "data/yolo_office.yaml"

# Evaluation image size (should match training size)
IMG_SIZE = 512

# Where to save the JSON metrics
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)


def main():
    """Load the trained model, run validation, and save metrics as JSON."""
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("[INFO] Running validation...")
    # model.val() returns a DetMetrics object with summary statistics
    metrics = model.val(
        data=DATA_CFG,
        imgsz=IMG_SIZE,
        split="val",      # use validation split from YAML
        plots=True,       # also save confusion matrix, PR curves, etc.
        device="cpu",     # everything runs on CPU for reproducibility
    )

    # Convert metrics to a plain Python dict (overall averages)
    overall = metrics.results_dict
    print("\n[INFO] Overall metrics:")
    for k, v in overall.items():
        # Some values are floats, some are other objects
        try:
            print(f"  {k}: {v:.4f}")
        except TypeError:
            print(f"  {k}: {v}")

    # Save the numeric metrics to docs/yolo_overall_metrics.json
    overall_path = DOCS_DIR / "yolo_overall_metrics.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(f"[INFO] Saved overall metrics to {overall_path}")


if __name__ == "__main__":
    main()
