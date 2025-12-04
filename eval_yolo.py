# eval_yolo.py
from ultralytics import YOLO
from pathlib import Path
import json

MODEL_PATH = "runs/detect/office_yolo4/weights/best.pt"
DATA_CFG = "data/yolo_office.yaml"
IMG_SIZE = 640

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

def main():
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("[INFO] Running validation...")
    metrics = model.val(
        data=DATA_CFG,
        imgsz=IMG_SIZE,
        split="val",
        plots=True,
        device="cpu",
    )

    overall = metrics.results_dict
    print("\n[INFO] Overall metrics:")
    for k, v in overall.items():
        try:
            print(f"  {k}: {v:.4f}")
        except TypeError:
            print(f"  {k}: {v}")

    overall_path = DOCS_DIR / "yolo_overall_metrics.json"
    with overall_path.open("w") as f:
        json.dump(overall, f, indent=2)
    print(f"[INFO] Saved overall metrics to {overall_path}")

if __name__ == "__main__":
    main()
