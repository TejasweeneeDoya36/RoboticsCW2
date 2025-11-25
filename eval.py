# eval.py
"""
Evaluate on the held-out test set and save:
  - metrics.json  (Accuracy, Precision (macro), Recall (macro), F1 (macro), AUC-ROC (macro OvR), MAE, per-class PR/F1)
  - confusion_matrix.png

Usage:
  python eval.py --data data/test --weights models/mobilenet_v2_office.pth --classes docs/classes.txt --out results/
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error
)
from utils import read_classes, save_json, plot_confmat
from model import load_model


def main(args):
    # --- Device & labels ---
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Load class list and count
    class_names = read_classes(args.classes)
    num_classes = len(class_names)

    # Transforms must match training normalization
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ImageFolder dataset + loader
    ds = datasets.ImageFolder(args.data, transform=tf)
    ld = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True
    )

    # Load model weights
    model = load_model(args.weights, num_classes=num_classes, device=device)
    model.eval()

    # Accumulators for metrics
    y_true = []           # ground-truth class indices
    y_pred = []           # predicted class indices
    prob_chunks = []      # predicted probability vectors (softmax)

    # Inference loop (no gradients needed)
    with torch.no_grad():
        for x, y in ld:
            x = x.to(device)

            # Forward pass -> logits
            logits = model(x)                  # shape [B, C]

            # Convert to probabilities for AUC/MAE
            probs = F.softmax(logits, dim=1)   # shape [B, C]

            # Predicted class ids
            preds = probs.argmax(dim=1).cpu().numpy()

            # Collect
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())
            prob_chunks.append(probs.cpu())

    # Convert to arrays
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    probs_np = torch.cat(prob_chunks, dim=0).numpy()  # shape [N, C]

    # Headline metrics (6)
    # 1) Accuracy (overall)
    acc = accuracy_score(y_true, y_pred)

    # 2) Precision (macro average)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)

    # 3) Recall (macro average)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # 4) F1 score (macro average)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # 5) AUC-ROC (macro, One-vs-Rest)
    #    Note: This needs probabilities. It may fail if some classes are missing in y_true,
    #    so we wrap it in try/except and return NaN in that case.
    try:
        auc_macro_ovr = roc_auc_score(y_true, probs_np, multi_class="ovr", average="macro")
    except Exception:
        auc_macro_ovr = float("nan")

    # 6) MAE between one‑hot truth and probabilities
    try:
        onehot = np.eye(probs_np.shape[1], dtype=float)[y_true]  # shape [N, C]
        mae = mean_absolute_error(onehot, probs_np)
    except Exception:
        mae = float("nan")

    # --- Per-class precision/recall/F1 (handy for diagnostics; not part of the 6 headline metrics) ---
    #     These stay to match your previous outputs.
    per_class = []
    # Compute per-class PR/F1 in a simple loop for readability
    for c in range(num_classes):
        # Binary vectors for "class c vs rest"
        y_true_c = (y_true == c).astype(int)
        y_pred_c = (y_pred == c).astype(int)
        p_c = precision_score(y_true_c, y_pred_c, zero_division=0)
        r_c = recall_score(y_true_c, y_pred_c, zero_division=0)
        f1_c = f1_score(y_true_c, y_pred_c, zero_division=0)
        per_class.append({
            "class": class_names[c],
            "precision": float(p_c),
            "recall": float(r_c),
            "f1": float(f1_c),
        })

    # Save outputs
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix PNG
    plot_confmat(y_true.tolist(), y_pred.tolist(), class_names, str(outdir / "confusion_matrix.png"))

    # JSON metrics with the 6 headline metrics plus per-class details
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "auc_macro_ovr": float(auc_macro_ovr),
        "mae": float(mae),
        "per_class": per_class,
        "num_samples": int(len(y_true)),
    }
    save_json(metrics, str(outdir / "metrics.json"))

    # Console summary
    print(
        f"Accuracy: {acc:.3f} | Precision(macro): {prec_macro:.3f} | "
        f"Recall(macro): {rec_macro:.3f} | F1(macro): {f1_macro:.3f} | "
        f"AUC(macro OvR): {auc_macro_ovr:.3f} | MAE: {mae:.4f}"
    )
    print(f"Saved metrics.json and confusion_matrix.png to {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test folder (ImageFolder layout)")
    ap.add_argument("--weights", required=True, help="model weights (.pth)")
    ap.add_argument("--classes", required=True, help="path to classes.txt (one label per line)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)

