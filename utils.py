# utils.py
"""
Utility helpers shared by training, evaluation, and the GUI.

- read_classes: read a classes.txt (one class name per line)
- set_seed: set Python/NumPy/PyTorch seeds for reproducibility
- save_json: write a Python object to a pretty JSON file
- plot_confmat: draw and save a confusion matrix image
"""
from pathlib import Path
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def read_classes(path: str):
    """Return list of class names from a text file (one per line)."""
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names

def set_seed(seed: int = 42):
    """Set seeds across libraries to make runs reproducible (best‑effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(obj, out_path: str):
    """Save arbitrary serializable `obj` to JSON at `out_path` (mkdirs included)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def plot_confmat(y_true, y_pred, class_names, out_path: str):
    """Plot a confusion matrix and save it as a PNG.

    Args:
    y_true: list/array of ground‑truth class ids
    y_pred: list/array of predicted class ids
    class_names: list of label names (for axes)
    out_path: where to save the image
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 6))

    # Show heatmap
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)

    # Ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate counts in each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
