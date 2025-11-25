# train.py
"""
Train MobileNetV2 on an ImageFolder dataset with train/val splits.

Usage:
  python train.py --data data --classes docs/classes.txt --out models/mobilenet_v2_office.pth

  Key points:
- Uses ImageNet normalization and light augmentations on the training set.
- Freezes convolutional backbone for speed; only the classifier is trained.
- Tracks and prints these validation metrics each epoch:
     Accuracy, Precision(macro), Recall(macro), F1(macro)
     AUC‑ROC (macro One‑vs‑Rest) using probabilities
     MAE between one‑hot labels and probability vectors
- Saves the best model by macro‑F1 to the path provided by --out.
- Writes a small .meta.json with classes and settings.
"""
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error
)
from utils import set_seed, read_classes, save_json
from model import get_mobilenet_v2

def make_loaders(data_root, img_size=224, batch_size=32):
    """Build train/val dataloaders from an ImageFolder tree.

    Expected structure under `data_root`:
    data_root/train/<class>/*.jpg
    data_root/val/<class>/*.jpg
    """
    common_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    # Training: augment + normalize to increase robustness
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), common_norm
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), common_norm
    ])
    train_ds = datasets.ImageFolder(Path(data_root)/"train", transform=train_tf)
    val_ds   = datasets.ImageFolder(Path(data_root)/"val",   transform=val_tf)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_ld, val_ld, train_ds.classes

def evaluate(model, loader, device):
    """Run a validation pass and compute loss + metrics.

    Returns:
    avg_loss, acc, prec_macro, rec_macro, f1_macro, auc_macro_ovr, mae
    """
    model.eval()
    ce = nn.CrossEntropyLoss()

    all_labels = []      # ground-truth class indices
    all_preds = []       # predicted class indices
    all_probs = []       # predicted probability vectors (softmax)
    total_loss = 0.0

    with torch.no_grad():  # disable gradients for faster eval
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)                            # shape: [B, C]
            loss = ce(logits, y)                         # scalar loss
            total_loss += loss.item() * x.size(0)        # accumulate (sum over samples)

            # Convert logits -> probabilities and predicted class
            probs = torch.softmax(logits, dim=1)         # shape: [B, C]
            preds = probs.argmax(dim=1)                  # shape: [B]

            # Move to CPU Python lists for sklearn
            all_labels.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu())

    # Basic metrics
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    val_recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    val_f1        = f1_score(all_labels, all_preds, average="macro")

    # Concatenate list of tensors -> numpy array of shape [N, C]
    probs_np = torch.cat(all_probs, dim=0).numpy()

    # AUC-ROC (macro, OvR). Can raise if a class is missing in labels.
    try:
        val_auc = roc_auc_score(all_labels, probs_np, multi_class="ovr", average="macro")
    except Exception:
        val_auc = float("nan")  # safe fallback

    # MAE between one‑hot labels and probabilities
    try:
        import numpy as np
        num_classes = probs_np.shape[1]
        y_onehot = np.eye(num_classes)[np.array(all_labels)]  # shape [N, C]
        val_mae = mean_absolute_error(y_onehot, probs_np)
    except Exception:
        val_mae = float("nan")  # safe fallback

    # Average loss per sample
    avg_loss = total_loss / len(loader.dataset)

    # Return everything needed by the training loop
    return avg_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, val_mae


def train(args):
    """Main training orchestration.
    - Builds dataloaders
    - Creates MobileNetV2 with ImageNet weights (for transfer learning)
    - Freezes conv features; trains classifier head with AdamW
    - Tracks validation metrics and saves the best model by macro‑F1
    """
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    train_ld, val_ld, class_names = make_loaders(args.data, args.img_size, args.batch_size)
    num_classes = len(class_names)

    model = get_mobilenet_v2(num_classes=num_classes, pretrained=True).to(device)

    # Freeze the feature extractor for speed/stability; fine‑tune head only
    for p in model.features.parameters():
        p.requires_grad = False

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best_f1, best_path = -1.0, Path(args.out)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        tr_loss = running/len(train_ld.dataset)

        # Validation pass + metrics
        va_loss, va_acc, va_prec, va_rec, va_f1, va_auc, va_mae = evaluate(model, val_ld, device)

        # Neat one‑line summary
        print(
            f"[Epoch {epoch}] "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"val_acc={va_acc:.3f} | val_prec={va_prec:.3f} | val_rec={va_rec:.3f} | "
            f"val_f1={va_f1:.3f} | val_auc={va_auc:.3f} | val_mae={va_mae:.4f}"
        )

        # Save by best macro‑F1
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ Saved best model to {best_path} (macro-F1 {best_f1:.3f})")

    # Save a small metadata file for reference
    meta = {"classes": class_names, "img_size": args.img_size, "best_val_macro_f1": best_f1}
    save_json(meta, best_path.with_suffix(".meta.json"))
    print("Done. Best weights:", best_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root folder containing train/ val/ test/ subfolders")
    ap.add_argument("--out", default="models/mobilenet_v2_office.pth")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)