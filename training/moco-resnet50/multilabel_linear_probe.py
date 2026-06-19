import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report
)
from monai.networks.nets import resnet as monai_resnet
import csv
import os
import argparse
import yaml
from torch.cuda import amp

from utils import args, load_config

from monai.transforms import Compose, RandSpatialCropd, EnsureTyped ,CenterSpatialCropd
import warnings
warnings.filterwarnings("ignore")

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))  # go two levels up
# src_dir = os.path.join(project_root, "src")
# sys.path.append(src_dir)

BASE_DIR = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/Evaluation"
MODEL_DIR = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/outputs/checkpoints/ResNet50_MoCo3D_last.pth"

def train_probe_transform(roi_size=(96,96,96)):
    return Compose([
        EnsureTyped(keys=['img']),
        RandSpatialCropd(keys=['img'], roi_size=roi_size, random_size=False),
    ])
    
def valid_probe_transform(roi_size=(96,96,96)):
    return Compose([
        EnsureTyped(keys=['img']),
        CenterSpatialCropd(
            keys=['img'], roi_size=roi_size
        ),
    ])


class LinearProbeNumpyDataset(Dataset):
    """
    Multi-label dataset for linear probing with optional multi-crop evaluation.
    If `num_crops>1`, returns multiple random crops per volume for evaluation.
    """
    def __init__(self, items, label_cols, roi_size=(96,96,96), transform=None, num_crops=1):
        self.items = items
        self.label_cols = label_cols
        self.roi_size = roi_size
        self.transform = transform
        self.num_crops = num_crops

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        arr = np.load(row['path']).astype(np.float32)  # shape: (1,128,256,256)
        label = np.array(
            [row[col] for col in self.label_cols],
            dtype=np.float32
        )
        
        crops = []
        for _ in range(self.num_crops):
            sample = {'img': arr.copy()}
            if self.transform:
                sample = self.transform(sample)
            crops.append(sample['img'])

        if self.num_crops == 1:
            return crops[0], label
        else:
            # Stack crops: shape (num_crops, C, D, H, W)
            return np.stack(crops, axis=0), label



# -----------------------------
# Linear Probe Model
# -----------------------------
class LinearProbe(nn.Module):
    def __init__(self, encoder, feat_dim=2048, num_classes=18):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(feat_dim, num_classes)

        # Freeze backbone
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()  # freeze BatchNorm running stats

    def forward(self, x):
        # Keep encoder frozen to save memory
        with torch.no_grad():
            f = self.encoder(x)
            f = f.view(f.size(0), -1)
        return self.fc(f)  # raw logits (no sigmoid)



# -----------------------------
# Load MoCo Encoder
# -----------------------------
def load_moco_encoder(checkpoint_path):
    backbone = monai_resnet.resnet50(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1
    )
    backbone.fc = nn.Identity()

    state = torch.load(checkpoint_path, map_location="cpu")

    encoder_state = {
        k.replace("encoder_q.", ""): v
        for k, v in state.items()
        if k.startswith("encoder_q.")
    }

    backbone.load_state_dict(encoder_state, strict=True)

    for p in backbone.parameters():
        p.requires_grad = False

    backbone.eval()
    return backbone


# -----------------------------
# Evaluation
# -----------------------------


@torch.no_grad()
def evaluate(model, loader, device, num_crops=1):
    """
    Evaluate a multi-label 3D model with optional multi-crop aggregation.

    Args:
        model: PyTorch model (expects 5D input for Conv3D: B, C, D, H, W)
        loader: DataLoader returning (x, y), where x is 6D if multi-crop: (B, N, C, D, H, W)
        device: torch.device
        num_crops: int, number of crops per sample

    Returns:
        metrics dict: auroc_macro, auroc_micro, auroc_per_class, f1_macro, f1_micro, report_dict
    """
    model.eval()

    y_true_all = []
    y_prob_all = []
    y_pred_all = []

    for x, y in loader:
        # Move to device
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Handle multi-crop inputs
        if num_crops > 1:
            B, N, C, D, H, W = x.shape
            assert N == num_crops, f"Expected num_crops={num_crops}, got N={N}"
            x = x.view(B * N, C, D, H, W)
            logits = model(x)                 # (B*N, num_classes)
            logits = logits.view(B, N, -1)   # (B, N, num_classes)
            logits = logits.mean(dim=1)      # average over crops
        else:
            # Single crop: remove crop dimension if exists
            if x.ndim == 6:  # (B, N=1, C, D, H, W)
                x = x.squeeze(1)
            logits = model(x)

        # Probabilities and thresholded predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        y_true_all.append(y.cpu())
        y_prob_all.append(probs.cpu())
        y_pred_all.append(preds.cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_prob = torch.cat(y_prob_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    # --- Metrics ---
    metrics = {
        "auroc_macro": roc_auc_score(
            y_true, y_prob, average="macro"
        ),
        "auroc_micro": roc_auc_score(
            y_true, y_prob, average="micro"
        ),
        "auroc_per_class": roc_auc_score(
            y_true, y_prob, average=None
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_micro": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "report_dict": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            digits=4,
            zero_division=0
        ),
    }

    return metrics




# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    train_df = pd.read_csv(BASE_DIR + "/Train_Test_Labels/train.csv")
    val_df = pd.read_csv(BASE_DIR + "/Train_Test_Labels/val.csv")

    # Load configuration
    cfg = load_config(args.config)

    # Get label columns from config
    label_cols = cfg["labels"]

    train_ds = LinearProbeNumpyDataset(
        items=train_df.to_dict('records'),
        label_cols=label_cols,
        roi_size=(96,96,96),
        transform=train_probe_transform(),
        num_crops=1
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,           # VRAM-friendly
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    

    val_ds = LinearProbeNumpyDataset(
        items=val_df.to_dict('records'),
        label_cols=label_cols,
        roi_size=(96,96,96),
        transform=valid_probe_transform(),
        num_crops=1   
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=2,           # process 1 volume at a time with multi-crops
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    encoder = load_moco_encoder(MODEL_DIR)
    model = LinearProbe(
        encoder, feat_dim=2048, num_classes=18
    ).to(device)

    optimizer = torch.optim.SGD(
        model.fc.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_macro_auc = -float("inf")
    best_metrics = None
    history = []  

    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)


        # Validation metrics (multi-label)
        metrics = evaluate(model, val_loader, device)  # should calculate AUROC, F1, etc.

        metrics_epoch = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "auroc_micro": metrics["auroc_micro"],
            "auroc_macro": metrics["auroc_macro"],
            "f1_micro": metrics["f1_micro"],
            "f1_macro": metrics["f1_macro"],
        }

        history.append(metrics_epoch)
        
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {avg_train_loss:.4f} | "
            f"AUROC(micro) {metrics['auroc_micro']:.4f} | "
            f"AUROC(macro) {metrics['auroc_macro']:.4f} | "
            f"F1(micro) {metrics['f1_micro']:.4f}"
        )

        if metrics["auroc_macro"] > best_macro_auc:
            best_macro_auc = metrics["auroc_macro"]
            best_metrics = metrics
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                BASE_DIR + "/checkpoints/ResNet50_best_linear_probe_multilabel.pth"
            )
            print(f"Saved best model at epoch {epoch} with AUROC_macro={best_macro_auc:.4f}")
            
    print("\nFinal classification report:")

    # -----------------------------
    # Save evaluation results
    # -----------------------------
    os.makedirs("results", exist_ok=True)

    # All epoch history
    history_df = pd.DataFrame(history)
    history_path = BASE_DIR + "/checkpoints/linearProbeTrainVal/linear_probe_epoch_metrics.csv"
    history_df.to_csv(history_path, index=False)

    assert best_metrics is not None, "No best model was saved!"

    #(A) Global metricscs
    global_metrics_path = BASE_DIR + "/checkpoints/linearProbeTrainVal/linear_probe_best_global_metrics.csv"
    with open(global_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["best_epoch", best_epoch])
        writer.writerow(["AUROC_micro", best_metrics["auroc_micro"]])
        writer.writerow(["AUROC_macro", best_metrics["auroc_macro"]])
        writer.writerow(["F1_micro", best_metrics["f1_micro"]])
        writer.writerow(["F1_macro", best_metrics["f1_macro"]])

    # (B) Per-class AUROC
    per_class_auc_path = BASE_DIR + "/checkpoints/linearProbeTrainVal/linear_probe_best_per_class_auroc.csv"
    with open(per_class_auc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "auroc"])
        for i, auc in enumerate(best_metrics["auroc_per_class"]):
            writer.writerow([i, auc])


    # (C) Classification report
    report_df = pd.DataFrame(best_metrics["report_dict"]).transpose()
    report_path = BASE_DIR + "/checkpoints/linearProbeTrainVal/linear_probe_best_classification_report.csv"
    report_df.to_csv(report_path)


    print("Saved evaluation results:")
    print(f" - {global_metrics_path}")
    print(f" - {per_class_auc_path}")
    print(f" - {report_path}")


if __name__ == "__main__":
    main()


