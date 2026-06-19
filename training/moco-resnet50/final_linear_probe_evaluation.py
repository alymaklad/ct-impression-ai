import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandSpatialCropd
)

from monai.networks.nets import resnet as monai_resnet

import argparse
import yaml

from utils import args, load_config


import warnings
warnings.filterwarnings("ignore")

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))  # go two levels up
# src_dir = os.path.join(project_root, "src")
# sys.path.append(src_dir)

BASE_DIR = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/Evaluation"
MODEL_DIR = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/Evaluation/checkpoints/ResNet50_best_linear_probe_multilabel.pth"


# -----------------------------
# Dataset
# -----------------------------

def get_test_transform(roi_size=(96, 96, 96)):
    return Compose([
        EnsureTyped(keys=["img"]),
        RandSpatialCropd(
            keys=["img"],
            roi_size=roi_size,
            random_size=False
        )
    ])


class MultiCropNumpyDataset(Dataset):
    def __init__(self, records, label_cols, transform, num_crops):
        self.records = records
        self.label_cols = label_cols
        self.transform = transform
        self.num_crops = num_crops

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        img = np.load(r["path"]).astype(np.float32)  # (1,128,256,256)
        label = np.array([r[c] for c in self.label_cols], dtype=np.float32)

        crops = []
        for _ in range(self.num_crops):
            data = {"img": img}
            out = self.transform(data)
            crops.append(out["img"])  # (1,96,96,96)

        crops = torch.stack(crops, dim=0)  # (N,1,96,96,96)

        return crops, torch.from_numpy(label)


# -----------------------------
# Linear Probe Model
# -----------------------------
class LinearProbe(nn.Module):
    def __init__(self, encoder, feat_dim=2048, num_classes=18):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            f = self.encoder(x)
            f = f.view(f.size(0), -1)
        return self.fc(f)


# -----------------------------
# Load MoCo Encoder
# -----------------------------
def load_moco_encoder(checkpoint_path):
    encoder = monai_resnet.resnet50(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1
    )
    encoder.fc = nn.Identity()

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location="cpu")

    # Extract encoder weights
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state.items()
        if k.startswith("encoder.")
    }

    print(f"Loaded {len(encoder_state)} encoder parameters")

    missing, unexpected = encoder.load_state_dict(
        encoder_state,
        strict=False
    )

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # Freeze encoder (linear probing)
    for p in encoder.parameters():
        p.requires_grad = False

    encoder.eval()
    return encoder





# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def evaluate_multicrop(model, loader, device, num_crops, threshold=0.5):
    model.eval()

    y_true_all, y_prob_all, y_pred_all = [], [], []

    for x, y in loader:
        # x: (B,N,1,96,96,96)
        x = x.to(device, non_blocking=True)
        y = y.to(device)

        B, N, C, D, H, W = x.shape
        assert N == num_crops

        x = x.view(B * N, C, D, H, W)

        logits = model(x)                 # (B*N,18)
        logits = logits.view(B, N, -1).mean(dim=1)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int()

        y_true_all.append(y.cpu())
        y_prob_all.append(probs.cpu())
        y_pred_all.append(preds.cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_prob = torch.cat(y_prob_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    metrics = {
        "auroc_micro": roc_auc_score(y_true, y_prob, average="micro"),
        "auroc_macro": roc_auc_score(y_true, y_prob, average="macro"),
        "auroc_per_class": roc_auc_score(y_true, y_prob, average=None),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "report_dict": classification_report(
            y_true, y_pred, output_dict=True, digits=4
        ),
        "y_true": y_true,
        "y_prob": y_prob
    }

    return metrics

def plot_per_class_auroc(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 4))
    sns.barplot(x="class_id", y="auroc", data=df)
    plt.axhline(0.5, linestyle="--")
    plt.ylabel("AUROC")
    plt.xlabel("Class ID")
    plt.title("Per-Class AUROC (Multi-Crop)")
    plt.tight_layout()
    plt.show()

def plot_micro_macro_roc(y_true, y_prob):
    fpr_micro, tpr_micro, _ = roc_curve(
        y_true.ravel(), y_prob.ravel()
    )
    auc_micro = auc(fpr_micro, tpr_micro)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_micro, tpr_micro, label=f"Micro AUROC = {auc_micro:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_micro_pr(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(
        y_true.ravel(), y_prob.ravel()
    )
    ap = average_precision_score(y_true, y_prob, average="micro")

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"Micro AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_overall_metrics(metrics, num_crops, out_path):
    df = pd.DataFrame([{
        "auroc_micro": metrics["auroc_micro"],
        "auroc_macro": metrics["auroc_macro"],
        "f1_micro": metrics["f1_micro"],
        "f1_macro": metrics["f1_macro"],
        "num_crops": num_crops,
        "threshold": 0.5
    }])
    df.to_csv(out_path, index=False)


def save_per_class_auroc(per_class, out_path):
    df = pd.DataFrame({
        "class_id": np.arange(len(per_class)),
        "auroc": per_class
    })
    df.to_csv(out_path, index=False)


def save_classification_report(report_dict, out_path):
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(out_path)


def Main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    NUM_CLASSES = 18
    NUM_CROPS = 5
    BATCH_SIZE = 2  # safe for 8GB VRAM

    # Load configuration
    cfg = load_config(args.config)

    # Get label columns from config
    label_cols = cfg["labels"]

    test_df = pd.read_csv(BASE_DIR + "/Train_Test_Labels/filtered_valid_predicted_labels.csv")

    test_ds = MultiCropNumpyDataset(
        records=test_df.to_dict("records"),
        label_cols=label_cols,
        transform=get_test_transform(),
        num_crops=NUM_CROPS
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    encoder = load_moco_encoder(MODEL_DIR)
    model = LinearProbe(encoder).to(device)

    metrics = evaluate_multicrop(
        model,
        test_loader,
        device,
        num_crops=NUM_CROPS
    )

    os.makedirs(BASE_DIR + "/FinalTestResults", exist_ok=True)

    save_overall_metrics(
        metrics, NUM_CROPS, BASE_DIR + "/FinalTestResults/overall_metrics.csv"
    )
    save_per_class_auroc(
        metrics["auroc_per_class"], BASE_DIR + "/FinalTestResults/per_class_auroc.csv"
    )
    save_classification_report(
        metrics["report_dict"], BASE_DIR + "/FinalTestResults/classification_report.csv"
    )

    plot_per_class_auroc(BASE_DIR + "/FinalTestResults/per_class_auroc.csv")
    plot_micro_macro_roc(metrics["y_true"], metrics["y_prob"])
    plot_micro_pr(metrics["y_true"], metrics["y_prob"])



if __name__ == "__main__":
    Main()