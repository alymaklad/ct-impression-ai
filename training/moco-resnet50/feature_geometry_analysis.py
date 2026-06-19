import torch
import torch.nn as nn
from monai.networks.nets import resnet as monai_resnet
import torch.nn.functional as F
from monai.transforms import CenterSpatialCropd,Compose,EnsureTyped
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from torch.utils.data import Dataset, DataLoader

import argparse
import yaml
from utils import args, load_config

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/Evaluation"
MODEL_DIR = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/Evaluation/checkpoints/ResNet50_best_linear_probe_multilabel.pth"


def get_test_transform(roi_size=(96, 96, 96)):
    return Compose([
    EnsureTyped(keys=['img']),
    CenterSpatialCropd(keys=['img'], roi_size=(96,96,96)),
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

@torch.no_grad()
def extract_features(model, loader, device, num_crops=1):
    """
    Returns:
        features: Tensor [N, D]
    """
    model.eval()
    feats_all = []

    for x, _ in loader:
        # x shape: [B, N, C, D, H, W] OR [B, C, D, H, W]
        x = x.to(device, non_blocking=True)

        if x.ndim == 6:
            B, N, C, D, H, W = x.shape
            x = x.view(B * N, C, D, H, W)
        else:
            B, C, D, H, W = x.shape
            N = 1

        feats = model(x)                  # (B*N, feat_dim)
        feats = feats.view(B, N, -1).mean(dim=1)  # aggregate crops

        feats_all.append(feats.cpu())

    features = torch.cat(feats_all, dim=0)
    return features


def feature_mean_norm(features):
    mean_feature = features.mean(dim=0)
    return torch.norm(mean_feature, p=2).item()


def feature_variance_per_dim(features):
    return features.var(dim=0, unbiased=False)  # [D]


def average_pairwise_cosine_similarity(features):
    features = F.normalize(features, dim=1)
    sim_matrix = features @ features.T

    N = sim_matrix.size(0)
    avg_cos_sim = (sim_matrix.sum() - N) / (N * (N - 1))
    return avg_cos_sim.item()


def save_global_metrics_csv(save_dir, metrics_dict):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "feature_geometry_metrics.csv")

    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics_dict.items():
            writer.writerow([k, v])


def save_variance_csv(save_dir, var_per_dim):
    path = os.path.join(save_dir, "feature_variance_per_dimension.csv")

    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dimension", "Variance"])
        for i, v in enumerate(var_per_dim.cpu().numpy()):
            writer.writerow([i, v])


def plot_variance_distribution(save_dir, var_per_dim):
    plt.figure()
    plt.hist(var_per_dim.cpu().numpy(), bins=50)
    plt.xlabel("Feature Variance")
    plt.ylabel("Count")
    plt.title("Feature Variance Distribution")
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "variance_distribution.png"))
    plt.close()


def plot_variance_spectrum(save_dir, var_per_dim):
    sorted_var = torch.sort(var_per_dim, descending=True).values.cpu().numpy()

    plt.figure()
    plt.plot(sorted_var)
    plt.xlabel("Feature Dimension (sorted)")
    plt.ylabel("Variance")
    plt.title("Feature Variance Spectrum")
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "variance_spectrum.png"))
    plt.close()


def plot_cosine_similarity_heatmap(save_dir, features, max_samples=300):
    features = F.normalize(features, dim=1)

    if features.size(0) > max_samples:
        idx = torch.randperm(features.size(0))[:max_samples]
        features = features[idx]

    sim_matrix = (features @ features.T).cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(sim_matrix)
    plt.colorbar()
    plt.title("Pairwise Cosine Similarity Heatmap")
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "cosine_similarity_heatmap.png"))
    plt.close()


def run_feature_geometry_analysis(
    encoder,
    val_loader,
    device,
    save_dir="feature_geometry_results"
):
    os.makedirs(save_dir, exist_ok=True)

    # Extract features
    features = extract_features(encoder, val_loader, device)

    # Metrics
    mean_norm = feature_mean_norm(features)
    var_per_dim = feature_variance_per_dim(features)
    avg_cos_sim = average_pairwise_cosine_similarity(features)

    metrics = {
        "feature_mean_norm": mean_norm,
        "mean_feature_variance": var_per_dim.mean().item(),
        "min_feature_variance": var_per_dim.min().item(),
        "max_feature_variance": var_per_dim.max().item(),
        "avg_pairwise_cosine_similarity": avg_cos_sim
    }

    # Save CSVs
    save_global_metrics_csv(save_dir, metrics)
    save_variance_csv(save_dir, var_per_dim)

    # Plots
    plot_variance_distribution(save_dir, var_per_dim)
    plot_variance_spectrum(save_dir, var_per_dim)
    plot_cosine_similarity_heatmap(save_dir, features)

    return metrics


def load_ssl_encoder(checkpoint_path, device):
    """
    Loads ONLY the encoder weights from a MoCo/SSL checkpoint.
    """
    # Build backbone exactly as in SSL training
    encoder = monai_resnet.resnet50(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1,
    )
    encoder.fc = nn.Identity()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # ---- Strip prefixes safely ----
    state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("encoder_q."):
            state_dict[k.replace("encoder_q.", "")] = v
        elif k.startswith("encoder."):
            state_dict[k.replace("encoder.", "")] = v

    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    encoder.to(device)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    NUM_CLASSES = 18
    NUM_CROPS = 1
    BATCH_SIZE = 2  


    # Example usage
    encoder = load_ssl_encoder(
        checkpoint_path=MODEL_DIR,
        device=device
    )

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

    encoder.to(device)

    metrics = run_feature_geometry_analysis(
        encoder=encoder,
        val_loader=test_loader,
        device=device,
        save_dir=BASE_DIR + "/feature_geometry_results/"
    )

    print(metrics)
