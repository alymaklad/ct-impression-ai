import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.cuda import amp
from monai.networks.nets import resnet as monai_resnet

from dataset import TwoViewNumpyDataset
from transforms import get_view_transforms
from moco3d import MoCo3D
from utils import seed_all, ensure_dir, load_config

import warnings
warnings.filterwarnings("ignore")


def build_backbone(pretrained_path=None, in_channels=1):
    # MONAI resnet50: num_classes=1 used but we'll take features before fc
    backbone = monai_resnet.resnet50(
        spatial_dims=3,
        n_input_channels=in_channels,
        num_classes=1
    )
    # remove final fc: replace with identity and ensure global pooling exists
    backbone.fc = torch.nn.Identity()

    if pretrained_path and os.path.exists(pretrained_path):
        sd = torch.load(pretrained_path, map_location="cpu")
        # load with strict=False to handle naming mismatches
        backbone.load_state_dict(sd, strict=False)

    return backbone


def load_checkpoint(resume_path, model, optimizer=None, scaler=None, device="cpu"):
    """
    Load checkpoint and restore model/optimizer/scaler states.
    Returns:
        start_epoch, best_val_loss
    """
    if not resume_path or not os.path.exists(resume_path):
        print(f"No valid checkpoint found at: {resume_path}")
        return 0, float("inf")

    print(f"Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"Resumed at epoch {start_epoch}")
    print(f"Best validation loss so far: {best_val_loss:.6f}")

    return start_epoch, best_val_loss


def train_one_epoch(model, dataloader, optimizer, scaler, device, use_amp=True):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        im_q = batch["img_q"].to(device)
        im_k = batch["img_k"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with amp.autocast():
                loss, logits = model(im_q, im_k)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, logits = model(im_q, im_k)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * im_q.size(0)

    return total_loss / len(dataloader.dataset)


def validate_one_epoch(model, dataloader, scaler, device, use_amp=True):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            im_q = batch["img_q"].to(device)
            im_k = batch["img_k"].to(device)

            if use_amp:
                with amp.autocast():
                    loss, logits = model(im_q, im_k)
            else:
                loss, logits = model(im_q, im_k)

            total_loss += loss.item() * im_q.size(0)

    return total_loss / len(dataloader.dataset)


def adjust_learning_rate(optimizer, epoch, cfg):
    """
    Adjust the learning rate according to the cosine schedule with warmup.
    """
    lr_max = float(cfg["lr"])
    lr_min = float(cfg["lr_min"])
    warmup_epochs = int(cfg["warmup_epochs"])
    total_epochs = int(cfg["num_epochs"])

    if epoch < warmup_epochs:
        lr = lr_max * (epoch + 1) / warmup_epochs
    else:
        curr_epoch = epoch - warmup_epochs
        remain_epochs = total_epochs - warmup_epochs
        lr = lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * curr_epoch / remain_epochs)
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def run_pretraining(cfg, fold_items, val_items=None, fold_name="ResNet50_MoCo3D"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg["checkpoint_dir"])

    viewA, viewB = get_view_transforms(roi_size=cfg["roi_size"])

    dataset = TwoViewNumpyDataset(
        fold_items,
        transform_q=viewA,
        transform_k=viewB
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True
    )
    

    val_dataloader = None
    if val_items:
        val_dataset = TwoViewNumpyDataset(
            val_items,
            transform_q=viewA,
            transform_k=viewB
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg["val_batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            drop_last=False,
            pin_memory=True
        )

    print(f"Training samples: {len(dataset)} | Validation samples: {len(val_items) if val_items else 0}")
    
    backbone = build_backbone(pretrained_path=cfg["pretrained_backbone"])
    model = MoCo3D(
        backbone,
        feat_dim=int(cfg["proj_dim"]),
        K=int(cfg["queue_size"]),
        m=float(cfg["momentum"]),
        T=float(cfg["temperature"])
    ).to(device)

    optimizer = SGD(
        model.parameters(),
        lr=float(cfg["lr"]),
        momentum=float(cfg["sgd_momentum"]),
        weight_decay=float(cfg["weight_decay"])
    )
    scaler = amp.GradScaler(enabled=cfg["use_amp"])

    # Resume support
    start_epoch = 0
    best_val_loss = float("inf")

    resume_path = cfg.get("resume_checkpoint", None)
    if resume_path:
        start_epoch, best_val_loss = load_checkpoint(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )

    print(f"Starting training for {fold_name} on {device}...")

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Training started at: {current_time}")

    metrics_path = os.path.join(cfg["checkpoint_dir"], f"{fold_name}_metrics.csv")

    # only create a new metrics file if starting from scratch
    if start_epoch == 0 or not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,train_loss,val_loss,time,best\n")

    min_delta = 1e-4
    epochs_no_improve = 0
    patience = 25

    training_start = time.time()

    for epoch in range(start_epoch, cfg["num_epochs"]):
        epoch_start = time.time()

        new_lr = adjust_learning_rate(optimizer, epoch, cfg)

        train_loss = train_one_epoch(
            model, dataloader, optimizer, scaler, device, cfg["use_amp"]
        )

        epoch_time_sec = time.time() - epoch_start
        total_time_sec = time.time() - training_start
        val_loss = 0.0

        if val_dataloader:
            val_loss = validate_one_epoch(
                model, val_dataloader, scaler, device, cfg["use_amp"]
            )
            print(
                f"Epoch {epoch+1}/{cfg['num_epochs']} [LR={new_lr:.6f}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Time: {total_time_sec/60:.2f} min (Epoch: {epoch_time_sec:.2f} sec)"
            )
        else:
            print(
                f"Epoch {epoch+1}/{cfg['num_epochs']} [LR={new_lr:.6f}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Time: {total_time_sec/60:.2f} min (Epoch: {epoch_time_sec:.2f} sec)"
            )

        best = val_loss < best_val_loss

        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{total_time_sec/60:.2f},{best}\n"
            )

        last_path = os.path.join(cfg["checkpoint_dir"], f"{fold_name}_last.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "cfg": cfg,
        }

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        torch.save(checkpoint, last_path)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(cfg["checkpoint_dir"], f"{fold_name}_best.pth")

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "cfg": cfg,
            }

            if scaler is not None:
                checkpoint["scaler_state_dict"] = scaler.state_dict()

            torch.save(checkpoint, best_path)
            print(f"Saved new best checkpoint to {best_path} (Val Loss: {val_loss:.4f})")
        else:
            if epoch > 35:
                epochs_no_improve += 1
                print(f"No improvement in val loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml"
    )
    parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    train_df = pd.read_csv(cfg["train_csv"], encoding="cp1252")
    fold_items = train_df.to_dict("records")

    val_df = pd.read_csv(cfg["val_csv"], encoding="cp1252")
    val_items = val_df.to_dict("records")

    run_pretraining(cfg, fold_items, val_items=val_items)