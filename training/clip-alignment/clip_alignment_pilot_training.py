from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import numpy as np
import pandas as pd
from tqdm import tqdm

import random
import ast
from typing import List
import os
from transformers import AutoTokenizer, AutoModel
from monai.networks.nets import resnet as monai_resnet
from monai.transforms import (
    Compose,
    CenterSpatialCropd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
import math
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR  = "C:/MTI Research/outputs/final_cleaned_checkpoint_128run_add200/ResNet50_MoCo3D_best.pth"
OUTPUT_DIR = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/FinalEvaluation/"

log_columns = [
    "epoch",
    "R@1_I2T", "R@5_I2T", "R@10_I2T",
    "R@1_T2I", "R@5_T2I", "R@10_T2I",
    "pos_neg_gap", "val_loss", "temp", "Best"
]


# ------------------------------------------------------------------ #
#  TRANSFORMS                                                          #
# ------------------------------------------------------------------ #
def train_transform(roi_size=(32, 128, 128)):
    return Compose([
        RandSpatialCropd(keys=["img"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["img"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=["img"], factors=0.1,  prob=0.5),
        RandShiftIntensityd(keys=["img"], offsets=0.05, prob=0.5),
    ])


def val_transform(roi_size=(32, 128, 128)):
    return Compose([
        CenterSpatialCropd(keys=["img"], roi_size=roi_size),
    ])


# ------------------------------------------------------------------ #
#  DATASET                                                             #
# ------------------------------------------------------------------ #
class CTReportDataset(Dataset):
    def __init__(self, df, tokenizer, roi_size=(32, 128, 128), augment=False):
        self.df        = df.reset_index(drop=True)
        self.roi_size  = roi_size
        self.transform = train_transform(roi_size) if augment else val_transform(roi_size)

        print("Pre-tokenizing reports...")
        self.tokens = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Tokenizing"):
            t = tokenizer(
                row["report_text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            self.tokens.append({k: v.squeeze(0) for k, v in t.items()})
        print("✅ Tokenization done\n")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = row['paths']

        try:
            p_list = ast.literal_eval(path)
        except (ValueError, SyntaxError):
            p_list = path

        p = random.choice(p_list) if isinstance(p_list, list) else p_list

        arr  = np.load(p)
        vol  = torch.tensor(arr, dtype=torch.float32)
        data = self.transform({"img": vol})
        vol  = data["img"].as_tensor() if hasattr(data["img"], "as_tensor") \
               else torch.as_tensor(data["img"]).clone()

        return {"vol": vol, "text": self.tokens[idx]}


# ------------------------------------------------------------------ #
#  COLLATE                                                             #
# ------------------------------------------------------------------ #
def collate_fn(batch):
    vols  = torch.stack([item["vol"] for item in batch])
    texts = {
        key: torch.stack([item["text"][key] for item in batch])
        for key in batch[0]["text"]
    }
    return {"vol": vols, "text": texts}


# ------------------------------------------------------------------ #
#  MODEL                                                               #
# ------------------------------------------------------------------ #
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class CTCLIP(nn.Module):
    def __init__(self, image_encoder, image_dim, text_model_name,
                 init_temp=0.05, debug=False):
        super().__init__()
        self.debug = debug
        self.image_encoder_frozen = True
        self.text_encoder_frozen  = True

        self.image_encoder = image_encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.text_encoder = AutoModel.from_pretrained(
            text_model_name, trust_remote_code=True
        )
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.image_proj  = ProjectionHead(in_dim=image_dim)
        self.text_proj   = ProjectionHead(in_dim=self.text_encoder.config.hidden_size)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temp), dtype=torch.float32)
        )

    def encode_image(self, x):
        if self.image_encoder_frozen:
            with torch.no_grad():
                f = self.image_encoder(x)
        else:
            f = self.image_encoder(x)

        if self.debug:
            assert f.dim() == 2 and f.size(1) == 2048, \
                f"Unexpected image feature shape: {f.shape}"

        return self.image_proj(f)

    def encode_text(self, tokens):
        tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}

        if self.text_encoder_frozen:
            with torch.no_grad():
                out = self.text_encoder(**tokens)
        else:
            out = self.text_encoder(**tokens)

        hidden = out.last_hidden_state
        mask   = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.text_proj(pooled)


# ------------------------------------------------------------------ #
#  LOSS                                                                #
# ------------------------------------------------------------------ #
def clip_loss(image_emb, text_emb, logit_scale, label_smoothing=0.1):
    scale  = logit_scale.exp().clamp(max=100)
    logits = image_emb @ text_emb.t() * scale
    labels = torch.arange(len(image_emb), device=image_emb.device)
    loss_i = F.cross_entropy(logits,     labels, label_smoothing=label_smoothing)
    loss_t = F.cross_entropy(logits.t(), labels, label_smoothing=label_smoothing)
    return (loss_i + loss_t) / 2


# ------------------------------------------------------------------ #
#  SCHEDULER                                                           #
# ------------------------------------------------------------------ #
def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / \
                   max(1, total_steps - warmup_steps)
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------ #
#  UNFREEZE HELPERS                                                    #
# ------------------------------------------------------------------ #
def unfreeze_last_resnet_blocks(vision_encoder: nn.Module,
                                n_blocks: int = 2) -> List[nn.Parameter]:
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        if not hasattr(vision_encoder, name):
            raise ValueError(f"Expected vision_encoder to have `{name}`")

    stages = [vision_encoder.layer1, vision_encoder.layer2,
              vision_encoder.layer3, vision_encoder.layer4]
    start  = max(0, len(stages) - n_blocks)

    trainable: List[nn.Parameter] = []
    for i, stage in enumerate(stages):
        req = i >= start
        for p in stage.parameters():
            p.requires_grad = req
            if req:
                trainable.append(p)
    return trainable


def _get_text_encoder_layers(model):
    enc = model.text_encoder
    if hasattr(enc, "bert") and hasattr(enc.bert, "encoder"):
        return enc.bert.encoder.layer
    elif hasattr(enc, "encoder"):
        return enc.encoder.layer
    else:
        raise AttributeError(
            f"Cannot find transformer layers in {type(enc).__name__}"
        )


def start_phase2(model, optimizer):
    print("Phase 2 — Unfreezing top encoder layers...")
    model.image_encoder_frozen = False
    model.text_encoder_frozen  = False

    vision_params = unfreeze_last_resnet_blocks(model.image_encoder, n_blocks=2)
    optimizer.add_param_group({
        "params":       vision_params,
        "lr":           1e-5,
        "weight_decay": 1e-2,
    })

    text_layers = _get_text_encoder_layers(model)
    text_params = []
    for layer in text_layers[-2:]:
        for p in layer.parameters():
            p.requires_grad = True
            text_params.append(p)
    optimizer.add_param_group({
        "params":       text_params,
        "lr":           5e-6,
        "weight_decay": 1e-2,
    })

    print(f"  Image params unfrozen : {sum(p.numel() for p in vision_params):,}")
    print(f"  Text params unfrozen  : {sum(p.numel() for p in text_params):,}")


# ------------------------------------------------------------------ #
#  OVERFITTING DETECTOR                                                #
# ------------------------------------------------------------------ #
class OverfitDetector:
    """
    Triggers when val_loss has been rising consistently for `patience`
    epochs AND the gap between train_loss and val_loss exceeds `gap_threshold`.

    This is separate from the R@1 patience — it specifically catches the case
    where the model memorizes training pairs (train loss → 0, val loss → ∞)
    even if R@1 keeps fluctuating upward.
    """
    def __init__(self, patience=5, gap_threshold=1.0, min_epochs=5):
        self.patience      = patience       # consecutive val_loss increases to trigger
        self.gap_threshold = gap_threshold  # train/val loss gap that signals memorization
        self.min_epochs    = min_epochs     # don't trigger in the first N epochs of a phase
        self.val_loss_history = []
        self.consecutive_increases = 0
        self.phase_epoch   = 0             # epochs elapsed in current phase

    def reset(self):
        """Call when a new phase starts."""
        self.val_loss_history      = []
        self.consecutive_increases = 0
        self.phase_epoch           = 0

    def update(self, train_loss: float, val_loss: float) -> bool:
        """
        Returns True if overfitting is detected and training should stop.
        """
        self.phase_epoch += 1

        # Don't trigger in the warmup window of a phase
        if self.phase_epoch <= self.min_epochs:
            self.val_loss_history.append(val_loss)
            return False

        # Check if val_loss is rising
        if len(self.val_loss_history) > 0 and val_loss > self.val_loss_history[-1]:
            self.consecutive_increases += 1
        else:
            self.consecutive_increases = 0

        self.val_loss_history.append(val_loss)

        # Gap check: train much lower than val = memorization
        gap_exceeded  = (val_loss - train_loss) > self.gap_threshold
        loss_rising   = self.consecutive_increases >= self.patience

        if gap_exceeded and loss_rising:
            print(
                f"  🔥 Overfit detected: val_loss rising for {self.consecutive_increases} "
                f"epochs, train/val gap={val_loss - train_loss:.3f} > {self.gap_threshold}"
            )
            return True

        return False


# ------------------------------------------------------------------ #
#  TRAINING LOOP                                                       #
# ------------------------------------------------------------------ #
def train_clip_pilot(
    model, trainLoader, valLoader, optimizer, device,
    start_epoch=1, epochs=75, best_r1=-1e9,
    patience=14,
    grad_accum_steps=2,
    label_smoothing=0.1,
    overfit_patience=5,
    overfit_gap=1.0,
):
    logs       = []
    scaler     = torch.cuda.amp.GradScaler()
    bad_epochs = 0

    steps_per_epoch = len(trainLoader)

    # ---- Determine if we're starting in Phase 1 or resuming in Phase 2 ----
    phase2_started = start_epoch > 15

    if not phase2_started:
        # Starting from Phase 1
        phase1_steps = 15 * steps_per_epoch
        warmup_steps = max(1, int(0.10 * phase1_steps))
        scheduler    = build_scheduler(optimizer, phase1_steps, warmup_steps)

        # Advance scheduler if resuming mid-phase-1
        if start_epoch > 1:
            steps_done = (start_epoch - 1) * steps_per_epoch
            for _ in range(steps_done):
                scheduler.step()

        print(f"Phase 1 scheduler: {warmup_steps} warmup / {phase1_steps} total steps")
        overfit_detector = OverfitDetector(
            patience=overfit_patience, gap_threshold=overfit_gap, min_epochs=5
        )
    else:
        # Resuming directly into Phase 2 — build Phase 2 scheduler
        # and advance it to where we left off
        phase2_total_steps = (epochs - 15) * steps_per_epoch
        p2_warmup          = max(1, int(0.10 * phase2_total_steps))
        scheduler          = build_scheduler(optimizer, phase2_total_steps, p2_warmup)

        steps_already_done = (start_epoch - 16) * steps_per_epoch
        for _ in range(steps_already_done):
            scheduler.step()

        print(f"Resumed Phase 2 scheduler at step "
              f"{steps_already_done}/{phase2_total_steps} "
              f"(warmup={p2_warmup})")

        overfit_detector = OverfitDetector(
            patience=overfit_patience, gap_threshold=overfit_gap,
            min_epochs=max(5, steps_already_done // steps_per_epoch)
        )

    for epoch in range(start_epoch, epochs + 1):

        # ---- Phase 2 transition (only fires once, from Phase 1) ----
        if epoch == 16 and not phase2_started:
            start_phase2(model, optimizer)
            phase2_started = True

            phase2_total_steps = (epochs - 15) * steps_per_epoch
            p2_warmup          = max(1, int(0.10 * phase2_total_steps))
            scheduler          = build_scheduler(optimizer, phase2_total_steps, p2_warmup)

            print(f"Phase 2 scheduler: {p2_warmup} warmup / {phase2_total_steps} total steps")

            # Reset both patience counters for Phase 2
            bad_epochs = 0
            overfit_detector.reset()
            print("  bad_epochs and overfit detector reset for Phase 2")

        # ---- Training ----
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(trainLoader, desc=f"Epoch {epoch}"), start=1):
            volumes = batch["vol"].to(device, non_blocking=True)
            texts   = batch["text"]

            with torch.cuda.amp.autocast():
                img  = model.encode_image(volumes)
                txt  = model.encode_text(texts)
                loss = clip_loss(img, txt, model.logit_scale,
                                 label_smoothing=label_smoothing) / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0 or step == len(trainLoader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * grad_accum_steps

        avg_loss = total_loss / len(trainLoader)
        val      = evaluate(model, valLoader, device)
        medR1    = (val["R@1_I2T"] + val["R@1_T2I"]) / 2
        phase    = 1 if epoch <= 15 else 2
        cur_lr   = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch} [Phase {phase}] | Train Loss: {avg_loss:.4f} | LR: {cur_lr:.2e}")
        print(
            f"Epoch {epoch:02d} | val_loss={val['val_loss']:.4f} | "
            f"R1_I2T={val['R@1_I2T']:.4f} R5={val['R@5_I2T']:.4f} "
            f"R10={val['R@10_I2T']:.4f} | R1_T2I={val['R@1_T2I']:.4f} | "
            f"temp={1.0 / model.logit_scale.exp().item():.4f}"
        )

        # ---- Save last checkpoint ----
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(
            {"epoch": epoch, "r1": medR1,
             "best_r1": best_r1,
             "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict(),
             "scheduler_state": scheduler.state_dict()},
            os.path.join(OUTPUT_DIR, "last_model_unfreeze.pt")
        )

        # ---- Best model tracking ----
        improved = medR1 > best_r1 + 1e-4
        if improved:
            best_r1    = medR1
            bad_epochs = 0
            torch.save(
                {"epoch": epoch, "best_r1": best_r1,
                 "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict()},
                os.path.join(OUTPUT_DIR, "best_model_unfreeze.pt")
            )
            print(f"✅ New best R@1={best_r1:.4f}! Saving model at epoch {epoch}")
        else:
            bad_epochs += 1

            # Phase 1: no early stopping — always run all 15 epochs
            if epoch < 16:
                print(f"  ⏳ No improvement. (Phase 1 — early stopping disabled)")
            else:
                print(f"  ⏳ No improvement. bad_epochs={bad_epochs}/{patience}")
                if bad_epochs >= patience:
                    print(f"🛑 Early stopping (no R@1 improvement) at epoch {epoch}.")
                    break

        # ---- Overfitting check (Phase 2 only) ----
        if epoch >= 16:
            if overfit_detector.update(avg_loss, val["val_loss"]):
                print(f"🛑 Early stopping (overfitting detected) at epoch {epoch}.")
                print(f"   Train loss={avg_loss:.4f} | Val loss={val['val_loss']:.4f} | "
                      f"Gap={val['val_loss'] - avg_loss:.4f}")
                break

        logs.append([
            epoch,
            val.get("R@1_I2T", 0), val.get("R@5_I2T", 0), val.get("R@10_I2T", 0),
            val.get("R@1_T2I", 0), val.get("R@5_T2I", 0), val.get("R@10_T2I", 0),
            val.get("pos_neg_gap", 0), val.get("val_loss", 0),
            1.0 / model.logit_scale.exp().item(),
            "Yes" if improved else "No"
        ])
        pd.DataFrame(logs, columns=log_columns).to_csv(
            os.path.join(OUTPUT_DIR, "logs.csv"), index=False
        )

    return best_r1


# ------------------------------------------------------------------ #
#  ENCODER LOADER                                                      #
# ------------------------------------------------------------------ #
def load_moco_encoder(checkpoint_path, in_channels=1):
    backbone    = monai_resnet.resnet50(spatial_dims=3,
                                        n_input_channels=in_channels,
                                        num_classes=1)
    backbone.fc = nn.Identity()

    ckpt        = torch.load(checkpoint_path, map_location="cpu")
    state_dict  = ckpt.get("model_state_dict", ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("encoder_q."):
            cleaned[k[len("encoder_q."):]] = v
        elif k.startswith("module.encoder_q."):
            cleaned[k[len("module.encoder_q."):]] = v

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    print("Missing keys   :", missing)
    print("Unexpected keys:", unexpected)
    return backbone


# ------------------------------------------------------------------ #
#  EVALUATION                                                          #
# ------------------------------------------------------------------ #
def _recall_at_k(sim_matrix, k):
    k       = min(k, sim_matrix.size(1))
    ranks   = sim_matrix.argsort(dim=1, descending=True)
    targets = torch.arange(len(sim_matrix))
    return (ranks[:, :k] == targets[:, None]).any(dim=1).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_img, all_txt = [], []
    total_loss, n    = 0.0, 0

    for batch in loader:
        vol   = batch["vol"].to(device, non_blocking=True)
        texts = batch["text"]

        with torch.cuda.amp.autocast():
            img  = model.encode_image(vol)
            txt  = model.encode_text(texts)
            loss = clip_loss(img, txt, model.logit_scale, label_smoothing=0.0)

        all_img.append(img.cpu())
        all_txt.append(txt.cpu())
        total_loss += loss.item() * vol.size(0)
        n          += vol.size(0)

    I   = torch.cat(all_img, dim=0)
    T   = torch.cat(all_txt, dim=0)
    sim = (I @ T.t()) * model.logit_scale.exp().detach().cpu()

    return {
        "R@1_I2T":     _recall_at_k(sim,      1),
        "R@5_I2T":     _recall_at_k(sim,      5),
        "R@10_I2T":    _recall_at_k(sim,     10),
        "R@1_T2I":     _recall_at_k(sim.t(),  1),
        "R@5_T2I":     _recall_at_k(sim.t(),  5),
        "R@10_T2I":    _recall_at_k(sim.t(), 10),
        "pos_neg_gap": (sim.diag().mean() -
                        sim[~torch.eye(len(sim), dtype=torch.bool)].mean()).item(),
        "val_loss":    total_loss / max(1, n)
    }


@torch.no_grad()
def evaluate_retrieval(model, loader, device):
    model.eval()
    all_img, all_txt = [], []

    for batch in loader:
        vol = batch["vol"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            all_img.append(model.encode_image(vol).cpu())
            all_txt.append(model.encode_text(batch["text"]).cpu())

    I   = torch.cat(all_img, dim=0)
    T   = torch.cat(all_txt, dim=0)
    sim = (I @ T.t()) * model.logit_scale.exp().detach().cpu()

    result = {
        "R@1_I2T":     _recall_at_k(sim,      1),
        "R@5_I2T":     _recall_at_k(sim,      5),
        "R@10_I2T":    _recall_at_k(sim,     10),
        "R@1_T2I":     _recall_at_k(sim.t(),  1),
        "R@5_T2I":     _recall_at_k(sim.t(),  5),
        "R@10_T2I":    _recall_at_k(sim.t(), 10),
        "pos_neg_gap": (sim.diag().mean() -
                        sim[~torch.eye(len(sim), dtype=torch.bool)].mean()).item()
    }
    return result, sim, I, T


# ------------------------------------------------------------------ #
#  ENTRY POINT                                                         #
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    TRAIN_CSV = "C:/MTI Research/M3D_CLIP/Data/train_val_merged.csv"
    VAL_CSV   = "C:/MTI Research/M3D_CLIP/Data/test_cleaned.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True
    )

    train_df = pd.read_csv(TRAIN_CSV, encoding="cp1252")
    val_df   = pd.read_csv(VAL_CSV,   encoding="cp1252")

    train_ds = CTReportDataset(train_df, tokenizer, roi_size=(32, 128, 128), augment=True)
    val_ds   = CTReportDataset(val_df,   tokenizer, roi_size=(32, 128, 128), augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              num_workers=2, pin_memory=False,
                              persistent_workers=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              num_workers=2, pin_memory=False,
                              persistent_workers=True, collate_fn=collate_fn)

    encoder = load_moco_encoder(MODEL_DIR)
    model   = CTCLIP(
        image_encoder=encoder,
        image_dim=2048,
        text_model_name="microsoft/BiomedVLP-CXR-BERT-specialized",
        init_temp=0.05,
        debug=False,
    ).to(device)

    optimizer = AdamW(
        [
            {"params": list(model.image_proj.parameters()), "lr": 1e-4, "weight_decay": 1e-4},
            {"params": list(model.text_proj.parameters()),  "lr": 1e-4, "weight_decay": 1e-4},
            {"params": [model.logit_scale],                 "lr": 1e-4, "weight_decay": 0.0},
        ]
    )

    # ----------------------------------------------------------------
    # Toggle below: fresh start vs resume
    # ----------------------------------------------------------------
    RESUME = True   # ← set False to train from scratch

    if RESUME:
        ckpt_path = os.path.join(OUTPUT_DIR, "best_model_unfreeze.pt")
        ckpt      = torch.load(ckpt_path, map_location=device)
        resume_epoch = ckpt["epoch"] + 1
        resume_r1    = ckpt["best_r1"]

        model.load_state_dict(ckpt["model_state"])

   
        if resume_epoch > 16:
            print("Re-attaching Phase 2 encoder params to optimizer...")
            start_phase2(model, optimizer)

        optimizer.load_state_dict(ckpt["optimizer_state"])

        print(f"Resumed from epoch {ckpt['epoch']} | best R@1: {resume_r1:.4f}")
        print(f"Optimizer param groups: {len(optimizer.param_groups)}")

        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])
    else:
        resume_epoch = 1
        resume_r1    = -1e9
        print("Starting training from scratch...")

    best_r1 = train_clip_pilot(
        model            = model,
        trainLoader      = train_loader,
        valLoader        = val_loader,
        optimizer        = optimizer,
        device           = device,
        start_epoch      = resume_epoch,
        epochs           = 75,          # extended — was 45
        best_r1          = resume_r1,
        patience         = 14,
        grad_accum_steps = 2,
        label_smoothing  = 0.1,
        overfit_patience = 5,           # consecutive val_loss increases to trigger
        overfit_gap      = 1.0,         # train/val gap that signals memorization
    )

    print(f"\nTraining complete. Best R@1: {best_r1:.4f}")

    # ---- Final evaluation on best checkpoint ----
    best_ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model_unfreeze.pt"),
                           map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print(f"Loaded best model from epoch {best_ckpt['epoch']} | "
          f"R@1={best_ckpt['best_r1']:.4f}")

    final_metrics, sim, img_emb, txt_emb = evaluate_retrieval(model, val_loader, device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame([final_metrics]).to_csv(
        OUTPUT_DIR + "retrieval_metrics_unfreeze.csv",  index=False)
    pd.DataFrame(sim.numpy()).to_csv(
        OUTPUT_DIR + "similarity_matrix_unfreeze.csv", index=False)
    pd.DataFrame(img_emb.numpy()).to_csv(
        OUTPUT_DIR + "image_embeddings_unfreeze.csv",  index=False)
    pd.DataFrame(txt_emb.numpy()).to_csv(
        OUTPUT_DIR + "text_embeddings_unfreeze.csv",   index=False)

    print(final_metrics)