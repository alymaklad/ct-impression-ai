import ast
from pyexpat import model
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from typing import List
import random
from utils import args, load_config, save_training_log

from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from monai.networks.nets import resnet as monai_resnet
from monai.transforms import Compose, RandSpatialCropd, EnsureTyped ,CenterSpatialCropd
import warnings
warnings.filterwarnings("ignore")
# def final_train_clip(
#     model,
#     train_loader,
#     val_loader,
#     device,
#     csv_path,
#     total_epochs=50,
#     warmup_epochs=5,
#     unfreeze_resnet_epoch=6,
#     unfreeze_resnet_layer3_epoch=16,
#     unfreeze_bert_epoch=26,
#     lr_proj=1e-3,
#     lr_resnet=1e-4,
#     lr_resnet_layer3=5e-5,
#     lr_bert=5e-5,
#     weight_decay=1e-4,
#     max_grad_norm=1.0
# ):
#     """
#     Final training loop for CTCLIP with progressive unfreezing and separate validation function.
#     Saves metrics to CSV after each epoch.
#     """
#     model.train()

#     # Freeze everything at start
#     for p in model.image_encoder.parameters():
#         p.requires_grad = False
#     for p in model.text_encoder.parameters():
#         p.requires_grad = False

#     optimizer = None
#     current_stage = None
#     best_val_r1 = 0.0
#     best_model_state = None
#     model.logit_scale.requires_grad = False

#     r1_list = []           # store average R@1 per epoch
#     pos_neg_gap_list = []   # store pos_neg_gap per epoch
#     epochs_list = []        # store epoch numbers
#     # CSV logging
#     log_columns = ["epoch", "stage", "train_loss", "R@1_I2T", "R@5_I2T", "R@10_I2T",
#                    "R@1_T2I", "R@5_T2I", "R@10_T2I", "pos_neg_gap"]
#     logs = []

#     for epoch in range(1, total_epochs + 1):
#         # ----------------------------
#         # Determine stage
#         # ----------------------------
#         if epoch <= warmup_epochs:
#             stage = 1
#         elif epoch < unfreeze_resnet_layer3_epoch:
#             stage = 2
#         elif epoch < unfreeze_bert_epoch:
#             stage = 3
#         else:
#             stage = 4

#         # ----------------------------
#         # Update optimizer if stage changed
#         # ----------------------------
#         if stage != current_stage:
#             param_groups = [{"params": model.image_proj.parameters(), "lr": lr_proj},
#                             {"params": model.text_proj.parameters(), "lr": lr_proj}]
            
#             if stage >= 2:
#                 for name, p in model.image_encoder.named_parameters():
#                     if "layer4" in name:
#                         p.requires_grad = True
#                 param_groups.append({"params": model.image_encoder.layer4.parameters(), "lr": lr_resnet})
                
#                 model.logit_scale.requires_grad = True


#             if stage >= 3:
#                 for name, p in model.image_encoder.named_parameters():
#                     if "layer3" in name:
#                         p.requires_grad = True
#                 param_groups.append({"params": model.image_encoder.layer3.parameters(), "lr": lr_resnet_layer3})

#             if stage >= 4:
#                 for i in [10, 11]:  # last 2 BERT layers
#                     for p in model.text_encoder.encoder.layer[i].parameters():
#                         p.requires_grad = True
#                 param_groups.append({"params": model.text_encoder.encoder.layer[10:].parameters(), "lr": lr_bert})

#             optimizer = AdamW(param_groups, weight_decay=weight_decay)
#             current_stage = stage
#             print(f"[Stage {stage} @ epoch {epoch}] Optimizer updated.")

#         # print(f"Training stage {current_stage} for epoch {epoch} with optimizer:")
#         # print(optimizer)
#         # print("Trainable image params:",
#         #     sum(p.requires_grad for p in model.image_encoder.parameters()))
#         # print("Trainable text params:",
#         #     sum(p.requires_grad for p in model.text_encoder.parameters()))

#         # ----------------------------
#         # Training loop
#         # ----------------------------
#         total_loss = 0.0
#         for crops, tokens in tqdm(train_loader, desc=f"Epoch {epoch}"):
#             crops = crops.to(device)
#             tokens = {k: v.to(device) for k, v in tokens.items()}

#             B, N, C, D, H, W = crops.shape
#             crops = crops.view(B * N, C, D, H, W)

#             img_emb = model.encode_image(crops)
#             img_emb = img_emb.view(B, N, -1).mean(dim=1)

#             txt_emb = model.encode_text(tokens)

#             loss = clip_loss(img_emb, txt_emb, model.logit_scale)

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch:03d} | Stage {current_stage} | Train Loss: {avg_loss:.4f}")

#         # ----------------------------
#         # Validation using separate function
#         # ----------------------------
#         metrics, sim, I, T = evaluate_retrieval(model, val_loader, device)
#         r1 = (metrics["R@1_I2T"] + metrics["R@1_T2I"]) / 2
#         print(f"Validation | R@1 avg: {r1:.4f} | pos_neg_gap: {metrics['pos_neg_gap']:.4f}")

#         r1_list.append(r1)
#         pos_neg_gap_list.append(metrics["pos_neg_gap"])
#         epochs_list.append(epoch)
        
#         save_training_log(epochs_list, r1_list, pos_neg_gap_list)
#         # ----------------------------
#         # Save best model
#         # ----------------------------
#         if r1 > best_val_r1:
#             best_val_r1 = r1
#             best_model_state = model.state_dict()
#             print(f"New best R@1! Saving model at epoch {epoch}")

#         # ----------------------------
#         # Log metrics to CSV
#         # ----------------------------
#         logs.append([
#             epoch, stage, avg_loss,
#             metrics.get("R@1_I2T", 0),
#             metrics.get("R@5_I2T", 0),
#             metrics.get("R@10_I2T", 0),
#             metrics.get("R@1_T2I", 0),
#             metrics.get("R@5_T2I", 0),
#             metrics.get("R@10_T2I", 0),
#             metrics.get("pos_neg_gap", 0)
#         ])
#         df = pd.DataFrame(logs, columns=log_columns)
#         os.makedirs(os.path.dirname(csv_path), exist_ok=True)
#         df.to_csv(csv_path, index=False)

#     # Load best model
#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)
#         print(f"Loaded best model with R@1={best_val_r1:.4f}")

#     return model

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def probe_transform(roi_size):
    return Compose([
        # EnsureTyped(keys=['img']),
        CenterSpatialCropd(keys=['img'], roi_size=roi_size),
    ])
    
def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def set_group_lrs(optimizer: torch.optim.Optimizer, base_lrs: List[float], mult: float):
    for pg, base in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base * mult
    
class GroupedCTReportDataset(Dataset):
    """
    Each item corresponds to one (patient, scan) case.
    It has multiple reconstruction paths; we sample one per __getitem__ call.
    """
    def __init__(self, grouped_df: pd.DataFrame, transform=None, test_mode: bool = False):
        self.df = grouped_df.reset_index(drop=True)
        self.transform = transform
        self.test_mode = test_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
      
        # Sample one reconstruction (handles 2, 5, etc.)
        if self.test_mode:
            path = row["path"]  # deterministic for validation; can also do random.choice for more robustness
        else:
            path = random.choice(row["paths"])
            # Return a dummy zero volume and empty text to avoid crashing; this item will be effectively
        vol = np.load(path)  # expected (D,H,W) or (C,D,H,W)
        vol = torch.tensor(vol, dtype=torch.float32)

        # MONAI-style dict transforms
        data = {"img": vol}
        if self.transform is not None:
            data = self.transform(data)
            vol = data["img"]

        # Ensure (C,D,H,W)
        if vol.dim() == 3:
            vol = vol.unsqueeze(0)
        elif vol.dim() != 4:
            raise ValueError(f"Unexpected vol shape: {tuple(vol.shape)} for path={path}")

        return {"vol": vol, "text": row["report_text"]}

class ResNet50_CXRText_CLIP(nn.Module):
    """
    CLIP-style contrastive model with LN + GELU projection heads
    """

    def __init__(
        self,
        text_ckpt: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        embed_dim: int = 768,
        max_text_len: int = 512,
        freeze_text_encoder: bool = True,
        freeze_vision_backbone: bool = True,
        init_temp: float = 0.07,
        vision_backbone=None,
    ):
        super().__init__()

        # --------------------
        # Vision encoder (ResNet-50)
        # --------------------
       
        # Remove classification head
        self.vision_backbone = vision_backbone
        self.vision_feat_dim = 2048

        # Vision projection head: 2048 -> 768 -> 768
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if freeze_vision_backbone:
            for p in self.vision_backbone.parameters():
                p.requires_grad = False

        # --------------------
        # Text encoder (CXR-BERT)
        # --------------------
        self.tokenizer = AutoTokenizer.from_pretrained(text_ckpt, use_fast=True, trust_remote_code=True)
        self.text_encoder = AutoModel.from_pretrained(text_ckpt, trust_remote_code=True)

        text_hidden = self.text_encoder.config.hidden_size  # usually 768

        # Text projection head: 768 -> 768 -> 768
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.max_text_len = max_text_len

        # Temperature
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))
        self.embed_dim = embed_dim

    # --------------------
    # Encoding
    # --------------------
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,3,H,W) or (B,1,H,W) if first conv adapted
        """
        x = self.vision_backbone(img)   # (B,2048,1,1)
        x = x.flatten(1)                # (B,2048)
        x = self.vision_proj(x)         # (B,768)
        x = F.normalize(x, dim=-1)      # L2 norm
        return x

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}

        if not any(p.requires_grad for p in self.text_encoder.parameters()):
            with torch.no_grad():
                out = self.text_encoder(**toks)
        else:
            out = self.text_encoder(**toks)

        cls = out.last_hidden_state[:, 0, :]  # (B,768)
        z = self.text_proj(cls)               # (B,768)
        z = F.normalize(z, dim=-1)            # L2 norm
        return z

    # --------------------
    # CLIP loss
    # --------------------
    def clip_loss(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (img_emb @ txt_emb.t())
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)
        )




def unfreeze_last_resnet_blocks(
    vision_encoder: nn.Module,
    n_blocks: int = 1
) -> List[nn.Parameter]:
    """
    Unfreeze the last N ResNet stages (layer4, layer3, ...).

    Args:
        vision_encoder: ResNet backbone (with layer1..layer4 attributes)
        n_blocks: number of stages to unfreeze from the end

    Returns:
        List of trainable parameters
    """

    # ----------------------------
    # Freeze everything first
    # ----------------------------
    # for p in vision_encoder.parameters():
    #     p.requires_grad = False

    # ----------------------------
    # Collect ResNet stages
    # ----------------------------
    required_layers = ["layer1", "layer2", "layer3", "layer4"]
    for name in required_layers:
        if not hasattr(vision_encoder, name):
            raise ValueError(
                f"Expected vision_encoder to have `{name}` (ResNet-style backbone)"
            )

    stages = [
        vision_encoder.layer1,
        vision_encoder.layer2,
        vision_encoder.layer3,
        vision_encoder.layer4,
    ]

    total = len(stages)
    start = max(0, total - n_blocks)

    trainable: List[nn.Parameter] = []

    # ----------------------------
    # Unfreeze last N stages
    # ----------------------------
    for i, stage in enumerate(stages):
        req = i >= start
        for p in stage.parameters():
            p.requires_grad = req
            if req:
                trainable.append(p)

    return trainable

@torch.no_grad()
def retrieval_metrics(img_emb: torch.Tensor, txt_emb: torch.Tensor) -> dict:
    """
    img_emb, txt_emb: (N, D), assumed L2-normalized.
    Returns:
      - Directional: R@1/5/10 for I2T and T2I
      - Summary: R1, R5, MedR (averaged across directions)
      - Similarity stats: pos_mean, neg_mean, pos_neg_gap
    """
    assert img_emb.ndim == 2 and txt_emb.ndim == 2, "Expected (N,D) embeddings"
    assert img_emb.shape[0] == txt_emb.shape[0], "Image/Text count mismatch"

    N = img_emb.shape[0]
    sim = img_emb @ txt_emb.t()  # (N,N)

    # --- ranks: Image->Text ---
    i2t_sorted = sim.argsort(dim=1, descending=True)         # (N,N)
    gt = torch.arange(N, device=sim.device).unsqueeze(1)    # (N,1)
    i2t_rank = (i2t_sorted == gt).nonzero()[:, 1] + 1       # 1-indexed

    # --- ranks: Text->Image ---
    t2i_sorted = sim.t().argsort(dim=1, descending=True)    # (N,N)
    t2i_rank = (t2i_sorted == gt).nonzero()[:, 1] + 1       # 1-indexed

    def r_at_k(ranks: torch.Tensor, k: int) -> float:
        return (ranks <= k).float().mean().item()

    # --- Directional R@K ---
    r1_i2t  = r_at_k(i2t_rank, 1)
    r5_i2t  = r_at_k(i2t_rank, 5)
    r10_i2t = r_at_k(i2t_rank, 10)

    r1_t2i  = r_at_k(t2i_rank, 1)
    r5_t2i  = r_at_k(t2i_rank, 5)
    r10_t2i = r_at_k(t2i_rank, 10)

    # --- Summary (average both directions) ---
    R1 = 0.5 * (r1_i2t + r1_t2i)
    R5 = 0.5 * (r5_i2t + r5_t2i)
    MedR = 0.5 * (torch.median(i2t_rank.float()).item() + torch.median(t2i_rank.float()).item())

    # --- Pos/Neg similarity stats ---
    pos = sim.diag()
    pos_mean = pos.mean().item()

    neg_mask = ~torch.eye(N, dtype=torch.bool, device=sim.device)
    neg = sim[neg_mask]
    neg_mean = neg.mean().item()

    return {
        # directional
        "R@1_I2T": r1_i2t,
        "R@5_I2T": r5_i2t,
        "R@10_I2T": r10_i2t,
        "R@1_T2I": r1_t2i,
        "R@5_T2I": r5_t2i,
        "R@10_T2I": r10_t2i,

        # summary
        "R1": R1,
        "R5": R5,
        "MedR": MedR,

        # similarity stats
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "pos_neg_gap": pos_mean - neg_mean,
    }

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True):
    model.eval()
    all_img, all_txt = [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        vol = batch["vol"].to(device, non_blocking=True)
        texts = batch["text"]

        img = model.encode_image(vol)
        txt = model.encode_text(texts, device)
        loss = model.clip_loss(img, txt)

        all_img.append(img)
        all_txt.append(txt)
        total_loss += loss.item() * vol.size(0)
        n += vol.size(0)

    img = torch.cat(all_img, dim=0)
    txt = torch.cat(all_txt, dim=0)
    metrics = retrieval_metrics(img, txt)
    metrics["val_loss"] = total_loss / max(1, n)
    return metrics

def train(model, cfg, device):
    seed_everything()
    

    # Unfreeze last N blocks of vision encoder + vision norm
    vision_tune_params = unfreeze_last_resnet_blocks(model.vision_backbone, n_blocks=cfg["unfreeze_last_blocks"])

    # Ensure projection heads trainable
    for p in model.vision_proj.parameters():
        p.requires_grad = True
    for p in model.text_proj.parameters():
        p.requires_grad = True
    model.logit_scale.requires_grad = False
    
    # Param groups
    proj_params = list(model.vision_proj.parameters()) + list(model.text_proj.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": proj_params, "lr": cfg["lr_proj"], "weight_decay": cfg["wd_proj"]},
            {"params": vision_tune_params, "lr": cfg["lr_vision"], "weight_decay": cfg["wd_vision"]},
        ],
        betas=cfg["betas"],
        eps=cfg["eps"],
    )
    base_lrs = [cfg["lr_proj"], cfg["lr_vision"]]

    # Scheduler steps
    steps_per_epoch = len(train_loader) // max(1, cfg["grad_accum"])
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["amp"] and device.type == "cuda"))

    best_r1 = -1e9
    bad_epochs = 0
    global_step = 0
    
    r1_list = []           # store average R@1 per epoch
    pos_neg_gap_list = []   # store pos_neg_gap per epoch
    epochs_list = []        # store epoch numbers
    logs = []

    # # Save config
    # with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
    #     json.dump(cfg.__dict__, f, indent=2)

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")
    print(f"Embed dim: {model.embed_dim}")

    num_epochs = int(cfg["epochs"])

    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"),start=1):
            vol = batch["vol"].to(device, non_blocking=True)
            texts = batch["text"]

            # LR schedule (per optimizer step)
            if (step - 1) % cfg["grad_accum"] == 0:
                mult = cosine_with_warmup(global_step, total_steps, warmup_steps)
                set_group_lrs(optimizer, base_lrs, mult)

            with torch.cuda.amp.autocast(enabled=(cfg["amp"] and device.type == "cuda")):
                img = model.encode_image(vol)
                txt = model.encode_text(texts, device)
                loss = model.clip_loss(img, txt) / cfg["grad_accum"]


            scaler.scale(loss).backward()

            if step % cfg["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                # clip all trainable params
                trainable = []
                trainable += proj_params
                trainable += vision_tune_params
                torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss.item() * vol.size(0) * cfg["grad_accum"]
            seen += vol.size(0)

        train_loss = running_loss / max(1, seen)
        val = evaluate(model, val_loader, device, amp=cfg["amp"])
        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val['val_loss']:.4f} | "
            f"R1={val['R1']:.4f} R5={val['R5']:.4f} MedR={val['MedR']:.2f} | "
            f"temp={1.0/model.logit_scale.exp().item():.4f} | time={dt:.1f}s"
        )
        
        r1_list.append(val["R1"])
        pos_neg_gap_list.append(val["pos_neg_gap"])
        epochs_list.append(epoch)
        
        
        save_path = os.path.join(cfg["out_dir"], cfg["save_name_last"])
        torch.save(
            {
                "epoch": epoch,
                "r1": val["R1"],
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
            },
            save_path,
        )
        
        
        improved = val["R1"] > best_r1 + cfg["min_delta"]
        if improved:
            best_r1 = val["R1"]
            bad_epochs = 0
            save_path = os.path.join(cfg["out_dir"], cfg["save_name"])
            torch.save(
                {
                    "epoch": epoch,
                    "best_r1": best_r1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg,
                },
                save_path,
            )
            print(f"✅ New best R@1! Saving model at epoch {epoch}")
        else:
            bad_epochs += 1
            print(f"  ⏳ No improvement. bad_epochs={bad_epochs}/{cfg['patience']}")

        if bad_epochs >= cfg["patience"]:
            print("🛑 Early stopping.")
            break
        
        logs.append([
            epoch, train_loss,
            val.get("R@1_I2T", 0),
            val.get("R@5_I2T", 0),
            val.get("R@10_I2T", 0),
            val.get("R@1_T2I", 0),
            val.get("R@5_T2I", 0),
            val.get("R@10_T2I", 0),
            val.get("R1", 0),
            val.get("R5", 0),
            val.get("MedR", 0),
            val.get("pos_mean", 0), 
            val.get("neg_mean", 0), 
            val.get("pos_neg_gap", 0), 
            val.get("val_loss", 0), 
            dt,
            "Yes" if improved else "No"
        ])
        df = pd.DataFrame(logs, columns=cfg["log_columns"])
        os.makedirs(os.path.dirname(cfg["csv_path"]), exist_ok=True)
        df.to_csv(cfg["csv_path"], index=False)

    save_training_log(epochs_list, r1_list, pos_neg_gap_list, cfg["out_dir"])
    print(f"Done. Best R1 = {best_r1:.4f}")

def load_moco_encoder(checkpoint_path):
    backbone = monai_resnet.resnet50(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1
    )
    
    # Remove classification head
    backbone.fc = nn.Identity()

    # Load MoCo v3 weights
    state = torch.load(checkpoint_path, map_location="cpu")

    encoder_state = {
        k.replace("encoder_q.", ""): v
        for k, v in state.items()
        if k.startswith("encoder_q.")
    }

    backbone.load_state_dict(encoder_state, strict=True)

    # IMPORTANT:
    # Do NOT freeze here
    # Do NOT call eval() here

    return backbone




    # Similarity matrix
    sim = I @ T.t()  # cosine similarity if embeddings are L2-normalized

    def recall_at_k(sim_matrix, k):
        """
        Computes Recall@k
        sim_matrix: (N_query, N_target)
        """
        ranks = sim_matrix.argsort(dim=1, descending=True)
        targets = torch.arange(len(sim_matrix), device=sim_matrix.device)
        return (ranks[:, :k] == targets[:, None]).any(dim=1).float().mean().item()

    metrics = {
        "R@1_I2T": recall_at_k(sim, 1),
        "R@5_I2T": recall_at_k(sim, 5),
        "R@10_I2T": recall_at_k(sim, 10),  # <-- NEW
        "R@1_T2I": recall_at_k(sim.t(), 1),
        "R@5_T2I": recall_at_k(sim.t(), 5),
        "R@10_T2I": recall_at_k(sim.t(), 10),  # <-- NEW
        "pos_neg_gap": (
            sim.diag().mean() -
            sim[~torch.eye(len(sim), device=sim.device, dtype=torch.bool)].mean()
        ).item()
    }

    return metrics, sim.cpu(), I.cpu(), T.cpu()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    cfg = load_config(args.config)


    # train_df = pd.read_csv(cfg["train_df"])
    # train_df["paths"] = train_df["paths"].apply(ast.literal_eval)
    # assert train_df["paths"].apply(lambda x: isinstance(x, list)).all()
    # train_transform = probe_transform(roi_size= cfg["roi_size"])
    # train_ds = GroupedCTReportDataset(train_df, transform=train_transform)
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=cfg["train_batch_size"],
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True,
    #     persistent_workers=True

    # )


    # val_df = pd.read_csv(cfg["val_df"])
    # val_df["paths"] = val_df["paths"].apply(ast.literal_eval)
    # assert val_df["paths"].apply(lambda x: isinstance(x, list)).all()
    # val_transform = probe_transform(roi_size= cfg["roi_size"])
    # val_ds = GroupedCTReportDataset(val_df, transform=val_transform)
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=cfg["batch_size"],
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
    
    test_df = pd.read_csv(cfg["test_df"])
    test_df_transform = probe_transform(roi_size= cfg["roi_size"])
    test_ds = GroupedCTReportDataset(test_df, transform=test_df_transform, test_mode=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    encoder = load_moco_encoder(cfg["MODEL_DIR"])  # your loader
    
    model = ResNet50_CXRText_CLIP(
       vision_backbone=encoder,
       text_ckpt=cfg["text_ckpt"],
    ).to(device)

    # train(model, cfg, device)

    # model = final_train_clip(
    #     model, train_loader, val_loader, device,
    #     csv_path= OUTPUT_DIR + "CTCLIP_training_log.csv",
    #     total_epochs=50
    # )

    ckpt_path = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/Evaluation/CLIP_ResNet50_BERT/best_CLIP_Imp.pt"

    checkpoint = torch.load(ckpt_path, map_location=device)

    # ---- Load weights ----
    model.load_state_dict(checkpoint["model_state"], strict=True)
    
    metrics = evaluate(model, test_loader, device)

    pd.DataFrame([metrics]).to_csv(cfg["out_dir"] + "/final_training_retrieval_metrics.csv", index=False)

    # pd.DataFrame(sim.numpy()).to_csv(cfg["out_dir"] + "final_training_similarity_matrix.csv", index=False)

    # pd.DataFrame(img_emb.numpy()).to_csv(cfg["out_dir"] + "final_training_image_embeddings.csv", index=False)  
    # pd.DataFrame(txt_emb.numpy()).to_csv(cfg["out_dir"] + "final_training_text_embeddings.csv", index=False)
    
    print(metrics)
