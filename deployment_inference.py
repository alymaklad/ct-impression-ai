from __future__ import annotations

import functools
import inspect
import os
import sys
from pathlib import Path
from typing import BinaryIO

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import resnet50 as monai_resnet50
from monai.transforms import Resize
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from raw_ct_preprocessing import preprocess_nifti_ct_volume


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model" / "merged-huggingface"
RUNTIME_MODEL_DIR = PROJECT_ROOT / ".runtime" / "merged-huggingface"
CLIP_ENCODER_PATH = PROJECT_ROOT / "model" / "clip-alignment" / "clip_alignment_best_model.pt"
PROJECTOR_PATH = MODEL_DIR / "projector.pt"
VISION_ENCODER_FT_PATH = MODEL_DIR / "vision_encoder_ft.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

VOL_D, VOL_H, VOL_W = 32, 128, 128
POOL_D, POOL_H, POOL_W = 1, 4, 4
NUM_VIS_TOKENS = POOL_D * POOL_H * POOL_W
IM_PATCH_TOKEN = "<im_patch>"
INSTRUCTION = "Please generate a radiology impression for this chest CT."


def prepare_runtime_model_dir() -> Path:
    RUNTIME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for source in MODEL_DIR.iterdir():
        target = RUNTIME_MODEL_DIR / source.name
        if source.is_file() and source.name == "config.json":
            import shutil

            shutil.copy2(source, target)
            continue
        if target.exists():
            continue
        if source.is_file():
            try:
                target.hardlink_to(source)
            except OSError:
                import shutil

                shutil.copy2(source, target)
    return RUNTIME_MODEL_DIR


def safe_load_state_dict(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    return checkpoint


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in [
            "vision_backbone.",
            "module.",
            "model.",
            "vision_model.",
            "visual.",
            "encoder.",
            "backbone.",
            "image_encoder.",
        ]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        cleaned[new_key] = value
    return cleaned


class ResNet3DBackbone(nn.Module):
    def __init__(self, state_dict_path: Path):
        super().__init__()
        net = monai_resnet50(spatial_dims=3, n_input_channels=1, num_classes=1)
        state_dict = clean_state_dict(safe_load_state_dict(state_dict_path))
        net.load_state_dict(state_dict, strict=False)

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.act = net.act
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.adapt_pool = nn.AdaptiveAvgPool3d((POOL_D, POOL_H, POOL_W))

        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        x = self.conv1(volume)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adapt_pool(x)
        x = x.flatten(2)
        return x.transpose(1, 2)


class ImageProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def patch_local_lamed_module() -> None:
    for module_name, module in list(sys.modules.items()):
        if module_name.endswith("modeling_m3d_lamed"):
            module.build_vision_tower = lambda config, **kwargs: nn.Identity()
            module.build_segmentation_module = lambda config, **kwargs: nn.Identity()


def load_final_llm():
    runtime_model_dir = prepare_runtime_model_dir()
    get_class_from_dynamic_module(
        "modeling_m3d_lamed.LamedPhi3ForCausalLM",
        str(runtime_model_dir),
        trust_remote_code=True,
    )
    patch_local_lamed_module()

    tokenizer = AutoTokenizer.from_pretrained(runtime_model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(runtime_model_dir, trust_remote_code=True)

    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            runtime_model_dir,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        raise RuntimeError(
            "This checkpoint is saved in bitsandbytes 4-bit NF4 format. "
            "CUDA is required to load it for inference. Export a non-quantized "
            "checkpoint if CPU deployment is required."
        )

    model.eval()
    return tokenizer, model


def load_vision_components(hidden_size: int):
    vision_encoder = ResNet3DBackbone(CLIP_ENCODER_PATH).to(DEVICE)
    if VISION_ENCODER_FT_PATH.exists():
        fine_tuned_state = torch.load(VISION_ENCODER_FT_PATH, map_location=DEVICE)
        vision_encoder.load_state_dict(fine_tuned_state)
    vision_encoder = vision_encoder.eval().to(dtype=DTYPE)

    projector = ImageProjector(2048, hidden_size).to(DEVICE)
    projector.load_state_dict(torch.load(PROJECTOR_PATH, map_location=DEVICE))
    projector = projector.eval().to(dtype=DTYPE)
    return vision_encoder, projector


def load_deployment_components():
    tokenizer, llm = load_final_llm()
    vision_encoder, projector = load_vision_components(llm.config.hidden_size)
    im_patch_id = tokenizer.convert_tokens_to_ids(IM_PATCH_TOKEN)
    if im_patch_id is None or im_patch_id < 0:
        raise ValueError(f"Tokenizer does not contain required token: {IM_PATCH_TOKEN}")
    return tokenizer, llm, vision_encoder, projector, im_patch_id


def load_npy_volume(source: str | Path | BinaryIO) -> torch.Tensor:
    volume = np.load(source)
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(f"Expected CT volume shape (1, D, H, W), got {volume.shape}")

    volume = volume[0].astype(np.float32)
    volume = np.clip(volume, -1.0, 1.0)
    volume = (volume + 1.0) * 0.5

    tensor = torch.from_numpy(volume)[None, ...]
    resize = Resize(spatial_size=(VOL_D, VOL_H, VOL_W), mode="trilinear")
    tensor = resize(tensor)
    if hasattr(tensor, "as_tensor"):
        tensor = tensor.as_tensor()
    return tensor.unsqueeze(0)


def load_preprocessed_volume_array(volume: np.ndarray) -> torch.Tensor:
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(f"Expected CT volume shape (1, D, H, W), got {volume.shape}")

    volume = np.clip(volume.astype(np.float32), -1.0, 1.0)
    volume = (volume[0] + 1.0) * 0.5
    tensor = torch.from_numpy(volume)[None, ...]
    resize = Resize(spatial_size=(VOL_D, VOL_H, VOL_W), mode="trilinear")
    tensor = resize(tensor)
    if hasattr(tensor, "as_tensor"):
        tensor = tensor.as_tensor()
    return tensor.unsqueeze(0)


def load_nifti_volume(source: str | Path, slope: float = 1.0, intercept: float = 0.0) -> torch.Tensor:
    normalized, _ = preprocess_nifti_ct_volume(source, slope=slope, intercept=intercept)
    return load_preprocessed_volume_array(normalized)


@torch.no_grad()
def generate_impression(
    tokenizer,
    llm,
    vision_encoder,
    projector,
    pixel_values: torch.Tensor,
    im_patch_id: int,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    patch_str = " ".join([IM_PATCH_TOKEN] * NUM_VIS_TOKENS)
    prompt = f"{patch_str} {INSTRUCTION}\n[IMPRESSION]"
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    inputs_embeds = llm.get_input_embeddings()(input_ids).clone()

    pixel_values = pixel_values.to(DEVICE, dtype=DTYPE)
    features = vision_encoder(pixel_values)
    image_tokens = projector(features)
    patch_positions = (input_ids[0] == im_patch_id).nonzero(as_tuple=True)[0]
    if len(patch_positions) != NUM_VIS_TOKENS:
        raise ValueError(f"Expected {NUM_VIS_TOKENS} image patch tokens, found {len(patch_positions)}")
    inputs_embeds[0, patch_positions] = image_tokens[0].to(inputs_embeds.dtype)

    original_forward = llm.forward
    valid_params = inspect.signature(original_forward).parameters

    @functools.wraps(original_forward)
    def forward_absorb_kwargs(*args, **kwargs):
        return original_forward(*args, **{key: value for key, value in kwargs.items() if key in valid_params})

    llm.forward = forward_absorb_kwargs
    try:
        with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=DEVICE == "cuda"):
            output_ids = GenerationMixin.generate(
                llm,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=15,
                num_beams=1,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or 0,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
    finally:
        llm.forward = original_forward

    generated_ids = output_ids[0]
    if len(generated_ids) > 0 and generated_ids[0] == tokenizer.bos_token_id:
        generated_ids = generated_ids[1:]

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if decoded.upper().startswith("[IMPRESSION]"):
        decoded = decoded[len("[IMPRESSION]"):].strip()
    return decoded
