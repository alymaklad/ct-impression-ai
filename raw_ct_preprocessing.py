from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
from scipy.ndimage import zoom


@dataclass(frozen=True)
class CTPreprocessingConfig:
    target_z_spacing: float = 1.5
    fixed_hw: int = 256
    target_depth: int = 128
    hu_clip: tuple[float, float] = (-1000.0, 1000.0)
    min_slices: int = 20


class RawCTPreprocessor:
    """Raw CT preprocessing extracted from dataset/preprocess_volumes.ipynb."""

    def __init__(self, config: CTPreprocessingConfig | None = None):
        self.config = config or CTPreprocessingConfig()

    def load_nifti_volume(self, file_path: str | Path) -> tuple[np.ndarray, tuple[float, ...]]:
        nii = nib.load(str(file_path))
        volume = nii.get_fdata().astype(np.float32)
        affine = nii.affine

        input_ornt = nio.io_orientation(affine)
        ras_ornt = nio.axcodes2ornt(("R", "A", "S"))
        transform = nio.ornt_transform(input_ornt, ras_ornt)
        volume = nio.apply_orientation(volume, transform)

        volume = volume.transpose(2, 1, 0)
        spacing = nii.header.get_zooms()

        if volume.shape[0] < self.config.min_slices:
            raise ValueError(f"Expected at least {self.config.min_slices} slices, got {volume.shape[0]}")

        return volume, spacing

    def apply_hu_conversion(self, volume: np.ndarray, slope: float = 1.0, intercept: float = -1024.0) -> np.ndarray:
        return (volume * float(slope)) + float(intercept)

    def resample_volume(self, volume: np.ndarray, current_spacing: tuple[float, ...]) -> np.ndarray:
        d, h, w = volume.shape
        scale_d = current_spacing[2] / self.config.target_z_spacing
        scale_h = self.config.fixed_hw / h
        scale_w = self.config.fixed_hw / w
        return zoom(volume, zoom=[scale_d, scale_h, scale_w], order=1)

    def crop_or_pad_depth(self, volume: np.ndarray, pad_value: float = -1.0) -> np.ndarray:
        current_d = volume.shape[0]
        target_d = self.config.target_depth

        if current_d == target_d:
            return volume

        if current_d > target_d:
            start = (current_d - target_d) // 2
            return volume[start : start + target_d, :, :]

        pad_total = target_d - current_d
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(
            volume,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )

    def clip_and_normalize(self, volume: np.ndarray) -> np.ndarray:
        hu_min, hu_max = self.config.hu_clip
        volume = np.clip(volume, hu_min, hu_max)
        return 2 * (volume - hu_min) / (hu_max - hu_min) - 1.0

    def preprocess_nifti(
        self,
        file_path: str | Path,
        slope: float = 1.0,
        intercept: float = -1024.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_volume, spacing = self.load_nifti_volume(file_path)
        hu_volume = self.apply_hu_conversion(raw_volume, slope=slope, intercept=intercept)
        resampled_hu = self.resample_volume(hu_volume, spacing)
        normalized = self.clip_and_normalize(resampled_hu)
        normalized = self.crop_or_pad_depth(normalized, pad_value=-1.0)
        normalized = np.expand_dims(normalized, axis=0).astype(np.float32)

        display_hu = self.crop_or_pad_depth(
            np.clip(resampled_hu, self.config.hu_clip[0], self.config.hu_clip[1]),
            pad_value=self.config.hu_clip[0],
        ).astype(np.float32)

        return normalized, display_hu


def load_npy_ct_volume(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    volume = np.load(file_path).astype(np.float32)
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(f"Expected .npy CT shape (1, D, H, W), got {volume.shape}")

    normalized = np.clip(volume, -1.0, 1.0)
    display_hu = ((normalized[0] + 1.0) * 0.5 * 2000.0) - 1000.0
    return normalized, display_hu.astype(np.float32)


def preprocess_nifti_ct_volume(
    file_path: str | Path,
    slope: float = 1.0,
    intercept: float = -1024.0,
) -> tuple[np.ndarray, np.ndarray]:
    return RawCTPreprocessor().preprocess_nifti(file_path, slope=slope, intercept=intercept)

