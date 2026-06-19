# src/transforms.py
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    EnsureTyped,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd
)


# The volumes are already preprocessed and have shape (1,128,256,256), values in [-1,1]
# We will perform geometric crops/flips/rotations and light intensity noise


# def get_view_transforms(roi_size=(96,96,96)):
#     base = [
#         EnsureTyped(keys=['img']),
#     ]

#     viewA = base + [
#         RandSpatialCropd(keys=['img'], roi_size=roi_size, random_size=False),
#         RandFlipd(keys=['img'], prob=0.5, spatial_axis=[0,1,2]),
#         RandRotate90d(keys=['img'], prob=0.5, max_k=3, spatial_axes=(1,2)),
#         RandGaussianNoised(keys=['img'], prob=0.2, mean=0.0, std=0.01),
#         RandShiftIntensityd(keys=["img"], offsets=0.1, prob=0.5),
#         RandScaleIntensityd(keys=["img"], factors=0.1, prob=0.5),
#     ]

#     viewB = base + [
#         RandSpatialCropd(keys=['img'], roi_size=roi_size, random_size=False),
#         RandFlipd(keys=['img'], prob=0.5, spatial_axis=[0,1,2]),
#         RandGaussianNoised(keys=['img'], prob=0.2, mean=0.0, std=0.01),
#         # RandRotate90d(keys=['img'], prob=0.5, max_k=3, spatial_axes=(1,2)),

#         # RandShiftIntensityd(keys=["img"], offsets=0.1, prob=0.5),
#         # RandScaleIntensityd(keys=["img"], factors=0.1, prob=0.5),
#     ]

#     return Compose(viewA), Compose(viewB)

def get_view_transforms(roi_size=(32,196,196)):

    base = [
        EnsureTyped(keys=['img']),
    ]

    viewA = base + [
        RandSpatialCropd(keys=['img'], roi_size=roi_size, random_size=False),
        RandFlipd(keys=['img'], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=['img'], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=['img'], prob=0.5, spatial_axis=[2]),
        RandRotate90d(keys=["img"], prob=0.3, max_k=3, spatial_axes=(1, 2)),
        RandGaussianNoised(keys=['img'], prob=0.2, mean=0.0, std=0.01),
        RandShiftIntensityd(keys=["img"], offsets=0.1, prob=0.5),
        RandScaleIntensityd(keys=["img"], factors=0.1, prob=0.5),
    ]

    viewB = base + [
        RandSpatialCropd(keys=['img'], roi_size=roi_size, random_size=False),
        RandFlipd(keys=['img'], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=['img'], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=['img'], prob=0.5, spatial_axis=[2]),
        RandRotate90d(keys=["img"], prob=0.2, max_k=3, spatial_axes=(1, 2)),
        RandGaussianNoised(keys=['img'], prob=0.1, mean=0.0, std=0.01),
    ]

    return Compose(viewA), Compose(viewB)
