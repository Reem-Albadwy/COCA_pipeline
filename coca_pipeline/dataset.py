import numpy as np
from pathlib import Path
from .split import create_splits
from configs.paths import RESAMPLED_IMAGES
from monai.transforms import (
    Compose,
    RandAffined,
    ResizeWithPadOrCropd,
    EnsureTyped,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged
)
from monai.data import Dataset

def get_transforms(window=(-300, 500), augment=False):
    """
    Build a MONAI Compose transform pipeline for COCA dataset.

    Parameters:
        window (tuple): HU window (min, max)
        augment (bool): whether to apply random augmentations

    Returns:
        Compose: MONAI composed transforms
    """
    base_transforms = [
        LoadImaged(keys=["image", "mask"]),  # load NIfTI images
        EnsureChannelFirstd(keys=["image", "mask"]),  # channel first for PyTorch
        ScaleIntensityRanged(
            keys=["image"],
            a_min=window[0],
            a_max=window[1],
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        ResizeWithPadOrCropd(
            keys=["image", "mask"],
            spatial_size=(171, 243, 243)  # ensure uniform size
        )
    ]

    if augment:
        base_transforms.append(
            RandAffined(
                keys=["image", "mask"],
                rotate_range=(np.pi/36, np.pi/36, 0),
                translate_range=(1, 1, 0),
                mode=("bilinear", "nearest"),
                prob=0.5
            )
        )

    base_transforms.append(EnsureTyped(keys=["image", "mask"]))  # final conversion to torch.Tensor

    return Compose(base_transforms)


def create_monai_dataset(df, augment=False):
    """
    Create a MONAI Dataset from a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'scan_id' column
        augment (bool): whether to apply augmentation transforms

    Returns:
        Dataset: MONAI Dataset object
    """
    root_dir = RESAMPLED_IMAGES

    data = []
    for scan_id in df["scan_id"]:
        data.append({
            "image": str(root_dir / scan_id / f"{scan_id}_img.nii.gz"),
            "mask": str(root_dir / scan_id / f"{scan_id}_seg.nii.gz")
        })

    transforms = get_transforms(augment=augment)
    return Dataset(data=data, transform=transforms)


if __name__ == "__main__":
    # quick test
    train_df, val_df, test_df = create_splits()
    dataset = create_monai_dataset(train_df, augment=True)
    print("Sample dataset item keys:", dataset[0].keys())