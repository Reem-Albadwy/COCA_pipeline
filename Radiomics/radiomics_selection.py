# radiomics_selection.py
import sys
from pathlib import Path
import pandas as pd
from coca_pipeline.split import create_splits
from configs.paths import DATASET_ROOT, RESAMPLED_IMAGES

NUM_SCANS = 25  

# Load train/val/test splits
train_df, val_df, test_df = create_splits()

# Filter only scans that contain calcium so that radiomics extraction works properly
train_with_calcium = train_df[train_df['has_calcium'] == True]

# Randomly sample NUM_SCANS from the calcium-containing scans
selected_scans = train_with_calcium.sample(n=NUM_SCANS, random_state=42)

# Check if image and mask files exist
missing_files = []
for scan_id in selected_scans['scan_id']:
    img_path = RESAMPLED_IMAGES / scan_id / f"{scan_id}_img.nii.gz"
    mask_path = RESAMPLED_IMAGES / scan_id / f"{scan_id}_seg.nii.gz"
    if not img_path.exists() or not mask_path.exists():
        missing_files.append(scan_id)

if missing_files:
    print(" Missing files for these scans:", missing_files)
else:
    print(f" All {NUM_SCANS} selected scans have image and mask files.")

# save selected scans to CSV for later use
selected_scans.to_csv(Path(__file__).parent / "selected_train_scans.csv", index=False)
print("Selected scans saved to selected_train_scans.csv")