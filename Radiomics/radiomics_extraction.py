# Radiomics/radiomics_extraction.py
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
import yaml
import numpy as np
from configs.paths import RESAMPLED_IMAGES

# Load selected scans
csv_path = Path(__file__).parent / "selected_train_scans.csv"
selected_scans = pd.read_csv(csv_path)
print(f"Loaded {len(selected_scans)} scans:\n", selected_scans.head())

# Load PyRadiomics parameters from YAML
yaml_path = Path(__file__).parent / "radiomics_params.yaml"
with open(yaml_path) as f:
    params = yaml.safe_load(f)

# Prepare extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# Disable all first, then enable selected features
extractor.disableAllFeatures()
extractor.enableFeaturesByName(**params['featureClasses'])
    
# Extraction loop
root_dir = RESAMPLED_IMAGES
features_list = []

for idx, row in selected_scans.iterrows():
    scan_id = row['scan_id']
    img_path = root_dir / scan_id / f"{scan_id}_img.nii.gz"
    mask_path = root_dir / scan_id / f"{scan_id}_seg.nii.gz"

    image = sitk.ReadImage(str(img_path))
    mask = sitk.ReadImage(str(mask_path))

    features = extractor.execute(image, mask)
    # Remove diagnostics metadata
    features = {
        k: v for k, v in features.items()
        if not k.startswith("diagnostics")
    }
    features["scan_id"] = scan_id
    
    # Max/mean calcium HU, total volume
    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    masked_voxels = img_arr[mask_arr > 0]
    if masked_voxels.size > 0:
        features['calcium_mean_HU'] = float(masked_voxels.mean())
        features['calcium_max_HU'] = float(masked_voxels.max())
        features['calcium_volume'] = float(masked_voxels.size)
    else:
        features['calcium_mean_HU'] = 0.0
        features['calcium_max_HU'] = 0.0
        features['calcium_volume'] = 0.0

    features_list.append(features)

#  Save to CSV
features_df = pd.DataFrame(features_list)
features_df.to_csv(Path(__file__).parent / "radiomics_features.csv", index=False)
print(" Radiomics features saved to radiomics_features.csv")