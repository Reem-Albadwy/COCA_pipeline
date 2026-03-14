# Radiomics/agatston_score.py
import pandas as pd
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from configs.paths import CANONICAL_IMAGES

# Load selected scans
csv_path = Path(__file__).parent / "selected_train_scans.csv"
scans_df = pd.read_csv(csv_path)

# Use canonical images to preserve ORIGINAL voxel spacing
root_dir = CANONICAL_IMAGES

agatston_scores = []

for _, row in scans_df.iterrows():

    scan_id = row["scan_id"]

    img_path = root_dir / scan_id / f"{scan_id}_img.nii.gz"
    mask_path = root_dir / scan_id / f"{scan_id}_seg.nii.gz"

    image = sitk.ReadImage(str(img_path))
    mask = sitk.ReadImage(str(mask_path))

    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)

    # Identify calcium voxels if:it lies inside the segmentation mask and Its CT attenuation is ≥130 HU (clinical calcium threshold)
    calcium = (mask_arr > 0) & (img_arr >= 130)

    if not np.any(calcium):
        agatston_scores.append(0)
        continue

    # Identify individual calcium lesions using connected component labeling
    calcium_img = sitk.GetImageFromArray(calcium.astype(np.uint8))
    calcium_img.CopyInformation(image)

    labeled = sitk.ConnectedComponent(calcium_img)
    labeled_arr = sitk.GetArrayFromImage(labeled)
     
    # Agatston scoring uses lesion area in mm²
    # We compute voxel area using the original in-plane spacing
    spacing = image.GetSpacing()
    voxel_area = spacing[0] * spacing[1]

    agatston = 0

    for lesion_id in np.unique(labeled_arr):

        if lesion_id == 0:
            continue

        lesion_mask = labeled_arr == lesion_id
        lesion_voxels = img_arr[lesion_mask]

        max_hu = lesion_voxels.max()

        # Determine weight
        if 130 <= max_hu < 200:
            weight = 1
        elif 200 <= max_hu < 300:
            weight = 2
        elif 300 <= max_hu < 400:
            weight = 3
        else:
            weight = 4

        lesion_area = lesion_voxels.size * voxel_area

        agatston += lesion_area * weight

    agatston_scores.append(agatston)

agatson_df = pd.DataFrame({
    "scan_id": scans_df["scan_id"],
    "agatston_score": agatston_scores
})



# Risk categories
def categorize(score):

    if score == 0:
        return "0"
    elif score < 100:
        return "1-99"
    elif score < 400:
        return "100-399"
    else:
        return ">=400"


agatson_df["agatston_category"] = agatson_df["agatston_score"].apply(categorize)

# Save
output_path = Path(__file__).parent / "agatston_scores.csv"
agatson_df.to_csv(output_path, index=False)

print("Agatston scores saved to agatston_scores.csv")
print(agatson_df.head())