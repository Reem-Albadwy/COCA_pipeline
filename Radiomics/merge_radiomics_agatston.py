import pandas as pd
from pathlib import Path

root = Path(__file__).parent

features = pd.read_csv(root / "radiomics_features.csv")
agatston = pd.read_csv(root / "agatston_scores.csv")

merged = features.merge(agatston, on="scan_id")

output = root / "radiomics_dataset.csv"
merged.to_csv(output, index=False)

print("Merged dataset saved to radiomics_dataset.csv")
print("Shape:", merged.shape)