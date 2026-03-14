from pathlib import Path
import os

# --------------------------------------------------
# Project root (COCA_pipeline folder)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------
# Dataset root
# Reads from environment variable if provided
# Otherwise assumes dataset is inside project/data
# --------------------------------------------------
DATASET_ROOT = Path(
    os.getenv("COCA_DATASET_ROOT", PROJECT_ROOT / "data")
)

# --------------------------------------------------
# Check dataset exists
# --------------------------------------------------
if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"""
Dataset folder not found.

Please set the dataset path using the environment variable:

For example Windows:
set COCA_DATASET_ROOT=D:\\Gated_release_final

For example Linux / Mac:
export COCA_DATASET_ROOT=/path/to/Gated_release_final

Current attempted path:
{DATASET_ROOT}
"""
    )

# --------------------------------------------------
# COCA dataset structure
# --------------------------------------------------
CANONICAL_IMAGES = DATASET_ROOT / "data_canonical" / "images"
CANONICAL_TABLES = DATASET_ROOT / "data_canonical" / "tables"
RESAMPLED_IMAGES = DATASET_ROOT / "data_resampled"

SCAN_INDEX = CANONICAL_TABLES / "scan_index.csv"