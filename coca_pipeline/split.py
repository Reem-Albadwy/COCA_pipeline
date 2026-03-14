import pandas as pd
from pathlib import Path
from configs.paths import SCAN_INDEX
from sklearn.model_selection import train_test_split

def create_splits(test_size=0.15, val_size=0.15, random_state=42):
    """
    Create stratified train/val/test splits for COCA dataset.

    Parameters:
        test_size (float): proportion of test set
        val_size (float): proportion of validation set
        random_state (int): seed for reproducibility

    Returns:
        train_df, val_df, test_df (pd.DataFrame)
    """
    df = pd.read_csv(SCAN_INDEX)
    
    # For radiomics we stratify based on calcium presence
    # Simple binary stratification: has_calcium or not
    df["has_calcium"] = df["voxels"] > 0  

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["has_calcium"], random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df["has_calcium"], random_state=random_state
    )

    print(f"Total scans: {len(df)}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = create_splits()
    
    print("\n=== Calcium Distribution ===")
    for name, df_split in zip(["Train", "Val", "Test"], [train_df, val_df, test_df]):
        print(f"\n{name} ratio:")
        print(df_split["has_calcium"].value_counts(normalize=True))