from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from .dataset import create_monai_dataset
from .split import create_splits
from monai.data import list_data_collate

def create_loader(df, batch_size=2, augment=False):
    """
    Create a PyTorch DataLoader with weighted sampling for class imbalance.

    Parameters:
        df (pd.DataFrame): dataframe with 'has_calcium' column
        batch_size (int)
        augment (bool)

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = create_monai_dataset(df, augment=augment)

    labels = df["has_calcium"].astype(int).tolist()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=list_data_collate  # handles variable-size tensors safely
    )
    return loader


def print_calcium_distribution(train_df, val_df, test_df):
    """
    Print percentage of scans with and without calcium for each split.
    """
    print("\n=== Calcium Distribution ===")
    for name, df_split in zip(["Train", "Validation", "Test"], [train_df, val_df, test_df]):
        counts = df_split["has_calcium"].value_counts()
        percents = df_split["has_calcium"].value_counts(normalize=True) * 100
        print(f"\n{name}:")
        print(f"  Has calcium: {counts.get(True,0)} ({percents.get(True,0):.1f}%)")
        print(f"  No calcium: {counts.get(False,0)} ({percents.get(False,0):.1f}%)")


if __name__ == "__main__":
    train_df, val_df, test_df = create_splits()

    #Verifying stratified split (calcium distribution per split) 
    print_calcium_distribution(train_df, val_df, test_df)

    train_loader = create_loader(train_df, batch_size=2, augment=True)
    val_loader = create_loader(val_df, batch_size=2, augment=False)
    test_loader = create_loader(test_df, batch_size=2, augment=False)

    # test a single batch
    batch = next(iter(train_loader))
    print("Batch image shape:", batch["image"].shape)
    print("Batch mask shape:", batch["mask"].shape)