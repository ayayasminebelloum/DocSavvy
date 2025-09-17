import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Use absolute path to the dataset
DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "text_dataset.csv"
DATA_PATH = Path(__file__).parent.parent.parent / "data"

def split_data(
        output_dir: str,
        training_split: float = 0.8, 
        test_split: float = 0.1, 
        validation_split: float = 0.1,
        random_state: int = 42,
        save_splits: bool = True,
        ) -> tuple:
    """
    Split the dataset into training, testing, and validation sets while maintaining
    class distribution.
    
    Parameters:
    -----------
    training_split : float
        Proportion of data for training (default: 0.8)
    test_split : float
        Proportion of data for testing (default: 0.1)
    validation_split : float
        Proportion of data for validation (default: 0.1)
    random_state : int
        Random seed for reproducibility
    save_splits : bool
        Whether to save the splits to CSV files
        
    Returns:
    --------
    tuple
        (train_df, test_df, val_df) DataFrames containing the splits
    """
    # Validate split proportions
    if not np.isclose(training_split + test_split + validation_split, 1.0):
        raise ValueError("Split proportions must sum to 1.0")
    
    # Read the dataset
    try:
        df_original = pd.read_csv(DATASET_PATH)
        print(f"Loaded dataset with {len(df_original)} records")
        
        # Convert labels to integers
        df_original['label'] = df_original['label'].astype(int)
        
        # Drop rows where label is 0
        df = df_original[df_original['label'] != 0].reset_index(drop=True)
        print(f"After dropping label 0: {len(df)} records")
        
        # Print class distribution after filtering
        class_counts = df['label'].value_counts().sort_index()
        print("\nClass distribution after dropping label 0:")
        for label, count in class_counts.items():
            print(f"Class {label}: {count} records ({count/len(df)*100:.1f}%)")
            
    except Exception as e:
        raise Exception(f"Error reading dataset: {e}")
    
    # First split: separate test set
    remaining_data, test_df = train_test_split(
        df,
        test_size=test_split,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: divide remaining data into train and validation
    validation_ratio = validation_split / (training_split + validation_split)
    train_df, val_df = train_test_split(
        remaining_data,
        test_size=validation_ratio,
        stratify=remaining_data['label'],
        random_state=random_state
    )
    
    # Print split statistics
    print("\nDataset split statistics:")
    print(f"Training set:   {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set:      {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
    
    # Print class distribution in each split
    print("\nClass distribution:")
    splits = {'Training': train_df, 'Validation': val_df, 'Test': test_df}
    for name, split_df in splits.items():
        print(f"\n{name} set:")
        class_dist = split_df['label'].value_counts().sort_index()
        for label, count in class_dist.items():
            percentage = count/len(split_df)*100
            print(f"Class {label}: {count} records ({percentage:.1f}%)")
    
    if save_splits:
        # Save splits to CSV
        output_dir = Path(output_dir)
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "validation.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        print(f"\nSplit datasets saved to {output_dir}")
    
    return train_df, test_df, val_df

if __name__ == "__main__":
    try:
        train_data, test_data, val_data = split_data(
            output_dir=DATA_PATH,
            training_split=0.6,
            test_split=0.2,
            validation_split=0.2,
            save_splits=True
        )
    except Exception as e:
        print(f"Error: {e}")