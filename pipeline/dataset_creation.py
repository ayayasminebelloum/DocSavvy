import os
import pandas as pd
import glob
import hashlib
from typing import List, Union, Optional
from pathlib import Path
from datetime import datetime

# Import the preprocessing modules
from preprocessing.preprocess_images import preprocess_document_image
from preprocessing.parse_pdf import parse_pdf
from preprocessing.cleaning_extracted_text import clean_text

def create_or_update_text_dataset(
    input_paths: List[str],
    dataset_path: str = "data/fitext_dataset.csv",
    classes_path: str = "data/classes.txt",
    mode: str = "append",
    id_type: str = "filename",
    preview: bool = False
) -> pd.DataFrame:
    """
    Create or update a pandas DataFrame with 'id' and 'text' columns,
    and integer label indices. Also generates classes.txt for model use.

    Parameters:
    -----------
    input_paths : List[str]
        List of file paths or directories to process
    dataset_path : str
        Path to the dataset CSV file (will be created if doesn't exist)
    classes_path : str
        Path to save class name to index mapping
    mode : str
        "append" - add new records to existing dataset
        "overwrite" - replace existing dataset with new records
        "production" - return DataFrame without saving
    id_type : str
        "filename" - use filename as ID
        "hash" - use content hash as ID
    preview : bool
        If True, visualize preprocessing steps during execution

    Returns:
    --------
    pandas.DataFrame
        DataFrame with 'id', 'text', 'label', 'timestamp' columns
    """
    existing_df = None
    processed_files = set()

    if mode == "append" and os.path.exists(dataset_path):
        try:
            existing_df = pd.read_csv(dataset_path)
            print(f"Loaded existing dataset with {len(existing_df)} records")
            processed_files = set(existing_df['id'].tolist())
        except Exception as e:
            print(f"Error reading existing dataset: {e}")
            existing_df = None

    # Create label mapping from input folder names
    label_names = [Path(p).name.lower() for p in input_paths]
    label_map = {name: idx for idx, name in enumerate(sorted(label_names))}

    # Save class index file WITH label IDs
    os.makedirs(os.path.dirname(classes_path), exist_ok=True)
    with open(classes_path, "w") as f:
        for name in sorted(label_map.keys(), key=lambda x: label_map[x]):
            f.write(f"{label_map[name]} {name}\n")
    print(f"Saved class labels to {classes_path}")

    # Init containers
    ids, filenames, texts, labels, timestamps = [], [], [], [], []

    for path in input_paths:
        label_name = Path(path).name.lower()
        label_id = label_map[label_name]

        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "**/*"), recursive=True)
        else:
            files = [path]

        for file_path in files:
            if not os.path.isfile(file_path):
                continue

            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()

            file_id = None if id_type == "hash" else file_name
            if mode == "append" and file_id in processed_files:
                print(f"Skipping already processed file: {file_name}")
                continue

            try:
                if file_ext in ['.pdf']:
                    text = parse_pdf(pdf_path=file_path)
                elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                    result = preprocess_document_image(file_path, preview=preview)
                    text = result['text']
                elif file_ext in ['.txt', '.md', '.csv']:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                elif file_ext in ['.xls', '.xlsx']:
                    try:
                        df = pd.read_excel(file_path, dtype=str)
                        text = df.fillna('').astype(str).apply(lambda row: ' '.join(row), axis=1).str.cat(sep=' ')
                    except Exception as e:
                        print(f"❌ Failed to process Excel: {file_path} → {e}")
                        continue
                else:
                    print(f"Skipping unsupported file type: {file_path}")
                    continue

                text = clean_text(text)

                if id_type == "hash":
                    file_id = hashlib.md5(text.encode('utf-8')).hexdigest()
                    if mode == "append" and file_id in processed_files:
                        print(f"Skipping duplicate content: {file_name}")
                        continue

                ids.append(file_id)
                filenames.append(file_name)
                texts.append(text)
                labels.append(label_id)
                timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                print(f"Processed: {file_name} → Label: {label_name} (ID: {label_id})")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    new_df = pd.DataFrame({
        'id': ids,
        'filename': filenames,
        'text': texts,
        'label': labels,
        'timestamp': timestamps
    })

    if mode == "append" and existing_df is not None:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"Added {len(new_df)} new records to dataset")
    else:
        final_df = new_df
        if mode == "overwrite":
            print(f"Created new dataset with {len(new_df)} records")

    if mode != "production" and dataset_path:
        final_df.to_csv(dataset_path, index=False)
        print(f"Dataset saved to {dataset_path}")

    return final_df

if __name__ == "__main__":
    input_dirs = [
        "unstructured_data/invoices",
        "unstructured_data/receipts",
        "unstructured_data/emails",
        "unstructured_data/contracts",
        "unstructured_data/bank_statements",
        "unstructured_data/other"
    ]

    dataset = create_or_update_text_dataset(
        input_paths=input_dirs,
        dataset_path="data/text_dataset.csv",
        classes_path="data/classes.txt",
        mode="append",
        id_type="filename",
        preview=False
    )

    print(f"\nDataset contains {len(dataset)} total records")
    print(f"Columns: {dataset.columns.tolist()}")
    print("\nSample data:")
    print(dataset.head())
