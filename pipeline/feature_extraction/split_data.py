import pandas as pd
import json
import os
import random
from pathlib import Path
from collections import defaultdict

# Set up base directory and paths
BASE_DIR = Path(__file__).parent.parent.parent
dataset_csv_path = BASE_DIR / "data/text_dataset.csv"
merged_json_path = BASE_DIR / "data/merged_invoice_annotations.json"
output_dir = BASE_DIR / "data/invoices"

print(f"BASE_DIR: {BASE_DIR}")
print(f"Dataset CSV path: {dataset_csv_path}")
print(f"Merged JSON path: {merged_json_path}")
print(f"CSV dataset exists: {dataset_csv_path.exists()}")
print(f"Merged JSON exists: {merged_json_path.exists()}")

os.makedirs(output_dir, exist_ok=True)

# 1. Load dataset and filter invoice images
df = pd.read_csv(dataset_csv_path)
invoice_df = df[df['label'] == 3]
invoice_filenames = invoice_df['id'].tolist()

# 2. Split into train, valid, test
random.seed(42)
random.shuffle(invoice_filenames)
total = len(invoice_filenames)
train_cut = int(0.7 * total)
valid_cut = int(0.85 * total)

splits = {
    "train": invoice_filenames[:train_cut],
    "valid": invoice_filenames[train_cut:valid_cut],
    "test": invoice_filenames[valid_cut:]
}

# Save CSV splits
for split_name, filenames in splits.items():
    split_df = invoice_df[invoice_df['id'].isin(filenames)]
    split_csv_path = output_dir / f"{split_name}.csv"
    split_df.to_csv(split_csv_path, index=False)
    print(f"Saved {split_name} CSV with {len(split_df)} entries to {split_csv_path}")

# 3. Load merged annotations
with open(merged_json_path, 'r') as f:
    merged = json.load(f)

# 4. Build lookup for image_id → file_name
image_id_to_data = {img['file_name']: img for img in merged['images']}
file_name_to_image_id = {img['file_name']: img['id'] for img in merged['images']}

# 5. Group annotations by image_id
annotations_by_image = defaultdict(list)
for ann in merged['annotations']:
    annotations_by_image[ann['image_id']].append(ann)

# 6. Create split JSON files
for split_name, filenames in splits.items():
    out = {
        "info": merged.get("info", {}),
        "licenses": merged.get("licenses", []),
        "categories": merged.get("categories", []),
        "images": [],
        "annotations": []
    }

    for fname in filenames:
        image_data = image_id_to_data.get(fname)
        if not image_data:
            continue  # skip if image metadata is missing

        image_id = image_data["id"]
        out["images"].append(image_data)
        out["annotations"].extend(annotations_by_image.get(image_id, []))

    # Save split JSON
    output_path = output_dir / f"{split_name}.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {split_name} annotations to {output_path}")

print(f"Done. Saved {len(splits['train'])} train, {len(splits['valid'])} valid, {len(splits['test'])} test entries.")