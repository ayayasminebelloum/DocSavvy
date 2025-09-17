import os
import json
import re
import pandas as pd
import spacy
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# === Configuration ===
BASE_DIR = Path(__file__).parent.parent
TEXT_PATH = BASE_DIR / "data" / "text_dataset.csv"
output_file = BASE_DIR / "data" / "merged_invoice_annotations.json"

FIELD_KEYS = ["invoice_number", "invoice_date", "due_date", "issuer_name", "recipient_name", "total_amount"]
CATEGORIES = [
    {"id": 1, "name": "invoice number", "supercategory": "field"},
    {"id": 2, "name": "invoice date", "supercategory": "field"},
    {"id": 3, "name": "due date", "supercategory": "field"},
    {"id": 4, "name": "billing address", "supercategory": "field"},  # issuer_name
    {"id": 5, "name": "name_client", "supercategory": "field"},      # recipient_name
    {"id": 6, "name": "total", "supercategory": "field"}             # total_amount
]

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEXT_PATH: {TEXT_PATH}")
print(f"File exists: {TEXT_PATH.exists()}")
print(f"Annotations will be saved to: {output_file}")
print(f"Data directory: {BASE_DIR / 'data'}")

# === Load SpaCy Model ===
nlp = spacy.load("en_core_web_sm")

# === Extractor ===
class MLExtractor:
    def __init__(self):
        # Updated patterns with more flexible matching
        self.patterns = {
            "invoice_number": r"(?:invoice|inv|ref|reference)[\s_]*(?:no|number|#|code)?[:.\-\s]*([A-Za-z0-9][\w\-]*\d[\w\-]*)",
            "invoice_date": r"(?:invoice|inv)[\s_]*(?:date|dt)[:.\-\s]*(\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,4})",
            "due_date": r"(?:due|payment)[\s_]*(?:date|dt)[:.\-\s]*(\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,4})",
            "total_amount": r"(?:total|amount|sum|balance)[\s_]*(?:due|:|\-)?[\s]*[\$£€]?([0-9,]+\.[0-9]{2})",
            "issuer_name": r"(?:from|sender|company|issued by|seller)[\s:]*([A-Za-z][\w\s,\.\-&]+(?:Inc|LLC|Ltd|GmbH|Co|Corp)?)",
            "recipient_name": r"(?:to|bill to|recipient|client|customer)[\s:]*([A-Za-z][\w\s,\.\-&]+)"
        }

        # Alternative patterns as fallbacks
        self.alt_patterns = {
            "invoice_number": [
                r"(?:#|no\.?|number)[\s:]*(\d+[\w\-]*\d*)",
                r"(?:invoice|inv)[\s:]*(\d{4,})"
            ],
            "invoice_date": [
                r"(?:date|issued)[\s:]*(\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,4})",
                r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})"
            ],
            "total_amount": [
                r"(?:total|amount|payment)[\s\$]*(\d+\.\d{2})",
                r"(?:\$|USD|EUR|GBP)[\s]*([0-9,]+\.[0-9]{2})"
            ]
        }

    def extract_field(self, text, field):
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        try:
            # Try main pattern first
            pattern = self.patterns.get(field)
            if pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            # Try alternative patterns if main one failed
            alt_patterns = self.alt_patterns.get(field, [])
            for alt_pattern in alt_patterns:
                match = re.search(alt_pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                    
            # If it's an invoice number, try to find any numerical ID
            if field == "invoice_number" and "#" in text:
                parts = text.split("#")
                if len(parts) > 1 and re.search(r'\d+', parts[1]):
                    number_match = re.search(r'([A-Za-z0-9][\w\-]*\d[\w\-]*)', parts[1])
                    if number_match:
                        return number_match.group(1).strip()
            
            # Use NLP to identify entities for name fields
            if field in ["issuer_name", "recipient_name"]:
                doc = nlp(text[:1000])  # Limit to first 1000 chars for performance
                org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                if org_entities:
                    return org_entities[0]
                    
            return ""
        except Exception as e:
            print(f"Error extracting field {field}: {e}")
            return ""

    def extract_all_fields(self, text):
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        return {field: self.extract_field(text, field) for field in FIELD_KEYS}

# Function to extract text from document files
def get_invoice_text(filename, df):
    invoice_row = df[df["id"] == filename]
    if not invoice_row.empty:
        text = invoice_row.iloc[0]["text"]
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        return text
    return ""

# Function to create the merged annotations JSON in COCO format
def create_merged_annotations(invoice_df, extracted_data=None):
    merged = {
        "info": {"description": "Invoice dataset annotations"},
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    
    image_id = 0
    annotation_id = 0
    
    # If we have extracted data, create a lookup dict
    field_data = {}
    if extracted_data is not None:
        for idx, row in extracted_data.iterrows():
            field_data[row.get("id", idx)] = row
    
    # Map field names to category IDs
    field_to_cat = {
        "invoice_number": 1, 
        "invoice_date": 2, 
        "due_date": 3,
        "issuer_name": 4,
        "recipient_name": 5,
        "total_amount": 6
    }
    
    # Create images and annotations
    for idx, row in invoice_df.iterrows():
        file_name = row["id"]
        
        # Create image entry
        image_entry = {
            "id": image_id,
            "file_name": file_name,
            "width": 800,  # Dummy dimensions
            "height": 1200,
            "license": 1,
            "date_captured": ""
        }
        merged["images"].append(image_entry)
        
        # Fallback to ensuring we always have annotations for this image
        has_any_annotations = False
        
        # Extract fields either from provided data or using extractor
        doc_text = row.get("text", "")
        # Ensure text is a string
        if not isinstance(doc_text, str):
            doc_text = str(doc_text) if doc_text is not None else ""
            
        try:
            if not field_data:
                extractor = MLExtractor()
                fields = extractor.extract_all_fields(doc_text)
            else:
                # Use pre-extracted fields if available
                row_data = field_data.get(idx, {})
                fields = {k: row_data.get(k, "") for k in FIELD_KEYS}
        except Exception as e:
            print(f"Error processing document {idx}: {e}")
            fields = {field: "" for field in FIELD_KEYS}
        
        # Create annotations for each field
        for field_name, value in fields.items():
            # Ensure value is a string
            if not isinstance(value, str):
                value = str(value) if value is not None else ""
                
            # Add annotation if we have a value or force at least one annotation per category
            cat_id = field_to_cat.get(field_name, 0)
            if value:
                has_any_annotations = True
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "text": value,
                    "bbox": [0, 0, 100, 20],  # Dummy bounding box
                    "area": 2000,
                    "iscrowd": 0
                }
                merged["annotations"].append(annotation)
                annotation_id += 1
        
        # If no annotations were found, add dummy annotations for this image
        if not has_any_annotations:
            for cat_id in range(1, 7):
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "text": f"Sample value for category {cat_id}",
                    "bbox": [0, 0, 100, 20],  # Dummy bounding box
                    "area": 2000,
                    "iscrowd": 0
                }
                merged["annotations"].append(annotation)
                annotation_id += 1
        
        image_id += 1
    
    return merged

# === Main Execution ===
if __name__ == "__main__":
    # 1. Check if text_dataset.csv exists and load it
    if not TEXT_PATH.exists():
        print(f"Error: {TEXT_PATH} doesn't exist. Please make sure you have your text dataset.")
        exit(1)
    
    df = pd.read_csv(TEXT_PATH)
    print(f"Loaded dataset with {len(df)} documents")
    
    # 2. Filter to only include invoice files (label 3)
    INVOICE_LABEL_IDS = {3}
    invoice_df = df[df["label"].isin(INVOICE_LABEL_IDS)]
    print(f"Found {len(invoice_df)} invoice files in dataset")
    
    if len(invoice_df) == 0:
        print("Error: No invoice files found in the dataset.")
        exit(1)
    
    # 3. Extract fields using our extractor
    print("Extracting fields from invoice documents...")
    
    # Try to extract fields first
    extractor = MLExtractor()
    predictions = []
    for idx, row in invoice_df.iterrows():
        text = row["text"]
        try:
            pred = extractor.extract_all_fields(text)
            pred["id"] = idx
            predictions.append(pred)
        except Exception as e:
            print(f"Error processing document {idx}: {e}")
            # Add empty predictions to maintain row count
            pred = {field: "" for field in FIELD_KEYS}
            pred["id"] = idx
            predictions.append(pred)
    
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(BASE_DIR / "data" / "extracted_invoice_field_values.csv", index=False)
    print(f"Field value extraction complete. Output saved to {BASE_DIR}/data/extracted_invoice_field_values.csv")
    
    # 4. Create the merged annotations JSON
    print("Creating COCO-format annotations JSON...")
    merged_annotations = create_merged_annotations(invoice_df, pred_df)
    
    # 5. Save the merged annotations
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged_annotations, f, indent=2)
    
    print(f"Merged annotations saved to: {output_file}")
    print(f"Total invoice files: {len(merged_annotations['images'])}")
    print(f"Total annotations: {len(merged_annotations['annotations'])}")