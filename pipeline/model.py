import os
import json
import random
import numpy as np
import pandas as pd
import spacy

# === Configuration ===
test_csv = "test_data/split_invoices/test/test.csv"
test_json = "test_data/split_annotations/test.json"

# Define multilingual field requirements and keyword variants
FIELD_MAP = {
    "invoice_number": [
        "invoice number", "no.", "invoice#", "numéro de facture", "número de factura",
        "numero de factura", "n° de facture", "n° factura", "factura num"
    ],
    "invoice_date": [
        "invoice date", "date issued", "facture date", "fecha factura",
        "fecha de emisión", "fecha de la factura", "date de la facture"
    ],
    "due_date": [
        "due date", "payment due", "date d'échéance", "fecha de vencimiento",
        "vencimiento", "plazo", "echeance", "fecha límite"
    ],
    "issuer_name": [
        "from", "seller", "issuer", "vendeur", "expéditeur", 
        "remitente", "vendedor", "emisor"
    ],
    "recipient_name": [
        "to", "bill to", "client", "customer", "acheteur", "cliente", 
        "destinatario", "comprador", "facturado a", "receptor"
    ],
    "total_amount": [
        "total", "amount due", "net to pay", "montant", "total a payer", 
        "importe total", "monto total", "importe", "suma total", "total a pagar"
    ]
}

# Load and initialize SpaCy with sentencizer
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# === Load Data for Extraction ===
def load_data(csv_path, json_path):
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].fillna('')
    texts = df['text'].astype(str).values
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return df, texts, annotations

# === Field Extraction Function ===
def extract_fields_from_text(text):
    doc = nlp(text)
    extracted = {}
    lowered = text.lower()

    for field, variants in FIELD_MAP.items():
        found = None
        for variant in variants:
            if variant.lower() in lowered:
                found = variant
                break
        if found:
            # Try to get span around the found term
            for sent in doc.sents:
                if found in sent.text.lower():
                    extracted[field] = sent.text.strip()
                    break
            else:
                extracted[field] = f"{found} (detected)"
        else:
            extracted[field] = "Not found"

    return extracted

# === Load Test Data and Extract Fields ===
test_df, test_texts, test_annotations = load_data(test_csv, test_json)

print("\n=== Sample Field Extraction ===")
rand_idx = random.randint(0, len(test_df) - 1)
sample_text = test_df.iloc[rand_idx]['text']
sample_id = test_df.iloc[rand_idx]['id']

print(f"File Name: {sample_id}")
print(f"Text: {sample_text}\n")
fields = extract_fields_from_text(sample_text)
print("Extracted Fields:")
for k, v in fields.items():
    print(f"- {k}: {v}")
