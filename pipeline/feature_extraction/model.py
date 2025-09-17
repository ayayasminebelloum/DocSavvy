import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import Model, load_model  # load_model added here
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import pickle
from pathlib import Path

# === Config ===
BASE_DIR = Path(__file__).parent.parent.parent
FIELD_KEYS = ["invoice_number", "invoice_date", "due_date", "issuer_name", "recipient_name", "total_amount"]
EMBED_DIM = 256
FF_DIM = 128
MAX_LEN = 250
EPOCHS = 15
BATCH_SIZE = 32
VOCAB_SIZE = 20000

# Paths
DATA_DIR = BASE_DIR / "data"
INVOICES_DIR = DATA_DIR / "invoices"
MODEL_OUTPUT_DIR = BASE_DIR / "pipeline" / "models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print(f"Using data from: {INVOICES_DIR}")
print(f"Models will be saved to: {MODEL_OUTPUT_DIR}")

# === Step 1: Convert annotation JSON + CSV to labeled dataset ===
def extract_fields_from_json(csv_path, json_path):
    df = pd.read_csv(csv_path).fillna("")
    with open(json_path) as f:
        ann_json = json.load(f)

    category_id_to_label = {cat["id"]: cat["name"] for cat in ann_json["categories"]}
    image_id_to_filename = {img["id"]: img["file_name"] for img in ann_json["images"]}

    field_map = defaultdict(lambda: defaultdict(str))
    for ann in ann_json["annotations"]:
        img_id = ann["image_id"]
        label = category_id_to_label.get(ann["category_id"], "").lower()
        value = ann.get("text", "") or ann.get("value", "")
        if value:
            field_map[img_id][label] = value.strip()

    rows = []
    for img in ann_json["images"]:
        file_id = img["id"]
        file_name = img["file_name"]
        row = df[df["id"] == file_name]
        if row.empty:
            continue

        row_dict = {
            "id": file_name,
            "text": row.iloc[0]["text"],
            "label": row.iloc[0]["label"],
            "invoice_number": field_map[file_id].get("invoice number", ""),
            "invoice_date": field_map[file_id].get("invoice date", ""),
            "due_date": field_map[file_id].get("due date", ""),
            "issuer_name": field_map[file_id].get("billing address", ""),
            "recipient_name": field_map[file_id].get("name_client", ""),
            "total_amount": field_map[file_id].get("total", "")
        }
        rows.append(row_dict)

    return pd.DataFrame(rows)

# === Step 2: Preprocessing ===
def tokenize_texts(texts, tokenizer=None, fit=False):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    if fit:
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    return padded, tokenizer

# === Step 3: Build Transformer Model ===
def transformer_block(embed_dim, num_heads, ff_dim):
    def block(x):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dense(embed_dim)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        return x
    return block

def build_model(vocab_size, embed_dim, ff_dim, output_dims):
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    x = transformer_block(embed_dim, num_heads=4, ff_dim=ff_dim)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)

    outputs = [Dense(128, activation="relu")(x) for _ in FIELD_KEYS]
    outputs = [Dense(1, activation="sigmoid", name=f"{field}_output")(o) for field, o in zip(FIELD_KEYS, outputs)]
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(1e-3),
        loss=["binary_crossentropy"] * len(FIELD_KEYS),
        metrics=["accuracy"] * len(FIELD_KEYS)
    )
    return model

# === Step 4: Prepare Labels as Binary Field Presence ===
def prepare_labels(df):
    labels = []
    for field in FIELD_KEYS:
        labels.append(df[field].apply(lambda x: 1 if str(x).strip() else 0).values)
    return labels

# === Step 5: Load and Build Everything ===
train_csv_path = INVOICES_DIR / "train.csv"
valid_csv_path = INVOICES_DIR / "valid.csv"
test_csv_path = INVOICES_DIR / "test.csv"
train_json_path = INVOICES_DIR / "train.json"
valid_json_path = INVOICES_DIR / "valid.json"
test_json_path = INVOICES_DIR / "test.json"

print(f"Training CSV exists: {train_csv_path.exists()}")
print(f"Training JSON exists: {train_json_path.exists()}")

train_df = extract_fields_from_json(train_csv_path, train_json_path)
valid_df = extract_fields_from_json(valid_csv_path, valid_json_path)
test_df = extract_fields_from_json(test_csv_path, test_json_path)

print(f"Loaded training data: {len(train_df)} samples")
print(f"Loaded validation data: {len(valid_df)} samples")
print(f"Loaded test data: {len(test_df)} samples")

X_train, tokenizer = tokenize_texts(train_df["text"], fit=True)
X_valid, _ = tokenize_texts(valid_df["text"], tokenizer)
X_test, _ = tokenize_texts(test_df["text"], tokenizer)

y_train = prepare_labels(train_df)
y_valid = prepare_labels(valid_df)
y_test = prepare_labels(test_df)

# === Step 6: Train Model ===
model = build_model(len(tokenizer.word_index) + 1, EMBED_DIM, FF_DIM, output_dims=1)
model.summary()

model.fit(X_train, y_train,
          validation_data=(X_valid, y_valid),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

# === Step 7: Predict on Test Set ===
predictions = model.predict(X_test)
for i, field in enumerate(FIELD_KEYS):
    test_df[f"predicted_{field}"] = (predictions[i] > 0.5).astype(int)

# === Step 8: Save Everything ===
model_path = MODEL_OUTPUT_DIR / "transformer_invoice_field_model.weights.h5"
tokenizer_path = MODEL_OUTPUT_DIR / "invoice_tokenizer.pkl"
predictions_path = MODEL_OUTPUT_DIR / "invoice_field_predictions.csv"

# Save weights instead of complete model to avoid serialization issues with custom layers
model.save_weights(model_path)
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

# Save model configuration and vocabulary size for reconstruction
model_config = {
    "vocab_size": len(tokenizer.word_index) + 1,
    "embed_dim": EMBED_DIM,
    "ff_dim": FF_DIM
}
config_path = MODEL_OUTPUT_DIR / "model_config.json"
with open(config_path, "w") as f:
    json.dump(model_config, f)

test_df.to_csv(predictions_path, index=False)

print(f"Model weights saved to: {model_path}")
print(f"Model config saved to: {config_path}")
print(f"Tokenizer saved to: {tokenizer_path}")
print(f"Predictions saved to: {predictions_path}")

# === Step 9: Load Model + Paul Data + Predict Single Entry ===
def load_model_from_weights(weights_path, config_path):
    """Load model by rebuilding architecture and loading weights"""
    try:
        # Load model configuration
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Rebuild the model with the same architecture
        rebuilt_model = build_model(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            ff_dim=config["ff_dim"],
            output_dims=1
        )
        
        # Load the saved weights
        rebuilt_model.load_weights(weights_path)
        return rebuilt_model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_paul_data(csv_path):
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].fillna('')
    return df

# paul_csv_path = DATA_DIR / "paul_data.csv"
# paul_df = load_paul_data(paul_csv_path)

# print("\nLoading the saved model...")
# config_path = MODEL_OUTPUT_DIR / "model_config.json"
# model = load_model_from_weights(model_path, config_path)
# print("Model loaded successfully.")

# print("\nLoading the tokenizer...")
# with open(tokenizer_path, "rb") as f:
#     tokenizer = pickle.load(f)
# print("Tokenizer loaded successfully.")

# # Predict a sample entry by index
# index = int(input("Enter the index to test: "))  
# if index < 0 or index >= len(paul_df):
#     print("Invalid index. Please provide a valid index.")
# else:
#     sample_text = paul_df.iloc[index]['text']
#     sample_id = paul_df.iloc[index]['id']

#     print(f"\n=== Sample Field Extraction ===")
#     print(f"File Name: {sample_id}")
#     print(f"Text: {sample_text}\n")

#     sample_sequence = tokenizer.texts_to_sequences([sample_text])
#     sample_padded = pad_sequences(sample_sequence, maxlen=250, padding='post')

#     predictions = model.predict(sample_padded)

#     print("Extracted Fields:")
#     for i, field in enumerate(FIELD_KEYS):
#         prediction = "Present" if predictions[i][0] > 0.5 else "Not Present"
#         print(f"- {field}: {prediction}")
