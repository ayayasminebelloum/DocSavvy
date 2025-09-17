import os
import sys
import pandas as pd
import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
import numpy as np
from werkzeug.utils import secure_filename
import json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add parent directory to path BEFORE any custom imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Added to sys.path: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

# Now import from pipeline
from pipeline.classification.HAN_classifier import HierarchicalAttentionNetwork # Import the HAN classifier instead of TwoStageClassifier
from pipeline.preprocessing.preprocess_images import preprocess_document_image
# Import the clean_extracted_text function from your updated module
from pipeline.preprocessing.parse_pdf import parse_pdf
from pipeline.preprocessing.preprocess_images import preprocess_document_image
from pipeline.preprocessing.cleaning_extracted_text import clean_text
from pipeline.feature_extraction.model import load_model_from_weights, tokenize_texts
from pipeline.feature_extraction.annotations_extraction import MLExtractor

# Flask app setup
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
           static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))
app.config['SECRET_KEY'] = 'document-classifier-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load class labels
class_mapping = {}
try:
    with open(os.path.join(PROJECT_ROOT, "data/classes.txt"), "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                idx, name = parts
                class_mapping[int(idx)] = name.capitalize()
                # Also map the string version of the index
                class_mapping[idx] = name.capitalize()
                # And map the name directly
                class_mapping[name] = name.capitalize()
    
    print("Class mapping loaded successfully:")
    for k, v in class_mapping.items():
        print(f"  {k} -> {v}")
except Exception as e:
    print(f"Error loading class mapping: {e}")
    # Fallback class mapping
    class_mapping = {
        1: 'Contract',
        '1': 'Contract',
        'contracts': 'Contract',
        2: 'Email',
        '2': 'Email',
        'emails': 'Email',
        3: 'Invoice',
        '3': 'Invoice',
        'invoices': 'Invoice',
        4: 'Other',
        '4': 'Other',
        'other': 'Other',
        5: 'Receipt',
        '5': 'Receipt',
        'receipts': 'Receipt'
    }

# Map internal class names to display names (ensure consistent capitalization)
display_classes = {
    'contracts': 'Contract',
    'emails': 'Email',
    'invoices': 'Invoice',
    'other': 'Other',
    'receipts': 'Receipt',
    'Contracts': 'Contract',
    'Emails': 'Email',
    'Invoices': 'Invoice',
    'Other': 'Other',
    'Receipts': 'Receipt'
}

# Define model paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "pipeline", "models")
HAN_CONFIG_PATH = os.path.join(MODEL_DIR, "han_config.joblib")
HAN_WEIGHTS_PATH = os.path.join(MODEL_DIR, "han_model_best.weights.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "han_tokenizer.pickle")

# Define invoice feature extraction model paths
INVOICE_MODEL_PATH = os.path.join(MODEL_DIR, "transformer_invoice_field_model.weights.h5")
INVOICE_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
INVOICE_TOKENIZER_PATH = os.path.join(MODEL_DIR, "invoice_tokenizer.pkl")

# Load the HAN model
print("Loading HAN classifier model")
model = None

# Add invoice feature extraction model loading
invoice_model = None
invoice_tokenizer = None
try:
    # Check if invoice model files exist
    if os.path.exists(INVOICE_MODEL_PATH) and os.path.exists(INVOICE_CONFIG_PATH) and os.path.exists(INVOICE_TOKENIZER_PATH):
        # Import the load_model_from_weights function
        from pipeline.feature_extraction.model import load_model_from_weights
        
        # Load invoice model
        invoice_model = load_model_from_weights(INVOICE_MODEL_PATH, INVOICE_CONFIG_PATH)
        
        # Load invoice tokenizer
        with open(INVOICE_TOKENIZER_PATH, "rb") as f:
            invoice_tokenizer = pickle.load(f)
            
        print("Successfully loaded invoice feature extraction model")
    else:
        print("Invoice model files not found. Checked paths:")
        print(f"  Model: {INVOICE_MODEL_PATH} - Exists: {os.path.exists(INVOICE_MODEL_PATH)}")
        print(f"  Config: {INVOICE_CONFIG_PATH} - Exists: {os.path.exists(INVOICE_CONFIG_PATH)}")
        print(f"  Tokenizer: {INVOICE_TOKENIZER_PATH} - Exists: {os.path.exists(INVOICE_TOKENIZER_PATH)}")
except Exception as e:
    print(f"Error loading invoice model: {str(e)}")
    import traceback
    traceback.print_exc()
    invoice_model = None

# Define the invoice field names for readability
INVOICE_FIELD_KEYS = ["invoice_number", "invoice_date", "due_date", "issuer_name", "recipient_name", "total_amount"]

try:
    # Initialize the HAN Classifier
    model = HierarchicalAttentionNetwork(
        max_features=20000,
        max_sentences=15,
        max_sentence_length=50,
        embedding_dim=100,
        gru_units=100,
        dropout_rate=0.3,
        model_dir=MODEL_DIR
    )
    
    # Check if all required files exist
    if os.path.exists(HAN_CONFIG_PATH) and os.path.exists(HAN_WEIGHTS_PATH) and os.path.exists(TOKENIZER_PATH):
        # Load the model with the appropriate files
        model.load_model(
            model_path=HAN_WEIGHTS_PATH,
            config_path=HAN_CONFIG_PATH
        )
        print("Successfully loaded HierarchicalAttentionNetwork model")
        print(f"Model type: {type(model)}")
    else:
        print("ERROR: Model files not found. Checked paths:")
        print(f"  Config: {HAN_CONFIG_PATH} - Exists: {os.path.exists(HAN_CONFIG_PATH)}")
        print(f"  Weights: {HAN_WEIGHTS_PATH} - Exists: {os.path.exists(HAN_WEIGHTS_PATH)}")
        print(f"  Tokenizer: {TOKENIZER_PATH} - Exists: {os.path.exists(TOKENIZER_PATH)}")
        model = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    import traceback
    traceback.print_exc()
    model = None
# Create a prediction function that uses only the best model
def predict_document_class(text):
    """Predict document class using the TwoStageClassifier"""
    print(f"Predicting class for text of length: {len(text)}")
    
    if model is None:
        print("ERROR: Model not loaded properly")
        return "Error"
    
    try:
        # Use the TwoStageClassifier's predict method directly
        raw_prediction = model.predict([text])[0]  # Get the first (and only) prediction
        print(f"Raw prediction: {raw_prediction} (type: {type(raw_prediction)})")
        
        predicted_class = class_mapping[raw_prediction]
        
        print(f"Final classification: {predicted_class}")
        return predicted_class
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error"

def predict_document_classes(texts):
    """Predict classes for multiple documents at once"""
    if model is None:
        print("ERROR: Model not loaded properly")
        return ["Error"] * len(texts)
    
    try:
        # Process the entire batch at once
        batch_predictions = model.predict(texts)
        
        # Map predictions to class names
        results = []
        for raw_prediction in batch_predictions:
            results.append(class_mapping[raw_prediction])
        
        return results
        
    except Exception as e:
        print(f"Error during batch classification: {str(e)}")
        import traceback
        traceback.print_exc()
        return ["Error"] * len(texts)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'txt', 'md', 'csv', 'xls', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_metadata(file_path):
    """Extract metadata from a file"""
    file_stat = os.stat(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    
    metadata = {
        'filename': os.path.basename(file_path),
        'file_type': file_ext[1:].upper(),  # Remove the dot
        'file_size': format_size(file_stat.st_size),
        'created_date': format_timestamp(file_stat.st_ctime),
        'modified_date': format_timestamp(file_stat.st_mtime),
        'full_path': file_path
    }
    
    # Additional metadata for specific file types
    if file_ext == '.pdf':
        try:
            from PyPDF2 import PdfReader
            with open(file_path, 'rb') as f:
                pdf = PdfReader(f)
                metadata['page_count'] = len(pdf.pages)
                if pdf.metadata:
                    metadata['author'] = pdf.metadata.get('/Author', 'Unknown')
                    metadata['creator'] = pdf.metadata.get('/Creator', 'Unknown')
                    metadata['creation_date'] = pdf.metadata.get('/CreationDate', 'Unknown')
        except Exception as e:
            print(f"Error extracting PDF metadata: {e}")
    
    return metadata

def format_size(size_bytes):
    """Format file size in bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def format_timestamp(timestamp):
    """Format timestamp to readable date"""
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def get_detailed_file_metadata(file_path):
    """Get detailed metadata for a file"""
    metadata = {}
    try:
        print(f"Getting metadata for file: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        # Basic file information
        file_stats = os.stat(file_path)
        file_size_bytes = file_stats.st_size
        print(f"File size in bytes: {file_size_bytes}")
        
        # Convert size to readable format
        if file_size_bytes < 1024:
            file_size = f"{file_size_bytes} bytes"
        elif file_size_bytes < 1024 * 1024:
            file_size = f"{file_size_bytes / 1024:.1f} KB"
        else:
            file_size = f"{file_size_bytes / (1024 * 1024):.1f} MB"
            
        # Get file extension and MIME type
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"File extension: {file_ext}")
        
        # Map extension to readable type
        file_type_map = {
            '.pdf': 'PDF Document',
            '.doc': 'Word Document',
            '.docx': 'Word Document',
            '.jpg': 'JPEG Image',
            '.jpeg': 'JPEG Image',
            '.png': 'PNG Image',
            '.txt': 'Text Document',
            '.csv': 'CSV Spreadsheet',
            '.xls': 'Excel Spreadsheet',
            '.xlsx': 'Excel Spreadsheet'
        }
        
        file_type = file_type_map.get(file_ext, f"{file_ext[1:].upper()} File")
        print(f"Detected file type: {file_type}")
        
        # Last modified time
        mod_time = datetime.datetime.fromtimestamp(file_stats.st_mtime)
        last_modified = mod_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Last modified: {last_modified}")
        
        # Compile metadata
        metadata = {
            'filename': os.path.basename(file_path),
            'size': file_size,
            'type': file_type,
            'extension': file_ext,
            'last_modified': last_modified
        }
        print(f"Final metadata: {metadata}")
    except Exception as e:
        print(f"Error getting file metadata: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        metadata = {
            'filename': os.path.basename(file_path),
            'size': 'Unknown',
            'type': 'Unknown',
            'extension': 'Unknown',
            'last_modified': 'Unknown'
        }
    
    return metadata

# Function to extract invoice fields
def extract_invoice_fields(text):
    """Extract invoice fields from text using both the ML model and regex extraction"""
    if invoice_model is None or invoice_tokenizer is None:
        print("ERROR: Invoice model or tokenizer not loaded properly")
        return {}
    
    try:
        # First use the ML model to detect which fields are present
        sequence = invoice_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=250, padding='post')
        predictions = invoice_model.predict(padded)
        
        # Initialize the extractor for getting actual values
        extractor = MLExtractor()
        
        # Extract fields based on predictions and get actual values
        fields = {}
        for i, field in enumerate(INVOICE_FIELD_KEYS):
            is_present = predictions[i][0] > 0.5
            if is_present:
                # If field is detected as present, try to extract its value
                value = extractor.extract_field(text, field)
                fields[field] = value if value else "Present but value not found"
            else:
                fields[field] = "Not Present"
        
        return fields
    except Exception as e:
        print(f"Error extracting invoice fields: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results')
def results():
    # Get file metadata from session
    file_data = session.get('file_metadata', {})
    
    filename = file_data.get('filename', '')
    classification = file_data.get('classification', 'Unknown')
    file_size = file_data.get('file_size', 'Unknown')
    file_type = file_data.get('file_type', 'Unknown')
    last_modified = file_data.get('last_modified', 'Unknown')
    
    # Get invoice fields if available
    invoice_fields = file_data.get('invoice_fields', {})
    
    return render_template('results.html', 
                          filename=filename,
                          classification=classification,
                          file_size=file_size,
                          file_type=file_type,
                          last_modified=last_modified,
                          invoice_fields=invoice_fields)

@app.route('/api/upload', methods=['POST'])
def api_upload_file():
    """API endpoint for file uploads - returns JSON response instead of redirecting"""
    print("==== API UPLOAD ENDPOINT CALLED ====")
    print(f"Request method: {request.method}")
    print(f"Request form data: {request.form}")
    print(f"Request files: {request.files}")
    
    response = {
        'success': False,
        'message': '',
        'redirect_url': None
    }
    
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            print("Error: No file part in the request")
            response['message'] = 'No file part in the request'
            return jsonify(response), 400
            
        file = request.files['file']
        print(f"File object: {file}")
        print(f"File name: {file.filename}")
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print("Error: No selected file (empty filename)")
            response['message'] = 'No selected file'
            return jsonify(response), 400
            
        if file and allowed_file(file.filename):
            print(f"File is allowed: {file.filename}")
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
            print(f"File saved to: {file_path}")
            
            # Get file metadata
            file_metadata = get_file_metadata(file_path)
            print(f"File metadata: {file_metadata}")
            
            # Process the document based on file type
            file_ext = os.path.splitext(file_path)[1].lower()
            text = ""
            
            if file_ext in ['.pdf']:
                # Parse PDF to extract text
                text = parse_pdf(file_path)
            elif file_ext in ['.txt', '.md']:
                # Read text file directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                # Process image to extract text
                result = preprocess_document_image(file_path)
                # Extract text from the result dictionary
                if isinstance(result, dict) and 'text' in result:
                    text = result['text']
                else:
                    text = str(result)  # Fallback if structure is unexpected
            elif file_ext in ['.csv', '.xls', '.xlsx']:
                # Handle spreadsheet files
                df = pd.read_excel(file_path) if file_ext in ['.xls', '.xlsx'] else pd.read_csv(file_path)
                text = df.to_string()
            else:
                print(f"Unsupported file type: {file_ext}")
                response['message'] = f'Unsupported file type: {file_ext}'
                return jsonify(response), 400
            
            # Clean the extracted text
            text = clean_text(text)
            print(f"Extracted and cleaned text length: {len(text)}")
            
            # Classify the document
            predicted_class = predict_document_class(text)
            predicted_class = display_classes.get(predicted_class, predicted_class)
            print(f"Document classified as: {predicted_class}")

            # Get detailed file metadata
            file_metadata = get_detailed_file_metadata(file_path)
            
            # Initialize invoice fields
            invoice_fields = {}
            
            # If document is classified as an invoice, extract invoice fields
            if predicted_class.lower() == 'invoice':
                print("Document classified as invoice. Extracting invoice fields...")
                invoice_fields = extract_invoice_fields(text)
                print(f"Extracted invoice fields: {invoice_fields}")
            
            # Store metadata in session
            session['file_metadata'] = {
                'filename': filename,
                'classification': predicted_class,
                'file_size': file_metadata['size'],
                'file_type': file_metadata['type'],
                'last_modified': file_metadata['last_modified'],
                'invoice_fields': invoice_fields
            }
            
            # Success response with redirect URL
            response['success'] = True
            response['message'] = f'File uploaded and processed successfully. Classified as: {predicted_class}'
            response['redirect_url'] = url_for('results')
            return jsonify(response)
            
        else:
            print("Invalid file type")
            response['message'] = 'Invalid file type'
            return jsonify(response), 400
            
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        response['message'] = f'Error processing document: {str(e)}'
        return jsonify(response), 500

@app.route('/api/upload_multiple', methods=['POST'])
def api_upload_multiple_files():
    """API endpoint for uploading multiple files - returns JSON response"""
    print("==== API UPLOAD MULTIPLE ENDPOINT CALLED ====")
    print(f"Request method: {request.method}")
    print(f"Request form data: {request.form}")
    print(f"Request files: {request.files}")
    
    response = {
        'success': False,
        'message': '',
        'redirect_url': None,
        'results': []
    }
    
    try:
        # Check if the post request has the files part
        if 'files[]' not in request.files:
            print("Error: No files part in the request")
            response['message'] = 'No files part in the request'
            return jsonify(response), 400
            
        files = request.files.getlist('files[]')
        print(f"Number of files: {len(files)}")
        
        if len(files) == 0 or all(file.filename == '' for file in files):
            print("Error: No selected files")
            response['message'] = 'No selected files'
            return jsonify(response), 400
        
        # List to store classification results
        classification_results = []
        
        # Process each file
        for file in files:
            if file and allowed_file(file.filename):
                print(f"Processing file: {file.filename}")
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)
                
                # Get file metadata
                file_metadata = get_file_metadata(file_path)
                
                try:
                    # Process the document based on file type
                    file_ext = os.path.splitext(file_path)[1].lower()
                    text = ""
                    
                    if file_ext in ['.pdf']:
                        result = parse_pdf(file_path)
                        # Extract just the text from the result
                        if isinstance(result, dict) and 'text' in result:
                            text = result['text']
                        else:
                            text = str(result)
                    elif file_ext in ['.txt', '.md']:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                        result = preprocess_document_image(file_path)
                        if isinstance(result, dict) and 'text' in result:
                            text = result['text']
                        else:
                            text = str(result)
                    elif file_ext in ['.csv', '.xls', '.xlsx']:
                        df = pd.read_excel(file_path) if file_ext in ['.xls', '.xlsx'] else pd.read_csv(file_path)
                        text = df.to_string()
                    else:
                        raise ValueError(f"Unsupported file type: {file_ext}")
                    
                    # Clean and classify the text
                    text = clean_text(text)
                    predicted_class = predict_document_class(text)
                    predicted_class = display_classes.get(predicted_class, predicted_class)
                    
                    # Initialize invoice fields
                    invoice_fields = {}
                    
                    # If document is classified as an invoice, extract invoice fields
                    if predicted_class.lower() == 'invoice':
                        print(f"File {filename} classified as invoice. Extracting invoice fields...")
                        invoice_fields = extract_invoice_fields(text)
                        print(f"Extracted invoice fields: {invoice_fields}")
                    
                    # Store result
                    file_result = {
                        'filename': filename,
                        'classification': predicted_class,
                        'file_size': file_metadata.get('file_size', 'Unknown'),
                        'file_type': file_metadata.get('file_type', 'Unknown'),
                        'invoice_fields': invoice_fields
                    }
                    classification_results.append(file_result)
                    
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
                    classification_results.append({
                        'filename': filename,
                        'classification': 'Error',
                        'error': str(e)
                    })
            else:
                print(f"Invalid file type: {file.filename}")
                classification_results.append({
                    'filename': file.filename,
                    'classification': 'Error',
                    'error': 'Invalid file type'
                })
        
        # Store all results in session
        session['classification_results'] = classification_results
        
        # Success response with redirect URL
        response['success'] = True
        response['message'] = f'Processed {len(classification_results)} files successfully'
        response['results'] = classification_results
        response['redirect_url'] = url_for('results_multiple')
        return jsonify(response)
            
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        response['message'] = f'Error processing files: {str(e)}'
        return jsonify(response), 500

@app.route('/results_multiple')
def results_multiple():
    """Display results for multiple classified documents"""
    # Get classification results from session
    classification_results = session.get('classification_results', [])
    
    return render_template('results_multiple.html', 
                          results=classification_results)

if __name__ == '__main__':
    app.run(debug=True, port=5001) 