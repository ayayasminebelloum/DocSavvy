import os
import PyPDF2
from io import BytesIO
from pdf2image import convert_from_path, convert_from_bytes
import tempfile

# Import your existing preprocessing function
from .preprocess_images import preprocess_document_image

def parse_pdf(pdf_path=None, pdf_bytes=None, tesseract_path=None):
    """
    Convert PDF to images and extract text using image preprocessing.
    
    Parameters:
    -----------
    pdf_path : str, optional
        Path to the PDF file
    pdf_bytes : bytes, optional
        PDF file as bytes
    tesseract_path : str, optional
        Path to Tesseract executable (if not in PATH)
        
    Returns:
    --------
    dict
        Dictionary with keys similar to preprocess_document_image:
        - 'text': Combined text from all pages
        - 'original': PDF metadata
        - 'steps': Processing steps (simplified for PDF)
        - 'processed_image': First page image (if available)
        - 'all_ocr_results': Results from all pages
    """
    # Create a temporary directory to store images
    temp_dir = tempfile.mkdtemp()
    
    # Convert PDF to images
    try:
        if pdf_path:
            images = convert_from_path(pdf_path, dpi=300)
        elif pdf_bytes:
            images = convert_from_bytes(pdf_bytes, dpi=300)
        else:
            raise ValueError("Either pdf_path or pdf_bytes must be provided")
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return {
            'text': "",
            'original': None,
            'steps': [],
            'processed_image': None,
            'all_ocr_results': []
        }
    
    # Process each page and combine results
    all_text = ""
    all_results = []
    all_ocr_results = []
    
    # Get PDF metadata if possible
    pdf_metadata = {}
    if pdf_path:
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                if pdf_reader.metadata:
                    pdf_metadata = dict(pdf_reader.metadata)
                pdf_metadata['page_count'] = len(pdf_reader.pages)
        except Exception as e:
            print(f"Error reading PDF metadata: {e}")
    
    # Store first page image for reference
    first_page_image = None
    steps = [("PDF Original", "PDF document converted to images")]
    
    for i, image in enumerate(images):
        # Save image to temporary file
        image_path = os.path.join(temp_dir, f"page_{i+1}.png")
        image.save(image_path, "PNG")
        
        try:
            # Process image with existing function
            result = preprocess_document_image(image_path, tesseract_path=tesseract_path)
            
            # Extract text
            if isinstance(result, dict) and 'text' in result:
                page_text = result.get('text', '')
                all_text += page_text + "\n\n"
                
                # Store first page image if not already set
                if i == 0 and 'processed_image' in result:
                    first_page_image = result['processed_image']
                
                # Collect OCR results
                if 'all_ocr_results' in result:
                    for ocr_item in result['all_ocr_results']:
                        ocr_item['page'] = i + 1
                        all_ocr_results.append(ocr_item)
            else:
                page_text = str(result)
                all_text += page_text + "\n\n"
                
            all_results.append(result)
            steps.append((f"Page {i+1} Processed", "Text extracted"))
            
        except Exception as e:
            print(f"Error processing page {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            steps.append((f"Page {i+1} Error", str(e)))
    
    # Return in the same format as preprocess_document_image
    return {
        'text': all_text.strip(),
        'original': pdf_metadata,
        'steps': steps,
        'processed_image': first_page_image,
        'all_ocr_results': all_ocr_results
    }

# Example usage
if __name__ == "__main__":
    # Using the simple approach - convert to image and process
    result = parse_pdf(pdf_path="unstructured_data/contracts/2_waycda.pdf")
    print(f"Extracted text: {result['text'][:200]}...")