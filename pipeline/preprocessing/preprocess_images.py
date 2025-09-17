import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract as tesserac
import os
import re

def preprocess_document_image(image_path, tesseract_path=None, preview=False):
    """
    Document image preprocessing pipeline optimized for OCR with multilingual and layout-aware tuning.
    Enhanced with table detection capabilities.
    
    Parameters:
    -----------
    image_path : str
        Path to the input document image
    tesseract_path : str, optional
        Path to the Tesseract executable (if not in PATH)
    preview : bool, optional
        Whether to display preprocessing steps visually
        
    Returns:
    --------
    dict
        A dictionary containing the original image, all preprocessing steps,
        the final processed image, and the extracted text (including any detected tables)
    """
    # Set Tesseract path if provided
    if tesseract_path:
        tesserac.pytesseract.tesseract_cmd = tesseract_path

    # Read the image
    print(image_path)
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Store all processing steps with their names
    steps = [("Original", cv2.cvtColor(original, cv2.COLOR_BGR2RGB))]

    # 1. Convert to grayscale
    if len(original.shape) == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original.copy()
    steps.append(("Grayscale", gray))

    # 2. Resize image to improve OCR results (ADDED adaptive resizing)
    scale_factor = 2.0  # force upscale for better OCR recognition
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    steps.append(("Resized", gray))

    # 3. Analyze image properties to determine best preprocessing
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    dynamic_range = (np.max(gray) - np.min(gray)) / 255

    # 4. Apply contrast enhancement if needed
    enhanced = gray.copy()
    if dynamic_range < 0.5 or std_val < 40:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(gray)
        clahe_std = np.std(clahe_enhanced)

        min_val, max_val = np.min(gray), np.max(gray)
        stretched = np.uint8(255 * ((gray - min_val) / (max(max_val - min_val, 1))))
        stretched_std = np.std(stretched)

        if clahe_std > stretched_std:
            enhanced = clahe_enhanced
            steps.append(("CLAHE Enhancement", enhanced))
        else:
            enhanced = stretched
            steps.append(("Contrast Stretching", enhanced))

    # 5. Gentle noise reduction
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    steps.append(("Edge-Preserving Denoising", denoised))

    # 6. Binarization
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    steps.append(("Otsu Thresholding", otsu))

    block_size = 21
    c_value = 9
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_value
    )
    steps.append((f"Adaptive Thresholding (block={block_size}, C={c_value})", adaptive))

    binary = adaptive

    # 7. Morphological noise removal
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small_components = np.sum(stats[1:, cv2.CC_STAT_AREA] < 5)

    if small_components > num_labels * 0.2:
        noise_kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_kernel)
        steps.append(("Opening (Noise Removal)", opening))
    else:
        opening = binary
        steps.append(("No Opening Applied", opening))

    processed = opening

    # 8. Invert to black text on white background
    inverted = cv2.bitwise_not(processed)
    steps.append(("Inverted for OCR", inverted))

    # 9. Sharpening
    sharpen_kernel = np.array([[-0.5, -0.5, -0.5],
                               [-0.5, 5.0, -0.5],
                               [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(inverted, -1, sharpen_kernel)
    steps.append(("Lightly Sharpened", sharpened))

    # 10. Table detection using Tesseract and image processing
    table_text = ""
    try:
        # Use horizontal and vertical line detection to find tables
        table_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding
        _, table_binary = cv2.threshold(table_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(table_binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        vertical_lines = cv2.morphologyEx(table_binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Find contours in the combined mask
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour as potential table
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Check if bounding box is large enough to be a table
            if w > inverted.shape[1] * 0.2 and h > inverted.shape[0] * 0.1:
                # Extract the table region
                table_region = inverted[y:y+h, x:x+w]
                
                # Draw the detected table
                table_visualization = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(table_visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
                steps.append(("Table Detection", table_visualization))
                
                # Use Tesseract to extract data from the table region with specialized config
                table_pil = Image.fromarray(table_region)
                # Try TSV output first for better structure
                try:
                    table_config = r'--oem 3 --psm 6 -l eng+fra+spa tsv'
                    tsv_output = tesserac.image_to_data(table_pil, config=table_config, output_type=tesserac.Output.DICT)
                    
                    # Process TSV data into rows and columns
                    n_boxes = len(tsv_output['text'])
                    lines = {}
                    
                    for i in range(n_boxes):
                        if int(float(tsv_output['conf'][i])) > 0:  # Only consider text with confidence > 0
                            text = tsv_output['text'][i]
                            if text.strip():  # Only consider non-empty text
                                line_num = tsv_output['line_num'][i]
                                if line_num not in lines:
                                    lines[line_num] = []
                                lines[line_num].append((tsv_output['left'][i], text))
                    
                    # Sort and format table text
                    table_content = "\n\n----- TABLE CONTENT START -----\n"
                    for line_num in sorted(lines.keys()):
                        # Sort items in line by x-coordinate
                        line_items = sorted(lines[line_num], key=lambda item: item[0])
                        line_text = "  ".join([item[1] for item in line_items])
                        table_content += line_text + "\n"
                    table_content += "----- TABLE CONTENT END -----\n\n"
                    table_text = table_content
                
                except Exception as e:
                    print(f"TSV extraction failed: {e}, trying regular OCR")
                    # Fallback to regular OCR
                    table_config = r'--oem 3 --psm 6 -l eng+fra+spa'
                    extracted_text = tesserac.image_to_string(table_pil, config=table_config)
                    
                    if extracted_text.strip():
                        table_text = "\n\n----- TABLE CONTENT START -----\n"
                        table_text += extracted_text
                        table_text += "\n----- TABLE CONTENT END -----\n\n"
    
    except Exception as e:
        print(f"Error during table detection: {e}")
        import traceback
        traceback.print_exc()

    # 11. OCR candidates with multilingual support and layout analysis
    ocr_candidates = [
        ("Standard", inverted),
        ("Sharpened", sharpened),
        ("Original Binary", cv2.bitwise_not(binary))
    ]

    ocr_results = []
    for name, img in ocr_candidates:
        pil_img = Image.fromarray(img)
        config = r'--oem 3 --psm 4 -l eng+fra+spa'  # multilingual OCR with layout detection
        try:
            text = tesserac.image_to_string(pil_img, config=config)
            char_count = len([c for c in text if c.isalnum()])
            ocr_results.append({
                'name': name,
                'text': text,
                'char_count': char_count
            })
        except Exception as e:
            print(f"OCR failed for {name}: {e}")

    if ocr_results:
        best_result = max(ocr_results, key=lambda x: x['char_count'])
        final_text = best_result['text']
        final_img = next(img for name, img in ocr_candidates if name == best_result['name'])
        steps.append((f"Final OCR Image ({best_result['name']})", final_img))
    else:
        final_text = "OCR failed for all processing methods"
        final_img = inverted
        steps.append(("Final OCR Image (Default)", final_img))
    
    # Combine the regular OCR text with any table text found
    if table_text:
        final_text = final_text + "\n" + table_text

    if preview:
        visualize_steps(steps)

    # Return the original format with the combined text
    return {
        'original': original,
        'steps': steps,
        'processed_image': final_img,
        'text': final_text,
        'all_ocr_results': ocr_results
    }

def visualize_steps(steps):
    """
    Display all preprocessing steps in a grid layout.
    
    Parameters:
    -----------
    steps : list
        List of (name, image) tuples representing each preprocessing step
    """
    n = len(steps)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(20, 5 * rows))

    for i, (name, img) in enumerate(steps):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 3:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()