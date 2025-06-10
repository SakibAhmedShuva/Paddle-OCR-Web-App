# /paddle-ocr-webapp/app.py

import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter
import io
# FIX 1: Import secure_filename to handle filenames with special characters safely
from werkzeug.utils import secure_filename

# --- Configuration ---
# Define the folder to store uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Global Model Initialization ---
# Initialize PaddleOCR. This is done once globally to avoid reloading the model on each request.
# The model files will be downloaded automatically on the first run.
print("Initializing PaddleOCR... This may take a moment on the first run.")
# Using use_angle_cls=False to match the notebook's behavior where the angle classifier was not used.
# Set show_log=False to keep the console clean.
ocr_model = PaddleOCR(lang='en', use_angle_cls=False, show_log=False)
print("PaddleOCR Initialized Successfully.")

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_variants(image_path):
    """
    Creates multiple preprocessed versions of an image for improved OCR accuracy.
    Returns a list of preprocessed image paths.
    """
    variants = []
    base_name = os.path.splitext(image_path)[0]
    
    try:
        # Load the original image
        original_image = Image.open(image_path)
        
        # Convert to RGB if needed
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Variant 1: Enhanced contrast and sharpened
        enhanced_image = original_image.copy()
        # Increase contrast
        contrast_enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = contrast_enhancer.enhance(1.5)
        # Sharpen the image
        enhanced_image = enhanced_image.filter(ImageFilter.SHARPEN)
        # Increase brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = brightness_enhancer.enhance(1.1)
        
        variant1_path = f"{base_name}_enhanced.jpg"
        enhanced_image.save(variant1_path, 'JPEG', quality=95)
        variants.append(('Enhanced (Contrast + Sharp)', variant1_path))
        
        # Variant 2: Grayscale with noise reduction (using OpenCV)
        cv_image = cv2.imread(image_path)
        # FIX: Check if cv_image was loaded correctly before processing
        if cv_image is None:
            print(f"Warning: Could not read image {image_path} with OpenCV.")
        else:
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            denoised = cv2.GaussianBlur(gray_image, (3, 3), 0)
            
            # Apply adaptive threshold for better text extraction
            adaptive_thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            variant2_path = f"{base_name}_denoised.jpg"
            cv2.imwrite(variant2_path, adaptive_thresh)
            variants.append(('Denoised Grayscale', variant2_path))
        
        # Variant 3: High contrast black and white
        # Convert to grayscale first
        bw_image = original_image.convert('L')
        
        # Enhance contrast significantly
        contrast_enhancer = ImageEnhance.Contrast(bw_image)
        bw_image = contrast_enhancer.enhance(2.0)
        
        # Apply threshold to create pure black and white
        threshold = 128
        bw_image = bw_image.point(lambda p: 255 if p > threshold else 0, mode='1')
        
        variant3_path = f"{base_name}_bw.jpg"
        bw_image.save(variant3_path, 'JPEG')
        variants.append(('High Contrast B&W', variant3_path))
        
        return variants
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return []


def combine_ocr_results(results_list):
    """
    Combines multiple OCR results using confidence-based selection and word frequency.
    Returns the best combined text result.
    """
    if not results_list:
        return "No text found."
    
    # Extract all text with confidence scores
    all_words = {}  # word -> {'count': int, 'total_confidence': float}
    all_texts = []
    
    for variant_name, result in results_list:
        if not result or not result[0]:
            continue
            
        variant_text = process_ocr_result(result)
        all_texts.append((variant_name, variant_text))
        
        # Extract individual words with confidence
        for item in result[0]:
            text = item[1][0].strip()
            confidence = item[1][1]
            
            # Split into words and process each
            words = text.split()
            for word in words:
                word_clean = word.lower().strip('.,!?;:"()[]{}')
                if len(word_clean) > 1:  # Ignore single characters
                    if word_clean not in all_words:
                        all_words[word_clean] = {'count': 0, 'total_confidence': 0.0, 'best_form': word}
                    all_words[word_clean]['count'] += 1
                    all_words[word_clean]['total_confidence'] += confidence
                    
                    # Keep the form with highest confidence
                    avg_conf = all_words[word_clean]['total_confidence'] / all_words[word_clean]['count']
                    if confidence > avg_conf * 0.9:  # Within 90% of average
                        all_words[word_clean]['best_form'] = word
    
    # Find the text with most words and good structure
    best_text = ""
    max_word_count = 0
    
    for variant_name, text in all_texts:
        word_count = len(text.split())
        if word_count > max_word_count and len(text.strip()) > 0:
            max_word_count = word_count
            best_text = text
    
    # If we have word frequency data, enhance the best text
    if all_words and best_text:
        # Replace words with their best forms based on confidence
        words_in_best = best_text.split()
        enhanced_words = []
        
        for word in words_in_best:
            word_clean = word.lower().strip('.,!?;:"()[]{}')
            if word_clean in all_words and all_words[word_clean]['count'] > 1:
                # Use the most confident form
                best_form = all_words[word_clean]['best_form']
                # Preserve punctuation from original
                if word != word_clean:
                    punct = word.replace(word_clean, '')
                    enhanced_words.append(best_form + punct)
                else:
                    enhanced_words.append(best_form)
            else:
                enhanced_words.append(word)
        
        best_text = ' '.join(enhanced_words)
    
    return best_text if best_text.strip() else "No text found."


# FIX 2: Corrected the cleanup_temp_files function to only clean up files.
def cleanup_temp_files(file_paths):
    """Clean up temporary original and preprocessing files."""
    for _, file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up {file_path}: {str(e)}")

# FIX 2: Moved the misplaced logic into its own correct function.
def process_ocr_result(result):
    """
    Processes the raw result from PaddleOCR to format the text logically.
    This logic sorts text first by vertical position and then by horizontal
    position to reconstruct lines correctly.
    """
    if not result or not result[0]:
        return "No text found."

    text_coordinates = []
    # The result from paddleocr is wrapped in a list, so we access result[0]
    for item in result[0]:
        # item: [bounding_box, (text, confidence_score)]
        box = item[0]
        text = item[1][0]
        
        # Calculate the middle y-coordinate and starting x-coordinate of the text box
        y_coord = (box[0][1] + box[2][1]) / 2
        x_coord = box[0][0]
        
        text_coordinates.append({'text': text, 'y': y_coord, 'x': x_coord})

    # Sort items primarily by their vertical position (y), and secondarily by horizontal (x)
    text_coordinates.sort(key=lambda item: (item['y'], item['x']))
    
    # Group text items into lines based on vertical proximity
    line_threshold = 18  # Threshold to determine if text is on the same line
    formatted_lines = []
    if not text_coordinates:
        return ""

    current_line = [text_coordinates[0]]
    for i in range(1, len(text_coordinates)):
        prev_item = current_line[-1]
        current_item = text_coordinates[i]
        
        # If the vertical distance is within the threshold, it's the same line
        if abs(current_item['y'] - prev_item['y']) <= line_threshold:
            current_line.append(current_item)
        else:
            # New line detected, process the previous line
            # Sort the items in the completed line by their x-coordinate
            current_line.sort(key=lambda x: x['x'])
            formatted_lines.append(' '.join([item['text'] for item in current_line]))
            
            # Start a new line
            current_line = [current_item]
            
    # Add the last line
    if current_line:
        current_line.sort(key=lambda x: x['x'])
        formatted_lines.append(' '.join([item['text'] for item in current_line]))

    return '\n'.join(formatted_lines)


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page for the frontend."""
    return render_template('index.html')

@app.route('/ocr-batch', methods=['POST'])
def upload_and_ocr_batch():
    """Handles multiple file uploads and performs OCR on each."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    
    files = request.files.getlist('files')
    
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected for uploading'}), 400
    
    results = []
    
    # Create the uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    for file in files:
        if file and allowed_file(file.filename):
            # FIX 1: Sanitize filename and create a unique name
            safe_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{safe_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                file.save(filepath)
                
                # Run OCR on the saved image file
                result = ocr_model.ocr(filepath, cls=False)
                
                # Process the result to get formatted text
                formatted_text = process_ocr_result(result)
                
                results.append({
                    'filename': file.filename,
                    'text': formatted_text,
                    'status': 'success'
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'text': '',
                    'status': 'error',
                    'error': str(e)
                })
            finally:
                # Clean up the uploaded file after processing
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            results.append({
                'filename': file.filename if file else 'Unknown',
                'text': '',
                'status': 'error',
                'error': 'File type not allowed'
            })
    
    return jsonify({'results': results})

@app.route('/ocr-augmented', methods=['POST'])
def upload_and_ocr_augmented():
    """Handles single file upload with augmented OCR processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
        
    if file and allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # FIX 1: Sanitize filename and create a unique name
        safe_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{safe_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        temp_files = [('Original', filepath)]  # Keep track for cleanup
        
        try:
            # Create preprocessed variants
            variants = preprocess_image_variants(filepath)
            temp_files.extend(variants)
            
            # Run OCR on original image
            original_result = ocr_model.ocr(filepath, cls=False)
            original_text = process_ocr_result(original_result)
            
            # Run OCR on each variant
            variant_results = []
            variant_details = []
            
            for variant_name, variant_path in variants:
                try:
                    variant_result = ocr_model.ocr(variant_path, cls=False)
                    variant_text = process_ocr_result(variant_result)
                    variant_results.append((variant_name, variant_result))
                    variant_details.append({
                        'name': variant_name,
                        'text': variant_text,
                        'word_count': len(variant_text.split()) if variant_text else 0
                    })
                except Exception as e:
                    variant_details.append({
                        'name': variant_name,
                        'text': f"Error processing variant: {str(e)}",
                        'word_count': 0
                    })
            
            # Add original to results for combination
            variant_results.append(('Original', original_result))
            
            # Combine results intelligently
            combined_text = combine_ocr_results(variant_results)
            
            return jsonify({
                'text': combined_text,
                'original_text': original_text,
                'variants': variant_details,
                'processing_info': {
                    'variants_processed': len(variants),
                    'combination_method': 'confidence_and_frequency_based'
                }
            })
            
        except Exception as e:
            return jsonify({'error': f'An error occurred during augmented OCR processing: {str(e)}'}), 500
        finally:
            # Clean up all temporary files
            cleanup_temp_files(temp_files)
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/ocr-batch-augmented', methods=['POST'])
def upload_and_ocr_batch_augmented():
    """Handles multiple file uploads with augmented OCR processing."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    
    files = request.files.getlist('files')
    
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected for uploading'}), 400
    
    results = []
    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    for file in files:
        if file and allowed_file(file.filename):
            # FIX 1: Sanitize filename and create a unique name
            safe_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{safe_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            temp_files = []
            
            try:
                file.save(filepath)
                temp_files.append(('Original', filepath))
                
                # Create preprocessed variants
                variants = preprocess_image_variants(filepath)
                temp_files.extend(variants)
                
                # Run OCR on original image
                original_result = ocr_model.ocr(filepath, cls=False)
                original_text = process_ocr_result(original_result)
                
                # Run OCR on each variant
                variant_results = []
                variant_details = []
                
                for variant_name, variant_path in variants:
                    try:
                        variant_result = ocr_model.ocr(variant_path, cls=False)
                        variant_text = process_ocr_result(variant_result)
                        variant_results.append((variant_name, variant_result))
                        variant_details.append({
                            'name': variant_name,
                            'text': variant_text,
                            'word_count': len(variant_text.split()) if variant_text else 0
                        })
                    except Exception as e:
                        variant_details.append({
                            'name': variant_name,
                            'text': f"Error: {str(e)}",
                            'word_count': 0
                        })
                
                # Add original to results for combination
                variant_results.append(('Original', original_result))
                
                # Combine results intelligently
                combined_text = combine_ocr_results(variant_results)
                
                results.append({
                    'filename': file.filename,
                    'text': combined_text,
                    'original_text': original_text,
                    'variants': variant_details,
                    'status': 'success',
                    'processing_info': {
                        'variants_processed': len(variants),
                        'combination_method': 'confidence_and_frequency_based'
                    }
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'text': '',
                    'status': 'error',
                    'error': str(e)
                })
            finally:
                # Clean up temporary files for this image
                cleanup_temp_files(temp_files)
        else:
            results.append({
                'filename': file.filename if file else 'Unknown',
                'text': '',
                'status': 'error',
                'error': 'File type not allowed'
            })
    
    return jsonify({'results': results})
    
@app.route('/ocr', methods=['POST'])
def upload_and_ocr():
    """Handles the file upload and performs OCR."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
        
    if file and allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # FIX 1: Sanitize filename before saving
        safe_filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        try:
            # Run OCR on the saved image file
            result = ocr_model.ocr(filepath, cls=False)
            
            # Process the result to get formatted text
            formatted_text = process_ocr_result(result)
            
            return jsonify({'text': formatted_text})
            
        except Exception as e:
            return jsonify({'error': f'An error occurred during OCR processing: {str(e)}'}), 500
        finally:
            # Clean up the uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'error': 'File type not allowed'}), 400

# --- Main Execution ---
if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)