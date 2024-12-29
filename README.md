# PaddleOCR Tutorial

This repository contains a tutorial implementation of PaddleOCR, demonstrating how to perform Optical Character Recognition (OCR) using the PaddlePaddle framework.

## Overview

PaddleOCR is a rich, leading OCR toolkit that supports multiple languages and offers excellent text detection and recognition capabilities. This tutorial shows how to set up and use PaddleOCR for text extraction from images, with specific focus on proper text alignment and formatting.

## Prerequisites

- Python 3.6+
- CUDA-compatible GPU (recommended for better performance)
- Basic understanding of OCR concepts

## Installation

Install the required packages using pip:

```bash
pip install paddlepaddle-gpu
pip install paddleocr
```

For CPU-only installation, use:
```bash
pip install paddlepaddle
pip install paddleocr
```

## Required Libraries

```python
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
```

## Usage

### 1. Initialize the OCR Model

```python
# Setup model with English language support
ocr_model = PaddleOCR(lang='en')
```

### 2. Load and Process Image

```python
# Specify the path to your image
img_path = os.path.join('.', '/content/denoised.jpg')

# Run OCR on the image
result = ocr_model.ocr(img_path)
```

### 3. Text Extraction and Formatting

The script includes sophisticated text extraction that:
- Extracts text, bounding boxes, and confidence scores
- Sorts text based on vertical position
- Groups text lines based on proximity
- Formats output with proper line breaks

```python
# Extract components
boxes = []
texts = []
scores = []
text_coordinates = []

# Process OCR results
for line in result:
    if line:
        for item in line:
            if len(item) == 2:
                box = item[0]
                text = item[1][0]
                score = item[1][1]
                y_coord = (box[0][1] + box[2][1]) / 2
                text_coordinates.append((text, y_coord))
```

### 4. Text Alignment

The script implements intelligent text alignment:
- Groups text elements that are vertically close
- Maintains proper line spacing
- Joins text elements on the same line
- Configurable threshold for line detection

## Customization

You can customize the text grouping by adjusting the `line_threshold` parameter:

```python
line_threshold = 10  # Adjust based on your image characteristics
```

## Output

The script outputs properly formatted text with:
- Preserved line breaks
- Maintained text order
- Grouped text elements
- Clean formatting

## Features

- Multi-language support (default: English)
- Automatic text detection and recognition
- Intelligent text alignment and grouping
- Confidence scoring for recognized text
- Customizable text grouping parameters
- Support for various image formats

## Best Practices

1. **Image Preparation**
   - Use clear, high-resolution images
   - Ensure good contrast between text and background
   - Pre-process images if necessary (denoising, contrast adjustment)

2. **Parameter Tuning**
   - Adjust `line_threshold` based on your specific use case
   - Consider image resolution when setting thresholds
   - Test with different confidence thresholds if needed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PaddlePaddle team for the excellent OCR toolkit
- Contributors to the PaddleOCR project

## Contact

For questions or feedback, please open an issue in the repository.
