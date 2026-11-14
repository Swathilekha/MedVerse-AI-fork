import os
from PIL import Image
import pytesseract

from ..config import TESSERACT_CMD

# Configure Tesseract executable
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def extract_text(image_path: str) -> str:
    """
    Extract text from a medical report image using Tesseract OCR.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    image = Image.open(image_path)

    text = pytesseract.image_to_string(image)

    return text