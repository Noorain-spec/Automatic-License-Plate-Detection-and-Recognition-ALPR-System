"""

# plate_reader.py
import easyocr
import cv2

# Initialize OCR reader only once
reader = easyocr.Reader(['en'], gpu=False)

def read_plate_text(image_path):
    '''
    Reads license plate number from an image using EasyOCR.
    Returns extracted text as string.
    '''
    print(f"\n🔍 Reading license plate text from: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error: Image not found.")
        return ""

    # OCR inference
    results = reader.readtext(img)

    plate_text = ""

    for box, text, score in results:
        # Avoid reading non-plate text, filter by confidence
        if score > 0.4:
            plate_text += text + " "

    plate_text = plate_text.strip()

    if plate_text == "":
        print("⚠️ No readable plate text detected.")
    else:
        print("✅ Detected Plate Number:", plate_text)

    return plate_text
"""
# plate_reader.py

from paddleocr import PaddleOCR
import cv2

# Initialize OCR (English + numbers only)
ocr = PaddleOCR(
    lang='en', 
    det=True, 
    rec=True, 
    use_angle_cls=True
)

def read_plate_text(cropped_plate):
    """
    Takes a cropped license plate image (numpy array)
    and returns extracted text.
    """

    if cropped_plate is None:
        return ""

    # Convert BGR → RGB
    img = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

    result = ocr.ocr(img, cls=True)

    if not result or not result[0]:
        return ""

    text = ""
    for line in result[0]:
        text_part = line[1][0]
        text += text_part + " "

    return text.strip()
