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
