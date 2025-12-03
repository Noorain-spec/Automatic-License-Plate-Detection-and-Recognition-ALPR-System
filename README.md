# Automatic-License-Plate-Detection-and-Recognition-ALPR-System

Create and activate virtual environment: 
    python venv venv
    venv/source/activate
    
install req.txt file using: pip install -r req.txt

run UI: streamlit /modesl/app.py 

Models Used

YOLOv8n / YOLOv12n (Ultralytics) for license plate detection.

EasyOCR / PaddleOCR for text extraction from detected plates.

OpenCV for image/video processing (cropping, pre-processing).

Streamlit for UI-based interactive deployment.

