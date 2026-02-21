import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import easyocr
import numpy as np

# ---------------------------------------------------------
# Load YOLO Model
# ---------------------------------------------------------
MODEL_PATH = "runs/detect/yolov12n-license-plate/weights/best.pt"   # <-- path to your trained weights
model = YOLO(MODEL_PATH)

# ---------------------------------------------------------
# Load OCR
# ---------------------------------------------------------
reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------------------------------------
# Extract text from cropped plate
# ---------------------------------------------------------
def read_plate_text(plate_img):
    results = reader.readtext(plate_img)
    text = ""
    for (_, txt, conf) in results:
        text += txt + " "
    return text.strip()

# ---------------------------------------------------------
# Plate Detection + OCR
# ---------------------------------------------------------
def detect_and_read(image):

    results = model(image)
    final_texts = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        for (x1, y1, x2, y2) in boxes:
            plate_crop = image[int(y1):int(y2), int(x1):int(x2)]

            if plate_crop.size == 0:
                continue

            text = read_plate_text(plate_crop)
            final_texts.append((plate_crop, text))

    return final_texts

# ---------------------------------------------------------
# STREAMLIT UI (Two-Column Layout)
# ---------------------------------------------------------
st.title("🚗 Automatic License Plate Detection and Recognition (ALPR) System")
st.write("Upload an image/video to detect plates and extract plate numbers")

option = st.radio("Select Input Type", ["Image", "Video"])

# Create layout columns
left_col, right_col = st.columns([1.2, 1])

# ---------------- IMAGE MODE ----------------
if option == "Image":
    with left_col:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        run_btn = st.button("🔍 Run Detection")

    if uploaded_file is not None:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        image = cv2.imread(tfile.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with left_col:
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

        if run_btn:
            with st.spinner("Detecting plates..."):
                outputs = detect_and_read(image)

            st.success("Detection complete!")

            with right_col:
                st.subheader("📌 Detected Plates & Extracted Text")
                for idx, (crop, text) in enumerate(outputs):
                    st.image(crop, caption=f"Cropped Plate #{idx+1}", width=300)
                    st.info(f"Extracted Text: **{text}**")

# ---------------- VIDEO MODE ----------------
else:
    with left_col:
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])
        run_btn = st.button("🎥 Process Video")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        with left_col:
            st.video(tfile.name)

        if run_btn:
            cap = cv2.VideoCapture(tfile.name)
            plate_texts = []

            stframe = left_col.empty()  # show video left

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                outputs = detect_and_read(frame)
                for (_, text) in outputs:
                    plate_texts.append(text)

                stframe.image(frame, channels="BGR")

            cap.release()

            st.success("Video processing complete!")

            with right_col:
                st.subheader("📌 Extracted Plates")
                for t in plate_texts:
                    st.write(f"- **{t}**")
