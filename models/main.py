import cv2
import os
from ultralytics import YOLO
from plate_reader import read_plate_text

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "runs/detect/yolov12n-license-plate/weights/best.pt"   # <-- path to your trained weights
#INPUT_IMAGE = "input/auto_num_plate_detection/images/N5.jpeg" 
INPUT_IMAGE ="input/auto_num_plate_detection/images/N47.jpeg"
#INPUT_IMAGE = "input/auto_num_plate_detection/images/N22.jpeg"
OUTPUT_DIR = "test_output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# ----------------------------
# RUN DETECTION
# ----------------------------
results = model(INPUT_IMAGE)

img = cv2.imread(INPUT_IMAGE)
h, w, _ = img.shape

plate_texts = []  # store extracted numbers

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Crop the detected plate
        crop = img[y1:y2, x1:x2]

        crop_path = os.path.join(OUTPUT_DIR, "plate_crop.jpg")
        cv2.imwrite(crop_path, crop)

        # OCR
        text = read_plate_text(crop_path)

        print("\nDetected Plate:")
        print("Coordinates:", x1, y1, x2, y2)
        print("OCR Result:", text)

        if text:
            plate_texts.append(text)

# ----------------------------
# FINAL OUTPUT
# ----------------------------
print("\n========================")
print(" LICENSE NUMBERS FOUND ")
print("========================")

if len(plate_texts) == 0:
    print("No license numbers detected.")
else:
    for i, t in enumerate(plate_texts):
        print(f"{i+1}. {t}")

        """

# Load YOLO
print("🔄 Loading detection model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded!")

def detect_and_read_plate(image_path):
    print(f"\n📌 Processing image: {image_path}")
    
    # Run detection
    results = model(image_path)
    detections = results[0].boxes.data

    img = cv2.imread(image_path)

    if len(detections) == 0:
        print("❌ No license plate detected.")
        return

    for i, box in enumerate(detections):
        x1, y1, x2, y2, score, cls = box.tolist()

        # Crop region
        crop = img[int(y1):int(y2), int(x1):int(x2)]

        crop_path = os.path.join(OUTPUT_DIR, f"plate_{i}.jpg")
        cv2.imwrite(crop_path, crop)

        print(f"\n🟥 Cropped plate saved → {crop_path}")

        # OCR
        text = read_plate_text(crop)
        print("🔍 OCR Result:", text)

    print("\n🎉 Processing complete.")

# Run
detect_and_read_plate(INPUT_IMAGE)
