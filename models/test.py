import cv2
from ultralytics import YOLO
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_PATH = "runs/detect/yolov12n-license-plate/weights/best.pt"   # <-- path to your trained weights

IMAGE_PATH = "input/auto_num_plate_detection/images/N5.jpeg"      # <-- single image path (NOT folder)
VIDEO_PATH = "input/auto_num_plate_detection/TEST/TEST.mp4"      # <-- change if you have video, else ignore

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
print("🔄 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!/n")


# ---------------------------------------------------------
# FUNCTION: Run inference on a single image
# ---------------------------------------------------------
def test_image(model, img_path):
    """Run inference on a single image."""
    print("/n🖼️ Running inference on image:", img_path)

    results = model(img_path, save=True)

    print("Inference complete. YOLO saved results here:")
    print(results[0].save_dir)


# ---------------------------------------------------------
# FUNCTION: Run inference on a video
# ---------------------------------------------------------
def test_video(model, video_path):
    """Run inference on a video."""
    print("/n🎥 Running video inference:", video_path)

    results = model(video_path, save=True)

    print("Video inference complete. Saved results here:")
    print(results[0].save_dir)


# ---------------------------------------------------------
# RUN TESTS
# ---------------------------------------------------------
# Run image test
test_image(model, IMAGE_PATH)

# -----------------------------
# Run video test (optional)
# -----------------------------
# Uncomment if you want video processing
test_video(model, VIDEO_PATH)
