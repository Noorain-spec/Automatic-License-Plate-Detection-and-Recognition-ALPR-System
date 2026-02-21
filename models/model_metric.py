from ultralytics import YOLO

# Load your trained model directly
model = YOLO("runs/detect/yolov12n-license-plate/weights/best.pt")

# Evaluate
metrics = model.val()

print("\n📊 Evaluation Metrics:")
print("Precision:", metrics.box.p)
print("Recall:", metrics.box.r)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("F1 Score:", 2 * (metrics.box.p * metrics.box.r) / (metrics.box.p + metrics.box.r))
