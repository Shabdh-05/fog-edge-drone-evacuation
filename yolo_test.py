from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Run detection on sample image (auto-download test image)
results = model("https://ultralytics.com/images/bus.jpg", show=True)

print("Detection completed")