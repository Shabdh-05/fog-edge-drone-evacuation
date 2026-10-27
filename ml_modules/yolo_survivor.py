from ultralytics import YOLO

class SurvivorDetector:
    def __init__(self):
        self.model = YOLO("runs/detect/train3/weights/best.pt")

    def detect(self, image_path):
        results = self.model(image_path, show=True)
        return results