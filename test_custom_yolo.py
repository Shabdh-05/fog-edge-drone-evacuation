from ml_modules.yolo_survivor import SurvivorDetector

detector = SurvivorDetector()
detector.detect("bus.jpg")

print("Custom Survivor Detection Completed")