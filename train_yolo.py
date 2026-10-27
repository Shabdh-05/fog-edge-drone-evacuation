from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="survivor.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        device=0
    )

if __name__ == "__main__":
    main()