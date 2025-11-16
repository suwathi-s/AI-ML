from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    augment=True,
    name="solarpanel_defect_detection",
    save=True,
    plots=True
)

print("Training complete! Check runs/detect/solarpanel_defect_detection for results.")
