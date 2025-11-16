from ultralytics import YOLO

# Load best model
model = YOLO("runs/detect/solarpanel_defect_detection/weights/best.pt")

# Run prediction on test images
model.predict(
    source="dataset/images/test",
    save=True,
    conf=0.25
)
