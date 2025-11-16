# Solar Panel Defect Detection 

This project is a complete end-to-end deep learning system for automatically detecting defects in solar panels using:

🔹 CNN Classification (TensorFlow / Keras)
🔹 YOLOv8 Object Detection (Ultralytics)
🔹 Streamlit Web Interface

The system can classify defects, detect defect regions, visualize results, and generate downloadable reports.

## 📁 Project Folder Structure 

solarpanel_defect_detection/
│
├── app.py                           # Streamlit Application
├── solar.env                         # Virtual environment (Python 3.13.5)
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── test/
│   │   └── val/
│   └── labels/
│       ├── train/
│       ├── test/
│       └── val/
│
├── Faulty_solar_panel/               # Raw dataset with defect categories
│   ├── Bird-drop
│   ├── Clean
│   ├── Dusty
│   ├── Electrical-damage
│   ├── Physical-Damage
│   └── Snow-Covered
│
├── runs/
│   └── detect/
│       ├── predict/
│       └── solarpanel_defect_detection/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── models/                           # All CNN models generated during training
│   ├── best_solar_model-01-valacc-0.2874.keras
│   ├── best_solar_model-02-valacc-0.4023.keras
│   ├── best_solar_model-03-valacc-0.4828.keras
│   ├── best_solar_model-04-valacc-0.6117.keras
│   ├── best_solar_model-05-valacc-0.5805.keras
│   ├── best_solar_model-07-valacc-0.6092.keras
│   └── best_solar_model-08-valacc-0.6379.keras
│
├── predict.py                        # CNN Prediction Helper Script
├── train_yolo.py                      # YOLOv8 Training Script
├── fine_tune.py                       # Model Fine-Tuning Script
├── evaluate.py                        # Model Evaluation Script
├── split_dataset.py                   # Train/Test/Val split
├── data_preprocessing.py              # Image preprocessing pipeline
├── data.yaml                          # YOLOv8 Dataset Configuration
│

.

## Key Features
🔹 1. CNN Classification

Trained using Keras/TensorFlow

Multi-class classifier

Works on uploaded RGB images

Outputs defect type & confidence score

🔹 2. YOLOv8 Object Detection

Trained using manually labeled YOLO-format dataset

Detects defect location + type

Draws bounding boxes on solar panels

🔹 3. Streamlit Web App

Upload multiple images

Run CNN / YOLO / Both

Display predictions

Download results as CSV

Real-time detection visualization

## Technologies Used

Deep Learning -	TensorFlow, Keras

Object Detection - YOLOv8 (Ultralytics)

Web App	Streamlit

Image Processing - OpenCV, PIL

Annotation - LabelImg

Python Versions -	3.13.5 (CNN), 3.11.9 (YOLO)

## Install Required Libraries

pip install numpy pandas tensorflow pillow opencv-python streamlit ultralytics matplotlib


## Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Loss curves

YOLO output:

mAP50

mAP50-95

Precision

Recall

.

##  Author
Suwathi

Solar Panel Defect Detection Project — Deep Learning + YOLOv8

