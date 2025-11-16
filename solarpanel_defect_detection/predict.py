# predict.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs

# Parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
MODEL_PATH = "fine_tuned_solar_model.keras"  # Best model from training
CLASS_INDICES_PATH = "class_indices.npy"

print("Loading model...")
model = load_model(MODEL_PATH)

# Load class labels
class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
inv_class_indices = {v: k for k, v in class_indices.items()}
print("Model and class labels loaded!")

def preprocess_img(img_path):
    """Load and preprocess image for prediction."""
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_single(img_path):
    """Predict class for a single image."""
    img_array = preprocess_img(img_path)
    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_label = inv_class_indices[pred_idx]

    print(f"\nPrediction for {os.path.basename(img_path)}:")
    print(f"   → Class: {predicted_label} | Confidence: {confidence:.2f}%")

    return predicted_label, confidence

if __name__ == "__main__":
    # Change this path to one of your test images
    test_image = r"C:\Users\91902\Desktop\solarpanel_defect_detection\Faulty_solar_panel\Electrical-damage\Electrical (3).png"
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
    else:
        predict_single(test_image)
