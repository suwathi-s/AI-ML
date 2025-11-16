# evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# ======================
# Paths
# ======================
MODEL_PATH = "fine_tuned_solar_model.keras"
TEST_DIR = "Faulty_solar_panel"   # adjust if your test folder path is different
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================
# Load model
# ======================
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# ======================
# Load test dataset
# ======================
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ======================
# Evaluate
# ======================
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}\n")

# ======================
# Predictions
# ======================
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
class_labels = list(test_generator.class_indices.keys())

# ======================
# Classification Report
# ======================
report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:\n")
print(report)

# Save report to file
report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Test Loss: {loss:.4f}\n\n")
    f.write(report)
print(f"Classification report saved at {report_path}")

# ======================
# Confusion Matrix
# ======================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved at {cm_path}")
