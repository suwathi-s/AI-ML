import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

# ========= LOAD ARTIFACTS ==========
model = load_model("models/best_lstm.h5")

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

X_test = np.load("models/X_test.npy")
y_test = np.load("models/y_test.npy")

# ========= PREDICT TEST SET ==========
pred_probs = model.predict(X_test)
pred_classes = np.argmax(pred_probs, axis=1)

# Convert encoded labels back to text
true_labels = label_encoder.inverse_transform(y_test)
pred_labels = label_encoder.inverse_transform(pred_classes)

# ========= METRICS ==========
accuracy = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels)
conf_matrix = confusion_matrix(true_labels, pred_labels)

print("\n Test Accuracy:", accuracy)
print("\n Classification Report:\n", report)
print("\n Confusion Matrix:\n", conf_matrix)

# ========= SAVE RESULTS ==========
# Save text report
with open("models/evaluationup_report.txt", "w", encoding="utf-8") as f:
    f.write("Test Accuracy:\n")
    f.write(str(accuracy) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))

# Save confusion matrix as CSV
cm_df = pd.DataFrame(
    conf_matrix,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)
cm_df.to_csv("models/confusionup_matrix.csv")

print("\n Evaluation results saved to:")
print("   → models/evaluationup_report.txt")
print("   → models/confusionup_matrix.csv")
