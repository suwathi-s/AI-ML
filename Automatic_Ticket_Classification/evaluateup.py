# evaluate.py
import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import tensorflow as tf

# ========= DEFINE focal loss (same as training) ==========
def focal_loss(gamma=2., alpha=0.25):

    def loss(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow((1 - y_pred), gamma)
        fl = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))

    return loss


# ========= LOAD LABEL ENCODER ==========
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)

# ========= LOAD MODEL WITH CUSTOM LOSS ==========
model = load_model(
    "models/bestupdated_lstm.h5",
    custom_objects={"loss": focal_loss(gamma=2, alpha=0.25)}
)

print(" Model Loaded Successfully")

# ========= LOAD TEST DATA ==========
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
