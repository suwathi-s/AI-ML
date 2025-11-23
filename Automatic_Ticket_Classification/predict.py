import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# ===============================  
# Load tokenizer, label encoder, model  
# ===============================
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = load_model("models/best_lstm.h5")

MAX_LEN = 300  # must match preprocess.py

# ===============================  
# Predict Function  
# ===============================
def predict_ticket(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    pred = model.predict(padded)
    class_index = np.argmax(pred)
    label = label_encoder.inverse_transform([class_index])[0]

    return label


# ===============================  
# Test Code  
# ===============================
if __name__ == "__main__":
    user_input = input("Enter a customer ticket text: ")
    result = predict_ticket(user_input)
    print("\n Predicted Category:", result)
