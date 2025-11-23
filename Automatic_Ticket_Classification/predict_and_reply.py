import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from dotenv import load_dotenv
import google.generativeai as genai

# ============================
# 1. LOAD API KEY
# ============================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found! Add GEMINI_API_KEY=your_key in .env file")

genai.configure(api_key=API_KEY)

# Working Gemini model from your environment:
GEMINI_MODEL = "models/gemini-2.5-pro"

# ============================
# 2. LOAD TOKENIZER & ENCODER
# ============================
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ============================
# 3. LOAD BEST MODEL
# ============================
model = load_model("models/best_lstm.h5")
MAX_LEN = 300   # match your training max_len

# ============================
# 4. CATEGORY PREDICTOR
# ============================
def predict_category(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    prediction = model.predict(padded)
    class_idx = np.argmax(prediction)
    category = label_encoder.inverse_transform([class_idx])[0]

    return category

# ============================
# 5. GEMINI REPLY GENERATOR
# ============================
def generate_reply(ticket_text, category):
    prompt = f"""
You are an automated customer support assistant.

Ticket Category: {category}

Customer Message:
"{ticket_text}"

Write a polite, empathetic, short reply (4–5 lines).
Do NOT provide advanced technical troubleshooting.
"""

    model = genai.GenerativeModel(GEMINI_MODEL)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# ============================
# 6. MAIN FUNCTION
# ============================
def predict_and_reply(ticket_text):
    category = predict_category(ticket_text)
    reply = generate_reply(ticket_text, category)
    return category, reply

# ============================
# 7. TEST
# ============================
if __name__ == "__main__":
    sample = "Can I change my flight date without extra charges?"
    cat, rep = predict_and_reply(sample)

    print("\nPredicted Category:", cat)
    print("\nAuto Reply:\n", rep)
