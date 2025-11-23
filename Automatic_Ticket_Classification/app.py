# app.py
import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from dotenv import load_dotenv
import google.generativeai as genai

# -------------------------------------------------
# Streamlit UI Setup (Dark Theme)
# -------------------------------------------------
st.set_page_config(page_title="Ticket Classifier", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0d1117;
    color: #ffffff;
}
section.main > div {
    background-color: #0d1117;
}
textarea, input, .stTextInput, .stTextArea {
    background-color: #161b22 !important;
    color: white !important;
    border-radius: 8px;
    border: 1px solid #30363d !important;
}
.stButton>button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #2ea043;
}
</style>
""", unsafe_allow_html=True)

st.title("📩 Automatic Ticket Classification & Customer Support Reply")
st.write("Effortlessly classify customer complaints & generate instant support replies.")

MAX_LEN = 300  
MODEL_PATH = "models/best_bilstm.h5"   # <-- Your final model
TOKENIZER_PATH = "models/tokenizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# -------------------------------------------------
# Load Assets
# -------------------------------------------------
@st.cache_resource
def load_assets():
    tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
    label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))
    model = load_model(MODEL_PATH, compile=False)
    return tokenizer, label_encoder, model

tokenizer, label_encoder, bilstm_model = load_assets()

# -------------------------------------------------
# Gemini AI Setup
# -------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
GEMINI_MODEL = "gemini-2.5-pro"

# -------------------------------------------------
# Predict Category (No Confidence)
# -------------------------------------------------
def predict_category(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    preds = bilstm_model.predict(padded, verbose=0)[0]

    idx = int(np.argmax(preds))
    label = label_encoder.inverse_transform([idx])[0]
    return label

# -------------------------------------------------
# Generate AI Reply
# -------------------------------------------------
def generate_reply(message, category):
    if not API_KEY:
        return "⚠ Gemini API key missing."

    prompt = f"""
You are a helpful customer support assistant.

Category: {category}

Customer Message:
\"\"\"{message}\"\"\"\"

Write a short (3–4 line), empathetic and professional support reply.
Avoid medical advice, deep troubleshooting, or long explanations.
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return "⚠ Could not generate AI reply."

# -------------------------------------------------
# UI Input
# -------------------------------------------------
st.subheader("✉ Enter Customer Ticket")
ticket_text = st.text_area("Customer Message:", height=160, placeholder="Type the customer complaint here...")

col1, col2 = st.columns([1.2, 0.8])

# -------------------------------------------------
# Classification + Reply
# -------------------------------------------------
with col1:
    if st.button("Classify & Generate Reply"):
        if not ticket_text.strip():
            st.warning("Please enter a message.")
        else:
            category = predict_category(ticket_text)
            st.success(f"Predicted Category: **{category}**")

            reply = generate_reply(ticket_text, category)
            st.subheader("🤖 Auto-Generated Support Reply")
            st.write(reply)

# -------------------------------------------------
# Save Ticket to CSV
# -------------------------------------------------
with col2:
    st.subheader("Save Ticket Result")

    if st.button("Save"):
        row = {
            "ticket": ticket_text,
            "predicted_category": category if "category" in locals() else ""
        }
        df = pd.DataFrame([row])
        out_path = "predictions_saved.csv"

        if os.path.exists(out_path):
            df.to_csv(out_path, mode="a", header=False, index=False)
        else:
            df.to_csv(out_path, index=False)

        st.success(f"Saved to {out_path}")

st.markdown("---")
st.caption("Powered by BiLSTM Model + Gemini AI • Smart Customer Support Automation ")
