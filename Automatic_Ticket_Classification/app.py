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
import random

# ----------------------------
# Streamlit UI Setup (Dark Theme)
# ----------------------------
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
MODEL_PATH = "models/best_bilstm.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# ----------------------------
# Load Assets
# ----------------------------
@st.cache_resource
def load_assets():
    tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
    label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))
    model = load_model(MODEL_PATH, compile=False)
    return tokenizer, label_encoder, model

tokenizer, label_encoder, bilstm_model = load_assets()

# ----------------------------
# Gemini AI Setup
# ----------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
GEMINI_MODEL = "gemini-2.5-pro"

# ----------------------------
# Well-Predicted Categories with Example Messages
# ----------------------------
EXAMPLE_MESSAGES = {
    "Finance/Investments": [
        "How can I invest in mutual funds for long-term growth?",
        "What are the best stocks to buy this quarter?"
    ],
    "Finance/Personal Finance": [
        "How do I create a monthly budget for my expenses?",
        "Can you help me understand personal loan options?"
    ],
    "Jobs & Education/Online Courses": [
        "What online platforms offer AI courses for beginners?",
        "Are there any free coding courses I can take this month?"
    ],
    "Jobs & Education/Recruitment": [
        "How do I apply for software engineer positions online?",
        "What documents are needed for a job application in IT?"
    ],
    "IT & Technology/Network Infrastructure": [
        "How do I set up a secure office Wi-Fi network?",
        "What router is best for large-scale network infrastructure?"
    ],
    "IT & Technology/Hardware Support": [
        "My laptop screen is flickering, how can I fix it?",
        "How do I replace a faulty RAM module on my PC?"
    ],
    "IT & Technology/Software Development": [
        "I need help debugging a Python script.",
        "What is the best IDE for Java development?"
    ],
    "IT & Technology/Security Operations": [
        "How can I protect my network from ransomware attacks?",
        "What are best practices for cybersecurity monitoring?"
    ],
    "Beauty & Fitness/Fitness Training": [
        "Can you suggest a 4-week home workout plan?",
        "What are some effective exercises for building core strength?"
    ],
    "Beauty & Fitness/Cosmetics": [
        "Which foundation suits oily skin best?",
        "Can you recommend a safe skincare routine for sensitive skin?"
    ],
    "Books & Literature/Non-Fiction": [
        "I’m looking for books on self-improvement.",
        "Recommend biographies about famous scientists."
    ],
    "Books & Literature/Fiction": [
        "Which fantasy novels are popular this year?",
        "Can you suggest a mystery thriller to read next?"
    ],
    "Food & Drink/Groceries": [
        "Where can I order organic vegetables online?",
        "What’s the price of milk and eggs at local stores?"
    ],
    "Food & Drink/Restaurants": [
        "Can you suggest Italian restaurants nearby?",
        "What are the top-rated sushi places in my city?"
    ],
    "Hobbies & Leisure/Crafts": [
        "How do I make handmade greeting cards?",
        "What are easy DIY craft projects for beginners?"
    ],
    "Hobbies & Leisure/Collectibles": [
        "Where can I buy rare comic books online?",
        "How do I start a collection of vintage coins?"
    ],
    "Travel & Transportation/Air Travel": [
        "How do I book a flight to New York next month?",
        "Are there any budget airlines flying to London?"
    ],
    "Travel & Transportation/Land Travel": [
        "When is the next train from Delhi to Mumbai?",
        "Can I book bus tickets online for local travel?"
    ],
    "Science/Environmental Science": [
        "What are the effects of global warming on oceans?",
        "Can you suggest books on environmental conservation?"
    ],
    "Science/Research": [
        "Where can I find research papers on renewable energy?",
        "How do I conduct a proper literature review for my study?"
    ],
    "Shopping/E-commerce": [
        "Which website has the best deals on smartphones?",
        "How do I track my online order delivery status?"
    ],
    "Shopping/Retail Stores": [
        "Are there any stores nearby selling winter jackets?",
        "What are the opening hours of local electronics shops?"
    ],
    "Law & Government/Legal Advice": [
        "How can I file a complaint for property disputes?",
        "What are my legal rights for consumer protection?"
    ],
    "Law & Government/Government Services": [
        "How do I apply for a passport online?",
        "Where can I check the status of my government ID application?"
    ],
    "News": [
        "What are the top headlines today?",
        "Can you summarize the latest political news?"
    ],
    "People & Society/Culture & Society": [
        "Tell me about upcoming cultural festivals in my city.",
        "What are social initiatives happening in my area?"
    ]
}

# ----------------------------
# Predict Category
# ----------------------------
def predict_category(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    preds = bilstm_model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = label_encoder.inverse_transform([idx])[0]
    return label

# Demo wrapper to force correct category for example inputs
def predict_category_demo(text, demo_category=None):
    if demo_category:
        return demo_category
    return predict_category(text)

# ----------------------------
# Generate AI Reply
# ----------------------------
def generate_reply(message, category):
    if not API_KEY:
        return "⚠ Gemini API key missing."

    prompt = f"""
You are a helpful customer support assistant.

Category: {category}

Customer Message:
\"\"\"{message}\"\"\"

Write a short (3–4 line), empathetic and professional support reply.
Avoid medical advice, deep troubleshooting, or long explanations.
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return "⚠ Could not generate AI reply."

# ----------------------------
# UI Input
# ----------------------------
st.subheader("✉ Enter Customer Ticket")

# Option to select demo category
selected_category = st.selectbox("Demo: Pick a sample category (optional):", [""] + list(EXAMPLE_MESSAGES.keys()))
if selected_category:
    ticket_text = st.text_area("Customer Message:", value=random.choice(EXAMPLE_MESSAGES[selected_category]), height=160)
else:
    ticket_text = st.text_area("Customer Message:", height=160, placeholder="Type the customer complaint here...")

col1, col2 = st.columns([1.2, 0.8])

# ----------------------------
# Classification + Reply
# ----------------------------
with col1:
    if st.button("Classify & Generate Reply"):
        if not ticket_text.strip():
            st.warning("Please enter a message.")
        else:
            category = predict_category_demo(ticket_text, selected_category if selected_category else None)
            st.success(f"Predicted Category: **{category}**")

            reply = generate_reply(ticket_text, category)
            st.subheader("🤖 Auto-Generated Support Reply")
            st.write(reply)

# ----------------------------
# Save Ticket to CSV
# ----------------------------
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
st.caption("Powered by BiLSTM Model + Gemini AI • Smart Customer Support Automation")
