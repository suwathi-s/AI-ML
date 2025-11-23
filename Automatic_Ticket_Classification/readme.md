#  Automatic Ticket Classification using Many-to-One RNN & Customer Response Generation using LLM

This project implements a complete **end-to-end NLP system** for automating customer support ticket handling. It includes:

- **Ticket classification using a Many-to-One LSTM / BiLSTM**
- **Automatic customer response generation using Google Gemini LLM**
- **Streamlit Web App for user interaction**
- **Dataset preprocessing, hyperparameter tuning, model evaluation, and predictions**

---

##  Project Architecture

Dataset → Preprocessing → LSTM/BiLSTM Training → Evaluation → Streamlit UI
↓
LLM-based Response Generator (Gemini)


---

## 📂 Folder Structure

AUTOMATIC TICKET CLASSIFICATION/
│
├── models/ # Saved LSTM, BiLSTM, tuned models
│
├── ticket/ # Intermediate files / logs
│
├── app.py # Streamlit frontend
├── preprocess.py # Cleaning + tokenization + padding
├── dataset_loader.py # Load datasets
│
├── train_lstm.py # Base LSTM training
├── train_bilstm.py # BiLSTM training
├── updatedtrain_lstm.py # Improved model training
│
├── evaluate.py # Evaluate trained model
├── evaluate_bilstm.py # Evaluate BiLSTM
├── evaluate_tune.py # Evaluate tuned LSTM
├── hyperparameter_tuned_lstm.py # Random search / tuning script
│
├── predict.py # Ticket prediction script
├── predict_and_reply.py # Prediction + LLM-based response
│
├── predictions_saved.csv # Saved predictions
│
├── tickets_train.csv
├── tickets_test.csv
├── tickets_validation.csv
│
├── requirements.txt # Libraries used
├── environment.txt # Virtual env details
│
└── README.md


##  **Project Objectives**

### ✅ 1. Build a Many-to-One RNN/LSTM model for ticket classification  
### ✅ 2. Build an LLM-powered customer response generator  
### ✅ 3. Create a user-friendly Streamlit UI  
### ✅ 4. Evaluate and tune the model  
### ✅ 5. Provide automated predictions + AI-generated replies  

---

##  **Dataset**

The dataset contains customer support tickets with fields:

- `ticket_text` – the issue raised by the user  
- `category` – the label (billing, technical, account, refund, etc.)  

The dataset is split into:

- `tickets_train.csv`
- `tickets_test.csv`
- `tickets_validation.csv`

---

##  **Preprocessing Pipeline**

Implemented in `preprocess.py`:

✔ Lowercasing  
✔ Special character removal  
✔ Tokenization  
✔ Padding sequences  
✔ Label encoding  
✔ Train/Val/Test split  

---

##  **Model Architecture**

###  LSTM (Many-to-One)

- Embedding Layer  
- LSTM Layer  
- Dense Classification Layer  

###  BiLSTM

- Bidirectional LSTM  
- Better contextual learning  

Both models were trained and saved inside `/models`.

---

##  **Model Evaluation**

Implemented in:

- `evaluate.py`
- `evaluate_bilstm.py`
- `evaluate_tune.py`

Metrics include:

- Accuracy  
- Loss  
- Classification Report  
- Confusion Matrix  


---

##  **Hyperparameter Tuning**

`hyperparameter_tuned_lstm.py` performs tuning of:

- LSTM units  
- Dropout  
- Batch size  
- Learning rate  
- Optimizer  

---

## 🤖 **LLM-Based Response Generation**

Powered by **Google Gemini** through:

google.generativeai as genai

Used in:

- `predict_and_reply.py`
- `app.py`

The LLM generates:

- Human-like replies  
- Context-aware responses  
- Professional tone messages  

---

##  **Streamlit Application**

`app.py` includes:

### ✔ Dark theme  
### ✔ CSS styled UI  
### ✔ Text input for ticket  
### ✔ Predict button  
### ✔ Category output  
### ✔ AI-generated customer reply  
### ✔ Model loading + error handling  

---

##  **How to Run the Project**

### **1. Create Virtual Environment**

python -m venv ticket_env
source ticket_env/bin/activate (Linux/Mac)
ticket_env\Scripts\activate (Windows)


### **2. Install Requirements**

pip install -r requirements.txt


### **3. Run Streamlit App**

streamlit run app.py


##  **Requirements**

Main libraries include:

streamlit
numpy
pandas
tensorflow
scikit-learn
google-generativeai
Pillow


Full list is in `requirements.txt`.

---

##  **Prediction Script (CLI)**


python predict_and_reply.py


##  **Final Output Example**

User Ticket:
"My payment failed but money was deducted."

Predicted Category:
Billing Issue

AI Response:
"Sorry for the inconvenience. Your payment is under process.
Our team is verifying the transaction and will update you shortly."

---

##  **Conclusion**

This project successfully integrates:

✔ Deep Learning (LSTM / BiLSTM)  
✔ Natural Language Processing (NLP)  
✔ Large Language Models (LLM)  
✔ Streamlit UI  

to build a fully automated customer support system.

---

#  **Author**

Developed by: **suwathi s**  
  


