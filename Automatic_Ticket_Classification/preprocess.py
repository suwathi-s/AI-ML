# preprocess.py
import os
import pandas as pd
import numpy as np
import re
import pickle
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from sklearn.preprocessing import LabelEncoder

# ======================== CONFIG ========================
MAX_WORDS = 20000
MAX_LEN = 300  # increased to capture more context
os.makedirs("models", exist_ok=True)

# Load Spacy for lemmatization
nlp = spacy.load("en_core_web_sm")

# ======================== TEXT CLEANING ========================
def clean_text(text):
    """
    Lowercase, remove newlines, non-alphanumerics, lemmatize, remove stopwords.
    """
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

# ======================== PREPROCESS FUNCTION ========================
def preprocess_csv(train_path, val_path, test_path):
    """
    Preprocess CSV files: clean text, tokenize, pad sequences, encode labels,
    and save artifacts for model training.
    """
    # Load CSVs
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    # Clean ticket body text
    for df in [train, val, test]:
        df['clean'] = df['body'].apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train['clean'])

    # Convert texts to padded sequences
    X_train = pad_sequences(tokenizer.texts_to_sequences(train['clean']), maxlen=MAX_LEN)
    X_val = pad_sequences(tokenizer.texts_to_sequences(val['clean']), maxlen=MAX_LEN)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test['clean']), maxlen=MAX_LEN)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train['queue'])
    y_val = label_encoder.transform(val['queue'])
    y_test = label_encoder.transform(test['queue'])

    # Save artifacts
    pickle.dump(tokenizer, open('models/tokenizer.pkl', 'wb'))
    pickle.dump(label_encoder, open('models/label_encoder.pkl', 'wb'))
    np.save('models/X_train.npy', X_train)
    np.save('models/X_val.npy', X_val)
    np.save('models/X_test.npy', X_test)
    np.save('models/y_train.npy', y_train)
    np.save('models/y_val.npy', y_val)
    np.save('models/y_test.npy', y_test)

    print("Preprocessing completed and all artifacts saved in 'models/' folder.")

# ======================== RUN PREPROCESS ========================
if __name__ == "__main__":
    preprocess_csv(
        train_path="tickets_train.csv",
        val_path="tickets_validation.csv",
        test_path="tickets_test.csv"
    )
