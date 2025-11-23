# hyperparameter_tuned_lstm.py

import numpy as np
import pickle
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from sklearn.utils.class_weight import compute_class_weight

# =========================== LOAD DATA ===========================
X_train = np.load("models/X_train.npy")
X_val = np.load("models/X_val.npy")
y_train = np.load("models/y_train.npy")
y_val = np.load("models/y_val.npy")

label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
num_classes = len(label_encoder.classes_)

# =========================== HYPERPARAMETERS ===========================
vocab_size = 20000
MAX_LEN = 300

EMBED_DIM = 256
LSTM_UNITS = 192
DENSE_UNITS = 160
LEARNING_RATE = 0.0007
BATCH_SIZE = 48
EPOCHS = 15

# =========================== CLASS WEIGHTS ===========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights)}

print("Class weights calculated:", class_weights)

# =========================== MODEL ===========================
model = Sequential([
    Embedding(input_dim=vocab_size,
              output_dim=EMBED_DIM,
              input_length=MAX_LEN),

    LSTM(LSTM_UNITS,
         return_sequences=False,
         dropout=0.3,
         recurrent_dropout=0.25),

    Dropout(0.35),

    Dense(DENSE_UNITS, activation='relu'),
    Dropout(0.25),

    Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

print(model.summary())

# =========================== CALLBACKS ===========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/best_lstm_tuned.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.4,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# =========================== TRAIN ============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

model.save("models/final_lstm_tuned.h5")

print("\n Fine-tuning Completed Successfully!")
print(" Saved best model: models/best_lstm_tuned.h5")
print(" Saved final model: models/final_lstm_tuned.h5")
