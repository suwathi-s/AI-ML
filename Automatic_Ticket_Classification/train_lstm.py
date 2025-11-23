# train_lstm.py
import numpy as np
import pickle
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.utils.class_weight import compute_class_weight

# =========================== LOAD DATA ===========================
X_train = np.load("models/X_train.npy")
X_val = np.load("models/X_val.npy")
y_train = np.load("models/y_train.npy")
y_val = np.load("models/y_val.npy")

label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
num_classes = len(label_encoder.classes_)

vocab_size = 20000
MAX_LEN = 300
EMBED_DIM = 200
LSTM_UNITS = 128

# =========================== CLASS WEIGHTS ===========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights)}

print("Class weights calculated")

# =========================== MODEL ===========================
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=MAX_LEN),
    LSTM(LSTM_UNITS, return_sequences=False),   # MANY-TO-ONE LSTM
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)

print(model.summary())

# =========================== CALLBACKS ===========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/best_lstm.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# =========================== TRAIN ===========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Save final model
model.save("models/final_lstm_model.h5")
print(" Training Completed! Model saved in models/ folder.")
