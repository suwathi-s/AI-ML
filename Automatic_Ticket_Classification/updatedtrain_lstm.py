# train_lstm.py
import numpy as np
import pickle
from collections import Counter
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import tensorflow as tf

# =========================== LOAD DATA ==========================
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

# =========================== CHECK ORIGINAL IMBALANCE ==========================
print("\nOriginal Train Distribution:", Counter(y_train))

# =========================== UPSAMPLING MINORITY CLASSES ==========================
X_train_balanced = []
y_train_balanced = []

for cls in np.unique(y_train):
    X_cls = X_train[y_train == cls]
    y_cls = y_train[y_train == cls]

    if len(X_cls) < 800:  # threshold - adjust based on your data
        X_up, y_up = resample(X_cls, y_cls, replace=True, n_samples=800, random_state=42)
        X_train_balanced.append(X_up)
        y_train_balanced.append(y_up)
    else:
        X_train_balanced.append(X_cls)
        y_train_balanced.append(y_cls)

X_train = np.vstack(X_train_balanced)
y_train = np.hstack(y_train_balanced)

print("\nAfter Upsampling:", Counter(y_train))

# =========================== CLASS WEIGHTS ==========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights)}

print("\nClass weights:", class_weights)

# =========================== FOCAL LOSS ==========================
def focal_loss(gamma=2., alpha=0.25):

    def loss(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow((1 - y_pred), gamma)
        fl = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))

    return loss

# =========================== MODEL ==========================
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=MAX_LEN),
    LSTM(LSTM_UNITS, return_sequences=False),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss=focal_loss(gamma=2, alpha=0.25),
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)

print(model.summary())

# =========================== CALLBACKS ==========================
early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

checkpoint = ModelCheckpoint(
    "models/bestupdated_lstm.h5",
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

# =========================== TRAIN ==========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

model.save("models/finalupdated_lstm_model.h5")
print("Training Completed! Model saved in models/ folder.")
