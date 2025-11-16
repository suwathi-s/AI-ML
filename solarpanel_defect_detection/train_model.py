# train_model.py
import os
import warnings
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---- Must set before tensorflow import ----
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Hide INFO/WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Disable oneDNN logs
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore

# ---------------- Config ----------------
DATA_DIR = r"C:\Users\91902\Desktop\solarpanel_defect_detection\Faulty_solar_panel"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

# Reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------- Load class indices ----------------
class_indices = np.load("class_indices.npy", allow_pickle=True).item()
num_classes = len(class_indices)
print("Loaded class mapping:", class_indices)

# ---------------- Data Generators ----------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

print(f"Training samples: {train_generator.samples}, Validation samples: {val_generator.samples}")

# ---------------- Model (Transfer Learning) ----------------
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- Callbacks ----------------
checkpoint_filepath = "best_solar_model-{epoch:02d}-valacc-{val_accuracy:.4f}.keras"
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-7
)

callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb]

# ---------------- Train ----------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ---------------- Save final model ----------------
final_model_name = "last_solar_model.keras"
model.save(final_model_name)
print(f"Final model saved as: {final_model_name}")

# ---------------- Report best validation accuracy ----------------
val_acc_history = history.history.get('val_accuracy', [])
if val_acc_history:
    best_val_acc = max(val_acc_history)
    best_epoch = val_acc_history.index(best_val_acc) + 1
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

ckpt_files = sorted(glob.glob("best_solar_model-*.keras"))
if ckpt_files:
    print("Saved checkpoint models:")
    for f in ckpt_files:
        print(" -", f)

# ---------------- Plot training curves ----------------
acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])

if acc and val_acc and loss and val_loss:
    epochs_range = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Training plot saved as training_results.png")
    plt.show()
