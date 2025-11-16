# fine_tune.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths
MODEL_PATH = "best_solar_model-17-valacc-0.7241.keras"   # your previously trained model
CLASS_INDICES_PATH = "class_indices.npy"
FINE_TUNED_MODEL_PATH = "fine_tuned_solar_model.keras"

# Parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-5   # small for fine-tuning

print("Loading base model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

# Load class indices
class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
num_classes = len(class_indices)

# Data generators (reuse preprocessing settings)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "Faulty_solar_panel",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    "Faulty_solar_panel",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Unfreeze some top layers for fine-tuning
base_model = model.layers[0]  # assuming first layer is base CNN
for layer in base_model.layers[-30:]:  # unfreeze last 30 layers
    layer.trainable = True

# Recompile with smaller LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
checkpoint = ModelCheckpoint(
    filepath="fine_tuned_best_model-{epoch:02d}-valacc-{val_accuracy:.4f}.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1,
    min_lr=1e-7
)

# Train again
print("Starting fine-tuning...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# Save final fine-tuned model
model.save(FINE_TUNED_MODEL_PATH)
print(f"Fine-tuned model saved as {FINE_TUNED_MODEL_PATH}")
