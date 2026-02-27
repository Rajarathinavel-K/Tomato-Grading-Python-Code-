import tensorflow as tf
import keras
from keras.applications import MobileNetV2
from keras.layers import Input, RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast, GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3 
EPOCHS = 15
FINE_TUNE_EPOCHS = 5

# Define Paths to Your Dataset
TRAIN_DIR = 'dataset/train/'
VAL_DIR = 'dataset/val/'
SAVE_PATH = 'C:/Trained_Models/tomato_grader_model.keras'

# Load Datasets
train_dataset = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels='inferred', label_mode='categorical',
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
)
val_dataset = keras.utils.image_dataset_from_directory(
    VAL_DIR, labels='inferred', label_mode='categorical',
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
)

print("Class names found by Keras (ensure these are alphabetical):", train_dataset.class_names)

# Create Enhanced Data Augmentation Layers
data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomBrightness(factor=0.2),
    RandomContrast(factor=0.2),
])

# Build the Transfer Learning Model
inputs = Input(shape=IMAGE_SIZE + (3,))
x = keras.layers.Rescaling(1./255)(inputs)
x = data_augmentation(x)

base_model = MobileNetV2(include_top=False, weights='imagenet')
base_model.trainable = False # Freeze initially

x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

# Compile and Train (Phase 1)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(SAVE_PATH, monitor='val_accuracy', save_best_only=True)

print("Starting initial model training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping, model_checkpoint]
)

# Fine-Tuning (Phase 2)
print("\nStarting fine-tuning phase...")
base_model.trainable = True

# Freeze the bottom 100 layers, unfreeze the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with a much lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_dataset,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping, model_checkpoint] 
)

print(f"\nTraining complete! Best model saved to: {SAVE_PATH}")