from google.colab import drive
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

# Mount Google Drive
drive.mount('/content/drive')

# Unzip dataset if required
!unzip -o /content/drive/MyDrive/grade4.zip -d /content/grade4

# Path to training and validation datasets
train_dir = '/content/grade4/train'  # Adjust these paths based on your dataset structure
val_dir = '/content/grade4/val'

# Preprocessing and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load the VGG16 model and custom layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Load pre-trained weights
weights_path = '/content/drive/MyDrive/vgg16_weights_tf_dim_ordering_tf_kernels_notop (1).h5'
if os.path.exists(weights_path):
    model.layers[0].load_weights(weights_path)
    print("Pre-trained weights loaded.")
else:
    print(f"Weights file not found at {weights_path}.")

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

# Train the model
epochs = 10  # Adjust based on your dataset and hardware
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Print accuracy
train_acc = history.history['accuracy'][-1] * 100
val_acc = history.history['val_accuracy'][-1] * 100
print(f"Final Training Accuracy: {train_acc:.2f}%")
print(f"Final Validation Accuracy: {val_acc:.2f}%")

# Save the model
model_save_path = '/content/drive/MyDrive/trained_model(1).h5'
model.save(model_save_path)
print(f"Model saved as {model_save_path}.")
