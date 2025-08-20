# lung_cancer_inceptionresnet.py
"""
Lung Cancer Detection using Transfer Learning (InceptionResNetV2)
Dataset: CT-scan images (train/test folders)
Author: Shubham
"""

import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.3, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory("lung_cancer/train", target_size=(224, 224), batch_size=32, class_mode="binary")
test_set = test_datagen.flow_from_directory("lung_cancer/test", target_size=(224, 224), batch_size=32, class_mode="binary")

# Base model
base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Custom model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_set, epochs=10, validation_data=test_set)

# Accuracy plot
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("InceptionResNetV2 Lung Cancer Accuracy")
plt.show()

# Save
model.save("lung_cancer_inceptionresnet.h5")
