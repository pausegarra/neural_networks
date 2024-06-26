import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Downloading dataset...")
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
print("Dataset downloaded")

class_names = [
  "T-Shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot"
]

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28,28)),
  keras.layers.Dense(128, activation="relu"),
  keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training model...")
model.fit(train_images, train_labels, epochs=5, verbose=0)
print("Training finished")

print("Saving model...")
model.save('models/image_classifier.h5')
print("Model saved")