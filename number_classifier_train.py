import tensorflow
from tensorflow import keras
import numpy as np

print("Downloading dataset...")
data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
print("Dataset downloaded")

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training model...")
model.fit(train_images, train_labels, epochs=10, verbose=0)
print("Training finished")

print("Saving model...")
model.save('models/number_classifier.h5')
print("Model saved")