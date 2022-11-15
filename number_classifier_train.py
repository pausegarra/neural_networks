import tensorflow
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print("Downloading dataset...")
data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
print("Dataset downloaded")

datagen = ImageDataGenerator(
  rotation_range = 30,
  width_shift_range = 0.25,
  height_shift_range = 0.25,
  zoom_range=[0.5,1.5],
)
datagen.fit(train_images.reshape(60000,28,28,1))

# fig1, axes1 = plt.subplots(4, 8, figsize=(1.5*8,2*4))
# for i in range(32):
#      ax = axes1[i//8, i%8]
#      ax.imshow(train_images[i].reshape(28,28), cmap='gray_r')
#      ax.set_title('Label: {}'.format(train_labels[i]))
# plt.tight_layout()
# plt.show()

# fig2, axes2 = plt.subplots(4, 8, figsize=(1.5*8,2*4))
# for X, Y in datagen.flow(train_images.reshape(60000,28,28,1),train_labels,batch_size=32,shuffle=False):
#   for i in range(0, 32):
#     ax = axes2[i//8, i%8]
#     ax.imshow(X[i].reshape(28,28), cmap='gray_r')
#     ax.set_title('Label: {}'.format(int(Y[i])))
#   break
# plt.tight_layout()
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3,3,), input_shape=(28,28,1), activation="relu"))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3,), input_shape=(28,28,1), activation="relu"))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training model...")
data_to_fit = datagen.flow(train_images.reshape(60000,28,28,1),train_labels,batch_size=32)
model.fit(data_to_fit, epochs=60, verbose=1, batch_size=32, validation_data=(test_images, test_labels))
print("Training finished")

model.evaluate(test_images, test_labels)

print("Saving model...")
model.save('models/number_classifier.h5')
print("Model saved")