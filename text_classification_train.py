import tensorflow as tf
from tensorflow import keras
import numpy as np

def decode_review(text):
  return " ".join([reverse_word_index.get(i, '?') for i in text])

print("Downloading dataset...")
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=80000)
print("Dataset downloaded")

words_index = keras.datasets.imdb.get_word_index()
words_index = {k:(v+3) for k, v in words_index.items()}
words_index["<PAD>"] = 0
words_index["<START>"] = 1
words_index["<UNK>"] = 2
words_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in words_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=words_index["<PAD>"], padding="post", maxlen=512)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=words_index["<PAD>"], padding="post", maxlen=512)

model = keras.Sequential()
model.add(keras.layers.Embedding(80000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

print("Training model...")
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=0)
print("Training finished")

print("Saving model...")
model.save('models/text_classification.h5')
print("Model saved")