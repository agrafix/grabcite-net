""" Ported from tflearn NLP CNN example
"""

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy

from load_data import load_data
from arch import cit_nocit_cnn

# configuration
max_words = 10000
max_sentence_len = 50

# Dataset loading

print("Loading dataset ... ")
train, test, _, _ = load_data(path='ref_bool.pkl', n_words=max_words,
                              valid_portion=0.1)

trainX, trainY = train
testX, testY = test

print("Train size: " + str(len(trainX)))
print("Test size: " + str(len(testX)))

print(testX[1:5])
print(testY[1:5])

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=max_sentence_len, value=0.)
testX = pad_sequences(testX, maxlen=max_sentence_len, value=0.)

print(testX[1:5])
print(testY[1:5])

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

print(testX[1:5])
print(testY[1:5])

# Train
model = cit_nocit_cnn(max_sentence_len, max_words)
model.fit(trainX, trainY, n_epoch = 10, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)
model.save("trained_model.tfl")
print("Wrote model to trained_model.tfl")