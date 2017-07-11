""" Ported from tflearn NLP CNN example
"""

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences, shuffle
import numpy
import random

from load_data import load_data, max_words, max_sentence_len
from arch import cit_nocit_rnn

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

# Evaluate data balance:
cT = 0
cF = 0
cTotal = len(trainY)

true_class = []
false_class = []

for idx, y in enumerate(trainY):
    if y == 1:
        true_class.append((trainX[idx], trainY[idx]))
        cT += 1
    else:
        false_class.append((trainX[idx], trainY[idx]))
        cF += 1

print("Train distribution: true=" + str(cT) + " (" + str(cT / cTotal) + ") / false=" + str(cF) + " (" + str(cF / cTotal) + ")")

trainX = []
trainY = []

random.shuffle(false_class)

if (cT / cTotal) < 0.4:
    print("Unbalanced data detected, adjust by over and undersampling")
    difference = cF - cT
    print("Difference is: " + str(difference))
    for tx, ty in true_class:
        trainX.append(tx)
        trainY.append(ty)

    for i in range(len(true_class), len(true_class) + difference):
        x, y = random.choice(true_class)
        trainX.append(x)
        trainY.append(y)

    for i in range(0, len(false_class) - 1):
        x, y = false_class[i]
        trainX.append(x)
        trainY.append(y)

    print("Completed balancing, train size now is: " + str(len(trainX)))

print("Shuffling data")
trainX, trainY = shuffle(trainX, trainY)

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=max_sentence_len, value=0.)
testX = pad_sequences(testX, maxlen=max_sentence_len, value=0.)

print(trainX[1:5])
print(trainY[1:5])

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

print(trainX[1:5])
print(trainY[1:5])

# Train
model = cit_nocit_rnn(max_sentence_len, max_words)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=64, n_epoch=2)
model.save("trained_model.tfl")
print("Wrote model to trained_model.tfl")