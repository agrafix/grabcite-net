""" Ported from tflearn NLP CNN example
"""

import tensorflow as tf
import keras
import keras.backend as K
import numpy
import random

import glove as g
import utils as u
from load_data import load_data, max_words, max_sentence_len
from arch import cit_nocit_rnn_pretrained

# Dataset loading

print("Loading dataset ... ")
train, test, _, word_dict, _ = load_data(path=u.getMostRecentOf("prepared-data/prepared", "pkl"), n_words=max_words,
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

# Data preprocessing
# Sequence padding
trainX = keras.preprocessing.sequence.pad_sequences(trainX, maxlen=max_sentence_len, value=0.)
testX = keras.preprocessing.sequence.pad_sequences(testX, maxlen=max_sentence_len, value=0.)

print(trainX[1:5])
print(trainY[1:5])

# Converting labels to binary vectors
trainY = keras.utils.to_categorical(trainY, num_classes=2)
testY = keras.utils.to_categorical(testY, num_classes=2)

print(trainX[1:5])
print(trainY[1:5])

# Load word vectors
embedding_vecs = g.loadEmbeddingVecs()
embedding_matrix = g.computeEmbeddingMatrix(embedding_vecs, word_dict)

# Train
model = cit_nocit_rnn_pretrained(max_sentence_len, max_words, word_dict, embedding_matrix)
model.fit(trainX, trainY, epochs=5, batch_size=64)

scores = model.evaluate(testX, testY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

if not os.path.exists("trained-models"):
    os.makedirs("trained-models")

model.save(u.makeTimedFilename('trained-models/trained', 'h5'))

K.clear_session()
