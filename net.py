""" Ported from tflearn NLP CNN example
"""

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
import numpy

from load_data import load_data

# configuration
max_words = 10000
max_sentence_len = 50

# IMDB Dataset loading
train, test, check, word_dict_rev = load_data(path='ref_bool.pkl', n_words=max_words,
                                              valid_portion=0.1)

trainX, trainY = train
testX, testY = test
checkX, checkY = check

print("Train size: " + str(len(trainX)))
print("Test size: " + str(len(testX)))
print("Check size: " + str(len(checkX)))

print(testX[1:5])
print(testY[1:5])

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=max_sentence_len, value=0.)
testX = pad_sequences(testX, maxlen=max_sentence_len, value=0.)
checkX = pad_sequences(checkX, maxlen=max_sentence_len, value=0.)

print(testX[1:5])
print(testY[1:5])

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
checkY = to_categorical(checkY, nb_classes=2)

print(testX[1:5])
print(testY[1:5])

# Building convolutional network
network = input_data(shape=[None, max_sentence_len], name='input')
network = tflearn.embedding(network, input_dim=max_words, output_dim=128)
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch = 10, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)
model.save("trained_model.tfl")
print("Wrote model to trained_model.tfl")

print(checkX[1:5])
results = model.predict(checkX)
for idx, val in enumerate(results):
    sentence_idx = checkX[idx]
    sentence = [word_dict_rev[i] if i in word_dict_rev else '<unk>' for i in sentence_idx]

    if val[0] < val[1]:
        print(sentence)
        print("Citation: " + str(val[1]))
