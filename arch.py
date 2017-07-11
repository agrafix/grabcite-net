import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool, max_pool_1d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.layers.recurrent import lstm

def cit_nocit_cnn(max_sentence_len, max_words):

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

    return model

def cit_nocit_rnn(max_sentence_len, max_words):

    # Building convolutional network
    network = input_data(shape=[None, max_sentence_len], name='input')
    network = tflearn.embedding(network, input_dim=max_words, output_dim=128)
    network = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    network = max_pool_1d(network, kernel_size=2)
    network = lstm(network, 128, dropout=0.2)
    network = fully_connected(network, 2, activation='sigmoid')
    network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)

    return model