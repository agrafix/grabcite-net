import six.moves.cPickle as pickle
import keras
import os
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, RepeatVector, TimeDistributed, Flatten

import utils as u

max_sentence_len = 100
hidden_size = 128
num_layers = 5

if __name__ == "__main__":
    path = u.getMostRecentOf("prepared-data/seqs", "pkl")
    print("Loading dataset " + str(path) + "... ")

    with open(path, 'rb') as f:
        train = pickle.load(f)
        validate = pickle.load(f)
        test = pickle.load(f)
        tokMapper = pickle.load(f)

    trainX, trainY = train
    testX, testY = validate

    print("Train size: " + str(len(trainX)))
    print("Test size: " + str(len(testX)))

    print(testX[1:5])
    print(testY[1:5])

    print("Padding dataset ...")
    def pad(s):
        return keras.preprocessing.sequence.pad_sequences(s, maxlen=max_sentence_len, value=0.)

    trainX = pad(trainX)
    trainY = pad(trainY)

    testX = pad(testX)
    testY = pad(testY)

    model = Sequential()

    # Creating encoder network
    model.add(Embedding(30000 + 1, 1000, input_length=max_sentence_len, mask_zero=True))
    model.add(LSTM(hidden_size))
    model.add(RepeatVector(max_sentence_len))

    # Creating decoder network
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(max_sentence_len))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(trainX, trainY, epochs=5, batch_size=64)

    scores = model.evaluate(testX, testY, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    if not os.path.exists("trained-models"):
        os.makedirs("trained-models")

    model.save(u.makeTimedFilename('trained-models/seq_trained', 'h5'))

    K.clear_session()