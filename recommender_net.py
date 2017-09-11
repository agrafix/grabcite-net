import numpy as np
import six.moves.cPickle as pickle
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, Merge, TimeDistributed
from keras.models import Model
import keras.backend as K
import keras
from keras.callbacks import ModelCheckpoint
import os

import word_mapper
import recommender_prepare as rp
import utils as u

maxTitleLen = 50
maxContextLen = rp.around * 2

def prepSet(x):
    return {
        'T': keras.preprocessing.sequence.pad_sequences(x['T'], maxlen=maxTitleLen, value=0.),
        'C': keras.preprocessing.sequence.pad_sequences(x['C'], maxlen=maxContextLen, value=0.),
        'R': keras.preprocessing.sequence.pad_sequences(x['R'], maxlen=maxTitleLen, value=0.),
        'L': keras.utils.to_categorical(x['L'], num_classes=2)
    }

if __name__ == "__main__":
    path = u.getMostRecentOf("prepared-data/recommender-v1", "pkl")
    print("Loading dataset " + str(path) + "... ")

    with open(path, 'rb') as f:
        train = pickle.load(f)
        validate = pickle.load(f)
        test = pickle.load(f)
        ref_dict = pickle.load(f)
        word_mapper = pickle.load(f)

    train = prepSet(train)
    validate = prepSet(validate)
    test = prepSet(test)

    print("Train size: " + str(len(train['T'])))
    print("Test size: " + str(len(test['T'])))
    print("Validation size: " + str(len(validate['T'])))

    print(train['T'][1:5])
    print(test['T'][1:5])
    print(validate['T'][1:5])

    wordCount = word_mapper.catSize()
    print("Got " + str(wordCount) + " words")

    print("Building the net")

    titleEmbLayer = Embedding(wordCount, 64, input_length=maxTitleLen, mask_zero=True)

    def LstmLayer(hidden, i):
        return LSTM(hidden, dropout=0.2, recurrent_dropout=0.2)(i)


    title_em = Input(shape=(maxTitleLen,), name='title_input')
    x = titleEmbLayer(title_em)
    title_out = LstmLayer(64, x) # LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)

    refTitle_em = Input(shape=(maxTitleLen,), name='ref_title_input')
    x = titleEmbLayer(refTitle_em)
    refTitle_out = LstmLayer(64, x) # LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)

    context_em = Input(shape=(maxContextLen,), name='context_input')
    x = Embedding(wordCount, 128, input_length=maxContextLen, mask_zero=True)(context_em)
    context_out = LstmLayer(128, x) # LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)

    x = keras.layers.concatenate([title_out, refTitle_out, context_out])
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=[title_em, refTitle_em, context_em], outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Let's train this...!")
    cptLoc = u.makeTimedFilename('trained-models/recommender-cpt-v1', 'h5')

    print("Checkpoints will go to " + cptLoc)
    checkpointer = ModelCheckpoint(filepath=cptLoc, verbose=1, save_best_only=True)

    model.fit([train['T'], train['R'], train['C']], train['L'], epochs=10, batch_size=32,
              validation_data=([validate['T'], validate['R'], validate['C']], validate['L']), callbacks=[checkpointer])

    print("Evaluating ...")
    scores = model.evaluate([validate['T'], validate['R'], validate['C']], validate['L'], verbose=0)
    print("Validation Accuracy: %.2f%%" % (scores[1] * 100))

    scores = model.evaluate([test['T'], test['R'], test['C']], test['L'], verbose=0)
    print("Test Accuracy: %.2f%%" % (scores[1] * 100))

    if not os.path.exists("trained-models"):
        os.makedirs("trained-models")

    print("Writing everything to disk")
    model.save(u.makeTimedFilename('trained-models/recommender-v1', 'h5'))

    print("K bye!")
    K.clear_session()