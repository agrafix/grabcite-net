from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import GRU, LSTM
from keras.layers import merge, Input, Dense, Lambda
from keras.layers.merge import Concatenate, concatenate
from keras import regularizers
import keras.backend as K
import glove as g

def cit_nocit_rnn_pretrained(max_sentence_len, max_words, word_dict, embedding_matrix):
    inp = Input(shape=(max_sentence_len,))
    emb = g.getEmbeddingLayer(word_dict, embedding_matrix, max_sentence_len)(inp)

    cnns = [Conv1D(g.getEmbeddingDim(), filter_length, activation='tanh', padding='same') for filter_length in [1, 2, 3, 5]]
    allCnns = concatenate([cnn(emb) for cnn in cnns])

    pooled = MaxPooling1D(pool_size=2)(allCnns)

    rnn = GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(pooled)
    rnn2 = GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(rnn)
    rnn3 = GRU(32, dropout=0.2, recurrent_dropout=0.2)(rnn2)

    dense = Dense(2, activation='softmax')(rnn3)

    model = Model(inputs=inp, outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def cit_nocit_rnn(max_sentence_len, max_words):
    inp = Input(shape=(max_sentence_len,))
    emb = Embedding(max_words, 128, input_length=max_sentence_len)(inp)

    cnns = [Conv1D(128, filter_length, activation='tanh', padding='same') for filter_length in [1, 2, 3, 5]]
    allCnns = concatenate([cnn(emb) for cnn in cnns])

    pooled = MaxPooling1D(pool_size=2)(allCnns)

    rnn = GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(pooled)
    rnn2 = GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(rnn)
    rnn3 = GRU(32, dropout=0.2, recurrent_dropout=0.2)(rnn2)

    dense = Dense(2, activation='softmax')(rnn3)

    model = Model(inputs=inp, outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def cit_nocit_rnn_rnn_cnn(max_sentence_len, max_words):
    inp = Input(shape=(max_sentence_len,))
    emb = Embedding(max_words, 128, input_length=max_sentence_len)(inp)

    fwd_rnn = LSTM(128, return_sequences=True)(emb)
    rev_rnn = LSTM(128, return_sequences=True, go_backwards=True)(emb)

    merged = concatenate([fwd_rnn, rev_rnn], axis=-1)

    cnns = [Conv1D(500, filter_length, activation='tanh', padding='same') for filter_length in [1, 2, 3, 5]]
    allCnns = concatenate([cnn(merged) for cnn in cnns])

    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    maxpool.supports_masking = True

    pooled = maxpool(allCnns)
    dense = Dense(2, activation='sigmoid')(pooled)

    model = Model(inputs=inp, outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def cit_nocit_rnn_try(max_sentence_len, max_words):
    inp = Input(shape=(max_sentence_len,))
    emb = Embedding(max_words, 128, input_length=max_sentence_len)(inp)

    branch1 = Conv1D(128, 3, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.01))(emb)
    branch2 = Conv1D(128, 4, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.01))(emb)
    branch3 = Conv1D(128, 5, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.01))(emb)

    comb = Concatenate()([branch1, ZeroPadding1D((0, 1))(branch2), ZeroPadding1D((0, 2))(branch3)])

    pooled = MaxPooling1D(pool_size=2)(comb)
    rnn = GRU(128, dropout=0.2, recurrent_dropout=0.2)(pooled)

    dense = Dense(2, activation='sigmoid')(rnn)

    model = Model(inputs=inp, outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def cit_nocit_rnn_simple(max_sentence_len, max_words):
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_sentence_len))
    model.add(Conv1D(128, 3, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model