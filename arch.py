from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import GRU, LSTM
from keras.layers import Input, Dense
from keras.layers.merge import Concatenate
from keras import regularizers

def cit_nocit_rnn(max_sentence_len, max_words):
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