from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras import regularizers

def cit_nocit_rnn(max_sentence_len, max_words):
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_sentence_len))
    model.add(Conv1D(128, 3, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model