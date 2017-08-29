import os
import numpy as np
from keras.layers import Embedding

from config import glove_path, glove_dim

def loadEmbeddingVecs():
    embeddings_index = {}
    f = open(os.path.join(glove_path))

    print('Reading glove vectors from ' + glove_path + ' ... ')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Parsed ' + str(len(embeddings_index)) + ' vectors')

    return embeddings_index

def computeEmbeddingMatrix(embeddings_index, word_dict):
    embedding_matrix = np.zeros((len(word_dict) + 1, glove_dim))
    for word, i in word_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def getEmbeddingLayer(word_dict, embedding_matrix, max_seq_len):
    return Embedding(len(word_dict) + 1, glove_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False)

def getEmbeddingDim():
    return glove_dim