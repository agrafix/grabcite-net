""" Ported from tflearn imdb dataset loader

Credits: LISA-lab, https://github.com/lisa-lab/
"""
from __future__ import print_function
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import tensorflow as tf

# configuration
max_words = 20000
max_sentence_len = 50

def remove_fully_unk(tset):
    x, y = tset
    xF = []
    yF = []

    for idx, sent in enumerate(x):
        if not all(word == 1 for word in sent) and len(sent) >= 5:
            notUnk = sum(word != 1 for word in sent)
            unk = sent.count(1)

            if unk <= notUnk:
                xF.append(sent)
                yF.append(y[idx])

    return (xF, yF)

def load_word_dicts(path="ref_bool.pkl"):
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    pickle.load(f)
    pickle.load(f)
    word_dict_rev = pickle.load(f)
    word_dict = pickle.load(f)

    f.close()
    return (word_dict, word_dict_rev)

def load_data(path="ref_bool.pkl", n_words=100000, valid_portion=0.1,
              maxlen=None,
              sort_by_len=True):
    '''Loads the dataset
    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    '''

    #############
    # LOAD DATA #
    #############
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = pickle.load(f)
    test_set = pickle.load(f)
    word_dict_rev = pickle.load(f)
    word_dict = pickle.load(f)

    f.close()
    if maxlen:
        print("Reducing total dataset to " + str(maxlen))

        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    print("Splitting training set into validation set")

    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    print("Removing unknown words")
    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    print("Cleanup noisy sentences")
    train = remove_fully_unk((train_set_x, train_set_y))
    valid = remove_fully_unk((valid_set_x, valid_set_y))
    test = remove_fully_unk((test_set_x, test_set_y))

    return train, valid, test, word_dict, word_dict_rev