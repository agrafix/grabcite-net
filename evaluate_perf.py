import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy

from load_data import load_data
from arch import cit_nocit_cnn

# configuration, keep in sync with trained model
max_words = 10000
max_sentence_len = 50

def ts_chunks(l, l2, n):
    for i in range(0, len(l), n):
        yield (l[i:i + n], l2[i:i + n])

# Dataset loading
print("Loading dataset ...")
_, _, check, word_dict_rev = load_data(path='ref_bool.pkl', n_words=max_words,
                                              valid_portion=0.1)

checkX, checkY = check

print("Check size: " + str(len(checkX)))

checkX = pad_sequences(checkX, maxlen=max_sentence_len, value=0.)

# Predict
print("Running predictions ...")
model = cit_nocit_cnn(max_sentence_len, max_words)
model.load("trained_model.tfl")
print("Loaded model from trained_model.tfl")

total = 0

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

print("Processing data in chunks of 1000")

for xchunk, ychunk in ts_chunks(checkX, checkY, 1000):
    results = model.predict(xchunk)

    for idx, val in enumerate(results):
        sentence_idx = xchunk[idx]
        sentence = [word_dict_rev[i] if i in word_dict_rev else '<unk>' for i in sentence_idx]

        predicted = 0
        if val[0] < val[1]:
            predicted = 1

        actual = ychunk[idx]

        total += 1
        if predicted == 1:
            if actual == 1:
                true_pos += 1
            else:
                false_pos += 1
        else: # predicted == 0
            if actual == 0:
                true_neg += 1
            else:
                false_neg += 1

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)

f1 = 2 * ((precision * recall) / (precision + recall))

print("Total: " + str(total))

print("TP: " + str(true_pos))
print("FP: " + str(false_pos))
print("TN: " + str(true_neg))
print("FN: " + str(false_neg))

print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))
