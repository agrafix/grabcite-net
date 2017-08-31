import numpy
import random
import keras
from keras.models import load_model
import keras.backend as K

import utils as u
from load_data import load_data, max_words, max_sentence_len

def ts_chunks(l, l2, n):
    for i in range(0, len(l), n):
        yield (l[i:i + n], l2[i:i + n])

# Dataset loading
print("Loading dataset ...")
_, _, check, _, word_dict_rev = load_data(path=u.getMostRecentOf("prepared-data/prepared", "pkl"), n_words=max_words,
                                              valid_portion=0.1)

checkX, checkY = check

print("Check size: " + str(len(checkX)))

checkX = keras.preprocessing.sequence.pad_sequences(checkX, maxlen=max_sentence_len, value=0.)

# Predict
print("Running predictions ...")
model = load_model(u.getMostRecentOf("trained-models/trained", "h5"))

total = 0

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

print("Processing data in chunks of 1000")

false_pos_list = []

for xchunk, ychunk in ts_chunks(checkX, checkY, 1000):
    results = model.predict(xchunk)

    for idx, val in enumerate(results):
        sentence_idx = xchunk[idx]
        sentence = [word_dict_rev[i] if i in word_dict_rev else '~' for i in sentence_idx]

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
                false_pos_list.append(sentence)
        else: # predicted == 0
            if actual == 0:
                true_neg += 1
            else:
                false_neg += 1

K.clear_session()

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

random.shuffle(false_pos_list)
short_list = false_pos_list[1:50]
for el in short_list:
    print(str(" ".join(el)).rstrip("~ ").lstrip("~ "))
