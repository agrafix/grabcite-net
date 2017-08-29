#
# Tool to generate data in the parallel text format useful for seq2seq learning
#
import glob
import os
import re
import nltk
import six.moves.cPickle as pickle
import string
import hashlib

from config import data_glob, refRegex, alphaChars

citMap = {}

def tokenize_data(sentence):
    rawToks = nltk.word_tokenize(sentence)
    toks = [w.lower() for w in rawToks if any(c in alphaChars for c in w)]
    good = len(toks) > 6
    return good, " ".join(toks)

def prepare_sentence(sentence_orig):
    refs = refRegex.findall(sentence_orig)
    refMap = {}
    sentence = sentence_orig
    for (ty, url) in refs:
        refHash = hashlib.sha224((ty + "/" + url).encode('utf-8')).hexdigest()
        refMap[refHash] = (ty, url)
        citKey = "<" + ty + ":" + url + ">"
        if citKey not in citMap.keys():
            citMap[citKey] = 0
        citMap[citKey] += 1
        sentence = sentence.replace(citKey, refHash)
    ok, withCit = tokenize_data(sentence)
    for refHash, (ty, url) in refMap.items():
        withCit = withCit.replace(refHash, "<" + ty + ":" + url + ">")

    sentence = refRegex.sub("", sentence_orig)
    ok2, noCit = tokenize_data(sentence)
    return ok and ok2, withCit, noCit

def build_dataset(targetFile):
    idx = 0
    print("Let's go")
    with open(targetFile + ".cit.txt", 'w') as citFile:
        with open(targetFile + ".plain.txt", 'w') as plainFile:
            with open(targetFile + ".cit.test.txt", 'w') as citFileTest:
                with open(targetFile + ".plain.test.txt", 'w') as plainFileTest:
                    with open(targetFile + ".cit.valid.txt", 'w') as citFileValid:
                        with open(targetFile + ".plain.valid.txt", 'w') as plainFileValid:
                            for file in glob.glob(data_glob):
                                with open(file, 'r') as myfile:
                                    data = myfile.read().split("\n============\n")
                                    for sentence in data:
                                        ok, withCit, noCit = prepare_sentence(sentence)
                                        if ok:
                                            if idx % 10 == 0:
                                                plainFileTest.write(noCit + "\n")
                                                citFileTest.write(withCit + "\n")
                                            elif idx % 10 == 1:
                                                plainFileValid.write(noCit + "\n")
                                                citFileValid.write(withCit + "\n")
                                            else:
                                                plainFile.write(noCit + "\n")
                                                citFile.write(withCit + "\n")
                                            idx += 1

    with open(targetFile + "_allCits.cit.txt", 'w') as citFile:
        for k, ct in citMap.items():
            citFile.write(k + "\t" + str(ct) + "\n")
    print("All done!")

if __name__ == "__main__":
    build_dataset("datasets/tiny")