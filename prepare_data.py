import glob
import os
import re
import nltk
import six.moves.cPickle as pickle
import string

data_glob = "data/*.txt"

refRegex = re.compile(r"<(DBLP|ARXIV|DOI|GC):([^>]*)>")

alphaChars = set(string.ascii_letters)

def prepare_sentence(sentence):
    refs = refRegex.findall(sentence)
    sentence = refRegex.sub("", sentence)
    rawToks = nltk.word_tokenize(sentence)
    rawToks.append("<EOS>")
    toks = [w.lower() for w in rawToks if any(c in alphaChars for c in w)]
    return refs, toks

def mk_output(dataSet):
    outputFormat = [[], []]
    for x, y in dataSet:
        outputFormat[0].append(x)
        outputFormat[1].append(0 if len(y) == 0 else 1)
    return outputFormat

def build_dataset(targetFile):
    refDict = {}
    refDictCtr = 1
    refDictRev = {}

    wordDict = {}
    wordDictCtr = 2 # start at 2, 1 means <unk> word
    wordDictRev = {}

    trainSet = []
    testSet = []

    idx = 0
    for file in glob.glob(data_glob):
        with open(file, 'r') as myfile:
            data = myfile.read().split("\n============\n")
            for sentence in data:
                refs, toks = prepare_sentence(sentence)

                Y = []
                for (refType, refId) in refs:
                    key = refType + ":" + refId
                    if key not in refDict:
                        refDict[key] = refDictCtr
                        refDictRev[refDictCtr] = key
                        refDictCtr += 1
                    Y.append(refDict[key])

                X = []
                for tok in toks:
                    if tok not in wordDict:
                        wordDict[tok] = wordDictCtr
                        wordDictRev[wordDictCtr] = tok
                        wordDictCtr += 1
                    X.append(wordDict[tok])

                if idx % 10 == 0:
                    testSet.append((X, Y))
                else:
                    trainSet.append((X, Y))
                idx += 1

    trainData = mk_output(trainSet)
    testData = mk_output(testSet)

    print("Train data size: " + str(len(trainData[0])))
    print("Test data size: " + str(len(testData[1])))

    with open(targetFile, 'wb') as f:
        pickle.dump(trainData, f)
        pickle.dump(testData, f)
        pickle.dump(wordDictRev, f)

build_dataset("ref_bool.pkl")