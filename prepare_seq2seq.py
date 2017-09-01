import six.moves.cPickle as pickle

import import_tools as it
from bidict import BiDict
from config import data_glob
import utils as u
import numpy as np
import pandas as pd

def handleToken(tok, occurs):
    if tok in occurs:
        occurs[tok] += 1
    else:
        occurs[tok] = 1

def buildDictionary(occurs, dictSizeMax=30000):
    refList = []
    fullList = []
    for key, value in occurs.items():
        if isinstance(key, it.Reference):
            refList.append((key, value))
        else:
            fullList.append((key, value))
    fullList.sort(key=lambda x: x[1], reverse=True)

    maxFull = dictSizeMax - len(refList)
    fullSource = refList + fullList[:maxFull]

    refIds = set()
    tokenMapper = BiDict()
    nextTokenId = 2
    for (tok, val) in fullSource:
        tokenId = tokenMapper.getFirst(tok)
        if tokenId == None:
            tokenMapper.insert(tok, nextTokenId)
            tokenId = nextTokenId
            nextTokenId += 1

        if isinstance(tok, it.Reference):
            refIds.add(tokenId)

    return (tokenMapper, refIds)

def toksToIdx(toks, tokenMapper):
    out = []
    for t in toks:
        tokenId = tokenMapper.getFirst(t)
        if tokenId == None:
            tokenId = 1
        out.append(tokenId)
    return out

if __name__ == "__main__":
    print("Ready, let's do it ...")
    occurs = {}
    allSentences = []
    def handler(idx, toks):
        allSentences.append(toks)
        for tok in toks:
            handleToken(tok, occurs)

    it.with_dataset(data_glob, handler)

    print("Building dictionary ...")
    (tokenDict, refIds) = buildDictionary(occurs)


    print("Generating training data ...")
    X = []
    Y = []

    for sent in allSentences:
        tokIds = toksToIdx(sent, tokenDict)
        X.append([tid for tid in tokIds if not tid in refIds])
        Y.append(tokIds)

    print("Splitting into train/validate/test")
    df = pd.DataFrame({'X': X, 'Y': Y})
    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    def writeFrame(f, frame):
        pickle.dump((frame['X'].tolist(), frame['Y'].tolist()), f)

    targetFile = u.makeTimedFilename("prepared-data/seqs", "pkl")
    print("Writing everything to file " + targetFile + " ...")
    with open(targetFile, 'wb') as f:
        writeFrame(f, train)
        writeFrame(f, validate)
        writeFrame(f, test)
        pickle.dump(tokenDict, f)