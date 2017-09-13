import glob
import nltk
from random import choice
import json
import os
import six.moves.cPickle as pickle
import pandas as pd
import numpy as np

import utils as u
from doc2vec import sentence_words, Reference, citation_contexts
from config import data_glob, refRegex, alphaChars
from word_mapper import WordMapper

around = 10
wordLimit = 30000

class PaperEntry:
    def __init__(self, title, url, contexts, refDict):
        self.title = title
        self.url = url
        self.contexts = contexts
        self.refDict = refDict

    def allWords(self):
        words = nltk.word_tokenize(self.title)
        for (refStr, myWords, refTitle) in self.contexts:
            words.extend(myWords)
            words.extend(nltk.word_tokenize(refTitle))
        return words

    def remapWords(self, wordMapper):
        self.title = wordMapper.listToId(nltk.word_tokenize(self.title))
        contexts = []
        for (refStr, words, refTitle) in self.contexts:
            words = wordMapper.listToId(words)
            refTitle = wordMapper.listToId(nltk.word_tokenize(refTitle))
            contexts.append((refStr, words, refTitle))
        self.contexts = contexts
        return self

    def anyRefTitle(self):
        if len(self.contexts) == 0:
            return None

        refStr, words, refTitle = choice(self.contexts)
        return refTitle

    def toVectors(self):
        vecs = []
        for (refStr, words, refTitle) in self.contexts:
            vecs.append(TrainingVector(self.title, words, refTitle, 1))
        return vecs

class TrainingVector:
    def __init__(self, paperTitle, context, refTitle, label):
        self.paperTitle = paperTitle
        self.context = context
        self.refTitle = refTitle
        self.label = label


def build_dataset(targetFile, around=10):
    all_papers = []
    all_refs = []

    print("Extracting contexts ...")
    for file in glob.glob(data_glob):
        base = os.path.splitext(file)[0]
        metaFile = base + ".meta"
        refsFile = base + ".refs"

        docTitle = None
        url = None

        if os.path.isfile(metaFile):
            with open(metaFile) as metaFileB:
                j = json.load(metaFileB)
                if "title" in j:
                    docTitle = j["title"]
                if "url" in j:
                    url = j["url"]

        refDict = {}
        if os.path.isfile(refsFile):
            with open(refsFile, 'r') as refsFileB:
                data = refsFileB.read().split("\n")
                for ln in data:
                    chunks = ln.split(";")
                    if len(chunks) >= 2:
                        key = chunks[0]
                        name = ";".join(chunks[1:])
                        refDict[key] = name
                        all_refs.append((key, name))

        local_contexts = []
        with open(file, 'r') as myfile:
            data = myfile.read().split("\n============\n")
            for sentence in data:
                toks = sentence_words(sentence)
                words = []
                refs = []
                for tok in toks:
                    if isinstance(tok, Reference):
                        refs.append(tok)
                    else:
                        words.append(tok)

                for ref in refs:
                    refStr = ref.noAngles()

                    if refStr in refDict:
                        local_contexts.append((refStr, words, refDict[refStr]))

        if docTitle != None:
            all_papers.append(PaperEntry(docTitle, url, local_contexts, refDict))

    print("Computing word mappings")
    wordMapper = WordMapper(all_papers, lambda x: x.allWords())
    wordMapper.restrictTo(wordLimit)

    print("Remapping word vectors")
    all_papers = [p.remapWords(wordMapper) for p in all_papers]

    print("Making training vectors from " + str(len(all_papers)) + " papers")
    all_vectors = []
    for paper in all_papers:
        myPositives = paper.toVectors()
        all_vectors.extend(myPositives)
        for mp in myPositives:
            pick = choice(all_papers)
            rt = pick.anyRefTitle()
            while pick == paper or rt == mp.refTitle or rt is None:
                pick = choice(all_papers)
                rt = pick.anyRefTitle()

            all_vectors.append(TrainingVector(mp.paperTitle, mp.context, rt, 0))

    print("That's all folks. Now we have " + str(len(all_vectors)) + " vectors. Breaking into train/valid/test")
    df = pd.DataFrame({
        'T': [v.paperTitle for v in all_vectors],
        'C': [v.context for v in all_vectors],
        'R': [v.refTitle for v in all_vectors],
        'L': [v.label for v in all_vectors]
    })
    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    def writeFrame(f, frame):
        pickle.dump({
            'T': frame['T'].tolist(),
            'C': frame['C'].tolist(),
            'R': frame['R'].tolist(),
            'L': frame['L'].tolist()
        }, f)

    print("Okay. All goes to " + targetFile)
    with open(targetFile, 'wb') as f:
        writeFrame(f, train)
        writeFrame(f, validate)
        writeFrame(f, test)
        pickle.dump(all_refs, f)
        pickle.dump(wordMapper, f)


if __name__ == "__main__":
    build_dataset(u.makeTimedFilename("prepared-data/recommender-v1", "pkl"), around)