import os
import glob
import six.moves.cPickle as pickle

from textblob import TextBlob
from gensim import corpora, models, similarities
from tqdm import tqdm

import utils as u
from doc2vec import sentence_words, Reference
from config import data_glob

num_factors=200

def getNouns(x):
    nouns = []
    for n in TextBlob(x).noun_phrases:
        nouns.append(n.string)
    return nouns

class QueryResult:
    def __init__(self, prob, papers, nouns):
        self.prob = prob
        self.papers = papers
        self.nouns = nouns

    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def toDict(self):
        return {
            'prob': self.prob,
            'papers': self.papers,
            'nouns': self.nouns
        }

class QueryEngine:
    def __init__(self):
        self.files = u.getMostRecentOfSet("prepared-data/recommender-lsi", ["dict", "tfidf", "lsi", "idx", "meta"])
        self.dict = corpora.Dictionary.load(self.files["dict"])
        self.tfidf = models.TfidfModel.load(self.files["tfidf"])
        self.lsi = models.LsiModel.load(self.files["lsi"])
        self.idx = similarities.MatrixSimilarity.load(self.files["idx"])
        with open(self.files["meta"], 'rb') as f:
            self.refMap = pickle.load(f)
            self.refDict = pickle.load(f)

    def simToRes(self, sim):
        docIdx = sim[0]
        prob = sim[1]

        out = []

        data = self.refMap[docIdx]
        for ref in data[0]:
            refStr = ref.noAngles()

            if refStr in self.refDict:
                out.append(self.refDict[refStr])
            else:
                out.append(refStr)

        return QueryResult(prob, out, data[1])

    def recommendCits(self, sentence):
        myNouns = getNouns(sentence)
        vec_bow = self.dict.doc2bow(myNouns)
        vec_tfidf = self.tfidf[vec_bow]
        vec_lsi = self.lsi[vec_tfidf]
        sims = self.idx[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        for i in range(0, 10):
            print(self.simToRes(sims[i]))

def build_dataset(nameMaker):
    all_refs = []

    print("Extracting contexts ...")
    idx = 0
    refMap = {}
    refDict = {}

    tokTexts = []

    for file in tqdm(glob.glob(data_glob), unit='files'):
        base = os.path.splitext(file)[0]
        refsFile = base + ".refs"

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

        with open(file, 'r') as myfile:
            data = myfile.read().split("\n============\n")
            for sentence in data:
                sentence = sentence.replace("<formula>", " ")
                toks = sentence_words(sentence)
                if any(isinstance(t, Reference) for t in toks):
                    nonRefToks = [x for x in toks if not isinstance(x, Reference)]
                    nouns = getNouns(" ".join(nonRefToks))
                    tokTexts.append(nouns)
                    refMap[idx] = ([x for x in toks if isinstance(x, Reference)], nouns)
                    idx += 1

    print("Building the index ...")
    dictionary = corpora.Dictionary(tokTexts)
    dictionary.save(nameMaker.getName("", "dict"))

    corpus = [dictionary.doc2bow(text) for text in tokTexts]
    corpora.MmCorpus.serialize(nameMaker.getName("", "mm"), corpus)

    tfidf = models.TfidfModel(corpus)
    tfidf.save(nameMaker.getName("", "tfidf"))

    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, num_topics=num_factors, id2word=dictionary)
    lsi.save(nameMaker.getName("", "lsi"))

    print("LSI Topics are: ")
    print(lsi.print_topics(10))

    corpus_lsi = lsi[corpus_tfidf]
    index = similarities.MatrixSimilarity(corpus_lsi)
    index.save(nameMaker.getName("", "idx"))

    with open(nameMaker.getName("", "meta"), 'wb') as f:
        pickle.dump(refMap, f)
        pickle.dump(refDict, f)

    print("Done.")

    print("Running a quick sanity test: ")

    myDoc = "the noise variance is super high"
    myNouns = getNouns(myDoc)
    print(myNouns)
    vec_bow = dictionary.doc2bow(myNouns)
    vec_tfidf = tfidf[vec_bow]
    vec_lsi = lsi[vec_tfidf]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sims[0:10])  # print sorted (document number, similarity score) 2-tuples

    print(refMap[sims[0][0]])
    print(refMap[sims[1][0]])
    print(refMap[sims[2][0]])
    print(refMap[sims[3][0]])
    print(refMap[sims[4][0]])
    print(refMap[sims[5][0]])

if __name__ == "__main__":
    nameMaker = u.FilenameMaker("prepared-data/recommender-lsi")
    build_dataset(nameMaker)

    qe = QueryEngine()
    qe.recommendCits("the noise variance is really high....")