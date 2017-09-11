from gensim.models import Doc2Vec
import glob
import nltk
from gensim.models.doc2vec import LabeledSentence
from random import shuffle

from doc2vec import sentence_words, citation_contexts, Reference
from config import data_glob

around = 10

if __name__ == "__main__":
    seek = []

    contexts = []
    print("Extracting contexts ...")
    noneIdx = 0
    for file in glob.glob(data_glob):
        with open(file, 'r') as myfile:
            data = myfile.read().split("\n============\n")
            all_words = []
            for sentence in data:
                all_words += sentence_words(sentence)
            ctxs = citation_contexts(all_words, around, True)
            for (ref, before, after) in ctxs:
                if isinstance(ref, Reference):
                    seek.append(((before + after), str(ref)))

    print("Load the model")
    model = Doc2Vec.load("trained-models/arxiv.doc2vec")

    hit = 0
    hit_5 = 0
    miss = 0


    print("Will check " + str(len(seek)) + " examples ")
    for (words, ref) in seek:
        new_doc = model.infer_vector(words)
        sims = model.docvecs.most_similar([new_doc], topn=5)
        num = 0
        ok = False
        for (sref, sperc) in sims:
            if sref == ref:
                if num == 0:
                    hit += 1
                    ok = True
                    break
                else:
                    hit_5 += 1
                    ok = True
                    break
            num += 1
        if not ok:
            miss += 1

    print("Hit: " + str(hit))
    print("Hit5: " + str(hit_5))
    print("Miss: " + str(miss))

    print("Perfect hits: " + str(hit / (hit + hit_5  + miss)))
    print("Top 5 hits: " + str((hit + hit_5) / (hit + hit_5  + miss)))