import glob
import os
import re
import nltk
import six.moves.cPickle as pickle
import string
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

data_glob = "data/*.txt"
refRegex = re.compile(r"<(DBLP|ARXIV|DOI|GC):([^>]*)>")
alphaChars = set(string.ascii_letters)

class Reference:
    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return self.content

def sentence_words(sentence):
    toks = nltk.word_tokenize(sentence)
    total_toks = len(toks)

    newToks = []
    idx = 0

    found = False

    while idx < total_toks:
        tok = toks[idx]
        if tok == "<" and idx + 4 < total_toks:
            refTy = toks[idx+1]
            if refTy == "DBLP" or refTy == "ARXIV" or refTy == "GC" or refTy == "DOI":
                full = "<" + refTy
                c = ""
                skipped = []
                idx += 2
                while c != ">" and idx - 1 < total_toks:
                    c = toks[idx]
                    full += c
                    skipped.append(c.lower())
                    idx += 1
                idx -= 1
                if c == ">":
                    found = True
                    newToks.append(Reference(full))
                else:
                    newToks.append("<")
                    newToks.append(refTy.lower())
                    newToks = newToks + skipped
            else:
                newToks.append(tok.lower())
        else:
            newToks.append(tok.lower())
        idx += 1

    outToks = [w for w in newToks if isinstance(w, Reference) or any(c in alphaChars for c in w)]

    return outToks

def citation_contexts(all_words, around):
    idx = around
    ctxs = []
    while idx < len(all_words) - around:
        tok = all_words[idx]
        if isinstance(tok, Reference):
            before = []
            xdi = idx
            while xdi > 0 and len(before) < around:
                xdi -= 1
                if not isinstance(all_words[xdi], Reference):
                    before.append(all_words[xdi])
            before = list(reversed(before))

            after = []
            xdi = idx
            while xdi < len(all_words) and len(after) < around:
                xdi += 1
                if not isinstance(all_words[xdi], Reference):
                    after.append(all_words[xdi])

            if len(before) == around and len(after) == around:
                ctxs.append((tok, before, after))
        idx += 1
    return ctxs

def build_dataset(targetFile, around=10):
    idx = 0
    contexts = []
    print("Extracting contexts ...")
    for file in glob.glob(data_glob):
        with open(file, 'r') as myfile:
            data = myfile.read().split("\n============\n")
            all_words = []
            for sentence in data:
                all_words += sentence_words(sentence)
            ctxs = citation_contexts(all_words, around)
            for (ref, before, after) in ctxs:
                contexts.append(LabeledSentence(words=(before + after), tags=[str(ref)]))

    print("Got " + str(len(contexts)) + " contexts ... Training")
    model = Doc2Vec(alpha=0.025, min_alpha=0.025, dm=1, dm_concat=1, size=100, window=around, negative=5, hs=0, min_count=2)
    model.build_vocab(contexts)
    for epoch in range(10):
        print("Epoch " + str(epoch))
        model.train(contexts, total_examples = len(contexts), epochs = 1)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    print("Training complete")

    new_doc = model.infer_vector("is to use machine learning".split(" "))
    print(new_doc)

    sims = model.docvecs.most_similar([new_doc], topn=5)
    print(sims)

    print(sims)

if __name__ == "__main__":
    build_dataset("wordvec2.pkl")