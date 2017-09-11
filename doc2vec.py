import glob
import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from random import shuffle

from config import data_glob, refRegex, alphaChars

class Reference:
    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return self.content

    def noAngles(self):
        return self.content[1:-1]

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

    outToks = newToks

    return outToks

def citation_contexts(all_words, around, only_cits=False):
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
            while xdi < len(all_words) - 1 and len(after) < around:
                xdi += 1
                if not isinstance(all_words[xdi], Reference):
                    after.append(all_words[xdi])

            if len(before) == around and len(after) == around:
                ctxs.append((tok, before, after))
        else:
            if idx > around and not only_cits:
                before = all_words[idx - around:idx]
                after = all_words[idx + 1:idx + around]

                noCit = all(not isinstance(w, Reference) for w in before) and all(not isinstance(w, Reference) for w in after)
                if noCit:
                    ctxs.append((tok, before, after))
                    idx += around - 1
        idx += 1
    return ctxs

def build_dataset(targetFile, around=10):
    contexts = []
    print("Extracting contexts ...")
    noneIdx = 0
    for file in glob.glob(data_glob):
        with open(file, 'r') as myfile:
            data = myfile.read().split("\n============\n")
            all_words = []
            for sentence in data:
                all_words += sentence_words(sentence)
            ctxs = citation_contexts(all_words, around)
            for (ref, before, after) in ctxs:
                if isinstance(ref, Reference):
                    contexts.append(LabeledSentence(words=(before + after), tags=[str(ref)]))
                else:
                    contexts.append(LabeledSentence(words=(before + [ref] + after), tags=["<none" + str(noneIdx) + ">"]))

    alpha, min_alpha, passes = (0.025, 0.001, 20)
    alpha_delta = (alpha - min_alpha) / passes

    print("Got " + str(len(contexts)) + " contexts ... Training")
    model = Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=4)
    model.build_vocab(contexts)
    for epoch in range(passes):
        shuffle(contexts)
        print("Epoch " + str(epoch) + "/" + str(passes) + ", alpha= " + str(alpha))
        model.train(contexts, total_examples=len(contexts), epochs=1)

        alpha -= alpha_delta
        model.alpha, model.min_alpha = alpha, alpha
    print("Training complete")
    print("Writing " + targetFile)
    model.save(targetFile)

    new_doc = model.infer_vector("is to use machine learning".split(" "))
    print(new_doc)

    sims = model.docvecs.most_similar([new_doc], topn=5)
    print(sims)

if __name__ == "__main__":
    build_dataset("arxiv.doc2vec")