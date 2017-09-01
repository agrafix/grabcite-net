import glob
import os
import nltk
import string

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

    return newToks

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
            while xdi < len(all_words) - 1 and len(after) < around:
                xdi += 1
                if not isinstance(all_words[xdi], Reference):
                    after.append(all_words[xdi])

            if len(before) == around and len(after) == around:
                ctxs.append((tok, before, after))
        idx += 1
    return ctxs

def with_dataset(sourceGlob, sentenceTokenHandler):
    idx = 0
    print("Reading from " + sourceGlob + " ...")
    for file in glob.glob(sourceGlob):
        with open(file, 'r') as myfile:
            data = myfile.read().split("\n============\n")
            for sentence in data:
                sentenceTokenHandler(idx, sentence_words(sentence))
                idx += 1
    print("Done")