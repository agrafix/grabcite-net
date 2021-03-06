import os

import glob
import glob
from tqdm import tqdm

from recommender_lsi import QueryEngine, QueryResult, getNouns
import utils as u
from doc2vec import sentence_words, Reference
from config import data_glob

resDict = {}

def compareResult(results, actualRefs, topN=1):
    ordered_cands = [y.noAngles() for x in results[0:topN] for y in x.paperRefs]
    candidates = set(ordered_cands)
    comp = set([y.noAngles() for y in actualRefs])

    rr_sum = 0
    rr_count = 0
    for x in comp:
        pos = 0
        found = False

        for oc in ordered_cands:
            pos += 1
            if oc == x:
                found = True
                break

        if found == True:
            rr = 1 / pos
        else:
            rr = 0
        rr_sum += rr
        rr_count += 1

    dictKey = "top" + str(topN)
    if dictKey not in resDict:
        resDict[dictKey] = {
            'full': 0,
            'partial': 0,
            'miss': 0,
            'rr_sum': 0,
            'rr_count': 0
        }

    if comp - candidates == set():
        resDict[dictKey]['full'] += 1
    elif candidates - comp != candidates:
        resDict[dictKey]['partial'] += 1
    else:
        resDict[dictKey]['miss'] += 1

    resDict[dictKey]['rr_sum'] += rr_sum
    resDict[dictKey]['rr_count'] += rr_count

def printResult():
    for dictKey, val in resDict.items():
        total = val['full'] + val['partial'] + val['miss']

        print("Matrix for " + dictKey)
        print("Full Matches: " + str(val['full'] / total))
        print("Partial Matches: " + str(val['partial'] / total))
        print("Full + Partial Matches: " + str((val['full'] + val['partial']) / total))
        print("Miss-Rate: " + str((val['miss']) / total))
        print("MRR: " + str((val['rr_sum'] / val['rr_count'])))

def evaluate_lsi():
    print("Loading query engine ...")
    qe = QueryEngine()

    print("Computing statistics ...")
    ctr = 0
    for file in tqdm(glob.glob(data_glob), unit='files'):
        with open(file, 'r') as myfile:
            data = myfile.read().split("\n============\n")
            for sentence in data:
                sentence = sentence.replace("<formula>", " ")
                toks = sentence_words(sentence)
                if any(isinstance(t, Reference) for t in toks):
                    ctr += 1
                    if ctr % 100 != 0:
                        continue

                    nonRefToks = [x for x in toks if not isinstance(x, Reference)]
                    plainSentence = " ".join(nonRefToks)
                    results = qe.recommendCits(plainSentence, 50)
                    actualRefs = [x for x in toks if isinstance(x, Reference)]
                    compareResult(results, actualRefs, 1)
                    compareResult(results, actualRefs, 5)
                    compareResult(results, actualRefs, 10)
                    compareResult(results, actualRefs, 25)
                    compareResult(results, actualRefs, 50)

    printResult()

if __name__ == "__main__":
    evaluate_lsi()
