# visualise a data set

# local imports
import prepare_data as pd
from config import data_glob

# package imports
import glob
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# std lib
from collections import Counter
from urllib.request import urlopen
import os
import time
import sys
import csv
from functools import lru_cache
import urllib.parse
import json

citMap = []
citTyMap = []
authorMap = []
journalMap = []
yearMap = []
hasCitMap = []
citDifMap = []

def countUp(d, key):
    if key != None and key != "":
        d.append(key)

def countEntries(d, entries):
    for e in entries:
        countUp(d, e)

def writeCsv(counter, csvName):
    print("Writing " + csvName)
    with open(csvName, 'w') as csvfile:
        cwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for (k, v) in counter.most_common():
            cwriter.writerow([k, v])

def plotHist(d, name, fname, sortStyle='key'):
    counter = Counter(d)
    cats = []
    counts = []

    mostCommon = counter.most_common()
    mostCommon.sort(key=lambda x: x[0] if sortStyle == 'key' else x[1])
    for (k, v) in tqdm(mostCommon, desc='Building histogram data', unit='dp'):
        cats.append(k)
        counts.append(v)

    if len(cats) < 5000:
        indexes = np.arange(len(cats))
        width = 0.7

        plt.figure(figsize=(40, 10))
        plt.gcf().subplots_adjust(bottom=0.25)

        plt.suptitle(name, fontsize=14, fontweight='bold')
        plt.bar(indexes, counts, width)
        plt.xticks(indexes + width * 0.5, cats, rotation='vertical', fontsize=6)
        # plt.tight_layout()  <-- this breaks atm

        print("Writing " + fname)
        plt.savefig(fname, dpi = (200))
        plt.close()
    else:
        print("Too many entries, skipping generation of " + fname)

    csvName = os.path.splitext(fname)[0] + ".csv"
    writeCsv(counter, csvName)

@lru_cache(maxsize=10000)
def getEntry(ref):
    recId = ref.replace("http://dblp.org/rec/", "")
    q = urllib.parse.urlencode({ "id": recId })
    url = "http://papergrep.com/api/get?" + q
    print("Fetching " + url)

    try:
        output = urlopen(url).read().decode('utf-8')
        entry = json.loads(output)
        return entry
    except:
        print(sys.exc_info())
        return None

dirName = "vis-" + str(time.time())
print("Output is going to go to " + dirName)
if not os.path.exists(dirName):
    os.makedirs(dirName)

countedPapers = set()

dataSetPapers = set()
referencePapers = {}

print("Working on " + data_glob)
for file in glob.glob(data_glob):
    metaName = os.path.splitext(file)[0] + ".meta"
    refName = os.path.splitext(file)[0] + ".refs"
    myUrl = None
    myMeta = None
    if os.path.exists(metaName):
        with open(metaName, 'r') as metaHandle:
            data = json.loads(metaHandle.read())
            myUrl = data["url"]
            if myUrl is not None:
                dataSetPapers.add(myUrl)
                myMeta = getEntry(myUrl)

    print("DBLP of " + file + " is " + str(myUrl))

    with open(file, 'r') as myfile:
        data = myfile.read().split("\n============\n")
        for sentence in data:
            refs, _ = pd.prepare_sentence(sentence)

            if len(refs) > 0:
                countUp(hasCitMap, "yes")
            else:
                countUp(hasCitMap, "no")

            for (ty, ref) in refs:

                countUp(citMap, ref)
                countUp(citTyMap, ty)

                if ty == 'DBLP':
                    if ref not in referencePapers:
                        referencePapers[ref] = 0
                    referencePapers[ref] += 1

                    entry = getEntry(ref)
                    if entry != None:
                        countEntries(authorMap, entry["authors"])
                        countEntries(journalMap, [entry["journal"]])
                        countEntries(yearMap, [str(entry["year"])])

                        if myMeta is not None and myMeta["year"] is not None and entry["year"] is not None:
                            yearDif = myMeta["year"] - entry["year"]

                            if yearDif >= 0:
                                countEntries(citDifMap, [yearDif])


# Plot all the things
plotHist(hasCitMap, "Sentences with Citation", dirName + "/cit_yn.png", sortStyle='key')
plotHist(citMap, "Citations", dirName + "/cits.png", sortStyle='value')
plotHist(citTyMap, "Citation Types", dirName + "/cit_ty.png", sortStyle='value')
plotHist(authorMap, "Authors", dirName + "/authors.png", sortStyle='value')
plotHist(journalMap, "Journals", dirName + "/journals.png", sortStyle='value')
plotHist(yearMap, "Years", dirName + "/years.png", sortStyle='key')
plotHist(citDifMap, "Year Difference", dirName + "/year-diff.png", sortStyle='key')

# Graph info
refKeys = set(referencePapers.keys())
inGraphRefs = { k: referencePapers[k] for k in dataSetPapers if k in referencePapers }

totalInner = 0
for r, ct in inGraphRefs.items():
    totalInner += ct

print("Total Inner-graph citations: " + str(totalInner))
print("Unique Inner-graph citations: " + str(len(refKeys.intersection(dataSetPapers))))
print("Unique Dangeling citations: " + str(len(refKeys.difference(dataSetPapers))))
print("Unique Uncited graph papers: " + str(len(dataSetPapers.difference(refKeys))))
