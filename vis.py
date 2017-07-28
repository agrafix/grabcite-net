# visualise a data set

# local imports
import prepare_data as pd

# package imports
import glob
import numpy as np
from tqdm import tqdm
from graphviz import Digraph

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

data_glob = "data-tiny/*.txt"

citMap = []
citTyMap = []
authorMap = []
journalMap = []
yearMap = []
hasCitMap = []

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

def plotHist(d, name, fname):
    counter = Counter(d)
    cats = []
    counts = []
    for (k, v) in tqdm(counter.most_common(), desc='Bulding histogram data', unit='dp'):
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
        output = urlopen(url).read()
        entry = json.loads(output)
        return entry
    except:
        print(sys.exc_info())
        return None

if not os.path.exists("vis"):
    os.makedirs("vis")

countedPapers = set()

graphMap = {}

print("Working on " + data_glob)
for file in glob.glob(data_glob):
    metaName = os.path.splitext(file)[0] + ".meta"
    myUrl = None
    if os.path.exists(metaName):
        with open(metaName, 'r') as metaHandle:
            data = json.loads(metaHandle.read())
            myUrl = data["info"]["url"]
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
                    if myUrl is not None:
                        if myUrl not in graphMap:
                            graphMap[myUrl] = set()
                        graphMap[myUrl].add(ref)

                    entry = getEntry(ref)
                    if entry != None:
                        countEntries(authorMap, entry["authors"])
                        countEntries(journalMap, [entry["journal"]])
                        countEntries(yearMap, [str(entry["year"])])
    break # TODO FIXME

# Plot all the things
plotHist(hasCitMap, "Sentences with Citation", "vis/cit_yn.png")
plotHist(citMap, "Citations", "vis/cits.png")
plotHist(citTyMap, "Citation Types", "vis/cit_ty.png")
plotHist(authorMap, "Authors", "vis/authors.png")
plotHist(journalMap, "Journals", "vis/journals.png")
plotHist(yearMap, "Years", "vis/years.png")

# Paint the graph
print("Building the full graph")
dot = Digraph(comment='Citation Graph')
allNodes = set(graphMap.keys())
for k in tqdm(graphMap.keys(), desc='Computing all nodes', unit='sources'):
    allNodes = allNodes.intersection(graphMap[k])

for n in tqdm(allNodes, desc='Adding all nodes', unit='nodes'):
    dot.node(n, n)

for k in graphMap.keys():
    for tgt in graphMap[k]:
        dot.edge(k, tgt)

dot.render('vis/graph.gv')