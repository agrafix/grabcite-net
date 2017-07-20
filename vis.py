# visualise a data set

# local imports
import prepare_data as pd
import seq2seq_data as s2s
import dblp as dblp

# package imports
import glob
import numpy as np
from tqdm import tqdm
import six.moves.cPickle as pickle
from lxml import etree

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# std lib
from collections import Counter
from xml.dom import minidom
from urllib.request import urlopen
import os
import time
import sys

data_glob = "data-tiny/*.txt"

if not os.path.exists("dblp.db"):
    print("Creating a DBLP lookup map... ")
    dblp.go('dblp.xml', 'dblp.db')

dblpMap = dblp.openMap('dblp.db')

citMap = []
citTyMap = []
authorMap = []
journalMap = []
yearMap = []

def countUp(d, key):
    if key != None and key != "":
        d.append(key)

def countEntries(d, entries):
    for e in entries:
        countUp(d, e)

def plotHist(d, name, fname):
    counter = Counter(d)
    cats = []
    counts = []
    for (k, v) in counter.most_common():
        cats.append(k)
        counts.append(v)

    indexes = np.arange(len(cats))
    width = 0.7

    plt.suptitle(name, fontsize=14, fontweight='bold')
    plt.bar(indexes, counts, width)
    plt.xticks(indexes + width * 0.5, cats, rotation='vertical')
    # plt.tight_layout()  <-- this breaks atm

    print("Writing " + fname)
    plt.savefig(fname)
    plt.close()

if not os.path.exists("vis"):
    os.makedirs("vis")

countedPapers = set()

print("Working on " + data_glob)
for file in glob.glob(data_glob):
    with open(file, 'r') as myfile:
        data = myfile.read().split("\n============\n")
        for sentence in data:
            refs, _ = pd.prepare_sentence(sentence)

            for (ty, ref) in refs:

                countUp(citMap, ref)
                countUp(citTyMap, ty)

                if ty == 'DBLP':
                    dblpId = ref.replace("http://dblp.org/rec/", "")
                    print("Handling " + dblpId)

                    try:
                        res = dblpMap.Get(dblpId.encode())
                        entry = pickle.loads(res)
                        countEntries(authorMap, entry["authors"])
                        countEntries(journalMap, [entry["journal"]])
                        countEntries(yearMap, [entry["year"]])
                    except:
                        print("Unknown dblp id: " + dblpId)
                        print(sys.exc_info()[0])


plotHist(citMap, "Citations", "vis/cits.png")
plotHist(citTyMap, "Citation Types", "vis/cit_ty.png")
plotHist(authorMap, "Authors", "vis/authors.png")
plotHist(journalMap, "Journals", "vis/journals.png")
plotHist(yearMap, "Years", "vis/years.png")