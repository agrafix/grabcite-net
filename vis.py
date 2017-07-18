# visualise a data set

# local imports
import prepare_data as pd
import seq2seq_data as s2s

# package imports
import glob
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# std lib
from collections import Counter
from xml.dom import minidom
from urllib.request import urlopen
import os

data_glob = "data/*.txt"

print("Working on " + data_glob)

citMap = []
citTyMap = []
authorMap = []
journalMap = []
yearMap = []

def countUp(d, key):
    if key != None and key != "":
        d.append(key)

def countXmlNodes(d, nodes):
    for node in nodes:
        ch = node.childNodes
        if len(ch) > 0:
            val = ch[0].nodeValue
            countUp(d, val)

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
    plt.tight_layout()

    print("Writing " + fname)
    plt.savefig(fname)
    plt.close()

if not os.path.exists("vis"):
    os.makedirs("vis")

for file in glob.glob(data_glob):
    with open(file, 'r') as myfile:
        data = myfile.read().split("\n============\n")
        for sentence in data:
            refs, _ = pd.prepare_sentence(sentence)

            for (ty, ref) in refs:

                countUp(citMap, ref)
                countUp(citTyMap, ty)

                if ty == 'DBLP':
                    url = ref.replace("dblp.org/rec", "dblp.org/rec/xml") + ".xml"
                    print("Fetching " + url)

                    try:
                        output = urlopen(url).read()
                        xmldoc = minidom.parseString(output)
                        authors = xmldoc.getElementsByTagName('author')
                        journal = xmldoc.getElementsByTagName('journal')
                        year = xmldoc.getElementsByTagName('year')
                        countXmlNodes(authorMap, authors)
                        countXmlNodes(journalMap, journal)
                        countXmlNodes(yearMap, year)
                    except:
                        print("Download failed!")


plotHist(citMap, "Citations", "vis/cits.png")
plotHist(citTyMap, "Citation Types", "vis/cit_ty.png")
plotHist(authorMap, "Authors", "vis/authors.png")
plotHist(journalMap, "Journals", "vis/journals.png")
plotHist(yearMap, "Years", "vis/years.png")