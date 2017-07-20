# parse and process a dblp XML into a manageable format

# package imports
from lxml import etree
from unidecode import unidecode
import leveldb

# std lib
import os, sys
import six.moves.cPickle as pickle

def processXml(context, func, *args, **kwargs):
    workTags = [u'www', u'phdthesis', u'inproceedings', u'incollection', u'proceedings', u'book', u'mastersthesis', u'article']
    authors = []
    title = ''
    journal = ''
    year = ''

    for event, elem in context:
        if elem.tag == 'journal' and elem.text:
            journal = unidecode(elem.text)

        if elem.tag == 'year' and elem.text:
            year = unidecode(elem.text)

        if elem.tag == 'author':
            authors.append(unidecode(elem.text))

        if elem.tag == 'title' and elem.text:
            title = unidecode(elem.text)

        if elem.tag in workTags:
            key = elem.attrib['key']
            if len(authors) is not 0 and title is not '':
                func(key, title, authors, journal, year, *args, **kwargs)

                title = ''
                del authors[:]
                journal = ''
                year = ''

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context

def handle(key, title, authors, journal, year, db):
    entry = {
        "key": key,
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year
    }

    print("Handled " + key)
    db.Put(key.encode(), pickle.dumps(entry))

def openMap(fOut):
    return leveldb.LevelDB(fOut)

def go(fIn, fOut):
    context = etree.iterparse(fIn, html=True, load_dtd=True)
    db = leveldb.LevelDB(fOut)
    processXml(context, handle, db)
    del db