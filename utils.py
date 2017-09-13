import time
import glob

class FilenameMaker:
    def __init__(self, baseName):
        self.now = time.time()
        self.base = baseName

    def getName(self, name, ext):
        return self.base + name + "-" + str(self.now) + "." + ext

def makeTimedFilename(name, ext):
    return FilenameMaker("").getName(name, ext)

def getMostRecentOf(name, ext):
    name = sorted(glob.glob(name + '-*.' + ext))[-1]
    print("Most recent is: " + name)
    return name

def getMostRecentOfSet(name, exts):
    d = {}
    for ext in exts:
        d[ext] = getMostRecentOf(name, ext)
    return d