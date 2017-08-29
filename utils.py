import time
import glob

def makeTimedFilename(name, ext):
    return name + "-" + str(time.time()) + "." + ext

def getMostRecentOf(name, ext):
    name = sorted(glob.glob(name + '-*.' + ext))[-1]
    print("Most recent is: " + name)
    return name