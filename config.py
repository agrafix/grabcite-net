import re
import string

data_glob = "data-tiny/*.txt"

refRegex = re.compile(r"<(DBLP|ARXIV|DOI|GC):([^>]*)>")
alphaChars = set(string.ascii_letters)

glove_path = 'tiny-glove.txt'
glove_dim = 100