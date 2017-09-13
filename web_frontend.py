from flask import Flask, render_template, request, jsonify
import keras
from keras.models import load_model
import nltk

import utils as u
from load_data import load_word_dicts, max_sentence_len
from prepare_data import prepare_sentence
from recommender_lsi import QueryEngine, QueryResult

# Dataset loading
print("Loading dataset ...")
wordDict, wordDictRev = load_word_dicts(path=u.getMostRecentOf("prepared-data/prepared", "pkl"))

print("Loading model ...")
model = load_model(u.getMostRecentOf("trained-models/trained", "h5"))

print("Loading query engine ...")
qe = QueryEngine()

print("Ready ...")

app = Flask(__name__, template_folder="web/templates")

@app.route("/")
def hello():
    return render_template('editor.html')

@app.route("/analyse", methods=["POST"])
def annotate():
    content = request.get_json(silent=True)
    body = content["body"]
    sentences = nltk.sent_tokenize(body)
    X = []
    for sentence in sentences:
        _, toks = prepare_sentence(sentence)
        xx = []
        for tok in toks:
            v = 1
            if tok in wordDict:
                v = wordDict[tok]
            xx.append(v)
        X.append(xx)

    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=max_sentence_len, value=0.)
    results = model.predict(X)

    out = []

    sentRes = zip(sentences, results)
    charPos = 0
    for (sent, result) in sentRes:
        posEnd = charPos + len(sent)
        quoteProb = result[1]
        out.append({ "rangeStart": charPos, "rangeEnd" : posEnd, "quote": quoteProb.item() })
        charPos = posEnd + 1

    print(out)
    return jsonify({"ranges": out})

@app.route("/recommend", methods=["POST"])
def recommend():
    content = request.get_json(silent=True)
    body = content["query"]
    res = qe.recommendCits(body)
    return jsonify({"recommendations": [r.toDict() for r in res]})