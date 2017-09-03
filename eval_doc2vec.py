from gensim.models import Doc2Vec

model = Doc2Vec.load("arxiv.doc2vec")
new_doc = model.infer_vector("is to use machine learning".split(" "))
print(new_doc)

sims = model.docvecs.most_similar([new_doc], topn=5)
print(sims)