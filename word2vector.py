from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


model = Word2Vec(LineSentence('example.txt'),
                 size=300,
                 window=5,
                 min_count=1,
                 workers=2)
model.save('./word2vec.model')
for i in model.wv.vocab.keys():  # vocab是dict
    print(type(i))
    print(i)
# model = Word2Vec.load('word2vec_model')
print(model.wv['球员'])
