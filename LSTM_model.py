"""
    古登堡计划文档范例生成
"""

import os
import numpy as np
import nltk
# import warnings
# warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from gensim.models.word2vec import Word2Vec
from seq2seq.models import Seq2Seq
from seq2seq.cells import LSTMDecoderCell, ExtendedRNNCell
from recurrentshop.engine import _OptionalInputPlaceHolder, RecurrentSequential, RNNCell


def define_model(seq_length):
    model = Sequential()
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, input_shape=(seq_length, 128)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
    return model


def seq2seq_model(batch=1, seq_length=1):
    seq_model = Seq2Seq(input_shape=(seq_length, 128), hidden_dim=10, output_length=1,
                    output_dim=128, depth=2, peek=True)
    seq_model.compile(loss='mse', optimizer='rmsprop')
    return seq_model


def predict_next(fitted_model, input_array):
    x = np.reshape(input_array, (-1,seq_length,128))
    y = fitted_model.predict(x)
    return y


def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = nltk.word_tokenize(raw_input)
    res = []
    for word in input_stream[(len(input_stream)-seq_length):]:
        res.append(w2v_model[word])
    return res


def y_to_word(y):
    # print('输出大小', y.shape)
    y = y.reshape((1, 128))
    # 找到最相似词
    word = w2v_model.most_similar(positive=y, topn=1)
    return word


def generate_article(init, fitted_model, rounds=30):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_word(predict_next(fitted_model, string_to_index(in_string)))
        in_string += ' ' + n[0][0]
    return in_string


# 数据生成器
def generate_array(train_x, train_y):
    count = 0
    X, Y = train_x, train_y
    X1 = []
    Y1 = []
    while True:
        for index in range(len(X)):
            X1.append(X[index])
            Y1.append(Y[index])
            count += 1
            if count >= batch_sizes:
                yield (np.array(X1), np.array(Y1))
                count = 0
                X1 = []
                Y1 = []


# 读取文本
raw_text = open('pg25990.txt').read()
raw_text = raw_text.lower()
sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
# 进行中文分词操作
sents = sentensor.tokenize(raw_text)
corpus = []
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))

print(corpus)
w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)

# 都放到一个list里面成为dataset
raw_input = [item for sublist in corpus for item in sublist]
print(raw_input)
text_stream = []
vocab = w2v_model.wv.vocab
for word in raw_input:
    if word in vocab:
        text_stream.append(word)
print('数据预览\n', text_stream[:10])

seq_length = 10
x = []
y = []
for i in range(0, len(text_stream) - seq_length):
    given = text_stream[i:i + seq_length]
    predict = text_stream[i + seq_length]
    x.append(np.array([w2v_model[word] for word in given]))
    y.append(w2v_model[predict])
# 我们可以看看做好的数据集的长相
print('数据集大小', len(x), len(y))  # 每个样本包含10x128维度
x = np.reshape(x, (-1, seq_length, 128))
y = np.reshape(y, (-1, 1, 128))

batch_sizes = 20
model = seq2seq_model(batch_sizes, seq_length)
model.fit(x, y, epochs=50, batch_size=batch_sizes)
"""
if os.path.exists('q2q_model.h5'):
    # 注意要载入自定义层
    model = load_model('q2q_model.h5',
                       custom_objects={'_OptionalInputPlaceHolder': _OptionalInputPlaceHolder,
                                       'RecurrentSequential': RecurrentSequential,
                                       'ExtendedRNNCell': ExtendedRNNCell,
                                       'RNNCell': RNNCell,
                                       'LSTMDecoderCell': LSTMDecoderCell,
                                       'LSTMDecoderCell.output_dim': RNNCell.output_dim})
else:
    model.fit(x, y, epochs=10, batch_size=batch_sizes)
    # model.fit_generator(generate_array(x, y), epochs=50, steps_per_epoch=12)
    model.save('q2q_model.h5')
"""

init = 'Creating the works from public domain print editions means that no ' \
       'one a United States copyright in these works, so the Foundation' \
       '(and you) can'
article = generate_article(init, fitted_model=model)
print(article)
