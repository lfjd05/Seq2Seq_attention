from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense
from model_pack import AttentionSeq2Seq
from layer_component import AttentionDecoder


# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique - 1) for _ in range(length)]


# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality, n_sample=5000):
    """
    :param n_in: 输入序列长度
    :param n_out: 输出序列长度
    :param cardinality:
    :param n_sample:
    :return:
    """
    X, y = list(), list()
    for _ in range(n_sample):
        # generate random sequence
        sequence_in = generate_sequence(n_in, cardinality)
        sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
        # print(sequence_out)
        # one hot encode
        src_encoded = one_hot_encode(sequence_in, cardinality)
        tar2_encoded = one_hot_encode(sequence_out, cardinality)
        # reshape as 3D
        src_encoded = src_encoded.reshape((src_encoded.shape[0], src_encoded.shape[1]))
        tar2_encoded = tar2_encoded.reshape((tar2_encoded.shape[0], tar2_encoded.shape[1]))
        X.append(src_encoded)
        y.append(tar2_encoded)
    return X, y


# define the encoder-decoder model
def baseline_model(n_timesteps_in, n_features):
    model = Sequential()
    model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
    model.add(RepeatVector(n_timesteps_in))
    model.add(LSTM(150, return_sequences=True))
    model.add(TimeDistributed(Dense(n_features, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


# define model, 这种结构是https://medium.com/datalogue/attention-in-keras-1892773a4f22，中的fig5
def attention_modle(n_timesteps_in, n_features):
    model = Sequential()
    model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
    model.add(AttentionDecoder(150, n_features))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


# configure problem
n_features = 50
n_timesteps_in = 10
n_timesteps_out = 4
epoch = 3

Seq2Seq = AttentionSeq2Seq(n_timesteps_in, n_timesteps_out, n_features, n_features, 128)
model = Seq2Seq.define_full_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
X, y = array(X), array(y)
print('数据维度', X.shape, y.shape)
# train LSTM
# generate new random sequence
# fit model for one epoch on this sequence
model.fit(X, y, verbose=2, epochs=epoch)
# evaluate LSTM
total, correct = 100, 0

for _ in range(total):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features, n_sample=1)
    X, y = array(X), array(y)
    X, y = X.reshape(1, X.shape[1], X.shape[2]), y.reshape(1, X.shape[1], X.shape[2])
    yhat = model.predict(X, verbose=0)
    if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
        correct += 1
print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
# spot check some examples
for _ in range(10):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features, n_sample=1)
    X, y = array(X), array(y)
    X, y = X.reshape(1, X.shape[1], X.shape[2]), y.reshape(1, X.shape[1], X.shape[2])
    yhat = model.predict(X, verbose=0)
    print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))

model = baseline_model(n_timesteps_in, n_features)
X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
X, y = array(X), array(y)
print('数据维度', X.shape, y.shape)
# train LSTM
# generate new random sequence
# fit model for one epoch on this sequence
model.fit(X, y, verbose=2, epochs=10*epoch)
# evaluate LSTM
total, correct = 100, 0

for _ in range(total):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features, n_sample=1)
    X, y = array(X), array(y)
    X, y = X.reshape(1, X.shape[1], X.shape[2]), y.reshape(1, X.shape[1], X.shape[2])
    yhat = model.predict(X, verbose=0)
    if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
        correct += 1
print('Base model Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
