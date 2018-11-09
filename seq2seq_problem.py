# coding=utf-8
"""
    输入序列：随机整数6个
    输出序列：输入序列前三个元素倒序，长度3
"""
from random import randint
from keras.utils import to_categorical
from numpy import array, argmax, array_equal
from example.model_pack import SimpleSeq2Seq, AttentionSeq2Seq


def generate_seq(length, n_unique):
    """
    :param length:
    :param n_unique: 序列值域
    :return:
    """
    return [randint(1, n_unique - 1) for _ in range(length)]  # 注意和numpy的不一样是闭区间的


def get_dataset(n_in, n_out, cardinality, n_samples):
    """

    :param n_in:
    :param n_out:
    :param cardinality:
    :param n_samples:
    :return:
    """
    X1, X2, y = list(), list(), list()
    for _ in range(n_samples):
        source = generate_seq(n_in, cardinality)
        target = source[:n_out]
        target.reverse()
        target_in = [0] + target[:-1]   # 添加起始字符0
        # 独热编码
        src_encoded = to_categorical(source, num_classes=cardinality)  # 输入参数为独热编码的维度
        tar_encoded = to_categorical(target, num_classes=cardinality)
        tar2_encoded = to_categorical(target_in, num_classes=cardinality)
        # store
        X1.append(src_encoded)
        X2.append(tar2_encoded)  # 解码器的输入，带start
        y.append(tar_encoded)
    return array(X1), array(X2), array(y)


# 解码函数
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# 独热编码的特征数
n_features = 50 + 1
n_steps_in = 6  # 输入时间步长度
n_steps_out = 3  # 输出时间步长度
# generate a single source and target sequence
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 50000)
print(X1.shape, X2.shape, y.shape)
print('X1=%s, X2=%s, y=%s' % (one_hot_decode(X1[0]), one_hot_decode(X2[0]), one_hot_decode(y[0])))

# Seq2Seq = SimpleSeq2Seq(n_features, n_features, 128)
Seq2Seq = AttentionSeq2Seq(n_features, n_features, 128)
train, infenc, infdec = Seq2Seq.define_models()
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# train model
train.fit([X1, X2], y, epochs=20)

# 评估模型
total, correct = 100, 0
for _ in range(total):
    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    target = Seq2Seq.predict_seq(infenc, infdec, X1, n_steps_out, n_features)
    if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
        correct += 1
print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))

# spot check some examples, 预测新的样本
for _ in range(10):
    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    target = Seq2Seq.predict_seq(infenc, infdec, X1, n_steps_out, n_features)
    print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))
