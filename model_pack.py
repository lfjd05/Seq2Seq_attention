# coding=utf-8
"""
    各种seq2seq模型的定义，序列的开始符号为"_"
    attention 模型定义参考
    https://github.com/uhauha2929/examples/blob/master/Hierarchical%20Attention%20Networks%20.ipynb
"""
from keras.models import Model
from keras.layers import Dense, Input, Permute, Lambda, RepeatVector, Multiply
from keras.layers import LSTM, TimeDistributed
from numpy import array
from keras import backend as K
from keras.engine.topology import Layer


# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False


class SimpleSeq2Seq:
    def __init__(self, n_input, n_output, n_uints):
        """
        :param n_input:输入维度
        :param n_output: 输出维度
        :param n_uints: 神经元128或者256
        """
        self.n_input = n_input
        self.n_output = n_output
        self.n_uints = n_uints

    def define_models(self):
        encoder_inputs = Input(shape=(None, self.n_input))   # 长度待定
        encoder = LSTM(self.n_uints, return_state=True)   # 返回状态
        encoder_output, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.n_output))
        decoder_lstm = LSTM(self.n_uints, return_state=True, return_sequences=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)   # 加状态
        decoder_dense = Dense(self.n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)    # 定义模型

        # 编码器接口
        encoder_model = Model(encoder_inputs, encoder_states)
        # 解码器接口
        decoder_state_input_h = Input(shape=(self.n_uints,))
        decoder_state_input_c = Input(shape=(self.n_uints,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
        return model, encoder_model, decoder_model

    def predict_seq(self, interface_en, interface_de, source, n_steps, cardinality):
        """
        :param interface_en:编码模型
        :param interface_de:解码模型
        :param source:解码的序列
        :param n_steps:每个序列步数量
        :param cardinality:每个时间步特征数
        :return:
        """
        state = interface_en.predict(source)   # 获得语义向量
        # 输入序列输入的开始
        target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # 收集预测
        output = list()
        for t in range(n_steps):
            yhat, h, c = interface_de.predict([target_seq]+state)
            output.append(yhat[0, 0, :])
            state = [h, c]
            # 更新目标序列
            target_seq = yhat
        return array(output)


class AttentionSeq2Seq:
    def __init__(self, n_input, n_output, n_uints):
        """
        :param n_input:输入维度
        :param n_output: 输出维度
        :param n_uints: 神经元128或者256
        """
        self.n_input = n_input
        self.n_output = n_output
        self.n_uints = n_uints

    def define_models(self):
        encoder_input = Input(shape=(None, self.n_input))   # 长度随便
        encoder = LSTM(self.n_uints, return_sequences=True)(encoder_input)
        # 维度扩展 (None, 200)
        encoder = TimeDistributed(Dense(200))(encoder)
        encoder_att = AttentionLayer()(encoder)
        sentEncoder = Model(encoder_input, encoder_att)

        decoder_inputs = Input(shape=(None, self.n_output))
        # attention作用于解码器输入
        context = TimeDistributed(sentEncoder)(decoder_inputs)
        decoder_lstm = LSTM(self.n_uints, return_state=False, return_sequences=True)(context)
        decoder_dense = Dense(self.n_uints, activation='softmax')(decoder_lstm)
        model = Model(decoder_inputs, decoder_dense)
        model.summary()


# 自定义的attention层
class AttentionLayer(Layer):
    def __init__(self, output_dim=0, **kwargs):
        self.output_dim = output_dim
        super(AttentionLayer, self).__init__(**kwargs)

    # 定义权重
    def build(self, input_shape):
        assert len(input_shape) == 3, "维度不是（time_step, dim）"
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    # 功能逻辑部分
    def call(self, inputs, mask=None):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))  # 维度交换
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs

    # 定义形状变化的逻辑，这让Keras能够自动推断各层的形状
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
