# coding=utf-8
"""
    各种seq2seq模型的定义，序列的开始符号为"_"
    attention 模型定义参考
    https://github.com/uhauha2929/examples/blob/master/Hierarchical%20Attention%20Networks%20.ipynb
"""
from keras.models import Model
from keras.layers import Dense, Input, Permute, Lambda, RepeatVector, Multiply, Reshape
from keras.layers import LSTM, Bidirectional
from numpy import array
from keras import backend as K
from layer_component import AttentionLayer, AttentionDecoder
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
        encoder_inputs = Input(shape=(None, self.n_input))  # 长度待定
        encoder = LSTM(self.n_uints, return_state=True)  # 返回状态
        encoder_output, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.n_output))
        decoder_lstm = LSTM(self.n_uints, return_state=True, return_sequences=True)
        print(encoder_states, decoder_inputs)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)  # 加状态
        decoder_dense = Dense(self.n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  # 定义模型

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
        state = interface_en.predict(source)  # 获得语义向量
        # 输入序列输入的开始
        target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # 收集预测
        output = list()
        for t in range(n_steps):
            yhat, h, c = interface_de.predict([target_seq] + state)
            output.append(yhat[0, 0, :])
            state = [h, c]
            # 更新目标序列
            target_seq = yhat
        return array(output)


class AttentionSeq2Seq:
    def __init__(self, n_input_len, n_output_len, n_input, n_output, n_uints):
        """
        :param n_input:输入维度
        :param n_output: 输出维度
        :param n_uints: 神经元128或者256
        """
        self.n_output_len = n_output_len
        self.n_input_len = n_input_len
        self.n_input = n_input  # 输入输出特征
        self.n_output = n_output
        self.n_uints = n_uints

    def define_final_models(self):
        """
        attention 加在解码器的输入，这样解码的时候可以用到attention
        :return:
        """
        encoder_inputs = Input(shape=(self.n_input_len, self.n_input))
        encoder = LSTM(self.n_uints, return_state=True)
        # 输出维度 （None, 200）
        encoder_output, state_h, state_c = encoder(encoder_inputs)
        # encoder_states = [state_h, state_c]

        # 解码器
        decoder_inputs = Input(shape=(None, self.n_output))
        # decoder_atten = attention_3d_block(decoder_inputs)
        decoder_atten = AttentionLayer()(encoder_inputs)
        # 公式推导见https://www.cnblogs.com/shixiangwan/p/7573589.html
        decoder_atten_state1 = Multiply(name='attention_mul1')([decoder_atten, state_c])
        decoder_atten_state2 = Multiply(name='attention_mul2')([decoder_atten, state_h])
        encoder_states = [decoder_atten_state2, decoder_atten_state1]
        decoder_lstm = LSTM(self.n_uints, return_sequences=True, return_state=True)
        # print('attention:', decoder_atten)
        # print('输入张量', decoder_inputs)
        # print(encoder_states)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)  # 加状态
        decoder_dense = Dense(self.n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  # 定义模型

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

        model.summary()
        return model, encoder_model, decoder_model

    def define_full_model(self):
        encoder_input = Input(shape=(self.n_input_len, self.n_input))
        encoder = Bidirectional(LSTM(self.n_uints, return_sequences=True))
        encoder_output = encoder(encoder_input)

        decoder = AttentionDecoder(self.n_uints, self.n_input)
        decoder_output = decoder(encoder_output)
        model = Model(encoder_input, decoder_output)
        model.summary()
        return model

    def final_model_predict_seq(self, interface_en, interface_de, source, n_steps, cardinality):
        """
        :param interface_en:编码模型
        :param interface_de:解码模型
        :param source:解码的序列
        :param n_steps:每个序列步数量
        :param cardinality:每个时间步特征数
        :return:
        """
        state = interface_en.predict(source)  # 获得语义向量
        # 输入序列输入的开始
        target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # 收集预测
        output = list()
        for t in range(n_steps):
            yhat, h, c = interface_de.predict([target_seq] + state)
            output.append(yhat[0, 0, :])
            state = [h, c]
            # 更新目标序列
            target_seq = yhat
        return array(output)


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, 3))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(3, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='attention_mul')([inputs, a_probs])
    return output_attention_mul


def model_attention_applied_before_lstm():
    inputs = Input(shape=(3, 51,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    print('attention输出', attention_mul)
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


# tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)
# model = model_attention_applied_before_lstm().summary()
