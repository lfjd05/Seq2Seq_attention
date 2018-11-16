# coding=utf-8
"""
    各种seq2seq模型的定义，序列的开始符号为"_"
    attention 模型定义参考
    https://github.com/uhauha2929/examples/blob/master/Hierarchical%20Attention%20Networks%20.ipynb
"""
from keras.models import Model
from keras.layers import Dense, Input, Permute, Lambda, RepeatVector, Multiply, Reshape
from keras.layers import LSTM, Recurrent, activations
from numpy import array
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints
import tensorflow as tf

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

    def define_models(self):
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


class AttentionLayer(Layer):
    """
        自attention, Q,和V都是inputs
    """

    def __init__(self, **kwargs):
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
        # inputs.shape = (batch_size, time_steps, dim)
        x = K.permute_dimensions(inputs, (0, 2, 1))  # 维度交换
        # print(x)  # (51,3)
        # test = K.dot(x, self.W)
        # print(test)    # (step, dim)
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        # print(a)  # 51,3
        a = K.permute_dimensions(a, (0, 2, 1))
        # a: (None, step, dim)
        # outputs = (a * inputs)    # 张量没有乘法，用这个做输出会有bug
        outputs = K.sum(a, axis=1)
        return outputs

    # 定义形状变化的逻辑，这让Keras能够自动推断各层的形状
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


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


tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)


class AttentionDecoder(Recurrent):

    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        self.return_sequences 默认是True
        :param units:
        :param output_dim:
        :param activation:
        :param return_probabilities:
        :param name:
        :param kernel_initializer:   初始权重
        :param recurrent_initializer:
        :param bias_initializer:
        :param kernel_regularizer: 正则化
        :param bias_regularizer:
        :param activity_regularizer:
        :param kernel_constraint:
        :param bias_constraint:
        :param kwargs:
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

    def build(self, input_shape):
        """
          model compile的时候运行的
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        """
            Matrices for creating the context vector
            attention计算Ct所用的权重 
        """
        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units,),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units,),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units,),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim,),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, inputs, initial_state=None, **kwargs):
        # store the whole sequence so we can "attend" to it at each timestep
        # 这里的输入不再是输入时间序列x，而是上个时刻的输入
        self.x_seq = inputs

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        # 计算方程1 得到e(j,t)
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        return super(AttentionDecoder, self).call(inputs)

    def get_initial_state(self, inputs):
        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x, states):
        """
            LSTM的几个表达式都在这
        :param x:
        :param states: 上个时刻的输出和隐层状态st
        :return:
        """
        ytm, stm = states

        # repeat the hidden state to the length of the sequence
        # 按照steps的维度重复n次，（sample, step, dim）
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        # softmax
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1 - zt) * stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(stm, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return None, self.timesteps, self.timesteps
        else:
            return None, self.timesteps, self.output_dim

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x
# model = model_attention_applied_before_lstm().summary()
