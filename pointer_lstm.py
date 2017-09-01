#
# Adapted from https://github.com/keon/pointer-networks
#
# Used under MIT, Copyright (c) 2017 Keon Kim
#
# Changes:
#
# - Support Keras v2.0
#

from keras.layers import TimeDistributed, Dense
from keras.activations import tanh, softmax
from keras.layers import LSTM
from keras.engine import InputSpec
import keras.backend as K
import keras

class PointerLSTM(LSTM):
    def __init__(self, hidden_shape, **kwargs):
        super(PointerLSTM, self).__init__(hidden_shape, **kwargs)
        self.hidden_shape = hidden_shape
        self.input_length = []

    def addOrthKern(self, name, shape):
        self.add_weight(name=name, shape=shape, initializer=keras.initializers.Orthogonal)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec[0] = InputSpec(shape=input_shape)
        #init = keras.initializers.get('orthogonal')
        self.W1 = self.addOrthKern('w1', (self.hidden_shape, 1))
        self.W2 = self.addOrthKern('w2', (self.hidden_shape, 1))
        self.vt = self.addOrthKern('vt', (input_shape[1], 1))
        self.trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1]-1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_state(x_input)

        constants = super(PointerLSTM, self).get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(PointerLSTM, self).step(x_input, states[:-1])

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        #Eij = TimeDistributed(Dense(1, kernel_initializer=self.W1))(en_seq)
        #Dij = TimeDistributed(Dense(1, kernel_initializer=self.W2))(dec_seq)

        timesteps = input_shape[1]
        input_dim = input_shape[2]
        Eij = keras.layers.recurrent._time_distributed_dense(en_seq, self.W1, output_dim=1, input_dim=input_dim, timesteps=timesteps)
        Dij = keras.layers.recurrent._time_distributed_dense(dec_seq, self.W2, output_dim=1, input_dim=input_dim, timesteps=timesteps)

        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def compute_output_shape(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])