import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, Dropout, Activation, LSTMCell
from keras.layers.merge import dot
# from misc import get_logger, Option
import tensorflow as tf
import os
import numpy as np
# opt = Option('./config.json')


class Decoder:
    def __init__(self):
        # self.logger = get_logger('Decoder')
        # self.encoder_vector_size = opt.encoder_vector_size     #TODO
        self.encoder_vector_size = 128     #TODO

        self.checkpoint_dir = None
        self.encoder_states_size = 1024
        self.n_hidden = 1024
        self.n_b_cate = 50
        self.n_m_cate = 500
        self.n_s_cate = 3000
        self.n_d_cate = 1000
        self.checkpoint_dir = 'nomatterwhatitis'

        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'model')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # self.encoder_states = None      # TODO encoder_states
        self.batch_size = 2      # TODO self.batch_size
        self.encoder_states = tf.Variable(tf.random_normal((self.batch_size, self.encoder_states_size)))  # TODO self.batch_size
        # self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, 4], name="decoder_inputs")# TODO decoder_inputs
        # self.decoder_outputs = tf.placeholder(tf.int64, [None, 4], name="decoder_outputs")      # TODO decoder_outputs
        self.decoder_inputs = tf.Variable([[1,341,2521],[253,1221,256]], trainable=False, name='decoder_inputs')
        self.decoder_outputs = tf.Variable([[1,341,2521,231], [42,253,1221,256]], trainable=False, name='decoder_outputs', dtype=tf.int64)

        # self.dec_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        # self.dec_cell.call(self.Wxbh, state=)
        # self.target_weights = tf.placeholder(tf.float32, [None, None], name="target_weights")

        # self.dec_cell = tf.Variable([None, self.n_hidden], name='dec_cell')
        #
        # def forward_propagation(self, x):
        #     # The total number of time steps
        #     T = len(x)
        #     # During forward propagation we save all hidden states in s because need them later.
        #     # We add one additional element for the initial hidden, which we set to 0
        #     s = np.zeros((T + 1, self.hidden_dim))
        #     s[-1] = np.zeros(self.hidden_dim)
        #     # The outputs at each time step. Again, we save them for later.
        #     o = np.zeros((T, self.word_dim))
        #     # For each time step...
        #     for t in np.arange(T):
        #         # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        #         s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
        #         o[t] = softmax(self.V.dot(s[t]))
        #     return [o, s]
        #
        #
        # def bptt(self, x, y):
        #     T = len(y)
        #     # Perform forward propagation
        #     o, s = self.forward_propagation(x)
        #     # We accumulate the gradients in these variables
        #     dLdU = np.zeros(self.U.shape)
        #     dLdV = np.zeros(self.V.shape)
        #     dLdW = np.zeros(self.W.shape)
        #     delta_o = o
        #     delta_o[np.arange(len(y)), y] -= 1.
        #     # For each output backwards...
        #     for t in np.arange(T)[::-1]:
        #         dLdV += np.outer(delta_o[t], s[t].T)
        #         # Initial delta calculation: dL/dz
        #         delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        #         # Backpropagation through time (for at most self.bptt_truncate steps)
        #         for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
        #             # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
        #             # Add to gradients at each previous step
        #             dLdW += np.outer(delta_t, s[bptt_step - 1])
        #             dLdU[:, x[bptt_step]] += delta_t
        #             # Update delta for next step dL/dz at t-1
        #             delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        #     return [dLdU, dLdV, dLdW]


        self.Whh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden, self.n_hidden], name="Whh", dtype=tf.float32)
        self.bh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden], name="bh", dtype=tf.float32)

        ''' 1st state to classfy big category'''

        self.Wxbh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.encoder_states_size, self.n_hidden], name="Wxbh", dtype=tf.float32)

        self.Whyb = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden, self.n_b_cate], name="Whyb", dtype=tf.float32)

        self.byb = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_b_cate], name="byb", dtype=tf.float32)


        self.dec_cell = tf.nn.tanh(
            tf.add(
                tf.add(
                    tf.matmul(self.encoder_states, self.Wxbh),
                    tf.matmul(self.encoder_states, self.Whh)),
                self.bh
            )
        )       #TODO self.decoder_inputs[0]

        print('self.bh.shape',self.bh.shape)    #(1024, )
        print('self.encoder_states.shape',self.encoder_states.shape)    #(2, 1024)
        print('self.Whh.shape', self.Whh.shape)   #(1024, 1024)
        print('self.Wxbh.shape', self.Wxbh.shape)  #(1024, 1024)
        print('tf.matmul(self.encoder_states, self.Wxbh)', tf.matmul(self.encoder_states, self.Wxbh))   #(2, 1024)
        print('tf.matmul(self.encoder_states, self.Whh)', tf.matmul(self.encoder_states, self.Whh))     #(2, 1024)
        print('self.dec_cell.shape', self.dec_cell.shape)  #(2, 1024)

        self.logits_yb = tf.add(tf.matmul(self.dec_cell, self.Whyb), self.byb)
        print('self.Whyb.shape',self.Whyb.shape)  #(1024, 50)
        print('self.logits_yb.shape', self.logits_yb.shape)  #(2, 50)

        self.pred_yb = tf.argmax(self.logits_yb, axis=1)
        print('self.pred_yb', self.pred_yb.shape)       #(2,)
        print(self.decoder_outputs[:, 0].shape)    #(2,)
        self.crossent_yb = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.squeeze(self.logits_yb),
                                                                          labels=self.decoder_outputs[:, 0]) #TODO self.decoder_outputs[0]
        self.cost_yb = tf.reduce_mean(self.crossent_yb)

        correct_pred_yb = tf.equal(self.pred_yb, self.decoder_outputs[:, 0])
        self.accuracy_yb = tf.reduce_mean(tf.cast(correct_pred_yb, "float"))
        tf.summary.scalar('accuracy_yb', self.accuracy_yb)


        ''' 2nd state to classfy medium category'''
        self.Wxmh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_b_cate, self.n_hidden], name="Wxmh", dtype=tf.float32)
        self.Whym = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden, self.n_m_cate], name="Whym", dtype=tf.float32)
        self.bym = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_m_cate], name="bym", dtype=tf.float32)

        self.dec_cell = tf.nn.tanh(
            tf.add(
                tf.add(
                    tf.matmul(tf.one_hot(indices=self.decoder_inputs[:, 0], depth=self.n_b_cate), self.Wxmh),
                    tf.matmul(self.dec_cell, self.Whh)),
                self.bh
            )
        )       #TODO self.decoder_inputs[0]
        self.logits_ym = tf.add(tf.matmul(self.dec_cell, self.Whym), self.bym)
        print(self.logits_ym)
        self.pred_ym = tf.argmax(self.logits_ym, axis=1)

        self.crossent_ym = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_ym, labels=self.decoder_outputs[:, 1]) #TODO self.decoder_outputs[0]
        self.cost_yb = tf.reduce_mean(self.crossent_ym)


        ''' 3rd state to classfy small category'''
        self.Wxsh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_m_cate, self.n_hidden], name="Wxsh")
        self.Whys = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden, self.n_s_cate], name="Whys", dtype=tf.float32)
        self.bys = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_s_cate], name="bys", dtype=tf.float32)

        self.dec_cell = tf.nn.tanh(
            tf.add(
                tf.add(
                    tf.matmul(tf.one_hot(indices=self.decoder_inputs[:, 1], depth=self.n_m_cate), self.Wxsh),
                    tf.matmul(self.dec_cell, self.Whh)),
                self.bh
            )
        )       #TODO self.decoder_inputs[0]
        self.logits_ys = tf.add(tf.matmul(self.dec_cell, self.Whys), self.bys)
        print(self.logits_ys)

        self.pred_ys = tf.argmax(self.logits_ys, axis=1)

        self.crossent_ys = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_ys, labels=self.decoder_outputs[:, 2]) #TODO self.decoder_outputs[0]
        self.cost_ys = tf.reduce_mean(self.crossent_ys)


        ''' 4th state to classfy detail category'''
        self.Wxdh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_s_cate, self.n_hidden], name="Wxdh", dtype=tf.float32)
        self.Whyd = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden, self.n_d_cate], name="Whyd", dtype=tf.float32)
        self.byd = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_d_cate], name="byd", dtype=tf.float32)

        self.dec_cell = tf.nn.tanh(
            tf.add(
                tf.add(
                    tf.matmul(tf.one_hot(indices=self.decoder_inputs[:, 2], depth=self.n_s_cate), self.Wxdh),
                    tf.matmul(self.dec_cell, self.Whh)),
                self.bh
            )
        )       #TODO self.decoder_inputs[0]
        self.logits_yd = tf.add(tf.matmul(self.dec_cell, self.Whyd), self.byd)
        self.pred_yd = tf.argmax(self.logits_yd, axis=1)

        self.crossent_yd = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_yd, labels=self.decoder_outputs[:, 2]) #TODO self.decoder_outputs[0]
        self.cost_yd = tf.reduce_mean(self.crossent_yd)



        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        a= sess.run(self.decoder_outputs[0])
        print(a)

if __name__ == '__main__':
    decoder = Decoder()
    # def get_model(self, num_classes, activation='sigmoid'):
    #     max_len = opt.max_len
    #     voca_size = opt.unigram_hash_size + 1
    #
    #     embd = Embedding(voca_size,
    #                  opt.embd_size,
    #                  name='uni_embd')
    #
    #
    #     lstm_cell = LSTMCell(128)
    #     b_input = Input(shape=(self.encoder_vector_size,))
    #     # b_output = Dense(num_classes, activation=activation)(relu)
    #     # b_model = Model(inputs=[b_input, start_input], outputs=b_output)
    #
    #     t_uni = Input((max_len,), name="input_1")
    #     t_uni_embd = embd(t_uni)  # token
    #
    #     w_uni = Input((max_len,), name="input_2")
    #     w_uni_mat = Reshape((max_len, 1))(w_uni)  # weight
    #
    #     uni_embd_mat = dot([t_uni_embd, w_uni_mat], axes=1)
    #     uni_embd = Reshape((opt.embd_size, ))(uni_embd_mat)
    #
    #     embd_out = Dropout(rate=0.5)(uni_embd)
    #     relu = Activation('relu', name='relu1')(embd_out)
    #     outputs = Dense(num_classes, activation=activation)(relu)
    #     model = Model(inputs=[t_uni, w_uni], outputs=outputs)
    #     optm = keras.optimizers.Nadam(opt.lr)
    #     model.compile(
    #         loss='binary_crossentropy',
    #         optimizer=optm,
    #         metrics=[top1_acc]
    #     )
    #     model.summary(print_fn=lambda x: self.logger.info(x))
    #     return model

