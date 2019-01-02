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
        self.time_step = 4
        # self.encoder_states = None      # TODO encoder_states
        self.batch_size = 2      # TODO self.batch_size
        self.encoder_states = tf.Variable(tf.random_normal((self.batch_size, self.encoder_states_size)))  # TODO self.batch_size
        # self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, 4], name="decoder_inputs")# TODO decoder_inputs
        # self.decoder_outputs = tf.placeholder(tf.int64, [None, 4], name="decoder_outputs")      # TODO decoder_outputs
        self.decoder_inputs = tf.Variable([[1,341,2521],[253,1221,256]], trainable=False, name='decoder_inputs')
        self.decoder_outputs = tf.Variable([[1,341,2521,231], [42,253,1221,256]], trainable=False, name='decoder_outputs', dtype=tf.int64)
        self.n_class = [self.n_b_cate, self.n_m_cate, self.n_s_cate, self.n_d_cate]

        def forward_propagation(self, x):
            # The total number of time steps
            T = len(x)
            # During forward propagation we save all hidden states in s because need them later.
            # We add one additional element for the initial hidden, which we set to 0
            s = np.zeros((T + 1, self.hidden_dim))
            s[-1] = np.zeros(self.hidden_dim)
            # The outputs at each time step. Again, we save them for later.
            o = np.zeros((T, self.word_dim))
            # For each time step...
            for t in np.arange(T):
                # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
                s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
                o[t] = softmax(self.V.dot(s[t]))
            return [o, s]


        def bptt(self, x, y):
            T = len(y)
            # Perform forward propagation
            o, s = self.forward_propagation(x)
            # We accumulate the gradients in these variables
            dLdU = np.zeros(self.U.shape)
            dLdV = np.zeros(self.V.shape)
            dLdW = np.zeros(self.W.shape)
            delta_o = o
            delta_o[np.arange(len(y)), y] -= 1.
            # For each output backwards...
            for t in np.arange(T)[::-1]:
                dLdV += np.outer(delta_o[t], s[t].T)
                # Initial delta calculation: dL/dz
                delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
                # Backpropagation through time (for at most self.bptt_truncate steps)
                for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                    # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                    # Add to gradients at each previous step
                    dLdW += np.outer(delta_t, s[bptt_step - 1])
                    dLdU[:, x[bptt_step]] += delta_t
                    # Update delta for next step dL/dz at t-1
                    delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
            return [dLdU, dLdV, dLdW]


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

        ''' 2nd state to classfy medium category'''
        self.Wxmh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_b_cate, self.n_hidden], name="Wxmh", dtype=tf.float32)
        self.Whym = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_hidden, self.n_m_cate], name="Whym", dtype=tf.float32)
        self.bym = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_m_cate], name="bym", dtype=tf.float32)

        ''' 3rd state to classfy small category'''
        self.Wxsh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    shape=[self.n_m_cate, self.n_hidden], name="Wxsh")
        self.Whys = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    shape=[self.n_hidden, self.n_s_cate], name="Whys", dtype=tf.float32)
        self.bys = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_s_cate], name="bys", dtype=tf.float32)
        ''' 4th state to classfy detail category'''
        self.Wxdh = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    shape=[self.n_s_cate, self.n_hidden], name="Wxdh", dtype=tf.float32)
        self.Whyd = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    shape=[self.n_hidden, self.n_d_cate], name="Whyd", dtype=tf.float32)
        self.byd = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[self.n_d_cate], name="byd", dtype=tf.float32)

        self.U = [self.Wxbh, self.Wxmh, self.Wxsh, self.Wxdh]
        self.W = [self.Whh, self.Whh, self.Whh, self.Whh]
        self.Wb = [self.bh, self.bh, self.bh, self.bh]
        self.V = [self.Whyb, self.Whym, self.Whys, self.Whyd]
        self.Vb = [self.byb, self.bym, self.bys, self.byd]





        self.decoder_states = [self.encoder_states] * 4
        self.decoder_inputs_v2 = [
            self.encoder_states,
            tf.one_hot(self.decoder_inputs[:, 0], depth=self.n_b_cate),
            tf.one_hot(self.decoder_inputs[:, 1], depth=self.n_m_cate),
            tf.one_hot(self.decoder_inputs[:, 2], depth=self.n_s_cate)
        ]
        self.logits = [None] * 4
        self.pred = [None] * 4
        self.cross_ent = [None] * 4
        self.cost = [None] * 4




        self.forward()
        self.bptt()

    def forward(self):
        for t in range(self.time_step):
            self.decoder_states[t] = tf.nn.tanh(
                tf.add(
                    tf.add(
                        tf.matmul(self.decoder_inputs_v2[t], self.U[t]),
                        tf.matmul(self.decoder_states[t-1], self.W[t])),
                    self.Wb[t]
                )
            )

            self.logits[t] = tf.nn.softmax(tf.add(tf.matmul(self.decoder_states[t], self.V[t]), self.Vb[t]))
            self.pred[t] = tf.argmax(self.logits[t], axis=1)
            self.cross_ent[t] = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.squeeze(self.logits[t]), labels=self.decoder_outputs[:, t])
            # TODO self.decoder_outputs[0]
            self.cost[t] = tf.reduce_mean(self.cross_ent[t])
            # print()
            # print(t)
            # print("decoder_states ", self.decoder_states[t])
            # print("logits ", self.logits[t])
            # print("pred ", self.pred[t])
            # print("cost ", self.cost[t])

    def bptt(self):
        dWhh = tf.zeros(shape=self.Whh.shape, name="Whh", dtype=tf.float32)
        dbh = tf.zeros(shape=self.bh.shape, name="bh", dtype=tf.float32)

        ''' 1st state to classfy big category'''
        dWxbh = tf.zeros(self.Wxbh.shape, name="dWxbh", dtype=tf.float32)
        dWhyb = tf.zeros(self.Whyb.shape, name="dWhyb", dtype=tf.float32)
        dbyb = tf.zeros(self.byb.shape, name="dbyb", dtype=tf.float32)

        ''' 2nd state to classfy medium category'''
        dWxmh = tf.zeros(self.Wxmh.shape, name="dWxmh", dtype=tf.float32)
        dWhym = tf.zeros(self.Whym.shape, name="dWhym", dtype=tf.float32)
        dbym = tf.zeros(self.bym.shape, name="dbym", dtype=tf.float32)

        ''' 3rd state to classfy small category'''
        dWxsh = tf.zeros(self.Wxsh.shape, name="dWxsh")
        dWhys = tf.zeros(self.Whys.shape, name="dWhys", dtype=tf.float32)
        dbys = tf.zeros(self.bys.shape, name="dbys", dtype=tf.float32)

        ''' 4th state to classfy detail category'''
        dWxdh = tf.zeros(self.Wxdh.shape, name="dWxdh", dtype=tf.float32)
        dWhyd = tf.zeros(self.Whyd.shape, name="dWhyd", dtype=tf.float32)
        dbyd = tf.zeros(self.byd.shape, name="dbyd", dtype=tf.float32)

        dU = [dWxbh, dWxmh, dWxsh, dWxdh]
        dW = [dWhh, dWhh, dWhh, dWhh]
        dWb = [dbh, dbh, dbh, dbh]
        dV = [dWhyb, dWhym, dWhys, dWhyd]
        dVb = [dbyb, dbym, dbys, dbyd]

        delta_o = [logit for logit in self.logits]
        # delta_t = [None] * 4

        # For each output backwards...
        for t in np.arange(self.time_step)[::-1]:
            delta_o[t] -= tf.one_hot(indices=self.decoder_outputs[:, t], depth=self.n_class[t])
            # print(dV[t])
            # print(delta_o[t])
            # print(self.decoder_states[t])
            # print(tf.matmul(tf.transpose(self.decoder_states[t]), delta_o[t]))
            dV[t] += tf.matmul(tf.transpose(self.decoder_states[t]), delta_o[t])

            # Initial delta calculation: dL/dz
            # delta_t = self.V.T.dot(delta_o[t]) * (1-(s[t]**2))
            # print(self.V[t])
            # print(delta_o[t])
            # print(self.decoder_states[t])
            # print(1-(tf.square(self.decoder_states[t])))
            # print(tf.matmul(delta_o[t], tf.transpose(self.V[t])))
            delta_t = tf.matmul(delta_o[t], tf.transpose(self.V[t])) * (1-(tf.square(self.decoder_states[t])))

            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(0, t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                # Add to gradients at each previous step

                # print(dW[t])
                # print(delta_t)
                # print(self.decoder_states[bptt_step-1])
                dW[t] += tf.matmul(tf.transpose(self.decoder_states[bptt_step-1]), delta_t)


                print(dU[t])
                print(delta_t)
                # dU[:, x[bptt_step]] += delta_t
                dU[t][:, self.decoder_inputs_v2[bptt_step]] += delta_t
                # Update delta for next step dL/dz at t-1
                # delta_t = self.W.T.dot(delta_t) * (1-s[bptt_step-1]**2)
                delta_t = tf.matmul(self.W[t], delta_t) * (1 - tf.square(self.decoder_states[bptt_step-1]))
        return [dU, dV, dW]

        # prev_s_t = np.zeros(self.hidden_dim)
        # diff_s = np.zeros(self.hidden_dim)
        # for t in range(0, T):
        #     delta_o = self.logits[t]
        #     delta_o[t][np.arange(len(y)), y] -= 1.
        #
        #     tf.dot
        #     input = np.zeros(self.word_dim)
        #     input[x[t]] = 1
        #     dprev_s, dU_t, dW_t, dV_t = layers[t].backward(input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv)
        #     prev_s_t = layers[t].s
        #     dmulv = np.zeros(self.word_dim)
        #     for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
        #         input = np.zeros(self.word_dim)
        #         input[x[i]] = 1
        #         prev_s_i = np.zeros(self.hidden_dim) if i == 0 else layers[i-1].s
        #         dprev_s, dU_i, dW_i, dV_i = layers[i].backward(input, prev_s_i, self.U, self.W, self.V, dprev_s, dmulv)
        #         dU_t += dU_i
        #         dW_t += dW_i
        #     dV += dV_t
        #     dU += dU_t
        #     dW += dW_t
        # return (dU, dW, dV)
            # correct_pred_yb = tf.equal(self.pred_yb, self.decoder_outputs[:, 0])
            # self.accuracy_yb = tf.reduce_mean(tf.cast(correct_pred_yb, "float"))
            # tf.summary.scalar('accuracy_yb', self.accuracy_yb)

        # def sgd_step(self, x, y, learning_rate):
        #     dU, dW, dV = self.bptt(x, y)
        #     self.U -= learning_rate * dU
        #     self.V -= learning_rate * dV
        #     self.W -= learning_rate * dW
        #
        #
        #
        # ''' 1st state to classfy big category'''
        #
        # self.dec_cell = tf.nn.tanh(
        #     tf.add(
        #         tf.add(
        #             tf.matmul(self.encoder_states, self.Wxbh),
        #             tf.matmul(self.encoder_states, self.Whh)),
        #         self.bh
        #     )
        # )       #TODO self.decoder_inputs[0]
        #
        # print('self.bh.shape',self.bh.shape)    #(1024, )
        # print('self.encoder_states.shape',self.encoder_states.shape)    #(2, 1024)
        # print('self.Whh.shape', self.Whh.shape)   #(1024, 1024)
        # print('self.Wxbh.shape', self.Wxbh.shape)  #(1024, 1024)
        # print('tf.matmul(self.encoder_states, self.Wxbh)', tf.matmul(self.encoder_states, self.Wxbh))   #(2, 1024)
        # print('tf.matmul(self.encoder_states, self.Whh)', tf.matmul(self.encoder_states, self.Whh))     #(2, 1024)
        # print('self.dec_cell.shape', self.dec_cell.shape)  #(2, 1024)
        #
        # self.logits_yb = tf.add(tf.matmul(self.dec_cell, self.Whyb), self.byb)
        # print('self.Whyb.shape',self.Whyb.shape)  #(1024, 50)
        # print('self.logits_yb.shape', self.logits_yb.shape)  #(2, 50)
        #
        # self.pred_yb = tf.argmax(self.logits_yb, axis=1)
        # print('self.pred_yb', self.pred_yb.shape)       #(2,)
        # print(self.decoder_outputs[:, 0].shape)    #(2,)
        # self.crossent_yb = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.squeeze(self.logits_yb),
        #                                                                   labels=self.decoder_outputs[:, 0]) #TODO self.decoder_outputs[0]
        # self.cost_yb = tf.reduce_mean(self.crossent_yb)
        #
        # correct_pred_yb = tf.equal(self.pred_yb, self.decoder_outputs[:, 0])
        # self.accuracy_yb = tf.reduce_mean(tf.cast(correct_pred_yb, "float"))
        # tf.summary.scalar('accuracy_yb', self.accuracy_yb)
        #
        #
        # ''' 2nd state to classfy medium category'''
        #
        # self.dec_cell = tf.nn.tanh(
        #     tf.add(
        #         tf.add(
        #             tf.matmul(tf.one_hot(indices=self.decoder_inputs[:, 0], depth=self.n_b_cate), self.Wxmh),
        #             tf.matmul(self.dec_cell, self.Whh)),
        #         self.bh
        #     )
        # )       #TODO self.decoder_inputs[0]
        # self.logits_ym = tf.add(tf.matmul(self.dec_cell, self.Whym), self.bym)
        # print(self.logits_ym)
        # self.pred_ym = tf.argmax(self.logits_ym, axis=1)
        #
        # self.crossent_ym = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_ym, labels=self.decoder_outputs[:, 1]) #TODO self.decoder_outputs[0]
        # self.cost_yb = tf.reduce_mean(self.crossent_ym)
        #
        #
        # ''' 3rd state to classfy small category'''
        #
        # self.dec_cell = tf.nn.tanh(
        #     tf.add(
        #         tf.add(
        #             tf.matmul(tf.one_hot(indices=self.decoder_inputs[:, 1], depth=self.n_m_cate), self.Wxsh),
        #             tf.matmul(self.dec_cell, self.Whh)),
        #         self.bh
        #     )
        # )       #TODO self.decoder_inputs[0]
        # self.logits_ys = tf.add(tf.matmul(self.dec_cell, self.Whys), self.bys)
        # print(self.logits_ys)
        #
        # self.pred_ys = tf.argmax(self.logits_ys, axis=1)
        #
        # self.crossent_ys = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_ys, labels=self.decoder_outputs[:, 2]) #TODO self.decoder_outputs[0]
        # self.cost_ys = tf.reduce_mean(self.crossent_ys)
        #
        #
        # ''' 4th state to classfy detail category'''
        #
        # self.dec_cell = tf.nn.tanh(
        #     tf.add(
        #         tf.add(
        #             tf.matmul(tf.one_hot(indices=self.decoder_inputs[:, 2], depth=self.n_s_cate), self.Wxdh),
        #             tf.matmul(self.dec_cell, self.Whh)),
        #         self.bh
        #     )
        # )       #TODO self.decoder_inputs[0]
        # self.logits_yd = tf.add(tf.matmul(self.dec_cell, self.Whyd), self.byd)
        # self.pred_yd = tf.argmax(self.logits_yd, axis=1)
        #
        # self.crossent_yd = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_yd, labels=self.decoder_outputs[:, 2]) #TODO self.decoder_outputs[0]
        # self.cost_yd = tf.reduce_mean(self.crossent_yd)
        #
        #
        #
        # sess = tf.Session()
        # init = tf.global_variables_initializer()
        # sess.run(init)
        # a= sess.run(self.decoder_outputs[0])
        # print(a)

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
    # w_uni = Input((max_len,), name="input_2")
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
