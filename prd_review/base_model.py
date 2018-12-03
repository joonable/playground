import pandas as pd
import tensorflow as tf
import os
import time
import abc


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config, file_name = None):
        self.lr = config['lr']
        self.n_hidden = config['n_hidden']
        self.total_epoch = config['total_epoch']
        self.batch_size = config['batch_size']
        self.embedding_size = config['embedding_size']
        # self.vocabulary_list = data_helper.vocabulary_list
        # self.vocabulary_dict = data_helper.vocabulary_dict
        self.n_class = self.n_input = self.dic_len = len(self.vocabulary_list)
        self.n_eval = config['n_eval']
        self.training_mode = True

        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.output_keep_prob = tf.placeholder(tf.float32)
        self.current_batch_size = tf.placeholder(dtype = tf.int32, shape = [], name = "current_batch_size")

        self.encoder_inputs = tf.placeholder(dtype = tf.int32, shape = [None, None], name = "encoder_inputs")
        self.decoder_inputs = tf.placeholder(dtype = tf.int32, shape = [None, None], name = "decoder_inputs")
        self.decoder_outputs = tf.placeholder(tf.int64, [None, None], name = "decoder_outputs")
        self.load_data_set(file_name)

    def build_model(self):
        '''you need to build enc, dec or etc in drived classes '''
        with tf.variable_scope('output'):
            self.logits = self.outputs.rnn_output
            self.prediction = tf.argmax(self.logits, axis = 2)

        with tf.variable_scope('Cost'):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits,
                                                                           labels = self.decoder_outputs)
            self.cost = (tf.reduce_mean(crossent * self.target_weights))
            tf.summary.scalar('cost', self.cost)

        with tf.variable_scope('Accuracy'):
            # predictions = self.prediction * self.target_weights
            correct_predictions = tf.equal(self.prediction, self.decoder_outputs)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('optimiser'):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.cost, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)
            lr = tf.train.exponential_decay(self.lr , global_step = self.global_step, decay_steps = self.n_eval,
                                            decay_rate = 0.999, staircase = True)
            opimiser = tf.train.AdamOptimizer(lr)
            self.train_op = opimiser.apply_gradients(
                zip(clipped_gradients, params), global_step = self.global_step)

        self.graph = tf.Graph()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter('./tensorboard', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.set_checkpoints()

    def load_data_set(self, file_name):
        if not file_name:
            file_name = 'with_noise'
        self.df_train = pd.read_csv('./dataset/df_train_' + file_name + '.csv')
        self.df_test = pd.read_csv('./dataset/df_test_' + file_name + '.csv')

    @abc.abstractmethod
    def build_embedding_layer(self):
        # with tf.variable_scope('embedding'):
        self.embedding = tf.get_variable("embedding_layer", [self.dic_len, self.embedding_size], trainable = True)
        self.encoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
        self.decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)

    @abc.abstractmethod
    def build_encoder(self):
        with tf.variable_scope('encode'):
            self.encoder_length = tf.placeholder(tf.int32, [None], name = "encoder_length")

            enc_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob = self.output_keep_prob)
            self.outputs, self.enc_states = tf.nn.dynamic_rnn(cell = enc_cell, inputs = self.encoder_emb_inp,
                                                              dtype = tf.float32, sequence_length = self.encoder_length)

    @abc.abstractmethod
    def build_decoder(self):
        with tf.variable_scope('decode'):
            self.decoder_length = tf.placeholder(tf.int32, [None], name = "decoder_length")
            self.target_weights = tf.placeholder(tf.float32, [None, None], name = "target_weights")

            self.dec_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            self.dec_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_cell, output_keep_prob = self.output_keep_prob)

    def set_checkpoints(self, timestamp = None):
        # Checkpoint files will be saved in this directory during training
        if not timestamp:
            timestamp = str(int(time.time()))

        self.timestamp = timestamp
        self.checkpoint_dir = './checkpoints_' + self.timestamp + '/'
        # if os.path.exists(self.checkpoint_dir):
        #     shutil.rmtree(self.checkpoint_dir)
        # os.makedirs(self.checkpoint_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'model')

    def set_feed_dict(self, batch, is_train = False):
        enc_input_batch, dec_input_batch, dec_output_batch, target_weights_batch, \
        enc_len_batch, dec_len_batch, current_batch_size_batch \
            = data_helper.make_batch(pd.DataFrame(batch, columns = ['x', 'y']))
        if is_train:
            output_keep_prob = 0.75
        else:
            output_keep_prob = 1

        feed_dict = {
            self.encoder_inputs:enc_input_batch,
            self.decoder_inputs:dec_input_batch,
            self.decoder_outputs:dec_output_batch,
            self.target_weights:target_weights_batch,
            self.encoder_length:enc_len_batch,
            self.decoder_length:dec_len_batch,
            self.output_keep_prob:output_keep_prob,
            self.current_batch_size:current_batch_size_batch
        }
        return feed_dict

    def train(self):
        x_train = self.df_train.x.tolist()
        y_train = self.df_train.y.tolist()
        train_batches = data_helper.batch_iter(
            data = list(zip(x_train, y_train)), batch_size = self.batch_size, num_epochs = self.total_epoch)
        train_loss, train_best_loss, val_best_loss, self.best_at_step = 0, 100, 100, 0

        log_msg_list = []

        for train_batch in train_batches:
            current_step = tf.train.global_step(self.sess, self.global_step)
            feed_dict = self.set_feed_dict(train_batch, True)
            self.merged_summaries = tf.summary.merge_all()
            _, loss, accuracy, summary = self.sess.run(
                [self.train_op, self.cost, self.accuracy, self.merged_summaries], feed_dict = feed_dict)
            self.train_writer.add_summary(summary = summary, global_step = current_step)

            log_msg = 'current_step = ', '{}'.format(current_step), \
                      ', cost = ', '{:.6f}'.format(loss), \
                      ', accuracy = ', '{:.6f}'.format(accuracy)+ '\n'
            log_msg_list += log_msg
            print(log_msg)

            train_loss += loss

            if current_step != 0 and current_step % self.n_eval == 0:
                val_loss, val_accuracy = self.test(
                    self.df_test.sample(frac = 0.2).reset_index(drop = True), False)
                train_loss /= (self.n_eval)
                log_msg = 'current_step = ', '{}'.format(current_step), \
                          ', val_cost = ', '{:.6f}'.format(val_loss), \
                          ', val_accuracy = ', '{:.6f}'.format(val_accuracy), \
                          ', train_cost = ', '{:.6f}'.format(train_loss)+ '\n'
                log_msg_list += log_msg
                print(log_msg)

                if train_loss < train_best_loss and val_loss < val_best_loss:
                    train_best_loss, val_best_loss, self.best_at_step = train_loss, val_loss, current_step

                    self.save_current_session(current_step)

                    print('Best cost {:.6f} and {:.6f} at step {}'.format(
                        train_best_loss, val_best_loss, self.best_at_step))

                with open('./logs/log_' + self.timestamp + '.txt', 'a') as f:
                    f.writelines(log_msg_list)

                log_msg_list = []
                train_loss = 0
                # conf = tf.ConfigProto()


    @abc.abstractmethod
    def save_current_session(self, current_step):
        pass

    @abc.abstractmethod
    def restore_best_session(self, best_at_step = None):
        pass

    def test(self, df, file_name, is_eval = True):
        feed_dict = self.set_feed_dict(df, True)

        results, loss, accuracy = self.sess.run([self.prediction, self.cost, self.accuracy], feed_dict = feed_dict)

        if is_eval:
            return loss, accuracy
        else:
            print('cost = ', '{:.6f}'.format(loss), ', accuracy = ', '{:.6f}'.format(accuracy))

            decoded_number = []
            for result in results:
                decoded_number.append([self.vocabulary_list[i] for i in result])

            decoded_jamo = []
            for result in decoded_number:
                try:
                    end = result.index('E')
                    decoded_jamo.append(''.join(result[:end]))
                except:
                    decoded_jamo.append(''.join(result))

            df['predict'] = [jamo.join_jamos(x) for x in decoded_jamo]
            df.to_csv('./model_result/' + file_name + '_result_' + self.timestamp + '.csv', index = False)