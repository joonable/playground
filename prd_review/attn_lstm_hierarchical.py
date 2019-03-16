from modules.attention import attention
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import tensorflow as tf
import time
from utils.prepare_data import *
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

# Hyperparameter
MAX_DOCUMENT_LENGTH = 32        # 256 -> 64
MAX_WORDS = 5000
EMBEDDING_SIZE = 128    # 128
HIDDEN_SIZE = 128       # 64
ATTENTION_SIZE = 128    # 64
lr = 1e-4               # 1e-3
BATCH_SIZE = 128        # 1024
KEEP_PROB = 0.8         # 0.5
LAMBDA = 0.0001
MAX_LABEL = 2
epochs = 6

# #
# alh 1
# 0.5 +  0.25 + 0.25 = 0.5
# sick 2.5
# with3 1
#
# # # load data
# x_train, y_train, x_test, y_test = load_data_pkl_with_sampling("../data/df_dataset.pkl", sample_ratio=1)
# print("1 x_train, x_test : ", len(x_train), len(x_test))
# #
# # data preprocessing
# x_train, x_test, vocab_size = data_preprocessing_v3(x_train, x_test, MAX_DOCUMENT_LENGTH,
#                                                     max_words=MAX_WORDS)
# print("2 x_train, x_test : ", len(x_train), len(x_test))
#
# print("vocab_size : {}".format(vocab_size))
#
# # split dataset to test and dev
# x_test, x_dev, y_test, y_dev, dev_size, test_size = \
#     split_dataset_v2(x_test, y_test, 0.2)
# print("train_size : {}".format(len(x_train)))
# print("test_size : {}".format(test_size))
# print("Validation size: ", dev_size)
#
# data=dict()
# data['x_train'] = x_train
# data['x_train'] = x_train
# data['y_train'] = y_train
# data['x_test'] = x_test
# data['y_test'] = y_test
# data['x_dev'] = x_dev
# data['y_dev'] = y_dev
# data['vocab_size'] = vocab_size
# data['dev_size'] = dev_size
# data['test_size'] = test_size
#
# with open('../data/prepared_data.pkl', 'wb') as pkl_file:
#     pickle.dump(data, pkl_file)

with open('../data/prepared_data.pkl', 'rb') as pkl_file:
    data = pickle.load(pkl_file)

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
x_dev = data['x_dev']
y_dev = data['y_dev']
vocab_size = data['vocab_size']
dev_size = data['dev_size']
test_size = data['test_size']

graph = tf.Graph()
with graph.as_default():
    batch_x = tf.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH])
    batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
    keep_prob = tf.placeholder(tf.float32)

    embeddings_var = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_x)
    # print(batch_embedded.shape)  # (?, 256, 100)
    rnn_outputs, _ = tf.nn.dynamic_rnn(BasicLSTMCell(HIDDEN_SIZE), batch_embedded, dtype=tf.float32)

    # Attention
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    drop = tf.nn.dropout(attention_output, keep_prob)
    shape = drop.get_shape()
    # shape = attention_output.get_shape()

    # Fully connected layer（dense layer)
    W = tf.Variable(tf.truncated_normal([shape[1].value, MAX_LABEL], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=batch_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Accuracy metric
    prediction = tf.argmax(tf.nn.softmax(y_hat), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))
    f1_score = tf.contrib.metrics.f1_score(tf.argmax(batch_y, 1), prediction)


with tf.Session(graph=graph) as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    print("Initialized!")

    print("Start trainning")
    start = time.time()

    for e in range(epochs):
        epoch_start = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, BATCH_SIZE):
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
            l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)

        epoch_finish = time.time()
        # print("Validation accuracy: ", sess.run([accuracy, loss], feed_dict={
        print("Validation accuracy: ", sess.run([f1_score, accuracy, loss], feed_dict={
            batch_x: x_dev,
            batch_y: y_dev,
            keep_prob: 1.0
        }))
        print("epoch finished, time consumed : ", time.time() - epoch_start, " s")

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    test_f1_0 = 0
    test_f1_1 = 0

    alpha_list= list()
    prediction_list = list()
    x_list = list()
    y_list = list()

    for x_batch, y_batch in fill_feed_dict_v2(x_test, y_test, BATCH_SIZE):
        fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: 1.0}
        acc, f1, alp, pred = sess.run([accuracy, f1_score, alphas, prediction], feed_dict=fd)
        # f1 = sess.run(feed_dict=fd)
        test_acc += acc
        test_f1_0 += f1[0]
        test_f1_1 += f1[1]
        cnt += 1

        alpha_list.extend(alp)
        prediction_list.extend(pred)
        x_list.extend(x_batch)
        y_list.extend(y_batch)


    print("len(alpha_list)", len(alpha_list))
    print("len(prediction_list)", len(prediction_list))

    result = pd.DataFrame()
    # test = test.head(n=len(alpha_list))
    result['x'] = x_list
    result['y'] = np.argmax(np.array(y_list), axis=1)
    result['alpha'] = alpha_list
    result['pred'] = prediction_list
    result.to_pickle('../data/df_result.pkl')

    print("Test accuracy : %f %f %f" % (test_acc/cnt * 100, test_f1_0/cnt, test_f1_1/cnt))
