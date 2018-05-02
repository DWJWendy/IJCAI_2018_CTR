# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/20 8:21
@Author  : dengxiongwen
@Email   : dengxiongwen@foxmail.com
@File    : tf_cnn.py
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import logging
import pickle
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from functools import reduce

random.seed(10)
tf.set_random_seed(10)

# configure the logging module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)
import gc


class CNN1(object):

    def __init__(self, n_input, n_output, x_shape, batch_size, load=0):
        if load == 1:
            saver = tf.train.import_meta_graph("../../data/model/cnn_model.ckpt.meta")
            self.sess = tf.Session()
            saver.restore(self.sess, "../../data/model/cnn_model.ckpt")
        else:
            logger.info('building the graph...')

            self.kb = 0.8
            self.batch_size = batch_size

            self.x = tf.placeholder(tf.float32, [None, n_input], name='input')
            self.y = tf.placeholder(tf.float32, [None, n_output], name='true_label')
            self.x_ = tf.reshape(self.x, shape=x_shape)

            # define the first convolution layer
            self.W_conv1 = self.weight_variable([2, 2, 1, 16])
            self.b_conv1 = self.bias_variable([16])
            self.h_conv1 = tf.nn.relu(self.conv2d(self.x_, self.W_conv1) + self.b_conv1)
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)

            # define the second convolution layer
            self.W_conv2 = self.weight_variable([2, 2, 16, 32])
            self.b_conv2 = self.bias_variable([32])
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)

            # transform the result of h_pool2 into 1D-form
            self.h_pool2_ = tf.reshape(self.h_pool2, [-1, (x_shape[1] // 4) * (x_shape[2] // 4) *32])
            h_pool2_shape = self.h_pool2_.get_shape()
            self.W_fc1 = self.weight_variable([h_pool2_shape[1].value, 500])
            self.b_fc1 = self.bias_variable([500])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_, self.W_fc1) + self.b_fc1)

            # add a dropout layer
            self.keep_prob = tf.placeholder(tf.float32, name='keep')
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

            # add a softmax layer, and get the final probability
            self.W_fc2 = self.weight_variable([500, n_output])
            self.b_fc2 = self.bias_variable([n_output])
            self.pred = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2, name='pred')

            # self.loss_func = tf.reduce_mean(- self.y * tf.log(self.pred), name='loss_func')
            self.loss_func = tf.reduce_mean(- self.y * tf.log(self.pred), name='loss_func') + \
                             0.001 * tf.nn.l2_loss(self.W_conv1) + \
                             0.001 * tf.nn.l2_loss(self.W_conv2) + \
                             0.001 * tf.nn.l2_loss(self.W_fc1) + \
                             0.001 * tf.nn.l2_loss(self.W_fc2)
            self.optm = tf.train.AdadeltaOptimizer(0.005).minimize(self.loss_func)
            self.init_op = tf.global_variables_initializer()

            self.sess = tf.Session()
            self.sess.run(self.init_op)

    @staticmethod
    def weight_variable(shape):
        """
        the method used to define the weight variables of the convolution layers
        :param shape:tuple or list, 该权重的形状
        :return:
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """
        the method used to define the weight variables of the bias of each convolution layer
        :param shape:
        :return:
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def get_batch_data(batch_size, train_data, label_data):
        total = len(train_data)
        if batch_size > 1:
            chose_samples = random.sample(range(total), batch_size)
        else:
            chose_samples = random.randint(0, total)
        return train_data[chose_samples], label_data[chose_samples]

    def train(self, train_data, label_data, epoches):
        train_data = np.array(train_data)
        label_data = np.array(label_data)
        # for i in range(epoches):
        #     logger.info('running the {}-th round of training process...'.format(i))
        #     attr_data, labe_data = self.get_batch_data(self.batch_size, train_data, label_data)
        #     _, loss = self.sess.run([self.optm, self.loss_func],
        #                             feed_dict={self.x: attr_data, self.y:labe_data, self.keep_prob: self.kb})
        #     if i + 1 == epoches:
        #         logger.info('finish training process and the loss is {}'.format(loss))
        #     elif (i + 10) % 10 == 0:
        #         logger.info('running the {}-th epoch and the loss is {}.'.format(i, loss))
        with tf.device("/gpu:0"):
            for i in range(epoches):
                logger.info('running the {}-th round of training process...'.format(i))
                attr_data, labe_data = self.get_batch_data(self.batch_size, train_data, label_data)
                _, loss = self.sess.run([self.optm, self.loss_func],
                                        feed_dict={self.x: attr_data, self.y:labe_data, self.keep_prob: self.kb})
                if i + 1 == epoches:
                    logger.info('finish training process and the loss is '.format(loss))
                elif (i + 100) % 100 == 0:
                    logger.info('running the {}-th epoch and the loss is {}.'.format(i, loss))

    def predict(self, test_data, test_label, mode='test'):
        logger.info('predicting the result...')
        if mode == 'test':
            pred, loss = self.sess.run([self.pred, self.loss_func], feed_dict={self.x: test_data, self.y: test_label, self.keep_prob: self.kb})
            return pred, loss
        elif mode == 'predict':
            result = self.sess.run(self.pred, feed_dict={self.x: test_data, self.keep_prob: self.kb})
            return result

    def load_model_predict(self, test_data, test_label, mode='test'):
        if mode == 'test':
            result = self.sess.run(['pred: 0', 'loss_func: 0'],  feed_dict={'input: 0': test_data, 'true_label: 0': test_label, 'keep: 0': 0.8})
            return  result
        elif mode == 'predict':
            result = self.sess.run('pred: 0', feed_dict={'input: 0': test_data, 'keep: 0': 0.8})
            return result

    def save_cnn_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)


def data_matrix(all_field_path, user_file, product_file, context_file, shop_file, mode=0):
    user_field = list(pd.read_csv(user_file, sep=',', nrows=3).columns)
    product_field = list(pd.read_csv(product_file, sep=',', nrows=3).columns)
    context_field = list(pd.read_csv(context_file, sep=',', nrows=3).columns)
    shop_field = list(pd.read_csv(shop_file, sep=',', nrows=3).columns)
    all_field_data = pd.read_csv(all_field_path, sep=',')

    all_field_attrs = list(all_field_data.columns)
    # exclude_attrs = ['user_id', 'item_id', 'context_id', 'shop_id']
    attrs = [user_field, product_field, context_field, shop_field]
    for field in attrs:
        for attr in field:
            if attr not in all_field_attrs:
                field.remove(attr)

    max_length = max([len(attr) for attr in attrs]) + 1
    label = 0
    for field in attrs:
        diff = max_length - len(field)
        if diff > 0:
            for i in range(label, label + diff):
                field.append('x' + str(i))
                all_field_data['x' + str(i)] = 0
            label += diff
        else:
            pass

    attrs_orders = reduce(lambda x, y: x + y, attrs, [])

    if mode == 0:
        return all_field_data[attrs_orders], max_length
    elif mode == 1:
        return all_field_data[attrs_orders], all_field_data.index


def split_train_test(data, label_data, ratio):
    data, label_data = np.array(data), np.array(label_data)
    total = len(data)
    train_chosen_samples = random.sample(range(total), int(ratio * total))
    test_chosen_samples = []
    for ind in range(total):
        if ind not in train_chosen_samples:
            test_chosen_samples.append(ind)
    train_set_attrs, train_set_target = data[train_chosen_samples], label_data[train_chosen_samples]
    test_set_attrs, test_set_target = data[test_chosen_samples], label_data[test_chosen_samples]
    return train_set_attrs, train_set_target, test_set_attrs, test_set_target


if __name__ == '__main__':
    all_field_path = '../../data/format_2/{}_all_field_one_hot.csv'
    user_field_file = '../../data/format_2/user_field_one_hot.csv'
    product_field_file = '../../data/format_2/product_field_one_hot.csv'
    context_field_file = '../../data/format_2/context_field_one_hot.csv'
    shop_field_file = '../../data/format_2/shop_field_one_hot.csv'

    logger.info('Reading the label_data...')
    label_data = pd.read_table('../../data/raw/train_data.csv', sep=' ', usecols=['is_trade'])['is_trade']
    onehot_model = OneHotEncoder()
    label_data = onehot_model.fit_transform(label_data.reshape([-1, 1])).toarray()
    feature_pos = list(onehot_model.active_features_).index(1)

    start, end = 0, 0
    logger.info('Getting the max_length of field features...')
    _, max_length = data_matrix(all_field_path=all_field_path.format(0), user_file=user_field_file,product_file=product_field_file, context_file=context_field_file, shop_file=shop_field_file)
    cnn_model = CNN1(n_input=4 * max_length, n_output=2, x_shape=[-1, 4, max_length, 1], batch_size=100)

    for i in range(9):
        logger.info('The {}-fold data...'.format(i))
        logger.info('add zero-element to the attrs to make a data_{} matrix...'.format(i))
        train_data, _ = data_matrix(all_field_path=all_field_path.format(i), user_file=user_field_file, product_file=product_field_file, context_file= context_field_file, shop_file=shop_field_file)

        end = int(end + len(train_data))
        indices = list(range(start, end))
        start = int(start + len(train_data))

        train_labels = label_data[indices]

        # logger.info('Spliting the data into training set and test set...')
        # train_set_attrs, train_set_target, test_set_attrs, test_set_target = split_train_test(all_field_data, label_data=label_data, ratio=0.8)

        cnn_model.train(train_data=train_data, label_data=train_labels, epoches=500)

    cnn_model.save_cnn_model('../../data/model/cnn_model.ckpt')
    exit(0)

    test_data, max_length = data_matrix(all_field_path=all_field_path.format(9), user_file=user_field_file, product_file=product_field_file, context_file=context_field_file, shop_file=shop_field_file)
    end += len(test_data)
    indices = list(range(start, end))
    test_labels = label_data[indices]


    pred_result, loss = cnn_model.predict(test_data=test_data, test_label=test_labels)

    # cnn_model = CNN1(n_input=4 * max_length, n_output=2, x_shape=[-1, 4, max_length, 1], batch_size=1000, load=1)
    # pred_result, loss = cnn_model.load_model_predict(test_data=test_set_attrs, test_label=test_set_target)

    print(loss)
    # cnn_model.save_cnn_model('../../data/model/cnn_model.ckpt')

    predict_data_path = '../../data/format_2/test/all_field_one_hot.csv'
    predict_data, max_length = data_matrix(all_field_path=predict_data_path, user_file=user_field_file, product_file=product_field_file, context_file=context_field_file, shop_file=shop_field_file)
    pred_test, instance_id = cnn_model.pred(test_data=predict_data, test_labels=None, mode='predict')
    # predict_data = np.array(predict_data)

    fn = open('../../data/result_cnn.txt', 'w', encoding='utf-8')
    fn.write('instance_id predicted_score\n')
    for i, ind in enumerate(instance_id):
        fn.write(str(instance_id) + ' ' + str(pred_test[i][feature_pos]) + '\n')
    fn.close()

    # iris = datasets.load_digits()
    # iris_attrs, iris_target = iris.data, iris.target
    # onehot_model = OneHotEncoder()
    # iris_target = onehot_model.fit_transform(iris_target.reshape([-1, 1])).toarray()

    # total = len(iris_target)
    # train_chosen_samples = random.sample(range(total), int(0.8 * total))
    # test_chosen_samples = []
    # for ind in range(total):
    #     if ind not in train_chosen_samples:
    #         test_chosen_samples.append(ind)

    # train_set_attrs, train_set_target = iris_attrs[train_chosen_samples], iris_target[train_chosen_samples]
    # test_set_attrs, test_set_target = iris_attrs[test_chosen_samples], iris_target[test_chosen_samples]

    # cnn_model = CNN1(n_input=64, n_output=10, x_shape=[-1, 8, 8, 1], batch_size=10)
    # cnn_model.train(train_data=train_set_attrs, label_data=train_set_target, epoches=500)
    # pred_result = cnn_model.predict(test_set_attrs)
