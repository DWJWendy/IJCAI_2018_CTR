# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/16 12:16
@Author  : dengxiongwen
@Email   : dengxiongwen@foxmail.com
@File    : model_tf_version.py
"""

import pandas as pd
import tensorflow as tf
import logging
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random

# configure the logging module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

random.seed(100)

class Model(object):

    # def __init__(self):
    def __init__(self, train_file, test_file):
        """
        initial the base class, and set the initial value to attributes
        """

        # self.train_x = pd.DataFrame([[1, 2, 3], [3, 1, 2], [2, 2, 1], [2, 1, 3]])
        # self.train_y = pd.DataFrame([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], dtype='float32')
        # self.test_x = pd.DataFrame([[1, 2, 3], [3, 1, 2], [3, 2, 1]])

        self.y_label = 'is_trade'
        self.train_x, self.train_y = self.read_data(train_file, read_type='train')
        self.test_x = self.read_data(test_file, read_type='test')
        self.test_instance_id = self.test_x['instance_id']
        self.test_x = self.test_x.reindex(columns=self.train_x.columns, fill_value=0)
        self.dimension = self.train_x.shape[1]
        self.batch_size = 5000
        self.indices = list(self.train_x.index)

        # transform the label into one-hot form
        onehot_model = OneHotEncoder()
        self.train_y = onehot_model.fit_transform(np.array(self.train_y)[:5000].reshape([-1, 1])).toarray()
        self.y_label_order = list(onehot_model.active_features_)
        print('The first dimension and the second dimension is respectively', self.y_label_order)

    def read_data(self, path, read_type='train'):
        """
        The method is used for read data of training set or test set.
        :param path: str, the file path of the data.
        :param read_type: str, default 'train', alternative 'test'. If read_type is 'train', the method return two
        dataframe, the data attributes and data labels respectively. If read_type is 'test', the method return only one
        dataframe of data attributes.
        :return:
        """
        with open('../../data/format/info_gain_attrs.csv', encoding='utf-8') as rd:
            info_gain = []
            while 1:
                line = rd.readline().replace('\ufeff', '').strip().split(',')
                if float(line[1]) < 0.01:
                    break
                else:
                    info_gain.append(line[0])

        logger.info('reading the ' + read_type + ' data...')
        data = pd.read_table(path, sep=',', dtype='float64')
        if read_type == 'train':
            # x_columns = list(filter(lambda x: True if x != 'instance_id' else False, data.columns))
            data_y = pd.read_table('../../data/raw/train_data.csv', sep=' ')[self.y_label]
            return data[info_gain], data_y
        elif read_type == 'test':
            return data
        else:
            raise ValueError('The parameter read_type should be "train" or "test"')

    @staticmethod
    def train(train_model, early_stop_threshold=0.0001, max_iter=10000):
        logger.info('training the data...')
        train_model.sess.run(train_model.init_op)
        loss_ = 0
        for i in range(max_iter):
            logger.info('the {}-th round...'.format(i))
            feed_dict = {train_model.input_matrix: train_model.select_data()}
            if i < 20:
                loss_ = train_model.sess.run(train_model.loss_func, feed_dict=feed_dict)
            else:
                loss = loss_
                loss_, _ = train_model.sess.run([train_model.loss_func, train_model.optimizer], feed_dict=feed_dict)
                ratio = (loss - loss_) / loss
                if ratio < early_stop_threshold:
                    break
        return train_model

    def select_data(self):
        chosen = random.sample(self.indices, self.batch_size)
        return self.train_x.loc[chosen]


    @staticmethod
    def save_model(train_model, path):
        logger.info('saving the tensorflow model...')
        # pickle.dump(train, path + '_pickle.pk')
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(train_model.sess, path + '_tf.ckpt')
        # train_model.sess.close()

    @staticmethod
    def predict_proba(train_model):
        # saver = tf.train.Saver()
        # sess = tf.Session()
        # train_model = saver.restore(sess, train_model_file)
        test_dict = {train_model.input_matrix: train_model.test_x}
        result = train_model.sess.run(train_model.predict_prob, feed_dict=test_dict)
        with open('../../data/model/predict_result.txt', 'w', encoding='utf-8') as wr:
            for i in range(len(train_model.test_instance_id)):
                wr.write(str(train_model.test_instance_id.iloc[i]) + ' ' + str(list(result[i])[train_model.y_label_order.index[1]]) + '\n')
        train_model.sess.close()


class LogisticRegression(Model):

    def __init__(self, train_file, test_file):
        Model.__init__(self, train_file, test_file)

        # build the graph
        self.input_matrix = tf.placeholder(tf.float64)
        self.weight_variables = tf.Variable(tf.zeros([self.dimension, self.train_y.shape[1]], dtype=tf.float64))
        self.intercept_variables = tf.Variable(tf.zeros([self.weight_variables.shape[1]], dtype=tf.float64))
        self.logits = tf.matmul(self.input_matrix, self.weight_variables) + self.intercept_variables
        self.predict_prob = tf.sigmoid(self.logits)
        self.loss_func = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.train_y)) + tf.nn.l2_loss(self.weight_variables)
        # self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.train_y))

        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss_func)

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()


class MultiLayerPerceptron(Model):
    def __init__(self, train_file, test_file, layer_num):
        """

        :param train_file:
        :param test_file:
        :param layer_num: list, 用于存放神经网络每层结构的列表。列表的长度表示隐层加输出层的层数，每个元素为整数，
        表示该层的神经元数量
        """
        Model.__init__(self, train_file, test_file)

        # build the graph
        self.input_matrix = tf.placeholder(dtype=tf.float64)
        self.hidden_layer = {}
        self.intercept_dict = {}
        self.intermd_dict = {}
        for i, num in enumerate(layer_num):
            self.hidden_layer['w' + str(i)] = tf.Variable(tf.zeros([self.dimension, layer_num[i]]))
            self.intercept_dict['b' + str(i)] = tf.Variable(tf.ones([layer_num[i]]))
            if i == 0:
                self.intermd_dict['layer' + str(i)] = tf.sigmoid(
                    tf.add(tf.matmul(self.input_matrix, self.hidden_layer['w' + str(i)]), self.intercept_dict['b' + str(i)]))
            elif i < len(layer_num) - 1:
                self.intermd_dict['layer' + str(i)] = tf.sigmoid(tf.add(tf.matmul(self.intercept_dict['layer' + str(i - 1)], self.hidden_layer['w' + str(i)]), self.intercept_dict['b' + str(i)]))
            else:
                self.intermd_dict['layer' + str(i)] = tf.add(tf.matmul(self.intercept_dict['layer' + str(i - 1)], self.hidden_layer['w' + str(i)]), self.intercept_dict['b' + str(i)])

        self.predict_prob = tf.sigmoid(self.intermd_dict['layer' + str(len(layer_num) - 1)])
        self.loss_func = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.intermd_dict['layer' + str(len(layer_num) - 1)], labels=self.train_y))
        for key, value in self.hidden_layer:
            self.loss_func += tf.nn.l2_loss(value)

        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss_func)
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()



class RandomForest(Model):
    pass

    # def __init__(self, train_file, test_file):
        # Model.__init__(self, train_file, test_file)


if __name__ == '__main__':
    train = '../../data/format/all_field_one_hot.csv'
    test = '../../data/format/test/all_field_one_hot.csv'

    # LR model
    lr_model = LogisticRegression(train_file=train, test_file=test)
    lr_model = lr_model.train(lr_model)
    lr_model.save_model(lr_model, '../../data/model/LogisticRegression_model')
    lr_model.predict_proba(lr_model)

    # NN model
    # nn_model = MultiLayerPerceptron(train_file=train, test_file=test, layer_num=[10, 5, 2])
    # nn_model = nn_model.train(nn_model)
    # nn_model.save_model(nn_model, '../../data/model/MultiLayerPerceotron_model.ckpt')