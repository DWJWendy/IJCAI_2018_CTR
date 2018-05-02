#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-19 下午3:20
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : model.py
"""

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import pandas as pd
import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import logging
from sklearn.metrics import log_loss
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


class Data_Preprocess(object):
    def __init__(self, train_path, test_path, raw_train_path, raw_test_path):
        """
        Read the data including the train_data/test_data of one hot, raw_train_data/test_data, and the label of
        raw_train_data.
        :param train_path:
        :param test_path:
        :param raw_train_path:
        :param raw_test_path:
        """
        self.raw_train_data = self.read_data(raw_train_path, data_type="raw")  # 获取is_trade
        # 需要把她它分为测试集和训练集
        self.X_data = self.read_data(train_path, data_type="one-hot").drop("instance_id", axis=1)
        self.Y_label = self.raw_train_data["is_trade"]
        self.predict_data = self.read_data(test_path, data_type="one-hot")
        self.predict_x = self.alignment_data()
        self.predict_index = self.read_data(raw_test_path, data_type="raw")["instance_id"]

        # 交叉验证数据集
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.cross_data()

    @staticmethod
    def read_data(path, data_type):
        """
        Read data according to the path of data
        :param data_type:
        :param path:
        :return:
        """
        if data_type == "raw":
            return pd.read_csv(path, sep=" ")
        elif data_type == "one-hot":
            return pd.read_csv(path, sep=",")

    def alignment_data(self):
        logger.info("数据对齐...")
        return self.predict_data.reindex(columns=self.X_data.columns, fill_value=0)

    @staticmethod
    def save_model(obj, path):
        pickle.dump(obj, open(path, "wb"))
        logger.info('The model has been saved to ' + path + '...')

    def cross_data(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X_data, self.Y_label, test_size=0.1, random_state=0)
        return X_train, X_test, Y_train, Y_test


class LR_Model(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def lr_model(self):
        """
        Method: logisticRegression
        :return: return the probability of test data with list format
        """
        logger.info('LR_model beginning ...')
        classifier = LogisticRegression(solver="sag", class_weight="balanced")
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/lr_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("LR_model evaluating...")
        logloss = log_loss(y_true, np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/lr_results.txt"):
        logger.info('Write_result finishing ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.lr_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('lr_model finished ...')


class Bayes_Model(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def bayes_model(self):
        logger.info('Bayes_model beginning ...')
        classifier = BernoulliNB()
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/bayes_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("Bayes_model evaluating...")
        logloss = log_loss(y_true, np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/bayes_results.txt"):
        logger.info('Write_result finishing ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.bayes_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('bayes_model finished ...')


class RandomTree(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def randomtree_model(self):
        logger.info('RandomTree_model beginning ...')
        classifier = RandomForestClassifier(class_weight="balanced")
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/random_tree_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("Random_tree_model evaluating...")
        logloss = log_loss(y_true,np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/random_tree_results.txt"):
        logger.info('Write_result finishing ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.randomtree_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('random_tree_model finished ...')


class GTB(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def gtb_model(self):
        logger.info('GTB_model beginning ...')
        classifier = GradientBoostingClassifier()
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/gtb_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("GTB_model evaluating...")
        logloss = log_loss(y_true, np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/gtb_results.txt"):
        logger.info('Write_result finishing ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.gtb_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('GTB_model finished ...')


class NeuralNetwork(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def nn_model(self):
        logger.info('NN_model beginning ...')
        classifier = MLPClassifier(solver="sgd", hidden_layer_sizes=(500, 3))
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/nn_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("NN_model evaluating...")
        logloss = log_loss(y_true, np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/nn_results.txt"):
        logger.info('Write_result beginning ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.nn_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('NN_model finished ...')


if __name__ == "__main__":
    train_path = "../../data/format_2/all_field_one_hot.csv"
    test_path = "../../data/format_2/test/all_field_one_hot.csv"
    raw_train_path = "../../data/raw/new_train_data.csv"
    raw_test_path = "../../data/raw/new_test_data.csv"
    logger.info("预处理数据...")
    data_results = Data_Preprocess(train_path=train_path, test_path=test_path,
                                   raw_train_path=raw_train_path, raw_test_path=raw_test_path)
    X_train, X_test, Y_train, Y_test = data_results.X_train, data_results.X_test, data_results.Y_train, \
                                       data_results.Y_test
    train_purchase_rate = sum(Y_train)/len(Y_train.index)
    test_purchase_rate = sum(Y_test)/len(Y_test.index)
    logger.info("训练集购买比例：" + str(train_purchase_rate))
    logger.info("测试集购买比例：" + str(test_purchase_rate))
    predict_x = data_results.predict_x
    predict_index = data_results.predict_index

    # lr_model
    # lr = LR_Model()
    # lr.run()
    # bayes_model
    # bayes = Bayes_Model()
    # bayes.run()
    # Random_tree model
    random_tree = RandomTree()
    random_tree.run()
    # GTB
    gtb = GTB()
    gtb.run()
    # NN_model
    nn_model = NeuralNetwork()
    nn_model.run()
