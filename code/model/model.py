# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/16 10:47
@Author  : dengxiongwen
@Email   : dengxiongwen@foxmail.com
@File    : model.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


class Model(object):
    def __init__(self, train_path, test_path, raw_train_path, raw_test_path):
        """
        Read the data including the train_data/test_data of one hot, raw_train_data/test_data, and the label of
        raw_train_data.
        :param train_path:
        :param test_path:
        :param raw_train_path:
        :param raw_test_path:
        """

        self.train_data = self.read_data(train_path, data_type="one-hot")
        self.test_data = self.read_data(test_path, data_type="one-hot")
        self.raw_train_data = self.read_data(raw_train_path, data_type="raw")  # 获取is_trade
        self.train_label = self.raw_train_data["is_trade"]
        self.test_index = self.read_data(raw_test_path, data_type="raw")["instance_id"]
        self.train_x = self.train_data.drop("instance_id", axis=1)
        self.test_x = self.alignment_data().drop("instance_id", axis=1)


    @staticmethod
    def read_data(path, data_type):
        """
        Read data according to the path of data
        :param path:
        :return:
        """
        if data_type == "raw":
            return pd.read_csv(path, sep=" ")
        elif data_type == "one-hot":
            return pd.read_csv(path, sep=",")

    def alignment_data(self):
        logger.info("数据对齐...")
        return self.test_data.reindex(columns=self.train_data.columns, fill_value=0)

    @staticmethod
    def save_model(obj, path):
        pickle.dump(obj, open(path, "wb"))
        logger.info('The model has been saved to ' + path + '...')


class LR(Model):
    # Read data, and acquire the train_x/train_y and test_x/test_x_index

    def __init__(self, train_path, test_path, raw_train_path, raw_test_path):
        """

        :param train_path:
        :param test_path:
        :param raw_train_path:
        :param raw_test_path:
        """
        Model.__init__(self, train_path, test_path, raw_train_path, raw_test_path)

    def lr_model(self):
        """
        Method: logisticRegression
        :return: return the probability of test data with list format
        """
        logger.info('LR_model finished ...')
        classifier = LogisticRegression()
        classifier.fit(self.train_x, self.train_label)
        index = list(classifier.classes_).index(1)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.test_x)))
        self.save_model(obj=classifier, path="../../data/results/lr_model.pk")
        return predict_y

    def write_result(self, predict_pro, path="../../data/results/lr_results.txt"):
        logger.info('Write_result finishing ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                f.write(str(self.test_index[i]) + " " + str(predict_pro[i]) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        predict_y = self.lr_model()
        self.write_result(predict_pro=predict_y)
        logger.info('lr_model finished ...')


class Bayes(Model):

    def __init__(self, train_path, test_path, raw_train_path, raw_test_path):
        Model.__init__(self, train_path, test_path, raw_train_path, raw_test_path)

    def bnb_model(self):
        classifier = BernoulliNB()
        classifier.fit(self.train_x, self.train_label)
        index = list(classifier.classes_).index(1)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.test_x)))
        self.save_model(classifier, path="../../data/results/Bayes_model.pk")
        logger.info('bnb_model finished ...')
        return predict_y

    def write_result(self, predict_pro, path="../../data/results/bayes_results.txt"):
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                f.write(str(self.test_index[i]) + " " + str(predict_pro[i]) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        predict_y = self.bnb_model()
        self.write_result(predict_pro=predict_y)
        logger.info("Bayes_model finished")


class RandomTree(Model):

    def __init__(self, train_path, test_path, raw_train_path, raw_test_path):
        Model.__init__(self, train_path, test_path, raw_train_path, raw_test_path)

    def RandomTree_model(self):
        classifier = RandomForestClassifier()
        classifier.fit(self.train_x, self.train_label)
        index = list(classifier.classes_).index(1)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.test_x)))
        self.save_model(classifier, path="../../data/results/RandomTree_model.pk")
        logger.info('RandomTree_model finished ...')
        return predict_y

    def write_result(self, predict_pro, path="../../data/results/RandomTree_results.txt"):
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                f.write(str(self.test_index[i]) + " " + str(predict_pro[i]) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        predict_y = self.RandomTree_model()
        self.write_result(predict_pro=predict_y)
        logger.info("RandomTree finished")


class GTB(Model):

    def __init__(self, train_path, test_path, raw_train_path, raw_test_path):
        Model.__init__(self, train_path, test_path, raw_train_path, raw_test_path)

    def gtb_model(self):
        classifier = GradientBoostingClassifier()
        classifier.fit(self.train_x, self.train_label)
        index = list(classifier.classes_).index(1)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.test_x)))
        self.save_model(classifier, path="../../data/results/GTB_model.pk")
        logger.info('gtb_model finished ...')
        return predict_y

    def write_result(self, predict_pro, path="../../data/results/gtb_results.txt"):
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                f.write(str(self.test_index[i]) + " " + str(predict_pro[i]) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        predict_y = self.gtb_model()
        self.write_result(predict_pro=predict_y)
        logger.info("GTB_model finished")


class NeuralNetwork(LR):
    def __init__(self, train_path, test_path, raw_train_path, raw_test_path):
        Model.__init__(self, train_path, test_path, raw_train_path, raw_test_path)

    def nn_model(self):
        classifier = MLPClassifier()
        classifier.fit(self.train_x, self.train_label)
        index = list(classifier.classes_).index(1)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.test_x)))
        self.save_model(classifier, path="../../data/results/NN_model.pk")
        logger.info('NN_model finished ...')
        return predict_y

    def write_result(self, predict_pro, path="../../data/results/nn_results.txt"):
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + "  " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                f.write(str(self.test_index[i]) + " " + str(predict_pro[i]) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        predict_y = self.nn_model()
        self.write_result(predict_pro=predict_y)
        logger.info("NN_model finished")


if __name__ == "__main__":
    train_path = "../../data/format/all_field_one_hot.csv"
    test_path = "../../data/format/test/all_field_one_hot.csv"
    raw_train_path = "../../data/raw/train_data.csv"
    raw_test_path = "../../data/raw/test_data.csv"
    # #  LR_model
    # LRmodel = LR(train_path=train_path, test_path=test_path, raw_train_path=raw_train_path, raw_test_path=raw_test_path)
    # LRmodel.run()
    # # Bayes_model
    # Bayesmodel = Bayes(train_path=train_path, test_path=test_path, raw_train_path=raw_train_path,
    #                          raw_test_path=raw_test_path)
    # Bayesmodel.run()

    # RandomTree_model
    RandomTreemodel = RandomTree(train_path=train_path, test_path=test_path, raw_train_path=raw_train_path,
                       raw_test_path=raw_test_path)
    RandomTreemodel.run()
    # GTB_model
    GTBmodel = GTB(train_path=train_path, test_path=test_path, raw_train_path=raw_train_path,
                     raw_test_path=raw_test_path)
    GTBmodel.run()
    # NeuralNetwork_model
    NeuralNetworkmodel = NeuralNetwork(train_path=train_path, test_path=test_path, raw_train_path=raw_train_path,
                               raw_test_path=raw_test_path)
    NeuralNetworkmodel.run()
