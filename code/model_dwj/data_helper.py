#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-19 下午3:19
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : data_helper.py
"""

from sklearn.cross_validation import train_test_split
import pandas as pd
import pickle
import logging

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
        self.predict_x = self.alignment_data().drop("instance_id", axis=1)
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


if __name__ == "__main__":
    train_path = "../../data/format/all_field_one_hot.csv"
    test_path = "../../data/format/test/all_field_one_hot.csv"
    raw_train_path = "../../data/raw/train_data.csv"
    raw_test_path = "../../data/raw/test_data.csv"
    data_results = Data_Preprocess(train_path=train_path, test_path=test_path,
                                   raw_train_path=raw_train_path, raw_test_path=raw_test_path)
    print(data_results.X_train, data_results.X_train, data_results.X_train, data_results.Y_train, data_results.Y_test)
