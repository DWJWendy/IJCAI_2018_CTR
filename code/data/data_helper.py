#!/usr/bin/env python
# -*- encoding:utf-8 -*-
"""
@author:毛毛虫_Wendy
@license:(c) Copyright 2017-
@contact:dengwenjun818@gmail.com
@file:data_helper.py
@time:18-4-3 上午8:55
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
import logging
from functools import reduce
from collections import defaultdict
import time
import datetime
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(lineno)d - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


class DataPreprocess(object):
    """
    Target: to clean the raw data and let the code user obtain the data with good format(One Hot Representation).

    Data Description:

    The data of the competition consists of four fields, i.e. user field, product field, context field and shop field.

    The user field contains 5 features -- user_id, user_gender_id, user_age_level, user_occupation_id, user_star --
    except user_id, all these features can directly transform to one hot format._level

    The product field contains 9 features -- item_id, item_category_list, item_property_list, item_brand_id, item_city_id,
    item_sales_level, item_collected_level and item_pv_level. Except item_id, item_category_list and item_property_list,
    all features can be directly transform to one hot format. As for item_category_list and item_property_list, we need
    to do something more.
    The feature item_category_list, we need to first split the string into a list, AND!!!!!!!!!!!!!!!
    The feature item_property_list, we split the string into a list as well, every unique element(property) we build a
    new feature for it using one hot representation.

    The context field contains 4 features -- context_id, context_timestamp, context_page_id, predict_category_property.
    Only context_page_id can be transform into one-hot representation. About context_timestamp, we can first transform
    it into levels, for example, every three hours can be a level.

    """

    def __init__(self, path, format_data_path):

        self.format_data_path = format_data_path

        self.raw_data = self.read_data(path)
        self.raw_data.to_csv('../../data/check_data_2.csv')
        self.modify_features = {'string': ['item_category_list',
                                           'item_property_list'],
                                'predict': ['predict_category_property'],
                                'time': ['context_timestamp'],
                                'continuous': ['shop_review_positive_rate',
                                               'shop_score_service',
                                               'shop_score_delivery',
                                               'shop_score_description']}
        self.data_id = {'product': 'item_id', 'user': 'user_id', 'context': 'context_id', 'shop': 'shop_id',
                        "instance": "instance_id"}
        self.field_features = {"product": ["item_brand_id", "item_city_id", "item_price_level",
                                           "item_sales_level", "item_collected_level", "item_pv_level"],
                               "user": ["user_gender_id", "user_age_level", "user_occupation_id",
                                        "user_star_level"],
                               "context": ["context_page_id"],
                               "shop": ["shop_review_num_level", "shop_star_level"]}
        self.field_dict = defaultdict(list)
        self.one_hot_model = OneHotEncoder()
        self.weekends = [4, 5, 6]
        self.holidays = ["12.31", "1.1", "1.2", "2.12", "2.13", "2.14", '3.1', "3.6", "3.7", "3.8", '3.14',
                         '4.5', "5.1", "5.2", "5.3", "5.4", '5.19', '5.20', '5.21', "5.31", "6.1", '6.16',
                         '6.17', '6.18', '9.1', "9.10", "10.1", "10.2", "10.3", "10.4", "10.5", "10.6", "10.7",
                         "11.9", "11.10", "11.11", "12.10", "12.11", "12.12", "12.23", "12.24", "12.25"]

    @staticmethod
    def read_data(path):
        """
        方法：读取全部csv数据
        :param path: 数据的地址
        :return: 返回读取数据（数据形式：数据框）
        """
        # print(datetime.datetime.now(), 'reading data...')
        logger.info('reading data...')
        raw_data = pd.read_table(path, sep=' ')
        for col in raw_data.columns:
            if len(raw_data[raw_data[col] == -1]) > 0:
                values = raw_data[col]
                min_value = values.min()
                raw_data[col] += (- min_value)
            else:
                pass
        return raw_data

    def split_feature(self):
        """
        string
        方法：根据属性值,分离多个特征,每个属性对应的特征形成list格式
        :param feature: 选择指定属性,然后分开多个特征,原数据为类型：189293;2378172;18972419;
        然后改成一个list列表[189293,2378172,18972419]
        :return: 返回新的数据split_data
        """
        data = self.raw_data[self.modify_features['string']]
        data_split_data = pd.DataFrame()
        for attr in data.columns:
            logger.info(attr + ' ing...')
            col_data = data[attr].apply(lambda string: set([num for num in string.split(';')]))
            data_split_data = pd.concat([data_split_data, col_data], axis=1)
        return data_split_data

    def split_predict_feature(self):
        """
        process the data of the feature named 'predict_category_property'
        :return:
        """
        data = self.raw_data[self.modify_features['predict']]
        data_split_predict = pd.DataFrame()
        for i, attr in enumerate(data.columns):
            col_data = data[attr].apply(lambda string: string.split(';'))
            predict_category_data = col_data.apply(lambda lst: set([item.split(':')[0] for item in lst]))
            predict_category_data.name = 'predict_category'
            predict_property_data = col_data.apply(self.split_predict_property)
            predict_property_data.name = 'predict_property'
            data_split_predict = pd.concat([data_split_predict, predict_category_data,
                                            predict_property_data], axis=1)
        return data_split_predict

    @staticmethod
    def split_predict_property(str_property):
        output = []
        for item in str_property:
            pro = item.split(':')
            if len(pro) < 2 or pro == '-1':
                continue
            else:
                pro = pro[1].split(',')
            output += pro
        output = list(filter(lambda x: True if x != '-1' else False, output))
        return set(output)

    @staticmethod
    def convert_interval_time(hour, size=3):
        """
        方法：把一天24小时按照力粒度为size大小进行分割
        :param hour:
        :param size:
        :return:
        """
        interval_time = [list(range(i, i + size)) for i in range(0, 24, size)]
        interval_time_factor = {}
        for i in range(len(interval_time)):
            interval_time_factor[i] = interval_time[i]
        for factor, h in interval_time_factor.items():
            if hour in h:
                return factor
            else:
                pass

    def time_to_factor(self):
        """
        方法：将时间戳转为区间因子
        tips: add two features, is_weekend and is_holiday
        :param data: 输入带时间戳的数据,
        :param size:将原始时间戳转化为粒度为size个小时一个因子,默认为size=3
        :return: 对于时间戳转为化为区间因子
        """
        data_time = self.raw_data[self.modify_features["time"]]
        data_time["format_time"] = data_time["context_timestamp"].apply(lambda x: time.localtime(x))
        convert_data_time = pd.DataFrame(columns=["interval_time", "is_weekends", "is_holidays"])
        convert_data_time["is_weekends"] = \
            data_time["format_time"].apply(lambda x: 1 if x[6] in self.weekends else 0)
        convert_data_time["is_holidays"] = \
            data_time["format_time"].apply(lambda x: 1 if str(x[1]) + "." + str(x[2]) in self.holidays else 0)
        convert_data_time["interval_time"] = \
            data_time["format_time"].apply(lambda x: self.convert_interval_time(hour=x[3]))
        return convert_data_time

    def continuous_var_to_factor(self, size=0.01):
        """
        方法：将连续数据按照粒度为size进行转化
        :param data_continuous: 输入带有连续变量的数据
        :param properity: 指定属性
        :param size： 粒度默认为0.01
        :return:
        """
        data_continuous = self.raw_data[self.modify_features['continuous']]
        data_continuous = (data_continuous / size).round()
        data_continuous = data_continuous.astype('int64')
        return data_continuous

    @staticmethod
    def combine_new_data(*dfs):
        """
        :return: 将所有属性的数据组合在一起返回new_data
        """
        return pd.concat(dfs, axis=1)

    def statistics(self, field, one_hot_data, type='value'):
        """
        方法：将new_data总数据每个属性的one_hot维度进行统计
        :param new_data: pd.DataFrame,
        :param type: str, default 'value', alternative 'lst'. Aimed at different forms, we can use this parameter to
        specify the method of statistics.
        :return: 返回字典{"属性1":"dim1","属性2":"dim2"}
        """
        self.field_dict[field] = list(one_hot_data.columns)

    def one_hot_represent(self, new_data, data_type='value'):
        """

        :param new_data: pd.DataFrame. The data used to transform to the one-hot form.
        :param data_type: str, default 'value', alternative 'lst'. Aimed at different forms, we can use this parameter to
        specify the method of handling.
        :return:
        """
        if data_type == 'value':
            df_one_hot = pd.DataFrame()
            for col in new_data.columns:
                # print(col)
                one_hot_matrix = self.one_hot_model.fit_transform(new_data[[col]])
                feature_names = [col + str(attr) for attr in self.one_hot_model.active_features_]
                one_hot_matrix = pd.DataFrame(one_hot_matrix.toarray(), columns=feature_names)
                df_one_hot = pd.concat([df_one_hot, one_hot_matrix], axis=1)
            return df_one_hot
            # return pd.DataFrame(df_one_hot.toarray(), columns=feature_names)
        elif data_type == 'lst':
            cols = new_data.columns
            df_one_hot = pd.DataFrame()
            for col in cols:
                # print(col)
                data = new_data[col]
                all_values = list(reduce(lambda x, y: x | y, data, set()))
                one_hot_dim = len(all_values)
                one_hot_matrix = []
                for line in data:
                    one_hot_vec = np.zeros(one_hot_dim)
                    for value in line:
                        one_hot_vec[all_values.index(value)] = 1
                    one_hot_matrix.append(one_hot_vec)
                one_hot_matrix = pd.DataFrame(one_hot_matrix, columns=all_values)
                df_one_hot = pd.concat([df_one_hot, one_hot_matrix], axis=1)
            return df_one_hot
        else:
            raise ValueError('Can\'t recognize the type, please enter \'value\' or \'lst\'')

    def is_match(self, df_product, df_predict):
        """

        :param df_product:
        :param df_predict:
        :return:
        """
        df_data = pd.concat([df_product, df_predict], axis=1)
        df_match = pd.DataFrame(columns=['is_match_category', 'is_match_property'])
        df_match['is_match_category'] = df_data.apply(self.is_intersect_tuple, axis=1, args=('category',))
        df_match['is_match_property'] = df_data.apply(self.is_intersect_tuple, axis=1, args=('property',))
        return df_match

    @staticmethod
    def cal_ent(*prob):
        ent = []
        for num in prob[0]:
            if num == 0:
                ent.append(0)
            else:
                ent.append(-1 * num * np.log2(num))
        return sum(ent)

    def cal_info_gain(self, data, independent_variable, dependent_variable):
        total = len(data)
        data = data[[independent_variable, dependent_variable]]
        dep_values, inde_values = data[dependent_variable].unique(), data[independent_variable].unique()
        # prob_dep = [len(data[data[dependent_variable] == value]) / total for value in dep_values]
        # ent1 = self.cal_ent(prob_dep)
        prob_indp = [len(data[(data[independent_variable] == ind_var) & (data[dependent_variable] == value)]) / total
                     for ind_var in inde_values for value in dep_values]
        ent2 = self.cal_ent(prob_indp)
        return ent2

    def feature_selection_with_info_gain(self, data, num_feature=500, feature='item_property_list'):
        print(os.path.exists('../../data/format_2/infomation_gain.txt'))
        if os.path.exists('../../data/format_2/infomation_gain.txt'):
            with open('../../data/format_2/infomation_gain.txt', 'r', encoding='utf-8') as r:
                selected_feature = []
                for i in range(num_feature):
                    line = r.readline().replace('\ufeff', '').strip().split(',')
                    selected_feature.append(line[0])
        else:
            fea_s = list(data.columns)
            fea_s.remove(feature)

            property = []
            for lst in data[feature]:
                for pro in lst:
                    if pro not in property:
                        property.append(pro)

            info_gain = pd.Series()
            for pro in property:
                series = pd.Series([1 if pro in lst else 0 for lst in data[feature]], index=data.index, name=pro)
                concat_data = pd.concat([series, self.raw_data['is_trade']], axis=1)
                info_gain[pro] = self.cal_info_gain(data=concat_data, independent_variable=pro,
                                                    dependent_variable='is_trade')

            info_gain = info_gain.sort_values(ascending=False)
            info_gain.to_csv('../../data/format_2/infomation_gain.txt', encoding='utf-8')
            selected_feature = list(info_gain.index[: num_feature])

        new_feature = []
        for lst in data[feature]:
            new_fea = []
            for pro in lst:
                if pro in selected_feature:
                    new_fea.append(pro)
            new_feature.append(set(new_fea))
        data[feature] = new_feature
        return data

    @staticmethod
    def is_intersect_tuple(record, key_word):
        t1, t2 = 0, 0
        if key_word == 'category':
            t1, t2 = 'item_category_list', 'predict_category'
        elif key_word == 'property':
            t1, t2 = 'item_property_list', 'predict_property'
        else:
            pass
        tuple1, tuple2 = record[t1], record[t2]
        if len(tuple1 & tuple2) > 1:
            return 1
        else:
            return 0

    def run(self):
        """
        方法：允许数据预处理中的方法
        After run the initial method, we need to run this method to complete the data cleaning.
        The idea is as follows:
        1. Extract the data from self.data according to self.modify_features['string'], then use the method --
        self.split_feature -- to split the string, change the value from string-type into list-type
        2. Extract the data from self.data according to self.modify_features['time'], do the method -- self.time_to_factor
        3. Extract the data from self.data according to self.modify_features['continuous'], do the method --
        self.continuous_var_to_factor
        4. Do the One-Hot transformation
        5. Concatenate the result from the procedure 2, 3 and 4.
        :return:
        """
        df_split_data = self.split_feature()
        df_split_predict = self.split_predict_feature()
        df_match = self.is_match(df_product=df_split_data, df_predict=df_split_predict)
        logger.info('正在利用信息增益进行特征选择，对item_property_list进行...')
        df_split_data = self.feature_selection_with_info_gain(data=df_split_data,
                                                              num_feature=500,
                                                              feature='item_property_list')
        logger.info('正在进行one-hot...')
        df_split_data = self.one_hot_represent(new_data=df_split_data, data_type='lst')
        df_time = self.one_hot_represent(new_data=self.time_to_factor())
        df_continuous_value = self.one_hot_represent(new_data=self.continuous_var_to_factor())
        logger.info("finish to deal with especial features...")

        # represent the rest data of each field
        df_user_rest = self.one_hot_represent(self.raw_data[self.field_features["user"]])
        df_shop_rest = self.one_hot_represent(self.raw_data[self.field_features["shop"]])
        df_context_rest = self.one_hot_represent(self.raw_data[self.field_features["context"]])
        df_product_rest = self.one_hot_represent(self.raw_data[self.field_features["product"]])

        # combine the field of one-hot data
        logger.info("finish to deal with the rest features...")
        df_field_user = df_user_rest
        df_field_shop = self.combine_new_data(df_shop_rest, df_continuous_value)
        df_field_context = self.combine_new_data(df_context_rest, df_time, df_match)
        df_field_product = self.combine_new_data(df_product_rest, df_split_data)
        df_all_field = self.combine_new_data(df_field_user, df_field_shop, df_field_context, df_field_product)

        # add the index of each field
        logger.info("finish to add the features of index...")
        df_field_user.index = self.raw_data[self.data_id['user']]
        df_field_shop.index = self.raw_data[self.data_id['shop']]
        df_field_context.index = self.raw_data[self.data_id['context']]
        df_field_product.index = self.raw_data[self.data_id['product']]
        df_all_field.index = self.raw_data[self.data_id["instance"]]

        # write data to the format of csv
        logger.info("write the each field of one-hot data to csv ...")
        df_field_user.to_csv(self.format_data_path + 'user_field_one_hot.csv')
        df_field_shop.to_csv(self.format_data_path + 'shop_field_one_hot.csv')
        df_field_context.to_csv(self.format_data_path + 'context_field_one_hot.csv')
        df_field_product.to_csv(self.format_data_path + 'product_field_one_hot.csv')
        df_all_field.to_csv(self.format_data_path + "all_field_one_hot.csv")
        return df_field_user, df_field_product, df_field_context, df_field_shop


class SVDReduce(object):
    """
    The class is used to reduce the dimension of the data outputed from the class DataPreprocess with SVD method.
    """

    def __init__(self, data, dimension=500):
        """
        Initialize the class with the parameters.
        :param data: pd.DataFrame, the output data from the class DataPreprocess.
        :param dimension: int, default 500. To specify the output dimension.
        """
        self.data = data
        self.target_dim = dimension
        self.format_data_path = '../../data/format_2/'
        self.field = ['user', 'product', 'context', 'shop']
        # self.field = ['product']

    def judge(self, data):
        """
        Abandon
        方法：判读大领域的维度
        标准维度,判断：不足补零,大于转为svd()
        :return:
        """
        logger.info("judge the dimension...")
        field_matrix_shape = data.shape
        dimension = field_matrix_shape[1]
        if dimension > self.target_dim:
            return True
        else:
            return False

    def svd(self, field_matrix):
        """
        方法：对大的领域数据进行降维
        :param field_matrix: list(2d) or np.array, 每一行(list)表示一条record
        :return: 返回领域的降维矩阵
        """
        logger.info("use svd to reduce the dimension")
        indices = field_matrix.index
        fm = field_matrix
        field_matrix = np.array(field_matrix)
        field_matrix_dim = field_matrix.shape
        print(field_matrix_dim)

        # 对维度进行判断是否需要降维
        if field_matrix_dim[1] <= self.target_dim:
            logger.info('Filed_matrix_dim if smaller than the target, no need to perform reduction, thus we'
                        'only add extra zero element to make up the dimension.')
            dim_make_up = self.target_dim - field_matrix_dim[1]
            matrix_make_up = np.zeros([field_matrix_dim[0], dim_make_up])
            matrix_make_up = pd.DataFrame(matrix_make_up, index=indices)
            return pd.concat([fm, matrix_make_up], axis=1)
        else:
            svd = TruncatedSVD(n_components=self.target_dim)
            return pd.DataFrame(svd.fit_transform(field_matrix), index=indices)

    def run(self):
        """
        1. Extract the one-hot-form data from the self.new_data_one_hot according to the field-instruction.
        2. Based on the given self.target_dimension, judge the field matrix whether satisfy the dimension requirement.
        3. If so, do the svd method, else add extra zero element to achieve the self.target_dimension.
        """
        output_matrix = []
        for i, field_data in enumerate(self.data):
            # field_data = self.split_field(field=item)
            svd_matrix = self.svd(field_matrix=field_data)
            svd_matrix.to_csv(self.format_data_path + 'svd_' + self.field[i] + '.csv')
            output_matrix.append(svd_matrix)
        return output_matrix


if __name__ == '__main__':
    logger.info("train_data representing...")
    preprocess_tool = DataPreprocess(path="../../data/raw/new_train_data.csv",
                                     format_data_path="../../data/format_2/")
    data = preprocess_tool.run()
    svd_model = SVDReduce(data=data, dimension=500)
    svd_model.run()
    logger.info("test_data representing...")
    preprocess_tool = DataPreprocess(path="../../data/raw/new_test_data.csv",
                                     format_data_path="../../data/format_2/test/")
    data = preprocess_tool.run()
    svd_model = SVDReduce(data=data, dimension=500)
    svd_model.run()
