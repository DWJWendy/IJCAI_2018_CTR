# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
from matplotlib import pyplot as plt
# import interval
from sklearn.decomposition import PCA
from sklearn import preprocessing

class DataStatistical(object):
    """
    用于对数据进行单变量分析，变量可以为名义变量，有序变量，数值变量，字符串变量和具有从属关系的 变量。
    分析过程：
    For nominal and ordinal variables：通过变量的值将数据划分为多个pd.DataFrame，然后分别统计每个pd.DataFrame中
    is_trade=1和is_trade=0的数量，并计算购买率
    For ordinal variables:先将数据利用所给的粒度分组统计后，重复名义变量和有序变量的处理过程
    For string variables: 按照数据的划分方式，将数据转化为one-hot的形式，然后进行二进制编码or自编码器or其他降维处理(???)
    For subordinate variables: 分别对三个粒度的从属关系构建三个属性，然后转化为one-hot形式，编码方式同string变量相同。

    最后输出csv，要求：
    1. 分隔符为','
    2. 编码方式统一使用'utf-8'
    3. 将索引作为一个属性加入pd.DataFrame中，to_csv中的参数index置为False
    """
    def __init__(self, ind_var, var_type='nominal'):
        """
        The initial function.
        :param data: pd.DataFrame, 用于分析的数据，需至少包含两列
        :param ind_var: str, 自变量的变量名，要求与data中的命名一致
        :param var_type: str, default value 'nominal', 表示变量的类型。'nominal'表示名义变量，'ordinal'表示顺序变量，
        'numeric'表示数值变量，'string'表示字符串变量，'subordinate'表示具有从属关系的变量
        :return:
        """
        self.train_data = pd.read_table('../data/raw/round1_ijcai_18_train_20180301.txt',sep=" ")
        self.test_data = pd.read_table('../data/raw/round1_ijcai_18_test_a_20180301.txt',sep=" ")
        self.attr = ind_var
        self.type = var_type

    def nominal_analysis(self, ind_var):
        """
        名义变量分析过程：
        1. 分别统计attr属性值下的is_trade=1和is_trade=0数量
        2. 分别计算每个属性值的购买率
        3. 将索引视为整数，从小到大对整个dataframe进行排序后，输出csv
        :return:
        """
        data = self.raw_data[[ind_var, 'is_trade']]
        ind_var_value = data[ind_var].unique()
        trade_matrix = pd.DataFrame(columns=["trade","no_trade","rate"],index=ind_var_value)

        for ind in ind_var_value:
            trade_matrix.loc[ind,["trade"]] = len(data[(data['is_trade'] == 1) & (data[ind_var] == ind)])
            trade_matrix.loc[ind,["no_trade"]] = len(data[(data['is_trade'] == 0) & (data[ind_var] == ind)])
        trade_matrix["rate"] = trade_matrix["trade"] / (trade_matrix["trade"]+ trade_matrix["no_trade"])
        return trade_matrix

    def ordinal_analysis(self,ind_var):
        """
        直接调用self.nominal_analysis方法即可
        :return:
        """
        trade_matrix = self.nominal_analysis(ind_var=ind_var)
        return trade_matrix

    @staticmethod
    def gen_granularity_interval(size):
        """
        生成粒度划分的区间
        :return:
        """
        a = 0
        b = round(a + 1 / size, 5)
        granularity = [(a, b)]
        for i in range(size - 1):
            a = b
            b = round(a + 1 / size, 5)
            granularity.append((a, b))
        return granularity

    def numeric_analysis(self, ind_var, size=100):
        """
        先将数据利用所给的粒度分组统计后，重复名义变量和有序变量的处理过程
        :return:
        """
        granularity = self.gen_granularity_interval(size)
        data = self.raw_data[[ind_var, "is_trade"]]
        trade_matrix = pd.DataFrame(columns=["trade", "no_trade", "rate"], index=granularity)
        for ind in granularity:
            if min(ind) < data[ind_var].min() or max(ind) > data[ind_var].max():
                continue
            else:
                condition_1 = (data['is_trade'] == 1) & (data[ind_var] >= ind[0]) & (data[ind_var] < ind[1])
                trade_matrix.loc[ind, ["trade"]] = len(data[condition_1])
                condition_2 = (data['is_trade'] == 0) & (data[ind_var] >= ind[0]) & (data[ind_var] < ind[1])
                trade_matrix.loc[ind, ["no_trade"]] = len(data[condition_2])
        trade_matrix['rate'] = trade_matrix['trade'] + trade_matrix['no_trade']
        trade_matrix = trade_matrix[trade_matrix['rate'] > 0]
        trade_matrix['rate'] = trade_matrix['trade'] / (trade_matrix['trade'] + trade_matrix['no_trade'])
        return trade_matrix

  
    def string_analysis(self, ind_var, attr_type=1):
        """
        按照数据的划分方式，将数据转化为one-hot的形式
        :type: type=1 is the second of attributes;type=2 is the mix of attributes
        :return:
        """

        data = self.train_data[[ind_var]]
        ind_var_value = list(map(lambda x: x.split(";"), data[ind_var].unique()))


        #### the single properity
        # index_value = []
        # for temp in ind_var_value:
        #     index_value += temp
        # index_value = list(set(index_value))

        # attr_matrix = pd.DataFrame(index=data.index, columns=index_value)
        # print(attr_matrix)

        # temp = list(map(lambda x: x.split(";"), data[ind_var]))
        #
        # for i in range(len(data.index)):
        #     for attr in temp[i]:
        #         attr_matrix.ix[data.index[i], attr] = 1
        # return attr_matrix

        #### the mixture of properity
        index_value = ind_var_value
        attr_matrix = pd.DataFrame(np.zeros((len(data.index), len(index_value))),
                                   index=data.index, columns=index_value)
        temp = list(map(lambda x: x.split(";"), data[ind_var]))
        for i in range(len(data.index)):
            attr_matrix.ix[data.index[i], temp[i]] = 1
        return attr_matrix


    def subordinate_analysis(self, ind_var, attr_type=1):
        """
        分别对三个粒度的从属关系构建三个属性，然后转化为one-hot形式，编码方式同string变量相同。
        :attr_type: attr_type=1 is the second of attributes;attr_type=2 is the mix of attributes
        :return:
        """
        data = self.train_data[[ind_var]]
        ind_var_value = list(map(lambda x: x.split(";"), data[ind_var].unique()))
        if attr_type == 1:
            index_value = [temp[1] for temp in ind_var_value]
            attr_matrix = pd.DataFrame(np.zeros((len(data.index), len(index_value))),
                                       index=data.index, columns=index_value)
            temp = list(map(lambda x: x.split(";")[1], data[ind_var]))
            for i in range(len(data.index)):
                attr_matrix.ix[data.index[i], temp[i]] = 1
            return attr_matrix

        elif attr_type == 2:
            index_value = [temp[1]+temp[2] for temp in ind_var_value if len(temp)==3]
            attr_matrix = pd.DataFrame(np.zeros((len(data.index), len(index_value))),
                                   index=data.index, columns=index_value)
            temp = list(map(lambda x: x.split(";"), data[ind_var]))
            for i in range(len(data.index)):
                if len(temp[i])==3:
                    attr_matrix.ix[data.index[i], temp[i][1]+temp[i][2]] = 1
                else:
                    pass
            return attr_matrix


    def reduce_dimension(self, data,n_components,copy=True):
        """
        将字符串变量或从属关系变量进行编码降维，二进制编码or自编码器or其他降维处理(???)
        :param n_components: the size of dimension eg:n_componenst =10
        :param data:
        :return:
        """
        pca  = PCA(n_components=n_components,copy=True)
        DataS = pca.fit_transform(data)
        return DataS


    def process(self):
        """
        数据处理部分的主程序，首先利用self.type判断需要进入哪类变量的分析
        然后将数据送入相应的分析方法中
        :return:
        """
        if self.type == 'nominal':
            format_data = self.nominal_analysis(ind_var=self.attr)
        elif self.type == 'ordinal':
            format_data = self.ordinal_analysis(ind_var=self.attr)
        elif self.type == 'string':
            format_data = self.string_analysis()
        elif self.type == 'subordinate':
            format_data = self.subordinate_analysis()
        self.plot(data=format_data, graph_type='line')

    def plot(self, data, graph_type='line'):
        """
        将整理好的数据进行画图
        :param data: pd.DataFrame，经过该对象的方法整理后的数据
        :param name: str, 文件的保存的名称及图像的title
        :param graph_type: str，作图的类型
        :return:
        """
        data = data['rate']
        plt.figure(1, figsize=(10, 10))
        data.plot(kind=graph_type)
        plt.xlabel(self.attr)
        plt.ylabel('purchase')
        plt.title(self.attr + ' _purchase')
        plt.savefig(self.attr + '_purchase_' + graph_type + '.pdf')



