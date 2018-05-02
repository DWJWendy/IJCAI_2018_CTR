# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/22 8:45
@Author  : dengxiongwen
@Email   : dengxiongwen@foxmail.com
@File    : cnn_model_predict.py
"""

import tf_cnn
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow as tf
import gc
import logging
import numpy as np

# configure the logging module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


def split_batch_data(data, batch_size=1000):
    data = np.array(data)
    data_batch, batch = [], []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            data_batch += [np.array(batch)]
            batch = []
        else:
            pass
    if len(batch) > 0:
        data_batch.append(batch)
    else:
        pass
    return data_batch

all_field_path = '../../data/format_2/{}_all_field_one_hot.csv'
user_field_file = '../../data/format_2/user_field_one_hot.csv'
product_field_file = '../../data/format_2/product_field_one_hot.csv'
context_field_file = '../../data/format_2/context_field_one_hot.csv'
shop_field_file = '../../data/format_2/shop_field_one_hot.csv'

print('正在载入模型...')
cnn_model = tf_cnn.CNN1(n_input=None, n_output=None, x_shape=None, batch_size=None, load=1)


print('数据处理中...')

train_data, max_length = tf_cnn.data_matrix(all_field_path=all_field_path.format(0), user_file=user_field_file, product_file=product_field_file, context_file=context_field_file, shop_file=shop_field_file)
print(train_data.shape)
test_data = pd.read_csv(all_field_path.format(9), dtype='float').reindex(columns=train_data.columns, fill_value=0)
print(test_data.shape)

label_data = pd.read_table('../../data/raw/train_data.csv', sep=' ', usecols=['is_trade'])['is_trade']
onehot_model = OneHotEncoder()
label_data = onehot_model.fit_transform(label_data.reshape([-1, 1])).toarray()
feature_pos = list(onehot_model.active_features_).index(1)

start = len(label_data) - len(test_data)
labels_data = label_data[start: ]
print(labels_data.shape)

print('正在对测试数据进行测试...')
# cnn_model = tf_cnn.CNN1(n_input=4 * max_length, n_output=2, x_shape=[-1, 4, max_length, 1], batch_size=1000, load=1)

loss_lst = []
test_data, label_data = split_batch_data(test_data), split_batch_data(labels_data)
# print(label_data)
for i, test_batch in enumerate(test_data):
    _, loss = cnn_model.load_model_predict(test_data=test_batch, test_label=label_data[i], mode='test')
    loss_lst.append(loss)
print('validation set func_loss:', np.array(loss_lst).mean())

print('正在对比赛数据进行预测...')
predict_data_path = '../../data/format_2/test/all_field_one_hot.csv'
predict_data = pd.read_csv(predict_data_path)
instance_id = predict_data['instance_id']
predict_data = predict_data.reindex(columns=train_data.columns, fill_value=0)


pred_prob = []
predict_data = split_batch_data(predict_data)
for batch in predict_data:
    pred_test = cnn_model.load_model_predict(test_data=batch, test_label=None, mode='predict')
    for item in pred_test:
        pred_prob.append(item[feature_pos])


fn = open('../../data/result_cnn.txt', 'w', encoding='utf-8')
fn.write('instance_id predicted_score\n')
for i, ind in enumerate(instance_id):
    fn.write(str(ind) + ' ' + str(pred_prob[i]) + '\n')
fn.close()
