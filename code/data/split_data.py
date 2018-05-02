# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/21 20:56
@Author  : dengxiongwen
@Email   : dengxiongwen@foxmail.com
@File    : split_data.py
"""

import pandas as pd

all_field_path = '../../data/format_2/all_field_one_hot.csv'
data = pd.read_csv(all_field_path, sep=',')

total = len(data)

for i in range(10):
    print(i)
    if i != 9:
        data_ = data.iloc[int(i / 10 * total): int((i + 1) / 10 * total), :]
    else:
        data_ = data.iloc[int(i / 10 * total):, :]
    data_.to_csv('../../data/format_2/{}_all_field_one_hot.csv'.format(i))

