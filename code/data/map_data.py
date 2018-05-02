#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-20 下午7:39
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : map_data.py
"""

import pandas as pd
import json


def read_data(path):
    return pd.read_table(path, sep=" ")


def map_field(train_data, test_data, path):
    train_data = set(train_data.unique())
    test_data = set(test_data.unique())
    all_property = list(train_data.union(test_data))
    print(len(all_property))
    map_data = {}
    for i in range(len(all_property)):
        map_data[str(all_property[i])] = i
    with open(path, "w", encoding="utf-8") as dump:
        json.dump(map_data, dump)
    print(map_data)
    return path


def update_data(raw_data, field_1, field_2, path_1, path_2):
    with open(path_1, "r", encoding="utf-8") as dump:
        update_field_1 = json.load(dump)
    with open(path_2, "r", encoding="utf-8") as dump:
        update_field_2 = json.load(dump)
    raw_data[field_1] = raw_data[field_1].apply(lambda x: update_field_1[str(x)])
    raw_data[field_2] = raw_data[field_2].apply(lambda x: update_field_2[str(x)])
    return raw_data


def write_data(raw_data, path):
    raw_data.to_csv(path, sep=" ")


if __name__ == "__main__":
    train_data = read_data(path="../../data/raw/round1_ijcai_18_train_20180301.txt")
    test_data = read_data(path="../../data/raw/round1_ijcai_18_test_b_20180418.txt")
    train_data_brand = train_data["item_brand_id"]
    train_data_city = train_data["item_city_id"]
    test_data_brand = test_data["item_brand_id"]
    test_data_city = test_data["item_city_id"]
    path_1 = map_field(train_data=train_data_brand, test_data=test_data_brand,
                       path="../../data/raw/item_brand_id_2.json")
    path_2 = map_field(train_data=train_data_city, test_data=test_data_city,
                       path="../../data/raw/item_city_id_2.json")
    train_data = update_data(train_data, field_1="item_brand_id", field_2="item_city_id", path_1=path_1, path_2=path_2)
    test_data = update_data(test_data, field_1="item_brand_id", field_2="item_city_id", path_1=path_1, path_2=path_2)
    write_data(train_data, path="../../data/raw/new_train_data.csv")
    write_data(test_data, path="../../data/raw/new_test_data.csv")
