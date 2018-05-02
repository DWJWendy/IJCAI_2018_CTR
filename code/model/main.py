#!/usr/bin/env python
# -*- encoding:utf-8 -*-
"""
@author:毛毛虫_Wendy
@license:(c) Copyright 2017-
@contact:dengwenjun818@gmail.com
@file:main.py
@time:18-3-28 上午11:28
"""
import random

import pandas as pd
def read_csv(path):
    data = pd.read_csv(path,sep=" ")
    return data

def test_result(path,data):

    with open(path,"w",encoding="utf-8") as f:
        f.write("instance_id"+" "+"predicted_score"+"\r")
        for temp in data:
            f.write(str(temp)+" "+ str(round(random.random(),1))+"\r")

if __name__ == "__main__":
    data = read_csv(path="../data/round1_ijcai_18_train_20180301.csv")

    print(len(set(data["shop_score_description"])))
    #test_result(path="../data/test.txt",data= data["instance_id"])