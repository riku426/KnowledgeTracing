# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-08 16:21:21
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 11:47:28
import torch
import torch.utils.data as Data
# from data.readdata import DataReader
# from data.readdata_v2 import DataReader
from data.readdata_v3 import DataReader
from collections import defaultdict

dataset = 'algebra'


def getDataLoader(batch_size, num_of_questions, max_step):
    # handle = DataReader('dataset/assist2009/4_Ass_09_train.csv',
    #                     'dataset/assist2009/4_Ass_09_test.csv', max_step,
    #                     num_of_questions)
    if dataset == 'algebra':
        handle = DataReader('dataset/algebra05/algebra_train.csv',
                        'dataset/algebra05/algebra_test.csv', max_step,
                        num_of_questions)
    elif dataset == 'assist2009':
        handle = DataReader('dataset/assist2009/4_Ass_09_train.csv',
                        'dataset/assist2009/4_Ass_09_test.csv', max_step,
                        num_of_questions)
    dtrain = torch.from_numpy(handle.getTrainData().astype(float)).float()
    dtest = torch.from_numpy(handle.getTestData().astype(float)).float()
    print(2)
    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return trainLoader, testLoader
