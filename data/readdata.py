# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-08 18:46:52
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 00:51:05
import numpy as np
import itertools


class DataReader():
    def __init__(self, train_path, test_path, maxstep, numofques):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques

    # def getData(self, file_path):
    #     data = []
    #     with open(file_path, 'r') as file:
    #         for len, skill, ans in itertools.zip_longest(*[file] * 3):
    #             len = int(len.strip().strip(','))
    #             skill = [int(q) for q in skill.strip().strip(',').split(',')]
    #             ans = [int(a) for a in ans.strip().strip(',').split(',')]
    #             slices = len//self.maxstep + (1 if len % self.maxstep > 0 else 0)
    #             for i in range(slices):
    #                 temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
    #                 if len > 0:
    #                     if len >= self.maxstep:
    #                         steps = self.maxstep
    #                     else:
    #                         steps = len
    #                     for j in range(steps):
    #                         if ans[i*self.maxstep + j] == 1:
    #                             temp[j][skill[i*self.maxstep + j]] = 1
    #                         else:
    #                             temp[j][skill[i*self.maxstep + j] + self.numofques] = 1
    #                     len = len - self.maxstep
    #                 data.append(temp.tolist())
    #         print('done: ' + str(np.array(data).shape))
    #     return data
    def getData(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for len, skill, ques, ans in itertools.zip_longest(*[file] * 4):
                divided_len = len.strip().strip(',').split(',')
                len = int(divided_len[0])
                problem_len = 20
                if len > problem_len:
                    student = int(divided_len[1])
                    skill = [int(q) for q in skill.strip().strip(',').split(',')]
                    ques = [int(q) for q in ques.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                    slices = len//self.maxstep + (1 if len % self.maxstep > 0 else 0)
                    for i in range(slices):
                        temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                        if len > 0:
                            if len >= self.maxstep:
                                steps = self.maxstep
                            else:
                                steps = len
                            for j in range(steps):
                                if ans[i*self.maxstep + j] == 1:
                                    temp[j][skill[i*self.maxstep + j]] = 1
                                else:
                                    temp[j][skill[i*self.maxstep + j] + self.numofques] = 1
                            len = len - self.maxstep
                        data.append(temp.tolist())
            print('done: ' + str(np.array(data).shape))
        return data

    def getTrainData(self):
        print('loading train data...')
        trainData = self.getData(self.train_path)
        return np.array(trainData)

    def getTestData(self):
        print('loading test data...')
        testData = self.getData(self.test_path)
        return np.array(testData)
