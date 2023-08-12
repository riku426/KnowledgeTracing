# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-08 18:46:52
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 00:51:05
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
dataset = 'algebra'


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
        
        if dataset == 'algebra':
            self.numofdiffs = 11
            data = []
            df = pd.read_csv('dataset/algebra05/train_data.csv')
            test_df = pd.read_csv('dataset/algebra05/test_data.csv')
            diff_set = defaultdict(int)
            prob = df['problems'].tolist()
            diff = df['problem_difficulty'].tolist()
            for i in range(len(prob)):
                diff_set[prob[i]] = diff[i]
            students = df['student_id'].unique().tolist()
            test_students = test_df['student_id'].unique().tolist()
            ability_set = defaultdict(list)
            cnt = defaultdict(int)
            count = 0
            for s in students:
                ability_set[s] = df[df['student_id'] == s]['ability_profile'].tolist()
            for s in test_students:
                ability_set[s] = test_df[test_df['student_id'] == s]['ability_profile'].tolist()
            # ability_set[3699].append(0)
            
            
            with open(file_path, 'r') as file:
                for length, ques, skill, ans in itertools.zip_longest(*[file] * 4):
                    divided_len = length.strip().strip(',').split(',')
                    length = int(divided_len[0])
                    problem_len = 20
                    if length > problem_len:
                        student = int(divided_len[1])
                        if file_path == 'dataset/algebra05/algebra_train.csv':
                            if student > 2260:
                                continue
                        elif file_path == 'dataset/algebra05/algebra_test.csv':
                            if student > 2840:
                                continue
                        

                        skill = [int(q) for q in skill.strip().strip(',').split(',')]
                        ques = [int(q) for q in ques.strip().strip(',').split(',')]
                        ans = [int(a) for a in ans.strip().strip(',').split(',')]
                        slices = length//self.maxstep + (1 if length % self.maxstep > 0 else 0)

                        for i in range(slices):
                            temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques+4])
                            if length > 0:
                                if length >= self.maxstep:
                                    steps = self.maxstep
                                else:
                                    steps = length
                                for j in range(steps):
                                    
                                    if ans[i*self.maxstep + j] == 1:
                                        temp[j][skill[i*self.maxstep + j]] = 1
                                    else:
                                        temp[j][skill[i*self.maxstep + j] + self.numofques] = 1
                                    try:
                                        if i==0 and j == 0:
                                            temp[j][-2] = 0
                                        else:
                                            temp[j][-2] = ability_set[student][i*self.maxstep + j - 1]
                                    except:
                                        print(student, skill[i*self.maxstep + j])

                                    

                                            
                                    cnt[temp[j][-2]] += 1
                                    temp[j][-1] = diff_set[ques[i*self.maxstep + j]]
                                    temp[j][-3] = skill[i*self.maxstep + j]
                                    temp[j][-4] = ans[i*self.maxstep + j]

                                length = length - self.maxstep
                            data.append(temp)
                print('cnt1', cnt)
                data = np.array(data)
                cnt = defaultdict(int)
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        cnt[data[i][j][-2]] += 1
                print(('cnt', cnt))
                print(data.shape)
                print('done: ' + str(np.array(data).shape))
            return data
        
        elif dataset == 'assist2009':
            self.numofdiffs = 11
            data = []
            df = pd.read_csv('dataset/assist2009/train_data.csv')
            test_df = pd.read_csv('dataset/assist2009/test_data.csv')
            diff_set = defaultdict(int)
            prob = df['problems'].tolist()
            diff = df['problem_difficulty'].tolist()
            for i in range(len(prob)):
                diff_set[prob[i]] = diff[i]
            students = df['student_id'].unique().tolist()
            test_students = test_df['student_id'].unique().tolist()
            ability_set = defaultdict(list)
            cnt = defaultdict(int)
            count = 0
            for s in students:
                ability_set[s] = df[df['student_id'] == s]['ability_profile'].tolist()
            for s in test_students:
                ability_set[s] = test_df[test_df['student_id'] == s]['ability_profile'].tolist()
            ability_set[3699].append(0)
            
            
            with open(file_path, 'r') as file:
                for length, skill, ques, ans in itertools.zip_longest(*[file] * 4):
                    divided_len = length.strip().strip(',').split(',')
                    length = int(divided_len[0])
                    problem_len = 20
                    if length > problem_len:
                        student = int(divided_len[1])
                        if file_path == self.test_path:
                            if student > 3699:
                                continue
                        if file_path == self.train_path:
                            if student > 4057:
                                continue
                        skill = [int(q) for q in skill.strip().strip(',').split(',')]
                        ques = [int(q) for q in ques.strip().strip(',').split(',')]
                        ans = [int(a) for a in ans.strip().strip(',').split(',')]
                        slices = length//self.maxstep + (1 if length % self.maxstep > 0 else 0)
                        for i in range(slices):
                            temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques+4])
                            if length > 0:
                                if length >= self.maxstep:
                                    steps = self.maxstep
                                else:
                                    steps = length
                                for j in range(steps):
                                    
                                    if ans[i*self.maxstep + j] == 1:
                                        temp[j][skill[i*self.maxstep + j]] = 1
                                    else:
                                        temp[j][skill[i*self.maxstep + j] + self.numofques] = 1
                                    try:
                                        if i==0 and j == 0:
                                            temp[j][-2] = 0
                                        else:
                                            temp[j][-2] = ability_set[student][i*self.maxstep + j - 1]
                                    except:
                                        print(i*self.maxstep + j - 1, student)
                                    cnt[temp[j][-2]] += 1
                                    temp[j][-1] = diff_set[ques[i*self.maxstep + j]]
                                    temp[j][-3] = skill[i*self.maxstep + j]
                                    temp[j][-4] = ans[i*self.maxstep + j]
                                length = length - self.maxstep
                            data.append(temp)
                print('cnt1', cnt)
                data = np.array(data)
                cnt = defaultdict(int)
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        cnt[data[i][j][-2]] += 1
                print(('cnt', cnt))
                print(data.shape)
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
