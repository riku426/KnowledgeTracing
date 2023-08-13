# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-10 00:29:34
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:14:50
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from collections import defaultdict

device = torch.device("mps" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device, dropout):
        print('input_dim', input_dim)
        super(RNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_dim+11+8,             #(input_dim) or 438
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.rnn2 = nn.LSTM(265,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = device
        self.iput_dim = input_dim
        
    def _get_next_pred(self, res, skill):
        
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
        

    def forward(self, x):  # shape of input: [batch_size, length, questions * 2]
        
        # V3
        
        new_x = []
        difficulty = []
        abilitys = []
        for i in range(x.shape[0]):
            new_x_2 = []
            difficulty_2 = []
            ability_2 = []

            for j in range(x.shape[1]):
                diff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ability = [0, 0, 0, 0, 0, 0, 0, 0]
                new_x_2.append(x[i][j][:self.input_dim])
                diff[int(x[i][j][-1])] = 1
                ability[int(x[i][j][-2])] = 1
                difficulty_2.append(torch.Tensor(diff))
                ability_2.append(torch.Tensor(ability))

            new_x_2 = torch.stack(new_x_2, dim=0)
            difficulty_2 = torch.stack(difficulty_2, dim=0)
            ability_2 = torch.stack(ability_2, dim=0)
            
            new_x.append(new_x_2)
            difficulty.append(difficulty_2)
            abilitys.append(ability_2)

        new_x = torch.stack(new_x, dim=0)
        difficulty = torch.stack(difficulty, dim=0)
        abilitys = torch.stack(abilitys, dim=0)
                
        
        new_x = torch.cat([new_x, difficulty, abilitys], dim=2) # [new_x, difficulty, abilitys]
        
        new_x = self.tanh(new_x)
        
        # shape: [num_layers * num_directions, batch_size, hidden_size]
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)  # shape: [num_layers * num_directions, batch_size, hidden_size]
        # hidden state
        h_0 = Variable(torch.zeros(self.layer_dim, new_x.size(0), self.hidden_dim))
        # cell state
        c_0 = Variable(torch.zeros(self.layer_dim, new_x.size(0), self.hidden_dim))

        
        out, _  = self.rnn(new_x, (h_0, c_0))  # shape of out: [batch_size, length, hidden_size]
        
        res = self.sig(self.fc(out))

        res = res[:, :-1, :]
        
        
        return res
