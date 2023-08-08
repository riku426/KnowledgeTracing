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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class EERNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device, dropout):
        super(EERNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(256+96,             #(input_dim) or 438
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.rnn2 = nn.LSTM(265,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = device
        self.iput_dim = input_dim
        self.fc1 = nn.Linear(267, 248) #(259, 248) or (267, 248)
        self.drop1 = nn.Dropout(dropout)
        self.tanh2 = nn.Tanh()
        self.fc2 = nn.Linear(150+96+96, 248)
        
        self.z1 = nn.Linear(256+9, 9)
        self.tanz1 = nn.Tanh()
        self.z2 = nn.Linear(256+9, 256)
        self.tanz2 = nn.Tanh()
        
        self.skill_emb = nn.Embedding(125, 256)
        self.skill_emb.weight.data[-1]= 0
        
        self.ans_emb = nn.Embedding(2, 96)
        self.ans_emb.weight.data[-1]= 0
        
        self.diff_emb = nn.Embedding(12, 96)
        self.diff_emb.weight.data[-1]= 0
        
        self.abili_emb = nn.Embedding(9, 96)
        self.abili_emb.weight.data[-1]= 0
        
        self.attention_dim = 256
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        
    def _get_next_pred(self, res, skill):
        
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    
    def attention_module(self, lstm_output):
        
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output

        

    def forward(self, x):  # shape of input: [batch_size, length, questions * 2]
        
        # V3
        skill = []
        dif = []
        abili = []
        ans = []
        
        new_x = []
        difficulty = []
        abilitys = []
        for i in range(x.shape[0]):
            new_x_2 = []
            difficulty_2 = []
            ability_2 = []
            skill_2 = []
            dif_2 = []
            abili_2 = []
            ans_2 = []
            for j in range(x.shape[1]):
                diff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ability = [0, 0, 0, 0, 0, 0, 0, 0]
                new_x_2.append(x[i][j][:248])
                diff[int(x[i][j][-1])] = 1
                ability[int(x[i][j][-2])] = 1
                difficulty_2.append(torch.Tensor(diff))
                ability_2.append(torch.Tensor(ability))
                skill_2.append(int(x[i][j][-3]))
                dif_2.append(int(x[i][j][-1]))
                abili_2.append(int(x[i][j][-2]))
                ans_2.append(int(x[i][j][-4]))
            new_x_2 = torch.stack(new_x_2, dim=0)
            difficulty_2 = torch.stack(difficulty_2, dim=0)
            ability_2 = torch.stack(ability_2, dim=0)
            
            skill_2 = torch.tensor(skill_2)
            dif_2 = torch.tensor(dif_2)
            abili_2 = torch.tensor(abili_2)
            ans_2 = torch.tensor(ans_2)
            new_x.append(new_x_2)
            difficulty.append(difficulty_2)
            abilitys.append(ability_2)
            skill.append(skill_2)
            dif.append(dif_2)
            abili.append(abili_2)
            ans.append(ans_2)
        new_x = torch.stack(new_x, dim=0)
        difficulty = torch.stack(difficulty, dim=0)
        abilitys = torch.stack(abilitys, dim=0)
        skill = torch.stack(skill, dim=0)
        dif = torch.stack(dif, dim=0)
        abili = torch.stack(abili, dim=0)
        ans = torch.stack(ans, dim=0)
        
        
        skill_emb = self.skill_emb(skill)
        diff_emb = self.diff_emb(dif)
        abili_emb = self.diff_emb(abili)
        ans_emb = self.ans_emb(ans)
        
        skill_answer=torch.cat((skill_emb,ans_emb), 2)
        answer_skill=torch.cat((ans_emb,skill_emb), 2)
        
        answer=ans.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_emb=torch.where(answer==1, skill_answer, answer_skill)
        
        new_x = torch.cat([skill_answer_emb], dim=2)
        
        new_x = self.tanh(new_x)
        
        
        # new_x = torch.cat([new_x, difficulty, abilitys], dim=2) # [64, 739, 258]
        # new_x = self.tanh(self.fc1(new_x))
        
        # shape: [num_layers * num_directions, batch_size, hidden_size]
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)  # shape: [num_layers * num_directions, batch_size, hidden_size]
        # hidden state
        h_0 = Variable(torch.zeros(self.layer_dim, new_x.size(0), self.hidden_dim))
        # cell state
        c_0 = Variable(torch.zeros(self.layer_dim, new_x.size(0), self.hidden_dim))

        
        out, _  = self.rnn(new_x, (h_0, c_0))  # shape of out: [batch_size, length, hidden_size]
        
        
        # z1 = self.z1(torch.cat([abilitys, out], dim=2))
        # z1 = self.tanz1(z1)
        # z2 = self.z2(torch.cat([abilitys, out], dim=2))
        # z2 = self.tanz2(z2)
        
        # ht = torch.cat([z1 * abilitys, z2 * out], dim=2)
        
        # new_out, _ = self.rnn2(ht)
        
                
        
        out=self.attention_module(out)
        res = self.sig(self.fc(out))

        res = res[:, :-1, :]
        
        
        # new_out = torch.cat([out, abilitys], dim=2)
        # new_out = self.tanh2(self.fc2(new_out))
        
        # hidden2 = self.init_hidden(new_out.size(0))  # shape: [num_layers * num_directions, batch_size, hidden_size]
        
        # new_out, hidden2 = self.rnn2(new_out, hidden2)
        
        
        # res = self.sig(self.fc(new_out))
        return res

        
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)  # shape: [num_layers * num_directions, batch_size, hidden_size]
        # out, hn = self.rnn(x)  # shape of out: [batch_size, length, hidden_size]
        # res = self.sig(self.fc(out))  # shape of res: [batch_size, length, question]
        # return res
