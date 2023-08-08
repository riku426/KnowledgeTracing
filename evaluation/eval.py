# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-09 13:42:11
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:33:06
import tqdm
import torch
import logging

import torch.nn as nn
from sklearn import metrics

logger = logging.getLogger('main.eval')


def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.detach().cpu().numpy(),
                          torch.round(prediction).detach().cpu().numpy())
    recall = metrics.recall_score(ground_truth.detach().cpu().numpy(),
                                  torch.round(prediction).detach().cpu().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().cpu().numpy(),
        torch.round(prediction).detach().cpu().numpy())
    logger.info('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' +
                str(recall) + ' precision: ' + str(precision))
    print('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) +
          ' precision: ' + str(precision))


class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device
        self.num_of_diff = 11

    def forward(self, pred, batch):
        loss = 0
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)
        for student in range(pred.shape[0]):
            # v3
            batch = torch.where(batch >= 1, 1.0, 0.0)
            # delta = batch[student][:, 0:self.num_of_questions] + batch[
            #     student][:, self.num_of_questions:]  # shape: [length, questions]
            # delta = batch[student][:, 0:self.num_of_questions] + batch[
            #     student][:, self.num_of_questions:self.num_of_questions*2] + batch[
            #     student][:, self.num_of_questions*2:self.num_of_questions*3] + batch[
            #     student][:, self.num_of_questions*3:self.num_of_questions*4] + batch[
            #     student][:, self.num_of_questions*4:self.num_of_questions*5] + batch[
            #     student][:, self.num_of_questions*5:self.num_of_questions*6] + batch[
            #     student][:, self.num_of_questions*6:self.num_of_questions*7] + batch[
            #     student][:, self.num_of_questions*7:self.num_of_questions*8] + batch[
            #     student][:, self.num_of_questions*8:self.num_of_questions*9] + batch[
            #     student][:, self.num_of_questions*9:self.num_of_questions*10] + batch[
            #     student][:, self.num_of_questions*10:self.num_of_questions*11] + batch[
            #     student][:, self.num_of_questions*11:self.num_of_questions*12] + batch[
            #     student][:, self.num_of_questions*12:self.num_of_questions*13] + batch[
            #     student][:, self.num_of_questions*13:self.num_of_questions*14] + batch[
            #     student][:, self.num_of_questions*14:self.num_of_questions*15] + batch[
            #     student][:, self.num_of_questions*15:self.num_of_questions*16] + batch[
            #     student][:, self.num_of_questions*16:self.num_of_questions*17] + batch[
            #     student][:, self.num_of_questions*17:self.num_of_questions*18] +  batch[
            #     student][:, self.num_of_questions*18:self.num_of_questions*19] + batch[
            #     student][:, self.num_of_questions*19:self.num_of_questions*20] + batch[
            #     student][:, self.num_of_questions*20:self.num_of_questions*21] + batch[
            #     student][:, self.num_of_questions*21:]    # shape: [length, questions]
            # v3
            delta = batch[student][:, 0:self.num_of_questions] + batch[
                student][:, self.num_of_questions:248]
            # ここまで修正した。
            # ToDo
            # 上のerrorを修正する
            # a = の部分も上と同じように修正
            # その後はまたerrorのでばっく
            temp = pred[student][:self.max_step - 1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step - 1)]],
                                 dtype=torch.long, device=self.device)
            p = temp.gather(0, index)[0]
            # v3
            batch = torch.where(batch >= 1, 1.0, 0.0)
            # a = (((batch[student][:, 0:self.num_of_questions] -
            #        batch[student][:, self.num_of_questions:]).sum(1) + 1) // 2)[1:]
            # a = (((batch[student][:, 0:self.num_of_questions] - batch[
            #     student][:, self.num_of_questions:self.num_of_questions*2] + batch[
            #     student][:, self.num_of_questions*2:self.num_of_questions*3] - batch[
            #     student][:, self.num_of_questions*3:self.num_of_questions*4] + batch[
            #     student][:, self.num_of_questions*4:self.num_of_questions*5] - batch[
            #     student][:, self.num_of_questions*5:self.num_of_questions*6] + batch[
            #     student][:, self.num_of_questions*6:self.num_of_questions*7] - batch[
            #     student][:, self.num_of_questions*7:self.num_of_questions*8] + batch[
            #     student][:, self.num_of_questions*8:self.num_of_questions*9] - batch[
            #     student][:, self.num_of_questions*9:self.num_of_questions*10] + batch[
            #     student][:, self.num_of_questions*10:self.num_of_questions*11] - batch[
            #     student][:, self.num_of_questions*11:self.num_of_questions*12] + batch[
            #     student][:, self.num_of_questions*12:self.num_of_questions*13] - batch[
            #     student][:, self.num_of_questions*13:self.num_of_questions*14] + batch[
            #     student][:, self.num_of_questions*14:self.num_of_questions*15] - batch[
            #     student][:, self.num_of_questions*15:self.num_of_questions*16] + batch[
            #     student][:, self.num_of_questions*16:self.num_of_questions*17] - batch[
            #     student][:, self.num_of_questions*17:self.num_of_questions*18] +  batch[
            #     student][:, self.num_of_questions*18:self.num_of_questions*19] - batch[
            #     student][:, self.num_of_questions*19:self.num_of_questions*20] + batch[
            #     student][:, self.num_of_questions*20:self.num_of_questions*21] - batch[
            #     student][:, self.num_of_questions*21:]).sum(1) + 1) //
            #      2)[1:]
            # v3
            a = (((batch[student][:, 0:self.num_of_questions] -
                   batch[student][:, self.num_of_questions:248]).sum(1) + 1) // 2)[1:]
            for i in range(len(p) - 1, -1, -1):
                if p[i] > 0:
                    p = p[:i + 1]
                    a = a[:i + 1]
                    break
            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
        return loss, prediction, ground_truth


def train_epoch(model, trainLoader, optimizer, loss_func, device):
    model.to(device)
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        pred = model(batch).to(device)
        loss, prediction, ground_truth = loss_func(pred, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, optimizer


def test_epoch(model, testLoader, loss_func, device):
    model.to(device)
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch = batch.to(device)
        pred = model(batch).to(device)
        loss, p, a = loss_func(pred, batch)
        prediction = torch.cat([prediction, p[:]])
        ground_truth = torch.cat([ground_truth, a])
    performance(ground_truth, prediction)
    return prediction, ground_truth
