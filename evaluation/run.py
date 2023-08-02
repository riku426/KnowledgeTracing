# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-09 21:50:46
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:20:09
"""
Usage:
    run.py (rnn|sakt) --hidden=<h> [options]

Options:
    --length=<int>                      max length of question sequence [default: 50]
    --questions=<int>                   num of question [default: 124]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 10]
    --cuda=<int>                        use GPU id [default: 0]
    --hidden=<int>                      dimention of hidden state [default: 128]
    --layers=<int>                      layers of rnn or transformer [default: 1]
    --heads=<int>                       head number of transformer [default: 8]
    --dropout=<float>                   dropout rate [default: 0.1]
    --model=<string>                    model type
"""

import os
import random
import logging
import torch

import torch.optim as optim
import numpy as np

from datetime import datetime
from docopt import docopt
# from data.readdata import DataReader
from data.readdata_v2 import DataReader
# from data.readdata_v3 import DataReader
from data.dataloader import getDataLoader
import torch.utils.data as Data

from evaluation import eval


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args = docopt(__doc__)
    length = int(args['--length'])
    questions = int(args['--questions'])
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = args['--cuda']
    hidden = int(args['--hidden'])
    layers = int(args['--layers'])
    heads = int(args['--heads'])
    dropout = float(args['--dropout'])
    if args['rnn']:
        model_type = 'RNN'
    elif args['sakt']:
        model_type = 'SAKT'

    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    handler = logging.FileHandler(
        f'log/{date.year}_{date.month}_{date.day}_{model_type}_result.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(list(args.items()))

    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainLoader, testLoade = getDataLoader(bs, questions, length)

    
    if model_type == 'RNN':
        from model.DKT.RNNModel import RNNModel
        # model = RNNModel(questions * 2, hidden, layers, questions, device)
        ## v2
        model = RNNModel(questions * 2 * 11, hidden, layers, questions, device)
    elif model_type == 'SAKT':
        from model.SAKT.model import SAKTModel
        model = SAKTModel(heads, length, hidden, questions, dropout)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = eval.lossFunc(questions, length, device)

    for epoch in range(epochs):
        print('epoch: ' + str(epoch))
        model, optimizer = eval.train_epoch(model, trainLoader, optimizer,
                                          loss_func, device)
        logger.info(f'epoch {epoch}')
        pred, truth = eval.test_epoch(model, testLoade, loss_func, device)
        
    print(pred.shape)
    print(truth.shape)
    import csv
    pred, truth = eval.test_epoch(model, testLoade, loss_func, device)
    save_truth = truth.tolist()[:49048]
    save_pred = pred.tolist()[:49048]
    with open('dataset/save_truth.csv', 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in save_truth:
            writer.writerow([i])
    with open('dataset/save_pred.csv', 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in save_pred:
            writer.writerow([i])
    handle = DataReader('dataset/assist2009/4_Ass_09_train.csv',
                        'dataset/assist2009/4_Ass_09_test.csv', length,
                        questions)
    dtrain = torch.from_numpy(handle.getTrainData().astype(float)).float()
    trainLoader = Data.DataLoader(dtrain, batch_size=bs, shuffle=False)
    pred, truth = eval.test_epoch(model, trainLoader, loss_func, device)
    save_truth = truth.tolist()[:197027]
    save_pred = pred.tolist()[:197027]
    with open('dataset/train_save_truth.csv', 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in save_truth:
            writer.writerow([i])
    with open('dataset/train_save_pred.csv', 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in save_pred:
            writer.writerow([i])
    
    
    
    


if __name__ == '__main__':
    main()
