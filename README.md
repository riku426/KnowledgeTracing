# KnowledgeTracing

## Introduction

Some implementations of knowledge tracing with pytorch

1. DKT, paper: <https://arxiv.org/abs/1506.05908>, reference: <https://github.com/chsong513/DeepKnowledgeTraccing-DKT-Pytorch>

## Usage

```bash
# at the root path
# python -m DKT.evaluation.run
 python -m evaluation.run rnn --hidden=128 --length=739
```

## 評価
ただのdkt
0.7575 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=10 --dropout=0.2

困難度をかける
0.7532 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=10 --dropout=0.2

困難度を並列にする
0.7545 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=15 --dropout=0.2
