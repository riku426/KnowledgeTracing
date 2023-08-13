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
# dkvmn
DKVMN
0.731

# EERNN
EERNNA
0.7726

EERNN + 困難度
0.7800

EERNNM + 困難度
0.7647

# dkt
ただのdkt
0.7586 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=15 --dropout=0.2

困難度をかける
0.7532 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=10 --dropout=0.2

困難度を並列にたす
0.7618 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=15 --dropout=0.2

困難度とskillをembeddingにしたものを横に足す v3
0.7620 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=15 --dropout=0.2

困難度とability_profileとskillをembeddingにしたものを横に足す v3
0.7630 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=15 --dropout=0.2

困難度とskillをembeddingにしたものを横に足して、attentionを加える
0.7661 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=15 --dropout=0.2

困難度とability_profileとskillをembeddingしたものを横に足して、attentionを加える
0.7698 python -m evaluation.run rnn --hidden=256 --length=739 --epochs=15 --dropout=0.2 --dataset=assist2009

## assist
python -m evaluation.run eernn --hidden=256 --length=739 --epochs=15 --dropout=0.2 --dataset=assist2009


## algebra
python -m evaluation.run eernn --hidden=256 --length=200 --epochs=40 --dropout=0.2 --questions=138 --bs=32 --dataset=algebra


