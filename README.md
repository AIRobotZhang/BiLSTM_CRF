# BiLSTM+CRF for Name Entity Recognition

## Requirements
* Python==3.7.4
* pytorch==1.3.1
* torchtext==0.3.1
* numpy
* tqdm
* gensim

## Input data format
Sample dataset can be available in dataset folder. The data format is as follows('\t' means TAB):

```
藏 书 本 来 就 是 所 有 传 统 收 藏 门 类 中 的 第 一 大 户 ， 只 是 我 们 结 束 温 饱 的 时 间 太 短 而 已 。 \t O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
...
```

## How to run
Train & Dev:
Original training dataset is randomly split into 90% for training and 10% for dev.
注：sgns.renmin.bigram-char 为预训练字向量文件，其维度应与 embedding_size 保持一致。 在 data 目录下，可以通过运行 python word2vec_model.py
train_corpus.txt vector.model sgns.renmin.bigram-char 得到预训练字向量
```
$  python main.py --epoch 100 --lr 1e-3 --train_data_path data/train.tsv --test_data_path data/test.tsv --embedding_path data/sgns.renmin.bigram-char --hidden_size 512 --train_batch_size 128 --embedding_size 300
```
Test:
注： 20_Model_best.pt 为在验证集（训练集的 10%用于验证调参）上的最优模型
```
$  python main.py --epoch 100 --lr 1e-3 --train_data_path data/train.tsv --test_data_path data/test.tsv --embedding_path data/sgns.renmin.bigram-char --hidden_size 512 --train_batch_size 128 --embedding_size 300 --load_model results/20_Model_best.pt
```

More detailed configurations can be found in `main.py`.
