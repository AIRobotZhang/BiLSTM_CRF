# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse
from model import BiLSTM_CRF
from utils import dataset, metrics
from itertools import islice
import copy
from tqdm import tqdm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_iter, dev_iter, epoch, lr, id_label):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    all_loss = 0.0
    model.train()
    ind = 0.0
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        origin_lens = batch.text[1]
        # print(batch.text[1])
        batch_size = text.size()[0]
        target = batch.label[0]
        
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
            origin_lens = origin_lens.cuda()
            
        optim.zero_grad()
        loss = model(text, target, origin_lens)

        loss.backward()
        # clip_gradient(model, 1e-1)
        optim.step()
        # eval_model(model, dev_iter, id_label)
        # p, r, f1, eval_loss = eval_model(model, dev_iter, id_label)
        if idx % 10 == 0:
            logger.info('Epoch:%d, Idx:%d, Training Loss:%.4f', epoch, idx, loss.item())
            # dev_iter_ = copy.deepcopy(dev_iter)
            # p, r, f1, eval_loss = eval_model(model, dev_iter, id_label)
        all_loss += loss.item()
        ind += 1

    p, r, f1, eval_loss = 0.0, 0.0, 0.0, 0.0
    p, r, f1, eval_loss, _ = eval_model(model, dev_iter, id_label)
    # return all_loss/ind
    return all_loss/ind, p, r, f1, eval_loss

def eval_model(model, val_iter, id_label):
    eval_loss = 0.0
    ind = 0.0
    score = metrics.Entity_Score(id_label)
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_iter)):
            text = batch.text[0]
            origin_lens = batch.text[1]
            batch_size = text.size()[0]
            target = batch.label[0]
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
                origin_lens = origin_lens.cuda()
            loss = model(text, target, origin_lens)
            eval_loss += loss.item()
            prediction = model.decode(text, origin_lens) # batch_size*seq_len
            score.update(target, prediction, origin_lens)
            ind += 1
        p, r, f1, all_assess = score.result()

    return p, r, f1, eval_loss/ind, all_assess

def main():
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument("--epoch", default=100, type=int,
                        help="the number of epoches needed to train")
    parser.add_argument("--lr", default=2e-5, type=float,
                        help="the learning rate")
    parser.add_argument("--train_data_path", default=None, type=str,
                        help="train dataset path")
    parser.add_argument("--dev_data_path", default=None, type=str,
                        help="dev dataset path")
    parser.add_argument("--test_data_path", default=None, type=str,
                        help="test dataset path")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="the batch size")
    parser.add_argument("--dev_batch_size", default=64, type=int,
                        help="the batch size")
    parser.add_argument("--test_batch_size", default=64, type=int,
                        help="the batch size")
    parser.add_argument("--embedding_path", default=None, type=str,
                        help="pre-trained word embeddings path")
    parser.add_argument("--embedding_size", default=300, type=int,
                        help="the word embedding size")
    parser.add_argument("--hidden_size", default=128, type=int,
                        help="the hidden size")
    parser.add_argument("--fine_tuning", default=True, type=bool,
                        help="whether fine-tune word embeddings")
    parser.add_argument("--early_stopping", default=15, type=int,
                        help="Tolerance for early stopping (# of epochs).")
    parser.add_argument("--load_model", default=None,
                        help="load pretrained model for testing")
    args = parser.parse_args()

    if not args.train_data_path:
        logger.info("please input train dataset path")
        exit()
    if not (args.dev_data_path or args.test_data_path):
        logger.info("please input dev or test dataset path")
        exit()

    TEXT, LABEL, vocab_size, word_embeddings, train_iter, dev_iter, test_iter, tag_dict = \
                dataset.load_dataset(args.train_data_path, args.dev_data_path, \
                 args.test_data_path, args.embedding_path, args.train_batch_size, \
                                        args.dev_batch_size, args.test_batch_size)

    idx_tag = {}
    for tag in tag_dict:
        idx_tag[tag_dict[tag]] = tag

    model = BiLSTM_CRF(args.embedding_size, args.hidden_size, vocab_size, tag_dict, word_embeddings)
    if torch.cuda.is_available():
        model = model.cuda()

    # cost_test = []
    # start = time.perf_counter()
    # train_dev_size = len(train_iter)
    # train_size = int(train_dev_size*0.9)
    train_data, dev_data = dataset.train_dev_split(train_iter, 0.9)
    # for batch in train_data:
    #     print(batch)
    #     exit()

    # train_data = lambda: islice(train_iter,0,train_size)
    # dev_data = lambda: islice(train_iter,train_size,train_dev_size)
    # train_data = islice(train_iter,0,train_size)
    # dev_data = islice(train_iter,train_size,train_dev_size)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        # p, r, f1, eval_loss, all_assess = eval_model(model, dev_data, idx_tag)
        # logger.info('Eval Loss:%.4f, Eval P:%.4f, Eval R:%.4f, Eval F1:%.4f', \
        #                             eval_loss, p, r, f1)
        p, r, f1, eval_loss, all_assess = eval_model(model,  test_iter, idx_tag)
        logger.info('LOC Test P:%.4f, Test R:%.4f, Test F1:%.4f', \
                all_assess['LOC']['P'], all_assess['LOC']['R'], all_assess['LOC']['F'])
        logger.info('PER Test P:%.4f, Test R:%.4f, Test F1:%.4f', \
                all_assess['PER']['P'], all_assess['PER']['R'], all_assess['PER']['F'])
        logger.info('ORG Test P:%.4f, Test R:%.4f, Test F1:%.4f', \
                all_assess['ORG']['P'], all_assess['ORG']['R'], all_assess['ORG']['F'])
        logger.info('Micro_AVG Test P:%.4f, Test R:%.4f, Test F1:%.4f', \
                                    p, r, f1)
        return 

    best_score = 0.0
    for epoch in range(args.epoch):
        # train_data_ = copy.deepcopy(train_data)
        # dev_data_ = copy.deepcopy(dev_data)
        # train_model(model, train_data_, dev_data_, epoch, args.lr, idx_tag)
        train_loss, p, r, f1, eval_loss = train_model(model, train_data, dev_data, epoch, args.lr, idx_tag)
        
        logger.info('Epoch:%d, Training Loss:%.4f', epoch, train_loss)
        logger.info('Epoch:%d, Eval Loss:%.4f, Eval P:%.4f, Eval R:%.4f, Eval F1:%.4f', \
                                    epoch, eval_loss, p, r, f1)
        # p, r, f1, eval_loss, all_assess = eval_model(model,  test_iter, idx_tag)
        # logger.info('Test Loss:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f', \
        #                             eval_loss, p, r, f1)
        if f1 > best_score:
            best_score = f1
            torch.save(model.state_dict(), 'results/%d_%s_%s.pt' % (epoch, 'Model', str(best_score)))
        # p, r, f1, eval_loss, all_assess = eval_model(model,  test_iter, idx_tag)
        # logger.info('Test Loss:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f', \
        #                             eval_loss, p, r, f1)
    # # consuming_time = time.perf_counter()-start
    # # logger.info('Best Eval Acc: %.5f, Time consuming: %.2f', acc_max, consuming_time)

if __name__ == "__main__":
    main()
    #python main.py  --epoch 100 --lr 2e-3 --train_data_path data/train.tsv --test_data_path data/test.tsv --embedding_path data/sgns.weibo.char/sgns.weibo.char  --hidden_size 256 --train_batch_size 128 --load_model results/35_Model_0.8138361758347344.pt