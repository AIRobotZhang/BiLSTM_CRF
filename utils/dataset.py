# -*- coding: utf-8 -*-
import torch
from torchtext import data
from torchtext.vocab import Vectors
import numpy as np
from tqdm import tqdm

def load_dataset(train_data_path, dev_data_path, test_data_path,\
                wordVectors_path, train_batch_size, dev_batch_size, test_batch_size):
    """
    :param train_data_path: train dataeset
    :param dev_data_path:  dev dataset
    :param test_data_path: test dataset
    :param wordVectors_path: pre-trained word embeddings
    :param train_batch_size: batch size of train
    :param dev_batch_size: batch size of dev
    :param test_batch_size: batch size of test
    :return:
    """

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, pad_token='<pad>',\
                                    lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=True, tokenize=tokenize, pad_token='O',\
                                   unk_token=None, include_lengths=True, batch_first=True)

    train_data = data.TabularDataset(path=train_data_path, format='tsv',
                                fields=[('text', TEXT), ('label', LABEL)])
    if dev_data_path:
        dev_data = data.TabularDataset(path=dev_data_path, format='tsv',
                                     fields=[('text', TEXT), ('label', LABEL)])
    if test_data_path:
        test_data = data.TabularDataset(path=test_data_path, format='tsv',
                                fields=[('text', TEXT), ('label', LABEL)])
    # wordVectors_path = 'data/glove.6B/glove.6B.300d.txt'
    if wordVectors_path:
        vectors = Vectors(name=wordVectors_path)
        TEXT.build_vocab(train_data, vectors=vectors)
        word_embeddings = TEXT.vocab.vectors
        # print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
        print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    else:
        TEXT.build_vocab(train_data)
        word_embeddings = None
    LABEL.build_vocab(train_data)
    

    train_iter = data.Iterator(train_data, batch_size=train_batch_size, \
                                        train=True, sort=False, repeat=False, shuffle=True)
    dev_iter = None
    if dev_data_path:
        dev_iter = data.Iterator(dev_data, batch_size=dev_batch_size, \
                             train=False, sort=False, repeat=False, shuffle=True)
    test_iter = None
    if test_data_path:
        test_iter = data.Iterator(test_data, batch_size=test_batch_size, \
                                     train=False, sort=False, repeat=False, shuffle=False)

    vocab_size = len(TEXT.vocab)

    label_dict = dict(LABEL.vocab.stoi)
    length = len(label_dict)
    label_dict["<START>"] = length
    label_dict["<STOP>"] = length+1

    return TEXT, LABEL, vocab_size, word_embeddings, train_iter, dev_iter, test_iter, label_dict

def get_label(label_file, tag_dict=None):
    if not tag_dict:
        tag_dict = {}
        tag_dict["<START>"] = 0
        tag_dict["<STOP>"] = 1
        ind = 2
    label_tag = []
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            tag_ = []
            for item in line:
                if item not in tag_dict:
                    tag_dict[item] = ind
                    ind += 1
                tag_.append(tag_dict[item])
            label_tag.append(tag_)
    
    return label_tag, tag_dict

def train_dev_split(train_iter, ratio):
    length = len(train_iter)
    train_data = []
    dev_data = []
    train_start = 0
    train_end = int(length*ratio)
    ind = 0
    for batch in train_iter:
        if ind < train_end:
            train_data.append(batch)
        else:
            dev_data.append(batch)
        ind += 1
    return train_data, dev_data


if __name__ == '__main__':
    train_data_path = '../data/demo.tsv'
    dev_data_path = None
    test_data_path = None
    wordVectors_path = 'D:/MyDocument/Project/Graph-Rel-Entity/dataset/glove.6B/glove.6B.300d.txt'
    train_batch_size = 2
    dev_batch_size = None
    test_batch_size = None
    TEXT, LABEL, vocab_size, word_embeddings, train_iter, dev_iter, test_iter = \
        load_dataset(train_data_path, dev_data_path, test_data_path,\
                wordVectors_path, train_batch_size, dev_batch_size, test_batch_size)
    label_dict = dict(LABEL.vocab.stoi)
    length = len(label_dict)
    label_dict["<START>"] = length
    label_dict["<STOP>"] = length+1
    print(label_dict)
    for item in train_iter:
        print(item.text)
        print(item.label)
