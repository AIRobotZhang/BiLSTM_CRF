# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .crf import CRF, sequence_mask
import numpy as np
import time
# from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tag_to_ix, \
                                weights, fine_tuning=True):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.requires_grad = fine_tuning

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        # print(type(weights))
        # self.word_embeddings.weight.requires_grad = False
        if isinstance(weights, torch.Tensor):
            self.word_embeddings.weight = nn.Parameter(weights, requires_grad=self.requires_grad)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim//2, num_layers=3, bidirectional=True)
        # self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.5)
        # self.batchnorm = nn.BatchNorm1d(hidden_dim)
        # self.relu = nn.ReLU(inplace=False)
        # Maps the output of the LSTM into tag space.
        # self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # torch.nn.init.xavier_normal_(self.hidden2tag.weight.data)
        # self.lstm.reset_parameters()
        # self.softmax = nn.Softmax(dim=-1)

        self.crf = CRF(self.tagset_size, self.tag_to_ix)

    def decode(self, sentence, lens):
        input_ = self.word_embeddings(sentence)
        # (batch_size, num_sequences, embedding_length)
        input_ = input_.permute(1, 0, 2)
        input_ = self.dropout1(input_)
        output, _ = self.lstm(input_)
        # output = output.permute(1,2,0)
        # output = self.relu(self.batchnorm(output))
        # output = output.permute(2,0,1)
        output = self.dropout2(output)
        # output = self.linear(output)
        # output, _ = self.lstm2(output)
        # output = self.relu(output)
        # output = torch.cat([input_,output],2)
        feats = self.hidden2tag(output)
        # feats = self.softmax(feats)
        _, best_path = self.crf._viterbi_decode(feats, lens) #batch_size*seq_len
        # best_path = np.array(best_path).reshape(-1, len(best_path))

        return best_path

    def forward(self, sentence, tags, lens):
        # print(sentence)
        input_ = self.word_embeddings(sentence)
        # (batch_size, num_sequences, embedding_length)
        # print(input_)
        # print(self.word_embeddings.weight[31])
        input_ = input_.permute(1, 0, 2)
        input_ = self.dropout1(input_)
        output, _ = self.lstm(input_)
        # output = output.permute(1,2,0)
        # output = self.relu(self.batchnorm(output))
        # output = output.permute(2,0,1)
        output = self.dropout2(output)
        # output = torch.cat([input_,output],2)
        # output = self.linear(output)
        # output, _ = self.lstm2(output)
        # output = self.relu(output)
        feats = self.hidden2tag(output)
        # feats = self.softmax(feats)
        # start = time.perf_counter()
        loss = self.crf.neg_log_likelihood(feats, tags, lens)
        # print(time.perf_counter()-start)
        return loss


# class BiLSTM_pytorch_CRF(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, tag_to_ix, \
#                                 weights, fine_tuning=True):
#         super(BiLSTM_pytorch_CRF, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.vocab_size = vocab_size
#         self.tag_to_ix = tag_to_ix
#         self.tagset_size = len(tag_to_ix)-2
#         self.requires_grad = fine_tuning

#         self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
#         # self.word_embeddings.weight.requires_grad = False
#         self.word_embeddings.weight = nn.Parameter(weights, requires_grad=self.requires_grad)
#         self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
#         self.dropout = nn.Dropout(0.5)
#         # self.batchnorm = nn.BatchNorm1d(hidden_dim)
#         # self.relu = nn.ReLU(inplace=False)
#         # Maps the output of the LSTM into tag space.
#         self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
#         torch.nn.init.xavier_normal_(self.hidden2tag.weight.data)
#         self.lstm.reset_parameters()

#         self.crf = CRF(self.tagset_size)
#         for p in self.crf.parameters():
#             _ = torch.nn.init.uniform_(p, -1, 1)

#     def decode(self, sentence, lens):
#         input_ = self.word_embeddings(sentence)
#         max_len = input_.size(1)
#         mask = sequence_mask(lens, max_len).permute(1, 0)
#         # (batch_size, num_sequences, embedding_length)
#         input_ = input_.permute(1, 0, 2)
#         output, _ = self.lstm(input_)
#         output = self.dropout(output)
#         feats = self.hidden2tag(output)
#         best_path = self.crf.decode(feats, mask=mask) #batch_size*seq_len
#         # best_path = np.array(best_path).reshape(-1, len(best_path))

#         return best_path

#     def forward(self, sentence, tags, lens):
#         # print(sentence)
#         input_ = self.word_embeddings(sentence)
#         max_len = input_.size(1)
#         mask = sequence_mask(lens, max_len).permute(1, 0)
#         # (batch_size, num_sequences, embedding_length)
#         # print(input_)
#         # print(self.word_embeddings.weight[31])
#         input_ = input_.permute(1, 0, 2)
#         output, _ = self.lstm(input_)
#         # output = output.permute(1,2,0)
#         # output = self.relu(self.batchnorm(output))
#         # output = output.permute(2,0,1)
#         output = self.dropout(output)
#         feats = self.hidden2tag(output)
#         # start = time.perf_counter()
#         loss = -self.crf(feats, tags.permute(1,0), mask=mask, reduction='mean')
#         # print(time.perf_counter()-start)
#         return loss
