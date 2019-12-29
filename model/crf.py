# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sequence_mask(lens, max_len):
    batch_size = lens.size(0)

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp # batch_size*max_len

    return mask

def log_sum_exp(vec, dim=0):
    max_score, max_ind = torch.max(vec, dim)
    max_exp = max_score.unsqueeze(-1).expand_as(vec)
    return max_score + torch.log(torch.sum(torch.exp(vec-max_exp), dim))
    # max_score = vec[:, argmax(vec)]
    # max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # return torch.log(torch.sum(torch.exp(vec), 1))

class CRF(nn.Module):
    def __init__(self, tagset_size, tag_to_ix):
        super(CRF, self).__init__()
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.tag_to_ix = tag_to_ix
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        torch.nn.init.normal_(self.transitions.data, 0, 1)
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[self.STOP_TAG]] = -10000

    def _forward_alg(self, feats, origin_lens):
        # print('forward')
        # print(feats.size())
        # feats: seq_len*batch_size*tagset_size
        # Do the forward algorithm to compute the partition function
        batch_size = feats.size(1)
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        if torch.cuda.is_available():
            forward_var = forward_var.cuda()

        lens = origin_lens.clone() # batch_size

        # Iterate through the sentence
        for feat in feats:
            forward_expand = forward_var.unsqueeze(1).expand(batch_size, self.tagset_size, self.tagset_size)
            feat_expand = feat.unsqueeze(-1).expand(batch_size, self.tagset_size, self.tagset_size)
            trans_expand = self.transitions.unsqueeze(0).expand_as(forward_expand)
            mat_inter = forward_expand+trans_expand+feat_expand
            forward_lse = log_sum_exp(mat_inter, dim=2).squeeze(-1)
            mask = (lens>0).float().unsqueeze(-1).expand_as(forward_var)
            forward_var = mask*forward_lse + (1-mask)*forward_var
            lens = lens-1

            # alphas_t = []  # The forward tensors at this timestep
            # for next_tag in range(self.tagset_size):
            #     # broadcast the emission score: it is the same regardless of
            #     # the previous tag
            #     emit_score = feat[:, next_tag].view(batch_size, -1).expand(batch_size, self.tagset_size)
            #     # the ith entry of trans_score is the score of transitioning to
            #     # next_tag from i
            #     trans_score = self.transitions[next_tag].view(1, -1).expand(batch_size, -1)
            #     # The ith entry of next_tag_var is the value for the
            #     # edge (i -> next_tag) before we do log-sum-exp
            #     next_tag_var = forward_var + trans_score + emit_score
            #     # The forward variable for this tag is log-sum-exp of all the
            #     # scores.
            #     all_ = log_sum_exp(next_tag_var)
            #     alphas_t.append(all_.view(-1, 1))

            # forward_var = torch.cat(alphas_t, 1).view(batch_size, -1)
        forward_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]].unsqueeze(0).expand_as(forward_var)
        all_score = log_sum_exp(forward_var, dim=1).squeeze(-1) # batch_szie
        # terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]].expand(batch_size, -1)
        # alpha = log_sum_exp(terminal_var).view(-1, 1)
        return all_score

    def _score_sentence(self, feats, tags, origin_lens):
        # print('score')
        # print(feats.size())
        # print(tags.size())
        # feats: seq_len*batch_size*tagset_size
        # tags: batch_size*seq_len
        # Gives the score of a provided tag sequence
        batch_size = feats.size(1)
        seq_len = tags.size(1)
        # print(tags.size())
        start_ = torch.tensor([self.tag_to_ix[self.START_TAG]],dtype=torch.long).expand(batch_size,1)
        stop_ = torch.tensor([self.tag_to_ix[self.STOP_TAG]],dtype=torch.long).expand(batch_size,1)
        stop_pad = torch.tensor([self.tag_to_ix[self.STOP_TAG]],dtype=torch.long).expand(batch_size,seq_len+2)
        if torch.cuda.is_available():
            start_ = start_.cuda()
            stop_ = stop_.cuda()
            stop_pad = stop_pad.cuda()

        tags = torch.cat([start_, tags, stop_], 1)
        # tags = torch.cat([start_, tags], 1)
        mask = sequence_mask(origin_lens+1, max_len=seq_len+2).long()
        tags = (1-mask)*stop_pad + mask*tags
        # batch_transitions = self.transitions.expand(batch_size, -1, -1)
        batch_transitions = self.transitions.unsqueeze(0).expand(batch_size, *self.transitions.size())

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        tag_r = tags[:, 1:]
        tag_expand = tag_r.unsqueeze(-1).expand(*tag_r.size(), batch_transitions.size(1))
        tag_row = torch.gather(batch_transitions, 1, tag_expand) #batch_size*(seq_len+1)*tagset_size

        tag_lexpand = tags[:, :-1].unsqueeze(-1) #batch_size*(seq_len+1)*1
        trn_scr = torch.gather(tag_row, 2, tag_lexpand)
        trn_scr = trn_scr.squeeze(-1) #batch_size*(seq_len+1)

        mask = sequence_mask(origin_lens+1, max_len=seq_len+1).float()
        trn_scr = trn_scr*mask
        trans_score = trn_scr.sum(1).squeeze(-1)

        tag_rexpand = tags[:, 1:-1].unsqueeze(-1)
        feats = feats.transpose(1,0)
        # print(feats.size())
        emit_score = torch.gather(feats, 2, tag_rexpand).squeeze(-1)
        mask = sequence_mask(origin_lens, max_len=seq_len).float()
        emit_score = emit_score * mask
        emit_score = emit_score.sum(1).squeeze(-1)

        score = trans_score+emit_score #batch_size

        # for i, feat in enumerate(feats):
        #     # tmp_trans = torch.FloatTensor([[self.transitions[tags[j][i+1]][tags[j][i]]] for j in range(batch_size)])
        #     for j in range(batch_size):
        #         score[j][0] += self.transitions[tags[j][i+1]][tags[j][i]]+feat[j][tags[j][i+1]]

        #     # print(i)
        #     # print(i+1)
        #     # score = score + feat[:, tags[i+1]]

        # for j in range(batch_size):
        #     score[j][0] += self.transitions[self.tag_to_ix[self.STOP_TAG]][tags[j][-1]]
        # # tmp_trans = torch.FloatTensor([[self.transitions[self.tag_to_ix[self.STOP_TAG]][tags[j][-1]]] \
        # #                                                for j in range(batch_size)])
        # # score = score + tmp_trans
        return score

    def _viterbi_decode(self, feats, origin_lens):
        backpointers = []
        batch_size = feats.size(1)
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((batch_size, self.tagset_size), -10000.)
        init_vvars[:, self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        if torch.cuda.is_available():
            forward_var = forward_var.cuda()

        lens = origin_lens.clone()

        for feat in feats:
            forward_expand = forward_var.unsqueeze(1).expand(batch_size, self.tagset_size, self.tagset_size)
            trans_expand = self.transitions.unsqueeze(0).expand(batch_size, -1, -1)
            total_score = forward_expand+trans_expand
            max_score, max_ind = total_score.max(2) #batch_size*tagset_size

            forward_var_ = max_score+feat #batch_size*tagset_size
            backpointers.append(max_ind.unsqueeze(0))

            mask = (lens>0).float().unsqueeze(-1).expand_as(forward_var)
            forward_var = mask*forward_var_ + (1-mask)*forward_var

            mask = (lens==1).float().unsqueeze(-1).expand_as(forward_var)
            forward_var += mask*self.transitions[self.tag_to_ix[self.START_TAG]].unsqueeze(0).expand_as(forward_var)
            lens = lens-1
        # trans_expand = self.transitions[self.tag_to_ix[self.START_TAG]].unsqueeze(0).expand(batch_size, -1)
        # score = forward_var+trans_expand
        max_score, max_ind = forward_var.max(1)
        paths = [max_ind.unsqueeze(-1)]

        backpointers = torch.cat(backpointers)
        pointers = reversed(backpointers)
        for argmax in pointers:
            max_ind = max_ind.unsqueeze(-1)
            max_ind = torch.gather(argmax,1,max_ind).squeeze(-1)
            paths.insert(0,max_ind.unsqueeze(-1))

        paths = torch.cat(paths[1:],1) #batch_size*seq_len

        return max_score, paths

        #     # bptrs_t = []  # holds the backpointers for this step
        #     # viterbivars_t = []  # holds the viterbi variables for this step

        #     # for next_tag in range(self.tagset_size):
        #     #     # next_tag_var[i] holds the viterbi variable for tag i at the
        #     #     # previous step, plus the score of transitioning
        #     #     # from tag i to next_tag.
        #     #     # We don't include the emission scores here because the max
        #     #     # does not depend on them (we add them in below)
        #     #     next_tag_var = forward_var + self.transitions[next_tag].expand(batch_size, -1)
        #     #     best_tag_id = argmax(next_tag_var).view(batch_size, -1)
        #     #     bptrs_t.append(best_tag_id)
        #     #     viterbivars_t.append(torch.gather(next_tag_var, 1, best_tag_id))
        #     # # Now add in the emission scores, and assign forward_var to the set
        #     # # of viterbi variables we just computed
        #     # forward_var = (torch.cat(viterbivars_t, 1) + feat).view(batch_size, -1)
        #     # backpointers.append(bptrs_t)

        # # Transition to STOP_TAG
        # terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]].expand(batch_size, -1)
        # best_tag_id = argmax(terminal_var).view(batch_size, -1)
        # path_score = torch.gather(terminal_var, 1, best_tag_id)

        # # Follow the back pointers to decode the best pat、h.
        # best_path = [best_tag_id.numpy()]
        # for bptrs_t in reversed(backpointers):
        #     tmp = []
        #     for j in range(batch_size):
        #         tmp.append(bptrs_t[best_tag_id[j][0]][j].numpy())

        #     best_tag_id = tmp
        #     best_path.append(best_tag_id)
        # # Pop off the start tag (we dont want to return that to the caller)
        # start = best_path.pop()
        # assert start[0][0] == self.tag_to_ix[self.START_TAG]  # Sanity check
        # best_path.reverse()
        # # return seq*batch_size*1
        # return path_score, best_path

    def neg_log_likelihood(self, feats, tags, lens):
        forward_score = self._forward_alg(feats, lens)
        gold_score = self._score_sentence(feats, tags, lens)

        return (forward_score - gold_score).mean() #1 scaler
























