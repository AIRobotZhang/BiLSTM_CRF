# -*- coding: utf-8 -*-
import torch
import numpy as np
from collections import Counter

def get_entity(path, tag_map):
    results = []
    record = {}
    for index, tag_id in enumerate(path):
        try:
            tag_id = tag_id.item()
        except:
            tag_id = tag_id
        tag = tag_map[tag_id]
        # if tag == 'O':
        #     continue
        if tag.startswith("B-"):
            if len(record) > 0:
                results.append(record)
            record = {}
            record['begin'] = index
            record['type'] = tag.split('-')[1]
        elif tag.startswith('I-') and record.get('begin') != None:
            tag_type = tag.split('-')[1]
            if tag_type == record['type']:
                record['end'] = index
        else:
            if len(record) > 0:
                results.append(record)
            record = {}
    if len(record) > 0:
        results.append(record)
    return results


class Entity_Score(object):
    def __init__(self, id_to_label):
        self.id_to_label = id_to_label
        self._reset()

    def _reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def _compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right/origin)
        precision = 0 if found == 0 else (right/found)
        f1 = 0. if recall+precision ==0 else (2*precision*recall)/(precision+recall)
        return precision, recall, f1

    def result(self):
        all_statistic = {'LOC':{'origin':0, 'found':0, 'right':0}, 'PER':{'origin':0, 'found':0, 'right':0},\
                                         'ORG':{'origin':0, 'found':0, 'right':0}}
        all_assess = {'LOC':{'P':0,'R':0,'F':0}, 'PER':{'P':0,'R':0,'F':0}, 'ORG':{'P':0,'R':0,'F':0}}

        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        for orig in self.origins:
            all_statistic[orig['type']]['origin'] += 1
        for fou in self.founds:
            all_statistic[fou['type']]['found'] += 1
        for rig in self.rights:
            all_statistic[rig['type']]['right'] += 1

        precision, recall, f1 = self._compute(origin, found, right)
        all_assess['LOC']['P'], all_assess['LOC']['R'], all_assess['LOC']['F'] = \
                  self._compute(all_statistic['LOC']['origin'], all_statistic['LOC']['found'], all_statistic['LOC']['right'])
        all_assess['PER']['P'], all_assess['PER']['R'], all_assess['PER']['F'] = \
                  self._compute(all_statistic['PER']['origin'], all_statistic['PER']['found'], all_statistic['PER']['right'])
        all_assess['ORG']['P'], all_assess['ORG']['R'], all_assess['ORG']['F'] = \
                  self._compute(all_statistic['ORG']['origin'], all_statistic['ORG']['found'], all_statistic['ORG']['right'])
        
        return precision, recall, f1, all_assess
        # origin_counter = Counter([x['type'] for x in self.origins])
        # found_counter = Counter([x['type'] for x in self.founds])
        # right_counter = Counter([x['type'] for x in self.rights])
        # for type, count in origin_counter.items():
        #     origin = count
        #     found = found_counter.get(type, 0)
        #     right = right_counter.get(type, 0)
        #     recall, precision, f1 = self._compute(origin, found, right)

            # print("Type: %s - precision: %.4f - recall: %.4f - f1: %.4f"%(type,recall,precision,f1))

    def update(self, label_paths, pred_paths, lens):
        '''
        :param label_paths: [[2,3,4,5,6,5,8,8,8,8,8,4],[2,3,7,7,7,8,8,8,8]]
        :param pred_paths: [[2,3,4,5,6,5,8,8,8,8,8,4],[2,3,7,7,7,8,8,8,8]]
        '''
        for label_path, pre_path, mask in zip(label_paths, pred_paths, lens):
            label_path = label_path[:mask]
            pre_path = pre_path[:mask]
            label_entities = get_entity(label_path,self.id_to_label)
            pre_entities = get_entity(pre_path, self.id_to_label)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])