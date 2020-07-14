# -*- coding: utf-8 -*-
import pickle as  cPickle
import json

import jieba
import random
import codecs, sys
import time
import numpy as np
import pandas as pd
from progressbar import ShowProcess

def seg_line(line):
    return list(jieba.cut(line))


def seg_data(path):
    # 还款 的 计息 方式 是 等额 本息 吗##217673
    # label qid docid query doc
    print ('start process ', path)
    texts = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        len_data = len(lines)
        process_bar = ShowProcess(len_data, '')
        for line in lines:
            process_bar.show_process()
            split = line.strip().split('##')
            query = split[0].replace(" ","")
            docid = split[1]               # true docid
            texts.append(query)
            labels.append(docid)
    data = {"text":texts, "label":labels}
    return data


def process_data(data_path):
    return pd.DataFrame(seg_data(data_path))
