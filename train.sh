#!/bin/bash
set -u
set -e
train_test(){
    biz=${1}
    unlink data
    rm -rf output
    ln -s ../bert_cls_duiqi/${biz} data
    python bert_text_classifier.py > log${biz}
}

train_test anxin
#train_test beidian
train_test huaxia
train_test mobai
train_test nonghang
train_test shengdoushi
train_test xiecheng
train_test youyongfenqi


