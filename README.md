# bert_cls_own
## 下载google-bert
wget https://redirector.gvt1.com/edgedl/tfhub-modules/google/bert_chinese_L-12_H-768_A-12/1.tar.gz
## 下载依赖库
pip install -r requirements.txt
## 数据格式
训练集train.txt，一行一个样本，query分词结果##docid
测试集valid.txt，同上
类别信息id_docId，一行一个类别，index[tab]docid[tab]docname
## 执行
sh train.sh