# bert_cls_own
## docker 镜像
docker pull tensorflow/tensorflow:1.15.0-gpu-py3
docker run -d --name container_name -it -v host_dir:docker_dir tensorflow/tensorflow:1.15.0-gpu-py3 bash
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