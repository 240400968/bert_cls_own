#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/wshuyi/demo-chinese-text-binary-classification-with-bert/blob/master/demo_chinese_text_binary_classification_with_bert.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# base code borrowed from [this Google Colab Notebook](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb).
# 
# Refactored by [Shuyi Wang](https://www.linkedin.com/in/shuyi-wang-b3955026/)
# 
# Please refer to [this Medium Article](https://medium.com/@wshuyi/how-to-do-text-binary-classification-with-bert-f1348a25d905) for the tutorial on how to classify English text data.
# 
# 

# In[ ]:


#get_ipython().system('pip install bert-tensorflow')


# In[ ]:

from preprocess import process_data
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# In[ ]:

model_path='./bert_chinese_L-12_H-768_A-12'
bert_model_spec = hub.load_module_spec(model_path)

def pretty_print(result):
    df = pd.DataFrame([result]).T
    df.columns = ["values"]
    return df


# In[ ]:


def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    #bert_module = hub.Module(bert_model_hub)
    bert_module = hub.Module(bert_model_spec, trainable=True)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

def make_features(dataset, label_list, MAX_SEQ_LENGTH, tokenizer, DATA_COLUMN, LABEL_COLUMN):
    input_example = dataset.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)
    features = bert.run_classifier.convert_examples_to_features(input_example, label_list, MAX_SEQ_LENGTH, tokenizer)
    #print("features:", features)
    return features

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(
      bert_model_spec,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)

# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels, num_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        recall_n = [0] * num_labels
        precision_n = [0] * num_labels
        update_op_rec_n = [[]] * num_labels
        update_op_pre_n = [[]] * num_labels
        for k in range(num_labels):
            recall_n[k], update_op_rec_n[k] = tf.metrics.recall(
                labels=tf.equal(label_ids, k),
                predictions=tf.equal(predicted_labels, k)
            )    
            precision_n[k], update_op_pre_n[k] = tf.metrics.precision(
                labels=tf.equal(label_ids, k),
                predictions=tf.equal(predicted_labels, k)
            )    
        recall_value = sum(recall_n) * 1.0 / num_labels
        precision_value = sum(precision_n) * 1.0 / num_labels
        update_op_rec = sum(update_op_rec_n) * 1.0 / num_labels
        update_op_pre = sum(update_op_pre_n) * 1.0 / num_labels
        recall = (recall_value, update_op_rec)
        precision = (precision_value, update_op_pre)
        return {
            "eval_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

      eval_metrics = metric_fn(label_ids, predicted_labels, num_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn

def estimator_builder(OUTPUT_DIR, SAVE_SUMMARY_STEPS, SAVE_CHECKPOINTS_STEPS, label_list, LEARNING_RATE, num_train_steps, num_warmup_steps, BATCH_SIZE):

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
      num_labels=len(label_list),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": BATCH_SIZE})
    return estimator, model_fn, run_config


# In[ ]:


def run_on_dfs(train, test, DATA_COLUMN, LABEL_COLUMN, 
               MAX_SEQ_LENGTH = 128,
              BATCH_SIZE = 32,
              LEARNING_RATE = 2e-5,
              NUM_TRAIN_EPOCHS = 3.0,
              WARMUP_PROPORTION = 0.1,
              SAVE_SUMMARY_STEPS = 100,
               SAVE_CHECKPOINTS_STEPS = 10000):

    label_list = train[LABEL_COLUMN].unique().tolist()
    
    tokenizer = create_tokenizer_from_hub_module()

    train_features = make_features(train, label_list, MAX_SEQ_LENGTH, tokenizer, DATA_COLUMN, LABEL_COLUMN)
    test_features = make_features(test, label_list, MAX_SEQ_LENGTH, tokenizer, DATA_COLUMN, LABEL_COLUMN)

    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    estimator, model_fn, run_config = estimator_builder(
                                  OUTPUT_DIR, 
                                  SAVE_SUMMARY_STEPS, 
                                  SAVE_CHECKPOINTS_STEPS, 
                                  label_list, 
                                  LEARNING_RATE, 
                                  num_train_steps, 
                                  num_warmup_steps, 
                                  BATCH_SIZE)

    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    result_dict = estimator.evaluate(input_fn=test_input_fn, steps=None)
    return result_dict, estimator
    


# In[ ]:


import random
random.seed(10)


# In[ ]:


OUTPUT_DIR = 'output'


# ----- you just need to focus from here ------

# ## Get your data

# In[ ]:


#get_ipython().system('wget https://github.com/wshuyi/demo-chinese-text-binary-classification-with-bert/raw/master/dianping_train_test.pickle')


# In[ ]:
data_path='./data/'
train = process_data(data_path+'train.txt')
test = process_data(data_path+'valid.txt')
# with open("dianping_train_test.pickle", 'rb') as f:
#     train, test = pickle.load(f)
# train.to_csv("ziyong_train.csv")
# test.to_csv("ziyong_test.csv")
# In[ ]:


# In[ ]:


#train.head()
print("train head", train.head())
print("train label", train['label'])

# In[ ]:


myparam = {
        "DATA_COLUMN": "text",
        "LABEL_COLUMN": "label",
        "LEARNING_RATE": 2e-5,
        "NUM_TRAIN_EPOCHS":8,
    }


# In[ ]:


result, estimator = run_on_dfs(train, test, **myparam)


# In[ ]:


print(pretty_print(result))


# In[ ]:




