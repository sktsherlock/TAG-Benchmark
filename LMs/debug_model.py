from datasets import load_metric
from transformers import AutoModel, EvalPrediction, TrainingArguments, Trainer, AutoTokenizer,BertModel,AutoModelForMaskedLM
import utils as uf
from model import *
from utils.data.datasets import *
import torch as th
from torch.utils.data import random_split
import os
#! 直接导入from_pretrained
#! Bert: bert-base-uncased ; bert-large-uncased; prajjwal1/bert-tiny
#! Deberta: microsoft/deberta-base; microsoft/deberta-large
#! Distilbert: distilbert-base-uncased
#! Electra: google/electra-small-discriminator/google/electra-base-discriminator
encoder = AutoModel.from_pretrained('microsoft/deberta-base')
print(encoder.config)
trainable_params = sum(
    p.numel() for p in encoder.parameters() if p.requires_grad
)
print(f" LM Model parameters are {trainable_params}")

bert = BertModel.from_pretrained('bert-base-uncased')
print(bert.config)
trainable_params = sum(
    p.numel() for p in bert.parameters() if p.requires_grad
)
print(f" bert Model parameters are {trainable_params}")

"""
AutoModel.from_pretrained('bert-base-uncased')  |  Architectures: BertForMaskedLM | trainable_params: 109482240 (30760)
AutoModel.from_pretrained('prajjwal1/bert-tiny') |                                | trainable_params: 4385920  4385920
AutoModel.from_pretrained('microsoft/deberta-base') |                                | trainable_params: 138601728 (30760)
基本可以确定，这种调用的就是一个encoder；
"""
bert_mask = AutoModelForMaskedLM.from_pretrained('prajjwal1/bert-tiny')
print(bert_mask.config)
trainable_params = sum(
    p.numel() for p in bert_mask.parameters() if p.requires_grad
)
print(f" bert_mask Model parameters are {trainable_params}")

"""
AutoModelForMaskedLM.from_pretrained('bert-base-uncased')  |  Architectures: BertForMaskedLM | trainable_params: 109514298
AutoModelForMaskedLM.from_pretrained('prajjwal1/bert-tiny') |                                | trainable_params: 4416698
AutoModelForMaskedLM.from_pretrained('microsoft/deberta-base') |                                | trainable_params: 139244121 
增加了一个cls头 这个函数可以参考着学一下
"""
trainer = Trainer(
    model=bert_mask,
)

trainer.save_model(output_dir='/home/data/yh/debug/')
#! Load from it
model = AutoModel.from_pretrained('/home/data/yh/debug/')
print(model.config)
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print(f" bert Model parameters are {trainable_params}")

"""
AutoModel.from_pretrained('/home/data/yh/debug/') |                                | trainable_params: 4385920 = 之前的
cls头被去掉了 
"""
encoder = AutoModel.from_pretrained('prajjwal1/bert-tiny')
print(encoder.config)
trainable_params = sum(
    p.numel() for p in encoder.parameters() if p.requires_grad
)
print(f" LM Model full learning parameters are {trainable_params}")


