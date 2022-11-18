# TAG-Benchmark 

## LM-Prt 预训练LM模型Finetune
### 几步走：
- 加载对应LM模型的CKPT
- 下游分类任务微调
- 输出accuracy等其他metrics

## LM-Prt 继续预训练 （在下游数据集上预训练 MLM）RoBERTa/BERT/DistilBERT
### 几步走：
- 加载对应LM模型的CKPT
- 先在下游数据集上做MLM
- 再在下游数据集上做分类任务微调
- 输出accuracy等其他metrics
