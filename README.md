# Benchmark工作

## LM-Prt 预训练LM模型Finetune
### 几步走：
- 加载对应LM模型的CKPT
- 下游任务微调
- 输出accuracy等其他metrics

主体函数脉络：
- trainLM 主函数
- model 定义分类model
- lm_trainer 定义trainingArguments
- lm_utils 
## 考虑到图结构(边)的LM 预训练

## 