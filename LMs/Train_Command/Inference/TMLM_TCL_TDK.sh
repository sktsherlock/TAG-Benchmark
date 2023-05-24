#! Arxiv
# Small Model
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model TinyBert --pretrain_path=  --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/TinyBert/' --dataset arxiv_TA --inf_batch_size 600
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Distilbert  --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/Distilbert/' --dataset arxiv_TA --inf_batch_size 800
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Electra  --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/Electra/Small/' --dataset arxiv_TA --inf_batch_size 800
# Base Model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model RoBerta --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Arxiv/RoBerta/Base/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/RoBert/Base/' --dataset arxiv_TA --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Bert  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Arxiv/Bert/Base/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/Bert/Base/' --dataset arxiv_TA --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Electra-base  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Arxiv/Electra/Base/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/Electra/Base/' --dataset arxiv_TA --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Deberta  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Arxiv/Deberta/Base/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/Deberta/Base/' --dataset arxiv_TA --inf_batch_size 600
# Large Model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Roberta-large --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Arxiv/RoBerta/Large/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/RoBert/Large/' --dataset arxiv_TA --inf_batch_size 300
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Bert-large  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Arxiv/Bert/Large/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/Bert/Large/' --dataset arxiv_TA --inf_batch_size 300
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Electra-large --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Arxiv/Electra/Large/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/Electra/Large/' --dataset arxiv_TA --inf_batch_size 300
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Deberta-large  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Arxiv/Deberta/Large/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/OGB/Arxiv/Deberta/Large/' --dataset arxiv_TA --inf_batch_size 300


#!Children
# Amazon-Books-Children/History
# Small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model TinyBert --PrtMode='TMLM_TCL_Deepwalk'  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/TinyBert/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Distilbert  --PrtMode='TMLM_TCL_Deepwalk' --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Distilbert/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Electra  --PrtMode='TMLM_TCL_Deepwalk' --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Electra/Small/' --dataset Children_DT --inf_batch_size 1000
# Base
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model RoBerta  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/RoBerta/Base/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/Amazon/Children/RoBert/Base/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Bert  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Bert/Base/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/Amazon/Children/Bert/Base/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Electra-base  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Electra/Base/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/Amazon/Children/Electra/Base/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Deberta  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Deberta/Base/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/Amazon/Children/Deberta/Base/' --dataset Children_DT --inf_batch_size 1000
# Large
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Roberta-large  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/RoBerta/Large/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/Amazon/Children/RoBert/Large/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Bert-large  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Bert/Large/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/Amazon/Children/Bert/Large/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Electra-large  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Electra/Large/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/Amazon/Children/Electra/Large/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Deberta-large  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Deberta/Large/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/Amazon/Children/Deberta/Large/' --dataset Children_DT --inf_batch_size 1000

1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model Deberta-large  --PrtMode='TMLM_TCL_Deepwalk' --pretrain_path='/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Deberta/Large/' --inference_dir '/mnt/v-wzhuang/TAG/TMLM_TCL_TDK_Finetune/Amazon/Children/Deberta/Large/' --dataset Children_DT --inf_batch_size 1000