# Small Model
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/TinyBert/' --dataset arxiv_TA --inf_batch_size 600
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Distilbert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Distilbert/' --dataset arxiv_TA --inf_batch_size 800
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Electra  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Electra/Small/' --dataset arxiv_TA --inf_batch_size 800
# Base Model
CUDA_VISIBLE_DEVICES=4,5,6,7 python Train_Command/inference_LM.py --model RoBert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/RoBert/Base/' --dataset arxiv_TA --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=4,5,6,7 python Train_Command/inference_LM.py --model Bert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Bert/Base/' --dataset arxiv_TA --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=4,5,6,7 python Train_Command/inference_LM.py --model Electra-base  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Electra/Base/' --dataset arxiv_TA --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=4,5,6,7 python Train_Command/inference_LM.py --model Deberta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Deberta/Base/' --dataset arxiv_TA --inf_batch_size 1000