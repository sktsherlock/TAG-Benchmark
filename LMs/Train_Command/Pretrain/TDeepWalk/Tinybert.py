#!不同模型对应的命令
""" TinyBert
CUDA_VISIBLE_DEVICES=0 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk \
--att_dropout=0.1 --cla_dropout=0.1 --dataset=arxiv_TA --dropout=0.1 --epochs=5 --eq_batch_size=30 --eval_patience=50000 --label_smoothing_factor=0.1 --lr=2e-05 --model=Deberta --warmup_epochs=1 --gpus=0

"""
#!Model
# Bert;Deberta;Electra-base;RoBerta

CUDA_VISIBLE_DEVICES=5,6 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --att_dropout=0.1 --cla_dropout=0.1 --dataset=arxiv_TA --dropout=0.1 --epochs=4 --eq_batch_size=100 --eval_patience=50000 --label_smoothing_factor=0.1 --lr=2e-05 --model=Distilbert --warmup_epochs=1 --gpus=5,6 #--freeze=1
