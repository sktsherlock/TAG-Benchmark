# MLM

# TDK
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=arxiv_TA --dropout=0.1 --epochs=5 --eq_batch_size=120 --per_device_bsz=60 --grad_steps=1 --lr=5e-05 --model=Bert-large --warmup_epochs=1 --gpus=0,1 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Arxiv/Bert/Large/

# TCL
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=arxiv_TA --dropout=0.1 --epochs=5 --eq_batch_size=120 --per_device_bsz=30 --grad_steps=1 --lr=5e-05 --model=Bert-large --warmup_epochs=1 --gpus=0,1,2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Arxiv/Bert/Large/
# TCL+TDK
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=arxiv_TA --dropout=0.1 --epochs=5 --eq_batch_size=120 --per_device_bsz=30 --grad_steps=1 --lr=5e-05 --model=Bert-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Arxiv/Bert/Large/

# MLM+TCL+TDK