# MLM

# TDK
CUDA_VISIBLE_DEVICES=0 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=Photo_RS --dropout=0.1 --epochs=5 --eq_batch_size=60 --per_device_bsz=60 --grad_steps=1 --lr=5e-05 --model=Deberta-large --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Photo/Deberta/Large/

# TCL
CUDA_VISIBLE_DEVICES=1,2 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=Photo_RS --dropout=0.1 --epochs=5 --eq_batch_size=60 --per_device_bsz=30 --grad_steps=1 --lr=5e-05 --model=Deberta-large --warmup_epochs=1 --gpus=1,2 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Photo/Deberta/Large/
# TCL+TDK
CUDA_VISIBLE_DEVICES=3,4 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=Photo_RS --dropout=0.1 --epochs=5 --eq_batch_size=60 --per_device_bsz=30 --grad_steps=1 --lr=5e-05 --model=Deberta-large --warmup_epochs=1 --gpus=3,4 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Photo/Deberta/Large/

# MLM+TCL+TDK
CUDA_VISIBLE_DEVICES=3,4 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=Photo_RS --dropout=0.1 --epochs=5 --eq_batch_size=60 --per_device_bsz=30 --grad_steps=1 --lr=5e-05 --model=Deberta-large --warmup_epochs=1 --gpus=3,4 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Photo/Deberta/Large/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/Deberta/Large/