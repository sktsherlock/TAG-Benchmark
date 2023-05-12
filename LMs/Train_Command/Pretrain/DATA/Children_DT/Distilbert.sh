# MLM

# TDK
CUDA_VISIBLE_DEVICES=0 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=240 --grad_steps=1 --lr=5e-05 --model=Distilbert --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/Distilbert/
CUDA_VISIBLE_DEVICES=0 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=240 --grad_steps=1 --lr=5e-05 --model=Electra --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/Electra/Small/
CUDA_VISIBLE_DEVICES=0 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=240 --grad_steps=1 --lr=5e-05 --model=TinyBert --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/TinyBert/

# TCL
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=Distilbert --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/Distilbert/
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=Electra --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/Electra/Small/
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=TinyBert --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/TinyBert/

# tcl+tdk
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=Distilbert --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/Distilbert/
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=Electra --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/Electra/Small/
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=TinyBert --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/TinyBert/



# TCL
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=Amazon_TA --dropout=0.1 --epochs=5 --eq_batch_size=120 --per_device_bsz=30 --grad_steps=1 --lr=5e-05 --model=Roberta-large --warmup_epochs=1 --gpus=0,1,2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/RoBerta/Large/
# TCL+TDK
  CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=Amazon_TA --dropout=0.1 --epochs=5 --eq_batch_size=120 --per_device_bsz=30 --grad_steps=1 --lr=5e-05 --model=Roberta-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/RoBerta/Large/

# MLM+TCL+TDK