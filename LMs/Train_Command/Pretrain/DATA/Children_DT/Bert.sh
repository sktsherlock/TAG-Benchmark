Small
CUDA_VISIBLE_DEVICES=0 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=240 --grad_steps=1 --lr=5e-05 --model=TinyBert --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/TinyBert/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/TinyBert/
CUDA_VISIBLE_DEVICES=1,2 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=Distilbert --warmup_epochs=1 --gpus=1,2 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Distilbert/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/Distilbert/
CUDA_VISIBLE_DEVICES=3,4 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=Electra --warmup_epochs=1 --gpus=3,4 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Electra/Small/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/Electra/Small/


Base
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=90 --grad_steps=1 --lr=5e-05 --model=Bert --warmup_epochs=1 --gpus=0,1 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/Bert/Base/
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=90 --grad_steps=1 --lr=5e-05 --model=Deberta --warmup_epochs=1 --gpus=0,1 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/Deberta/Base/
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=90 --grad_steps=1 --lr=5e-05 --model=RoBerta --warmup_epochs=1 --gpus=0,1 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/RoBerta/Base/
CUDA_VISIBLE_DEVICES=2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=90 --grad_steps=1 --lr=5e-05 --model=Electra-base --warmup_epochs=1 --gpus=2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/Electra/Base/

CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Bert --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/Bert/Base/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Deberta --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/Deberta/Base/ No
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=RoBerta --warmup_epochs=1 --gpus=0,1,2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/RoBerta/Base/ No
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Electra-base --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/Electra/Base/ No

CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Bert --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/Bert/Base/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Deberta --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/Deberta/Base/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=RoBerta --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/RoBerta/Base/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Electra-base --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/Electra/Base/

CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Bert --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Bert/Base/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/Bert/Base/
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Deberta --warmup_epochs=1 --gpus=0,1,2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Deberta/Base/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/Deberta/Base/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=RoBerta --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/RoBerta/Base/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/RoBerta/Base/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Electra-base --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Electra/Base/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/Electra/Base/

Large
CUDA_VISIBLE_DEVICES=4,5 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=90 --grad_steps=1 --lr=5e-05 --model=Bert-large --warmup_epochs=1 --gpus=4,5 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/Bert/Large/
CUDA_VISIBLE_DEVICES=4,5 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=90 --grad_steps=1 --lr=5e-05 --model=Deberta-large --warmup_epochs=1 --gpus=4,5 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/Deberta/Large/
CUDA_VISIBLE_DEVICES=4,5 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=90 --grad_steps=1 --lr=5e-05 --model=Roberta-large --warmup_epochs=1 --gpus=4,5 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/RoBerta/Large/
CUDA_VISIBLE_DEVICES=4,5 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_DPK.py --PrtMode=Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=90 --grad_steps=1 --lr=5e-05 --model=Electra-large --warmup_epochs=1 --gpus=4,5 --cache_dir=/mnt/v-wzhuang/TAG/Prt/DeepWalk/Amazon/Children/Electra/Large/

CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Bert-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/Bert/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Deberta-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/Deberta/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Roberta-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/RoBerta/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Electra-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Children/Electra/Large/

CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Bert-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/Bert/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Deberta-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/Deberta/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Roberta-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/RoBerta/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Electra-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Children/Electra/Large/

CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Bert-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Bert/Large/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/Bert/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Deberta-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Deberta/Large/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/Deberta/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Roberta-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/RoBerta/Large/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/RoBerta/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TMLM_TCL_Deepwalk  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Children_DT --dropout=0.1 --epochs=5 --freeze=4 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Electra-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TMLM_TCL_Deepwalk/Amazon/Children/Electra/Large/ --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Children/Electra/Large/
