CUDA_VISIBLE_DEVICES=2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_Tlink.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=DBLP_T --dropout=0.1 --epochs=10 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=Distilbert --warmup_epochs=1 --gpus=2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TLink/DBLP/Distilbert/