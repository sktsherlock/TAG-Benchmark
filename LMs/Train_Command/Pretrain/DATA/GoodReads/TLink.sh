# 3 PLMs
TinyBert
CUDA_VISIBLE_DEVICES=0 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_TLink.py --PrtMode=TLink  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Good_T --dropout=0.1 --epochs=5 --eq_batch_size=5000 --per_device_bsz=5000 --grad_steps=1 --lr=5e-05 --model=TinyBert --warmup_epochs=1 --gpus=0 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TLink/Good/TinyBert/

Distilbert
CUDA_VISIBLE_DEVICES=1,2,3,4 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_TLink.py --PrtMode=TLink  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Good_T --dropout=0.1 --epochs=5 --eq_batch_size=4000 --per_device_bsz=1000 --grad_steps=1 --lr=5e-05 --model=Distilbert --warmup_epochs=1 --gpus=1,2,3,4 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TLink/Good/Distilbert/
Roberta-base
CUDA_VISIBLE_DEVICES=0,1,2,3,4 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_TLink.py --PrtMode=TLink  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Good_T --dropout=0.1 --epochs=5 --eq_batch_size=4000 --per_device_bsz=800 --grad_steps=1 --lr=5e-05 --model=RoBerta --warmup_epochs=1 --gpus=0,1,2,3,4 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TLink/Good/RoBerta/Base/

Bert-base
CUDA_VISIBLE_DEVICES=1,2,3,4,5 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_TLink.py --PrtMode=TLink  --att_dropout=0.1 --cla_dropout=0.1 --dataset=Good_T --dropout=0.1 --epochs=5 --eq_batch_size=4000 --per_device_bsz=800 --grad_steps=1 --lr=5e-05 --model=Bert --warmup_epochs=1 --gpus=1,2,3,4,5 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TLink/Good/Bert/Base/

# Inference THE Emb
CUDA_VISIBLE_DEVICES=0,1,2,3 python Train_Command/inference_LM.py --PrtMode=TLink --model=TinyBert --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TLink/Good/TinyBert/  --inference_dir=/mnt/v-wzhuang/TAG/TLink/Good/TinyBert/ --dataset=Good_T  --inf_batch_size 5000
CUDA_VISIBLE_DEVICES=0,1,2,3 python Train_Command/inference_LM.py --PrtMode=TLink --model=Distilbert --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TLink/Good/Distilbert/ --inference_dir=/mnt/v-wzhuang/TAG/TLink/Good/Distilbert/ --dataset=Good_T  --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --PrtMode=TLink --model=RoBerta --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TLink/Good/RoBerta/Base/ --inference_dir=/mnt/v-wzhuang/TAG/TLink/Good/RoBerta/Base/ --dataset=Good_T  --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --PrtMode=TLink --model=Bert --pretrain_path=/mnt/v-wzhuang/TAG/Prt/TLink/Good/Bert/Base/ --inference_dir=/mnt/v-wzhuang/TAG/TLink/Good/Bert/Base/ --dataset=Good_T  --inf_batch_size 1000