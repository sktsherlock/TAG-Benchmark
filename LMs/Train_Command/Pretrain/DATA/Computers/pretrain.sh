# TMLM
# Small
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=prajjwal1/bert-tiny --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/TinyBert/ --overwrite_output_dir=True --per_device_train_batch_size=30 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True --max_seq_length=512
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=google/electra-small-discriminator --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/Electra/Small/ --overwrite_output_dir=True --per_device_train_batch_size=30 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=distilbert-base-uncased  --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/Distilbert/ --overwrite_output_dir=True --per_device_train_batch_size=30 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True

# Base
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=microsoft/deberta-base --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/Deberta/Base/ --overwrite_output_dir=True --per_device_train_batch_size=12 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=bert-base-uncased --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/Bert/Base/ --overwrite_output_dir=True --per_device_train_batch_size=20 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=google/electra-base-discriminator  --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/Electra/Base/ --overwrite_output_dir=True --per_device_train_batch_size=20 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=roberta-base --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/RoBerta/Base/ --overwrite_output_dir=True --per_device_train_batch_size=20 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True

# Large
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=bert-large-uncased --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/Bert/Large/ --overwrite_output_dir=True --per_device_train_batch_size=8 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=roberta-large --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/RoBerta/Large/ --overwrite_output_dir=True --per_device_train_batch_size=8--train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=microsoft/deberta-large  --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/Deberta/Large/ --overwrite_output_dir=True --per_device_train_batch_size=5 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=google/electra-large-discriminator --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Computers/Electra/Large/ --overwrite_output_dir=True --per_device_train_batch_size=8 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Computers_RS/Computers_RS_TMLM.txt --warmup_steps=2500 --line_by_line=True

#! TCL
#Small
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=Distilbert --warmup_epochs=1 --gpus=0,1 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/Distilbert/

CUDA_VISIBLE_DEVICES=2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=120 --grad_steps=1 --lr=5e-05 --model=Electra --warmup_epochs=1 --gpus=2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/Electra/Small/
CUDA_VISIBLE_DEVICES=4 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --dropout=0.1 --epochs=5 --eq_batch_size=240 --per_device_bsz=240 --grad_steps=1 --lr=5e-05 --model=TinyBert --warmup_epochs=1 --gpus=4 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/TinyBert/
# Base
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Deberta --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/Deberta/Base/
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=RoBerta --warmup_epochs=1 --gpus=0,1,2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/RoBerta/Base/
CUDA_VISIBLE_DEVICES=5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=60 --grad_steps=1 --lr=5e-05 --model=Electra-base --warmup_epochs=1 --gpus=5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/Electra/Base/
CUDA_VISIBLE_DEVICES=1,2,3,4 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Bert --warmup_epochs=1 --gpus=1,2,3,4 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/Bert/Base/
# Large
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --freeze=4 --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Deberta-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/Deberta/Large/
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --freeze=4 --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Roberta-large --warmup_epochs=1 --gpus=0,1,2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/RoBerta/Large/
CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --freeze=4 --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Electra-large --warmup_epochs=1 --gpus=4,5,6,7 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/Electra/Large/

CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL.py --PrtMode=TCL --att_dropout=0.1 --cla_dropout=0.1 --dataset=Computers_RS --freeze=4 --dropout=0.1 --epochs=5 --eq_batch_size=180 --per_device_bsz=45 --grad_steps=1 --lr=5e-05 --model=Bert-large --warmup_epochs=1 --gpus=0,1,2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL/Amazon/Computers/Bert/Large/

# TDK

# TCL

# TCL+TDK
CUDA_VISIBLE_DEVICES=2,3 /usr/bin/env python sweep/dist_runner.py LMs/Train_Command/train_CL_DK.py --PrtMode=TCL_Deepwalk --freeze=4 --att_dropout=0.1 --cla_dropout=0.1 --dataset=Photo_RS --dropout=0.1 --epochs=5 --eq_batch_size=60 --per_device_bsz=30 --grad_steps=1 --lr=5e-05 --model=Bert-large --warmup_epochs=1 --gpus=2,3 --cache_dir=/mnt/v-wzhuang/TAG/Prt/TCL_Deepwalk/Amazon/Photo/Bert/Large/
