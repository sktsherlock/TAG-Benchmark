
# Model name
# Small: prajjwal1/bert-tiny  | distilbert-base-uncased | google/electra-small-discriminator
# Base: bert-base-uncased | microsoft/deberta-base | google/electra-base-discriminator | roberta-base
# Large: bert-large-uncased | microsoft/deberta-large | google/electra-large-discriminator | roberta-large

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=prajjwal1/bert-tiny --num_train_epochs=5 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/TinyBert/ --overwrite_output_dir=True --per_device_train_batch_size=20 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True --max_seq_length=512
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=distilbert-base-uncased --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/Distilbert/ --overwrite_output_dir=True --per_device_train_batch_size=20 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=google/electra-small-discriminator  --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/Electra/Small/ --overwrite_output_dir=True --per_device_train_batch_size=20 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=microsoft/deberta-base --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/Deberta/Base/ --overwrite_output_dir=True --per_device_train_batch_size=12 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=bert-base-uncased --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/Bert/Base/ --overwrite_output_dir=True --per_device_train_batch_size=20 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=google/electra-base-discriminator  --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/Electra/Base/ --overwrite_output_dir=True --per_device_train_batch_size=20 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=roberta-base --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/RoBerta/Base/ --overwrite_output_dir=True --per_device_train_batch_size=20 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=bert-large-uncased  --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/Bert/Large/ --overwrite_output_dir=True --per_device_train_batch_size=8 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=roberta-large  --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/RoBerta/Large/ --overwrite_output_dir=True --per_device_train_batch_size=8 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=microsoft/deberta-large  --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/Deberta/Large/ --overwrite_output_dir=True --per_device_train_batch_size=5 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /usr/bin/env python sweep/dist_runner.py LMs/train_MLM.py --do_train=True --learning_rate=5e-05 --model_name_or_path=google/electra-large-discriminator  --num_train_epochs=1 --output_dir=/mnt/v-wzhuang/TAG/Prt/TMLM/Amazon/Photo/Electra/Large/ --overwrite_output_dir=True --per_device_train_batch_size=8 --train_file=/mnt/v-wzhuang/TAG-Benchmark/data/Photo_RS/photo_TMLM.txt --warmup_steps=1000 --line_by_line=True


base
Deberta 12
Bert  20
RoBerta


train_file=/mnt/v-wzhuang/TAG-Benchmark/data/arxiv_TA/arxiv_TA_TMLM.txt





/mnt/v-wzhuang/TAG/Prt/TMLM/Arxiv/Deberta/Base/
/mnt/v-wzhuang/TAG/Prt/TMLM/Arxiv/RoBerta/Base/
/mnt/v-wzhuang/TAG/Prt/TMLM/Arxiv/Electra/Base/
/mnt/v-wzhuang/TAG/Prt/TMLM/Arxiv/Bert/Base/

/mnt/v-wzhuang/TAG/Prt/TMLM/Arxiv/Deberta/Large/
/mnt/v-wzhuang/TAG/Prt/TMLM/Arxiv/RoBerta/Large/
/mnt/v-wzhuang/TAG/Prt/TMLM/Arxiv/Electra/Large/
/mnt/v-wzhuang/TAG/Prt/TMLM/Arxiv/Bert/Large/-