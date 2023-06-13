# Small Model
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/TinyBert/' --dataset arxiv_TA --inf_batch_size 600
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Distilbert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Distilbert/' --dataset arxiv_TA --inf_batch_size 800
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Electra  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Electra/Small/' --dataset arxiv_TA --inf_batch_size 800
# Base Model
CUDA_VISIBLE_DEVICES=4,5,6,7 python Train_Command/inference_LM.py --model RoBerta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/RoBert/Base/' --dataset arxiv_TA --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=4,5,6,7 python Train_Command/inference_LM.py --model Bert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Bert/Base/' --dataset arxiv_TA --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=4,5,6,7 python Train_Command/inference_LM.py --model Electra-base  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Electra/Base/' --dataset arxiv_TA --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=4,5,6,7 python Train_Command/inference_LM.py --model Deberta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Deberta/Base/' --dataset arxiv_TA --inf_batch_size 1000
# Large Model
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python Train_Command/inference_LM.py --model Roberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/RoBert/Large/' --dataset arxiv_TA --inf_batch_size 300
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python Train_Command/inference_LM.py --model Bert-large   --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Bert/Large/' --dataset arxiv_TA --inf_batch_size 300
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python Train_Command/inference_LM.py --model Electra-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Electra/Large/' --dataset arxiv_TA --inf_batch_size 300
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python Train_Command/inference_LM.py --model Deberta-large --inference_dir '/mnt/v-wzhuang/TAG/Finetune/OGB/Arxiv/Deberta/Large/' --dataset arxiv_TA --inf_batch_size 300

# Amazon-Books-Children/History
# Small
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/TinyBert/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Distilbert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Distilbert/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Electra  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Electra/Small/' --dataset Children_DT --inf_batch_size 1000
# Base
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model RoBerta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/RoBert/Base/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Bert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Bert/Base/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Electra-base  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Electra/Base/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Deberta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Deberta/Base/' --dataset Children_DT --inf_batch_size 1000
# Large
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Roberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/RoBert/Large/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Bert-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Bert/Large/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Electra-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Electra/Large/' --dataset Children_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Deberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Children/Deberta/Large/' --dataset Children_DT --inf_batch_size 1000


# Small
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/TinyBert/' --dataset History_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Distilbert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/Distilbert/' --dataset History_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Electra  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/Electra/Small/' --dataset History_DT --inf_batch_size 1000
# Base
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model RoBerta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/RoBert/Base/' --dataset History_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Bert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/Bert/Base/' --dataset History_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Electra-base  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/Electra/Base/' --dataset History_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Deberta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/Deberta/Base/' --dataset History_DT --inf_batch_size 1000
# Large
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Roberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/RoBert/Large/' --dataset History_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Bert-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/Bert/Large/' --dataset History_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Electra-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/Electra/Large/' --dataset History_DT --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model Deberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/History/Deberta/Large/' --dataset History_DT --inf_batch_size 1000

# Sports-Fitness
# Small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/TinyBert/' --dataset Fitness_TT --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Distilbert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/Distilbert/' --dataset Fitness_TT --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/Electra/Small/' --dataset Fitness_TT --inf_batch_size 2000
# Base
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model RoBerta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/RoBert/Base/' --dataset Fitness_TT --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Bert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/Bert/Base/' --dataset Fitness_TT --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra-base  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/Electra/Base/' --dataset Fitness_TT --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Deberta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/Deberta/Base/' --dataset Fitness_TT --inf_batch_size 2000
# Large
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Roberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/RoBert/Large/' --dataset Fitness_TT --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Bert-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/Bert/Large/' --dataset Fitness_TT --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/Electra/Large/' --dataset Fitness_TT --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Deberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Fitness/Deberta/Large/' --dataset Fitness_TT --inf_batch_size 2000


#! ELECTRONICS-Photo
# Sports-Fitness
# Small
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/TinyBert/' --dataset Photo_RS --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Distilbert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/Distilbert/' --dataset Photo_RS --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/Electra/Small/' --dataset Photo_RS --inf_batch_size 1000
# Base
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model RoBerta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/RoBert/Base/' --dataset Photo_RS --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Bert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/Bert/Base/' --dataset Photo_RS --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra-base  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/Electra/Base/' --dataset Photo_RS --inf_batch_size 1000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Deberta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/Deberta/Base/' --dataset Photo_RS --inf_batch_size 1000
# Large
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Roberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/RoBert/Large/' --dataset Photo_RS --inf_batch_size 300
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Bert-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/Bert/Large/' --dataset Photo_RS --inf_batch_size 300
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/Electra/Large/' --dataset Photo_RS --inf_batch_size 300
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Deberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Photo/Deberta/Large/' --dataset Photo_RS --inf_batch_size 300

#! Digital Muics for link prediction need the feature
CUDA_VISIBLE_DEVICES=6,7 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Music/TinyBert/' --dataset Music_T --inf_batch_size 1000

#! DBLP
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/TinyBert/' --dataset DBLP_T --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Distilbert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/Distilbert/' --dataset DBLP_T --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/Electra/Small/' --dataset DBLP_T --inf_batch_size 2000
# Base
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model RoBerta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/RoBert/Base/' --dataset DBLP_T --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Bert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/Bert/Base/' --dataset DBLP_T --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra-base  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/Electra/Base/' --dataset DBLP_T --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Deberta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/Deberta/Base/' --dataset DBLP_T --inf_batch_size 2000
# Large
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Roberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/RoBert/Large/' --dataset DBLP_T --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Bert-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/Bert/Large/' --dataset DBLP_T --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/Electra/Large/' --dataset DBLP_T --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Deberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/Deberta/Large/' --dataset DBLP_T --inf_batch_size 2000

#! Computers
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/TinyBert/' --dataset Computers_RS --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Distilbert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/Distilbert/' --dataset Computers_RS --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/Electra/Small/' --dataset Computers_RS --inf_batch_size 2000
# Base
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model RoBerta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/RoBert/Base/' --dataset Computers_RS --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Bert  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/Bert/Base/' --dataset Computers_RS --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra-base  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/Electra/Base/' --dataset Computers_RS --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Deberta  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/Deberta/Base/' --dataset Computers_RS --inf_batch_size 500
# Large
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Roberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/RoBert/Large/' --dataset Computers_RS --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Bert-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/Bert/Large/' --dataset Computers_RS --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Electra-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/Electra/Large/' --dataset Computers_RS --inf_batch_size 2000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python Train_Command/inference_LM.py --model Deberta-large  --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Amazon/Computers/Deberta/Large/' --dataset Computers_RS --inf_batch_size 2000

#! Goodreads
CUDA_VISIBLE_DEVICES=0,1 python Train_Command/inference_LM.py --model TinyBert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Goodreads/TinyBert/' --dataset Good_T --inf_batch_size 5000
CUDA_VISIBLE_DEVICES=0,1,2,3 python Train_Command/inference_LM.py --model Bert --inference_dir '/mnt/v-wzhuang/TAG/Finetune/Goodreads/Bert/Base/' --dataset Good_T --inf_batch_size 2000