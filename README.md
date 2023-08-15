# CS-TAG 
Text-attributed Graph Datasets and Benchmark 

## Datasets and tasks
In our CS-TAG benchmark, we collect and make 8 text-attributed graph datasets from ogbn-arxiv, amazon, dblp and goodreads.
Except for ogbn-arxiv related datasets(Arxiv-TA), the rest of the datasets are constructed by us and uploaded to [google drive](https://drive.google.com/drive/folders/1bdBWkaIzRfbREN7dSndLcL-sKmQd4IqK).
You can download the datasets we process through the [google drive](https://drive.google.com/drive/folders/1bdBWkaIzRfbREN7dSndLcL-sKmQd4IqK) link. (You can use **gdown** to download the file you wanted in Linux.)

In CS-TAG, we perform the supervised node classification and link prediction tasks common on graphs on these datasets. 
However, depending on the characteristics of the different datasets, many other types of tasks can be continued to be mined in TAG. 
Here are some of the tasks we will be focusing on
- [] Self-supervised learning on TAGs
- [] Study of Robustness on TAGs
- [] Graph structure learning on TAGs

## Directory Layout
```bash
./GNN                
|---- model/                
|        |---- Dataloader.py    # Load the data from CS-TAG     	
|        |---- GNN_arg.py       # GNN settings (e.g. dropout, n-layers, n-hidden)
|        |---- GNN_library.py   # CS-TAG GNN baselines(e.g., mlp, GCN, GAT)
|---- GNN.py                    # .py for node classification task
|---- GNN_Link.py                    # .py for link prediction task
./LMs
|---- Model/
|        |---- Bert    # Save the config for the TinyBert, Bert-base and Bert-large
|        |---- Deberta    # Save the config for the Deberta-base and Deberta-large
|        |---- Distilbert    # Save the config for the Distilbert
|        |---- Electra    # Save the config for the Electra-small, Electra-base and Electra-large
|---- Train_Command/
|        |---- Pretrain/    # Save the scripts for the topological pretraining 
|                |---- Scripts/    # Save the scripts for the topological pretraining 
|                       |---- TCL.sh   #  Scripts for the TCL
|                       |---- TMLM.sh   #  Scripts for the TMLM
|                       |---- TDK.sh   #  Scripts for the TDK
|                       |---- TMDC.sh   #  Scripts for the TMDC
|        |---- Co-Train.py    # .py for the Co-Training strategy
|        |---- Toplogical_Pretrain.py    # .py for the toplogical pretraining strategy (e.g., TCL,TDK,TMLM, TCL+TDK)
|---- Trainer/
|        |---- Inf_trainer.py            # .py for getting node embedding from the PLMs
|        |---- TCL_trainer.py            # Trainer (following the huggingface) for the TCL strategy
|        |---- TDK_trainer.py            # Trainer (following the huggingface) for the TDK strategy
|        |---- TMDC_trainer.py            # Trainer (following the huggingface) for the TMDC strategy
|        |---- TLink_trainer.py            # Trainer (following the huggingface) for the TCL in the Link prediction tasks 
|        |---- lm_trainer.py                 # Trainer for node classification tasks
|        |---- train_MLM.py                 #  .py for the TMLM tasks (following the huggingface)
|---- utils/
|        |---- data/    # Save the scripts for the topological pretraining 
|                |---- data_augmentation.py # the .py for generating the corpus for the TMLM tasks
|                |---- datasets.py #  The defined dataset class for different tasks
|                |---- preprocess.py #  Some commands for preprocessing the data (e.g. tokenize_graph, split_graph)
|        |---- function 
|                |---- dgl_utils.py   # Some commands from dgl 
|                |---- hf_metric.py   # Some metric used in this benchmark (e.g. accuracy, f1)
|        |---- modules
|                |---- conf_utils.py
|                |---- logger.py
|        |---- settings.py    # Some config for the datasets. You can creat your dataset in this file!  
|---- model.py       # Define the model for the donstream tasks
|---- lm_utils.py    # Define the config for the PLM pipeline
|---- trainLM.py     # Running for the node classification tasks
|---- dist_runner.py  # Parallel way to training the model
```


## Environments
You can quickly install the corresponding dependencies
```shell
conda env create -f environment.yml
```

## Main experiments in CS-TAG
Representation learning on the TAGs often depend on the two type models: Graph Neural Networks and Language Models.
For the latter, we often use the Pretrained Language Models (PLMs) to encode the text.
For the GNNs, we follow the [DGL](https://www.dgl.ai/) toolkit and implement them in the GNN library.
For the PLMs, we follow the [huggingface](https://huggingface.co/) trainer to implement the PLMs in a same pipeline.
We know that there are no absolute fair between the two type baselines.

We use the [wandb](https://wandb.ai/site) to log the results of our experiments.
We make public the logs of some of our experiments done and organized to promote more researchers to study TAG.
- [x] [Node classification from GNN](https://wandb.ai/csu_tag/OGB-Arxiv-GNN/reports/GNN-Accuracy--Vmlldzo0MjcyMzk4)
- [x] [LM related in Ele-computers](https://wandb.ai//csu_tag/Computers/reports/Ele-Computers--Vmlldzo0NjMxNTA4)

### Future work
In the CS-TAG, we mainly explore the form of classification tasks on TAGs, so we mainly use the mask language models.
But in recent years, the autoregressive language models have recently evolved rapidly, with models with increasingly larger and 
models that work increasingly well on the generative tasks.
![LLM](LLM.png)
To this end, in the future we will explore some suitable forms of generative tasks on TAGs to analyze the performance performance of different large language models(ChatGPT, GPT-4, LLaMA, and so on.).

