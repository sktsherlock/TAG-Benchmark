# Welcome to the scripts used to extract text features! 😎

## Quick Star 🚀

### 1. Download the datasets from [CSTAG](https://huggingface.co/datasets/Sherirto/CSTAG). 👐

```bash
cd ../data/
sudo apt-get update && sudo apt-get install git-lfs && git clone https://huggingface.co/datasets/Sherirto/CSTAG
cd CSTAG && ls 
```
Now, you can see the **Arxiv**, **Children**, **CitationV8**, **Computers**, **Fitness**, **Goodreads**, **History** and  **Photo** under the **''data/CSTAG''** folder.

### 2. Extract features on the datasets you care about with PLM on huggingface. 👋

```bash
cd ../../FeatureExtractor/
# Extract features by LM4Feature.py 
CUDA_VISIBLE_DEVICES=0 python LM4Feature.py --csv_file 'data/CSTAG/Arxiv/Arxiv.csv' --model_name 'bert-base-uncased' --name 'Arxiv' --path 'data/CSTAG/Arxiv/Feature/' --max_length 512 --batch_size 500  
# If you have multiple GPUs, you can simply execute this code in parallel.If you have multiple GPUs, you can simply execute this code in parallel.
CUDA_VISIBLE_DEVICES=0,1,2,3 python LM4Feature.py --csv_file 'data/CSTAG/Arxiv/Arxiv.csv' --model_name 'bert-base-uncased' --name 'Arxiv' --path 'data/CSTAG/Arxiv/Feature/' --max_length 512 --batch_size 500  
cd ../data/CSTAG/Arxiv/Feature/ && ls
```

If you follow the example code above, then you can see the feature file named <font color=#00ffff>"Arxiv_bert_base_uncased_512_cls.npy"</font>. Where <font color=#00ffff>'Arxiv'</font> is determined by the **--name 'Arxiv'** in the script; <font color=#00ffff>'512'</font> is determined by --max_length, and <font color=#00ffff>'cls'</font> is the default text representation.



