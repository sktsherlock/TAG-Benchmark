# Welcome to the scripts used to extract text features 😎

## Quick Star 🚀

### 1. Download the datasets from [CSTAG](https://huggingface.co/datasets/Sherirto/CSTAG) 👐

```bash
cd ../data/
sudo apt-get update && sudo apt-get install git-lfs && git clone https://huggingface.co/datasets/Sherirto/CSTAG
cd CSTAG && ls 
```
Now, you can see the **Arxiv**, **Children**, **CitationV8**, **Computers**, **Fitness**, **Goodreads**, **History** and  **Photo** under the **''data/CSTAG''** folder.

### 2. Extract features from the dataset you care about 👋

```bash
cd ../../FeatureExtractor/
# Extract features by LM4Feature.py 
CUDA_VISIBLE_DEVICES=0 python LM4Feature.py --csv_file 'data/CSTAG/Arxiv/Arxiv.csv' --model_name 'bert-base-uncased' --name 'Arxiv' --path 'data/CSTAG/Arxiv/Feature/' --max_length 512 --batch_size 500  
```


