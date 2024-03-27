import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer,BertTokenizer
import utils.function as uf
from utils.settings import *
from tqdm import tqdm
from utils.function.os_utils import mkdir_p

def _tokenize_amazon_datasets(d):
    if not osp.exists(osp.join(d.data_root, f'{d.data_name}.csv')):
        mkdir_p(d.data_root)
        print(f'Please check there is a file in the {d.data_root}')
    #! Tokenize the data
    else:
        df = pd.read_csv(osp.join(d.data_root, f'{d.data_name}.csv'))
        text = df['text'].tolist()
        # text = pd.read_csv(osp.join(d.data_root, f'{d.data_name}.txt'), header=None, sep='\t')
        # text = text[0]
        # Look at
    #! For debug
    tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
    tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=d.max_length,
                          return_token_type_ids=True).data
    mkdir_p(d._token_folder)
    for k in tokenized:
        with open(osp.join(d._token_folder, f'{k}.npy'), 'wb') as f:
            np.save(f, tokenized[k])
    uf.pickle_save('processed', d._processed_flag['token'])
    return