import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer,BertTokenizer
import utils.function as uf
from utils.settings import *
from tqdm import tqdm
from utils.function.os_utils import mkdir_p

def _tokenize_webkb_datasets(d):
    #! 创建目录
    print(d.data_root)
    if not osp.exists(osp.join(d.data_root, f'{d.data_name}.txt')):
        mkdir_p(d.data_root)
        raise{f'Please input the txt to the {d.data_root}'}
    #! Tokenize the data
    else:
        text = pd.read_csv(osp.join(d.data_root, f'{d.data_name}.txt'), header=None, sep='\t')
        text = text[0]
        # Look at
    #! For debug
    tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
    tokenized = tokenizer(text.tolist(), padding='max_length', truncation=True, max_length=d.max_length,
                          return_token_type_ids=True).data
    mkdir_p(d._token_folder)
    for k in tokenized:
        with open(osp.join(d._token_folder, f'{k}.npy'), 'wb') as f:
            np.save(f, tokenized[k])
    uf.pickle_save('processed', d._processed_flag['token'])
    return