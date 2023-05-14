import gc
import os
import time

import dgl
import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer
import time
import utils.function as uf
from utils.function.dgl_utils import sample_nodes
from utils.settings import *
from tqdm import tqdm
from copy import deepcopy
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
from utils.data.OGB.arxiv import _tokenize_ogb_arxiv_datasets
from utils.data.Amazon.Amazon_data import _tokenize_amazon_datasets


def plot_length_distribution(node_text, tokenizer, g):
    sampled_ids = np.random.permutation(g.nodes())[:10000]
    get_text = lambda n: node_text.iloc[n]['text'].tolist()
    tokenized = tokenizer(get_text(sampled_ids), padding='do_not_pad').data['input_ids']
    node_text['text_length'] = node_text.apply(lambda x: len(x['text'].split(' ')), axis=1)
    pd.Series([len(_) for _ in tokenized]).hist(bins=20)
    import matplotlib.pyplot as plt
    plt.show()


def tokenize_graph(cf):
    # = Tokenization on Full Graph
    full_dict = deepcopy(cf.model_conf)
    full_dict['dataset'] = '_'.join(full_dict['dataset'].split('_')[:2])
    full_cf = cf.__class__(SN(**full_dict)).init()
    d = full_cf.data
    if not d.is_processed('token'):
        if cf.local_rank <= 0:
            # ! Load full-graph
            print(f'Processing data on LOCAL_RANK #{cf.local_rank}...')
            print(f'Loaded graph structure, start tokenization...')
            if d.md['type'] == 'ogb':
                if d.ogb_name == 'ogbn-arxiv':
                    _tokenize_ogb_arxiv_datasets(d)
                else:
                    raise NotImplementedError
            elif d.md['type'] == 'amazon':
                _tokenize_amazon_datasets(d)
            else:
                raise NotImplementedError
            print(f'Tokenization finished on LOCAL_RANK #{cf.local_rank}')
        else:
            # If not main worker (i.e. Local_rank!=0), wait until data is processed and load
            print(f'Waiting for tokenization on LOCAL_RANK #{cf.local_rank}')
            while not d.is_processed('token'):
                time.sleep(2)  # Check if processed every 2 seconds
            print(f'Detected processed data, LOCAL_RANK #{cf.local_rank} start loading!')
            time.sleep(5)  # Wait for file write for 5 seconds
    else:
        cf.log(f'Found processed {cf.dataset}.')

def split_graph(nodes_num, train_ratio, val_ratio):

    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_ids = indices[:train_size]
    val_ids = indices[train_size:train_size+val_size]
    test_ids = indices[train_size+val_size:]

    return train_ids, val_ids, test_ids

def split_time(g, train_year=2016, val_year=2017):
    np.random.seed(42)
    year = list(np.array(g.ndata['year']))
    indices = np.arange(g.num_nodes())
    # 1999-2014 train
    train_ids = indices[:year.index(train_year)]
    val_ids = indices[year.index(train_year):year.index(val_year)]
    test_ids = indices[year.index(val_year):]

    return train_ids, val_ids, test_ids


def load_ogb_graph_structure_only(cf):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(cf.data.ogb_name, root=uf.init_path(cf.data.raw_data_path))
    g, labels = data[0]
    split_idx = data.get_idx_split()
    labels = labels.squeeze().numpy()
    return g, labels, split_idx

def load_amazon_graph_structure_only(cf):
    import dgl
    g = dgl.load_graphs(f"{cf.data.data_root}{cf.data.amazon_name}.pt")[0][0]
    labels = g.ndata['label'].numpy()
    if cf.splits == 'random':
        split_idx = split_graph(g.num_nodes(), cf.train_ratio, cf.val_ratio)
    elif cf.splits == 'time':
        split_idx = split_time(g, 2015, 2016)
    else:
        raise ValueError('Please check the split datasets way')


    return g, labels, split_idx

def load_TAG_info(cf):
    d = cf.data
    # ! Process Full Graph
    if not d.is_processed('g_info'):
        if cf.local_rank <= 0:
            # 根据data中的type来实现不同的图数据加载代码
            if d.md['type'] == 'ogb':
                g, labels, split_idx = load_ogb_graph_structure_only(cf)
                # Process and save supervision
                splits = {**{f'{_}_x': split_idx[_].numpy() for _ in ['train', 'valid', 'test']}, 'labels': labels}
                # g, splits = _subset_graph(g, cf, splits)
                g_info = SN(splits=splits, labels=labels, n_nodes=g.num_nodes())
            elif d.md['type'] == 'amazon':
                g, labels, split_idx = load_amazon_graph_structure_only(cf)
                splits = {'train_x': split_idx[0], 'valid_x': split_idx[1], 'test_x': split_idx[2]}
                g_info = SN(splits=splits, labels=labels, n_nodes=g.num_nodes())
            else:
                raise NotImplementedError #
            d.save_g_info(g_info)
            del g
        else:
            # If not main worker (i.e. Local_rank!=0), wait until data is processed and load
            print(f'Waiting for feature processing on LOCAL_RANK #{cf.local_rank}')
            while not d.is_processed('g_info'):
                time.sleep(2)  # Check if processed every 2 seconds
            print(f'Detected processed feature, LOCAL_RANK #{cf.local_rank} start loading!')
            time.sleep(5)  # Wait f
    g_info = uf.pickle_load(d._g_info_file)
    return g_info


def tokenize_NP_graph(cf):
    from utils.data.OGB.arxiv import _tokenize_NP_ogb_arxiv_datasets
    full_dict = deepcopy(cf.model_conf)
    full_dict['dataset'] = '_'.join(full_dict['dataset'].split('_')[:2])
    full_cf = cf.__class__(SN(**full_dict)).init()
    d = full_cf.data
    if not d.is_processed('TNP_token'):
        if cf.local_rank <= 0:
            # ! Load full-graph
            print(f'Processing data on LOCAL_RANK #{cf.local_rank}...')
            g_info = load_TAG_info(full_cf)
            if d.md['type'] == 'ogb':
                if d.ogb_name == 'ogbn-arxiv':
                    _tokenize_NP_ogb_arxiv_datasets(d, g_info.labels, NP=True)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            print(f'Tokenization finished on LOCAL_RANK #{cf.local_rank}')
        else:
            # If not main worker (i.e. Local_rank!=0), wait until data is processed and load
            print(f'Waiting for tokenization on LOCAL_RANK #{cf.local_rank}')
            while not d.is_processed('TNP_token'):
                time.sleep(2)  # Check if processed every 2 seconds
            print(f'Detected processed data, LOCAL_RANK #{cf.local_rank} start loading!')
            time.sleep(5)  # Wait for file write for 5 seconds
    else:
        cf.log(f'Found processed NP {cf.dataset}.')

def tokenize_TRP_graph(cf):
    from utils.data.OGB.arxiv import _tokenize_TRP_ogb_arxiv_datasets
    full_dict = deepcopy(cf.model_conf)
    full_dict['dataset'] = '_'.join(full_dict['dataset'].split('_')[:2])
    full_cf = cf.__class__(SN(**full_dict)).init()
    d = full_cf.data
    if not d.is_processed('TRP_token'):
        if cf.local_rank <= 0:
            # ! Load full-graph
            print(f'Processing data on LOCAL_RANK #{cf.local_rank}...')
            g_info = load_TAG_info(full_cf)
            if d.md['type'] == 'ogb':
                if d.ogb_name == 'ogbn-arxiv':
                    _tokenize_TRP_ogb_arxiv_datasets(d, g_info.labels)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            print(f'Tokenization finished on LOCAL_RANK #{cf.local_rank}')
        else:
            # If not main worker (i.e. Local_rank!=0), wait until data is processed and load
            print(f'Waiting for tokenization on LOCAL_RANK #{cf.local_rank}')
            while not d.is_processed('TRP_token'):
                time.sleep(2)  # Check if processed every 2 seconds
            print(f'Detected processed data, LOCAL_RANK #{cf.local_rank} start loading!')
            time.sleep(5)  # Wait for file write for 5 seconds
    else:
        cf.log(f'Found processed TRP {cf.dataset}.')