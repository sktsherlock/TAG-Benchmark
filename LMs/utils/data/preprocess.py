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
            elif d.md['type'] in {'amazon', 'dblp', 'good'}:
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
    val_ids = indices[train_size:train_size + val_size]
    test_ids = indices[train_size + val_size:]

    return train_ids, val_ids, test_ids


def split_time(g, train_year=2016, val_year=2017):
    np.random.seed(42)
    year = list(np.array(g.ndata['year']))
    indices = np.arange(g.num_nodes())
    # 1999-2014 train
    # Filter out nodes with label -1
    print(f'train year: {train_year}')
    valid_indices = [i for i in indices if g.ndata['label'][i] != -1]

    # Filter out valid indices based on years
    train_ids = [i for i in valid_indices if year[i] < train_year]
    val_ids = [i for i in valid_indices if year[i] >= train_year and year[i] < val_year]
    test_ids = [i for i in valid_indices if year[i] >= val_year]


    train_length = len(train_ids)
    val_length = len(val_ids)
    test_length = len(test_ids)

    print("Train set length:", train_length)
    print("Validation set length:", val_length)
    print("Test set length:", test_length)

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
    g = dgl.load_graphs(f"{cf.data.data_root}{cf.data.data_name}.pt")[0][0]
    labels = g.ndata['label'].numpy()
    if cf.splits == 'random':
        split_idx = split_graph(g.num_nodes(), cf.train_ratio, cf.val_ratio)
    elif cf.splits == 'time':
        split_idx = split_time(g, cf.data.train_year, cf.data.val_year)
    else:
        raise ValueError('Please check the split datasets way')

    return g, labels, split_idx


def load_dblp_graph_structure_only(cf):
    import dgl
    g = dgl.load_graphs(f"{cf.data.data_root}{cf.data.data_name}.pt")[0][0]
    return g


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
            elif d.md['type'] == 'dblp':
                g = load_dblp_graph_structure_only(cf)
                g_info = SN(n_nodes=g.num_nodes())
            elif d.md['type'] == 'good':
                g = load_dblp_graph_structure_only(cf)
                g_info = SN(n_nodes=g.num_nodes())
            else:
                raise NotImplementedError  #
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

