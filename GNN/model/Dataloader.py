from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import numpy as np
import torch as th

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
    year = list(np.array(g.ndata['year']))
    indices = np.arange(g.num_nodes())
    # 1999-2014 train
    train_ids = indices[:year.index(train_year)]
    val_ids = indices[year.index(train_year):year.index(val_year)]
    test_ids = indices[year.index(val_year):]

    return train_ids, val_ids, test_ids

def load_data(name, train_ratio=0.6, val_ratio=0.2):
    if name == 'ogbn-arxiv':
        data = DglNodePropPredDataset(name=name)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        graph, labels = data[0]
        labels = labels[:, 0]
    elif name == 'amazon-children':
        graph = dgl.load_graphs('/mnt/v-wzhuang/Amazon/Books/Amazon-Books-Children.pt')[0][0]
        labels = graph.ndata['label']
        train_idx, val_idx, test_idx = split_graph(graph.num_nodes(), 0.6, 0.2)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == 'amazon-history':
        graph = dgl.load_graphs('/mnt/v-wzhuang/Amazon/Books/Amazon-Books-History.pt')[0][0]
        labels = graph.ndata['label']
        train_idx, val_idx, test_idx = split_graph(graph.num_nodes(), 0.6, 0.2)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == 'amazon-fitness':
        graph = dgl.load_graphs('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Sports/Fitness/Sports-Fitness.pt')[0][0]
        labels = graph.ndata['label']
        train_idx, val_idx, test_idx = split_graph(graph.num_nodes(), train_ratio, val_ratio)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    elif name == 'amazon-photo':
        graph = dgl.load_graphs('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Electronics/Photo/Electronics-Photo.pt')[0][0]
        labels = graph.ndata['label']
        train_idx, val_idx, test_idx = split_graph(graph, 2015, 2016)
        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)
    else:
        raise ValueError('Not implemetned')
    return graph, labels, train_idx, val_idx, test_idx
