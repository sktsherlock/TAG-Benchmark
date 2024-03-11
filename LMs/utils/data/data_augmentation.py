import pandas as pd
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import copy
import random
import os.path as osp
import argparse


def args_init():
    argparser = argparse.ArgumentParser(
        "Topology augmentation of the text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default='ogbn-arxiv',
        help="Which dataset to be augmented",
    )
    argparser.add_argument("--path", type=str, default=None, required=True, help="Path to save the augmented-text")
    argparser.add_argument("--stat_path", type=str, default=None, required=True, help="Path to save the augmented-text")
    argparser.add_argument("--epochs", type=int, default=5, required=True, help="The sampling runs")
    argparser.add_argument("--max_length", type=int, default=512, required=True, help="The sampling runs")

    return argparser


def Toplogy_Augment(text, edge0, edge1):
    edge0 = np.array(edge0)
    edge1 = np.array(edge1)
    assert len(edge0) == len(edge1)
    ans = copy.copy(text)
    for i, j in zip(edge0, edge1):
        ans[i] += ' Neighbour: ' + text[j]
    text = ans
    return text


def load_data(name):
    if name == 'ogbn-arxiv':
        data = DglNodePropPredDataset(name=name)
        graph, labels = data[0]


    elif name == 'amazon-children':
        graph = dgl.load_graphs('/mnt/v-wzhuang/Amazon/Books/Amazon-Books-Children.pt')[0][0]
        labels = graph.ndata['label']

    elif name == 'amazon-history':
        graph = dgl.load_graphs('/mnt/v-wzhuang/Amazon/Books/Amazon-Books-History.pt')[0][0]
        labels = graph.ndata['label']

    elif name == 'amazon-fitness':
        graph = dgl.load_graphs('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Sports/Fit/Sports-Fitness.pt')[0][0]
        labels = graph.ndata['label']

    elif name == 'amazon-photo':
        graph = dgl.load_graphs('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Electronics/Photo/Electronics-Photo.pt')[0][0]
        labels = graph.ndata['label']
    elif name == 'amazon-computer':
        graph = dgl.load_graphs('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Electronics/Computers/Electronics-Computers.pt')[0][0]
        labels = graph.ndata['label']
    elif name == 'webkb-cornell':
        graph = dgl.load_graphs('data/WebKB/Cornell/Cornell.pt')[0][0]
        labels = graph.ndata['label']
    elif name == 'webkb-texas':
        graph = dgl.load_graphs('data/WebKB/Texas/Texas.pt')[0][0]
        labels = graph.ndata['label']
    elif name == 'webkb-washington':
        graph = dgl.load_graphs('data/WebKB/Washington/Washington.pt')[0][0]
        labels = graph.ndata['label']
    elif name == 'webkb-wisconsin':
        graph = dgl.load_graphs('data/WebKB/Wisconsin/Wisconsin.pt')[0][0]
        labels = graph.ndata['label']
    else:
        raise ValueError('Not implemetned')
    return graph, labels


def main():
    argparser = args_init()
    args = argparser.parse_args()
    graph, labels = load_data(name=args.dataset)

    graph.remove_self_loop()
    graph = dgl.to_bidirected(graph)
    neighbours = list(graph.adjacency_matrix_scipy().tolil().rows)  # 一阶邻居 获得
    if args.dataset == 'ogbn-arxiv':
        text = pd.read_csv('/mnt/v-wzhuang/TAG-Benchmark/data/ogb/ogbn_arxiv/ogbn-arxiv.txt', sep='\t', header=None)
        text = text[0]
    elif args.dataset == 'amazon-photo':
        text = pd.read_csv('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Electronics/Photo/Electronics-Photo.txt', sep='\t', header=None)
        text = text[0]
    elif args.dataset == 'amazon-children':
        text = pd.read_csv('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Books/Children/Books-Children.txt', sep='\t', header=None)
        text = text[0]
    elif args.dataset == 'amazon-history':
        text = pd.read_csv('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Books/History/Books-History.txt', sep='\t', header=None)
        text = text[0]
    elif args.dataset == 'amazon-fitness':
        text = pd.read_csv('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Sports/Fit/Sports-Fitness.txt', sep='\t', header=None)
        text = text[0]
    elif args.dataset == 'amazon-computer':
        text = pd.read_csv('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Electronics/Computers/Electronics-Computers.txt', sep='\t', header=None)
        text = text[0]
    elif args.dataset == 'webkb-cornell':
        text = pd.read_csv('data/WebKB/Cornell/Cornell.txt', sep='\t', header=None)
        text = text[0]
    elif args.dataset == 'webkb-texas':
        text = pd.read_csv('data/WebKB/Texas/Texas.txt', sep='\t', header=None)
        text = text[0]
    elif args.dataset == 'webkb-washington':
        text = pd.read_csv('data/WebKB/Washington/Washington.txt', sep='\t', header=None)
        text = text[0]
    elif args.dataset == 'webkb-wisconsin':
        text = pd.read_csv('data/WebKB/Wisconsin/Wisconsin.txt', sep='\t', header=None)
        text = text[0]
    else:
        raise ValueError('Not implemented.')
    # Augmentation
    new_dataset = []
    for index, the_text in enumerate(text):
        neighbour = neighbours[index]
        sampled_neighbours = np.random.choice(neighbour, args.epochs)

        # 将当前节点和每个邻居的文本拼接起来
        for neighbour_id in sampled_neighbours:
            new_text = the_text + ' ' + text[neighbour_id]

            # 将新数据加入到新数据集中
            new_dataset.append(new_text)
    # 打乱数据
    random.shuffle(new_dataset)
    df = pd.DataFrame(new_dataset)
    # 截断数据
    df[0] = df.apply(lambda x: ' '.join(x[0].split(' ')[:args.max_length]), axis=1)
    length = df.apply(lambda x: len(x[0].split(' ')), axis=1)
    text_stat = pd.DataFrame(length.describe())
    text_stat.index.rename('Statics', inplace=True)
    text_stat.columns = ["Length"]
    text_stat.to_csv(args.stat_path)
    # 保存text，text_len 文件
    df.to_csv(args.path, sep='\t', header=None, index=False)


if __name__ == '__main__':
    main()
