import numpy as np
import os.path as osp
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
from ogb.nodeproppred import DglNodePropPredDataset

def check_dir(file_name=None):
    dir_name = osp.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def plot_embedding(args, labels):
    embedding_path = args.embedding_path
    figure_path = args.figure_path
    if not osp.exists(embedding_path):
        raise ValueError('No embedding path')
    check_dir(figure_path)

    embeddings = np.load(osp.join(embedding_path, "emb.npy"))
    tsne = TSNE(init='pca', random_state=0)

    tsne_features = tsne.fit_transform(embeddings)

    xs = tsne_features[:, 0]
    ys = tsne_features[:, 1]

    plt.scatter(xs, ys, c=labels)
    figure_name = osp.join(figure_path, args.dataset.lower() + ".pdf")
    check_dir(figure_name)
    plt.savefig(figure_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSNE")
    parser.add_argument("--dataset", type=str, default='ogbn-arxiv')
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--figure_path", type=str, default=None)
    args = parser.parse_args()

    dataset = DglNodePropPredDataset('ogbn-arxiv', root='/mnt/v-haoyan1/CirTraining/data/ogb/')
    g, labels = dataset[0]

    plot_embedding(args, labels=labels)