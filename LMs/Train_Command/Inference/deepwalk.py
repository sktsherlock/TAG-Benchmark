import numpy as np
import os
import dgl
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset
from gensim.models import Word2Vec
import networkx as nx
from karateclub import DeepWalk

print("ok")
# root_dir = os.path.abspath(os.path.expanduser('/mnt/v-wzhuang/TAG-Benchmark/data/ogb/'))
# dataset = DglNodePropPredDataset('ogbn-arxiv', root=root_dir)
g = dgl.load_graphs('/mnt/v-wzhuang/Amazon/Books/Amazon-Books-History.pt')[0][0]
g = dgl.to_bidirected(g)
nx_G = g.to_networkx()
model = DeepWalk(walk_number=10,walk_length=80,dimensions=128,workers=4,window_size=5,epochs=1,learning_rate=0.05,min_count=1,seed=42)  # node embedding algorithm
model.fit(nx_G)  # fit it on the graph
embedding = model.get_embedding()  # extract embeddings
np.save('/mnt/v-wzhuang/TAG-Benchmark/data/amazon/Books/History/deepwalk_feat.npy', embedding)
print("finish")