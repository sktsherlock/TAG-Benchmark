#%%
import itertools
import os
from sklearn.metrics import roc_auc_score
os.environ["DGLBACKEND"] = "pytorch"
from dgl.nn import SAGEConv
from model.GNN_library import GraphSAGE
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
import dgl

g = dgl.load_graphs("/mnt/v-wzhuang/Amazon/Books/Amazon-Books-History.pt")[0][0]
g = dgl.to_bidirected(g)
#%%
# Split edge set for training and testing
u, v = g.edges()
np.random.seed(42)
eids = np.arange(g.num_edges())
eids = np.random.permutation(eids) # 打乱边顺序

test_size = int(len(eids) * 0.2)
train_size = g.num_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

#! 选择10000个作为负样本
neg_eids = np.random.choice(len(neg_u), g.num_edges())

test_neg_u, test_neg_v = (
    neg_u[neg_eids[:test_size]],
    neg_v[neg_eids[:test_size]],
)
train_neg_u, train_neg_v = (
    neg_u[neg_eids[test_size:]],
    neg_v[neg_eids[test_size:]],
)
#%%
device = f'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

train_g = dgl.remove_edges(g, eids[:test_size]).to(device)
#%%



# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model

#%%
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes()).to(device)
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes()).to(device)

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes()).to(device)
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes()).to(device)
#%%
import dgl.function as fn


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]
#%%
feat = torch.from_numpy(np.load('/mnt/v-wzhuang/TAG/Finetune/Amazon/History/TinyBert/emb.npy').astype(np.float32)).to(device)
in_feats = feat.shape[1]

model = GraphSAGE(in_feats, 16, n_classes=16, n_layers=2, activation= F.relu,
                  dropout=0.2, aggregator_type='mean').to(device)
# You can replace DotPredictor with MLPPredictor.
# pred = MLPPredictor(16)
pred = DotPredictor().to(device)


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).cpu().numpy()
    return roc_auc_score(labels, scores)
#%%
# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=0.01
)

with torch.no_grad():
    h = model(train_g, feat).to(device)
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print("AUC", compute_auc(pos_score, neg_score))

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(100):
    # forward
    h = model(train_g, feat).to(device)
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

# ----------- 5. check results ------------------------ #


with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print("AUC", compute_auc(pos_score, neg_score))

