import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_undirected

from ogb.linkproppred import PygLinkPropPredDataset
from model.Dataloader import Evaluator, split_edge_MMR, from_dgl
from model.GNN_arg import Logger
import dgl
import numpy as np
import wandb
import os

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, x, graph, edge_split, optimizer, batch_size):
    model.train()
    predictor.train()

    source_edge = edge_split['train']['source_node'].to(x.device)
    target_edge = edge_split['train']['target_node'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(source_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, graph.adj_t)

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, x.size(0), src.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, graph, edge_split, evaluator, batch_size, neg_len):
    model.eval()
    predictor.eval()

    h = model(x, graph.adj_t)

    def test_split(split, neg_len):
        source = edge_split[split]['source_node'].to(h.device)
        target = edge_split[split]['target_node'].to(h.device)
        target_neg = edge_split[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, neg_len).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, neg_len)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train', neg_len)
    valid_mrr = test_split('valid', neg_len)
    test_mrr = test_split('test', neg_len)

    return train_mrr, valid_mrr, test_mrr


def main():
    parser = argparse.ArgumentParser(description='Link-Prediction PLM/TCL')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=5)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gnn_model', type=str, help='GNN MOdel', default='GCN')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--neg_len', type=int, default=2000)
    parser.add_argument("--use_PLM", type=str, default="/mnt/v-wzhuang/TAG/Finetune/DBLP/CitationV8/TinyBert/emb.npy",
                        help="Use LM embedding as feature")
    parser.add_argument("--path", type=str, default="/mnt/v-wzhuang/TAG/Link_Predction/DBLP-2015/",
                        help="Path to save splitting")
    parser.add_argument("--graph_path", type=str, default="/mnt/v-wzhuang/DBLP/Citation-2015.pt",
                        help="Path to load the graph")
    args = parser.parse_args()
    wandb.config = args
    wandb.init(config=args, reinit=True)
    print(args)

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    graph = dgl.load_graphs(f'{args.graph_path}')[0][0]

    edge_split, train_G = split_edge_MMR(graph, time=2015, random_seed=42, neg_len=args.neg_len, path=args.path)


    x = torch.from_numpy(np.load(args.use_PLM).astype(np.float32)).to(device)
    x = x.to(device)

    torch.manual_seed(42)
    idx = torch.randperm(edge_split['train']['source_node'].numel())[:len(edge_split['valid']['source_node'])]
    edge_split['eval_train'] = {
        'source_node': edge_split['train']['source_node'][idx],
        'target_node': edge_split['train']['target_node'][idx],
        'target_node_neg': edge_split['valid']['target_node_neg'],
    }
    #len(edge_split['train']['source_node']) = 6110221  valid 5338 test 5338


    train_G  = from_dgl(train_G)
    train_G = T.ToSparseTensor()(train_G) #完整数据
    train_G.adj_t = train_G.adj_t.to_symmetric()
    train_G = train_G.to(device)


    if args.gnn_model == 'SAGE':
        model = SAGE(x.size(1), args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.gnn_model == 'GCN':
        model = GCN(x.size(1), args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.gnn_model == 'GAT':
        model = GAT(x.size(1), args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.heads,
                    args.dropout).to(device)
    else:
        raise ValueError('Not implemented')

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='DBLP')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, x, train_G, edge_split, optimizer,
                         args.batch_size)
            wandb.log({'Loss': loss})
            if epoch % args.eval_steps == 0:
                result = test(model, predictor, x, train_G, edge_split, evaluator,
                               args.batch_size, args.neg_len)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_mrr, valid_mrr, test_mrr = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_mrr:.4f}, '
                          f'Valid: {valid_mrr:.4f}, '
                          f'Test: {test_mrr:.4f}')

        logger.print_statistics(run)

    logger.print_statistics(key='mrr')


if __name__ == "__main__":
    main()


