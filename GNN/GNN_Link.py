# -*- coding: utf-8 -*-
import argparse
import time
import torch
import os
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce, SparseTensor
from torch_cluster import random_walk
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from plnlp.logger import Logger
from plnlp.model import BaseModel, adjust_lr
from plnlp.utils import gcn_normalization, adj_normalization


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='SAGE')
    parser.add_argument('--predictor', type=str, default='MLP')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_func', type=str, default='AUC')
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--data_name', type=str, default='ogbl-ddi')
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--eval_metric', type=str, default='hits')
    parser.add_argument('--walk_start_type', type=str, default='edge')
    parser.add_argument('--res_dir', type=str, default='')
    parser.add_argument('--pretrain_emb', type=str, default='')
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--emb_hidden_channels', type=int, default=256)
    parser.add_argument('--gnn_hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--grad_clip_norm', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--walk_length', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--year', type=int, default=-1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_lr_decay', type=str2bool, default=False)
    parser.add_argument('--use_node_feats', type=str2bool, default=False)
    parser.add_argument('--use_coalesce', type=str2bool, default=False)
    parser.add_argument('--train_node_emb', type=str2bool, default=True)
    parser.add_argument('--train_on_subgraph', type=str2bool, default=False)
    parser.add_argument('--use_valedges_as_input', type=str2bool, default=False)
    parser.add_argument('--eval_last_best', type=str2bool, default=False)
    parser.add_argument('--random_walk_augment', type=str2bool, default=False)
    args = parser.parse_args()
    return args

def main():
    args = argument()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.data_name, root=args.data_path)
    data = dataset[0]

    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)

    data = T.ToSparseTensor()(data)
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)

    if hasattr(data, 'num_features'):
        num_node_feats = data.num_features
    else:
        num_node_feats = 0

    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    else:
        num_nodes = data.adj_t.size(0)

    split_edge = dataset.get_edge_split()

    print(args)

    # create log file and save args
    log_file_name = 'log_' + args.data_name + '_' + str(int(time.time())) + '.txt'
    log_file = os.path.join(args.res_dir, log_file_name)
    with open(log_file, 'a') as f:
        f.write(str(args) + '\n')

    if hasattr(data, 'x'):
        if data.x is not None:
            data.x = data.x.to(torch.float)

    if args.data_name == 'ogbl-citation2':
        data.adj_t = data.adj_t.to_symmetric()

    if args.data_name == 'ogbl-collab':
        # only train edges after specific year
        if args.year > 0 and hasattr(data, 'edge_year'):
            selected_year_index = torch.reshape(
                (split_edge['train']['year'] >= args.year).nonzero(as_tuple=False), (-1,))
            split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
            split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
            split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
            train_edge_index = split_edge['train']['edge'].t()
            # create adjacency matrix
            new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0],
                                      col=new_edge_index[1],
                                      value=new_edge_weight.to(torch.float32))
            data.edge_index = new_edge_index

        # Use training + validation edges
        if args.use_valedges_as_input:
            full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
            full_edge_weight = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=-1)
            # create adjacency matrix
            new_edges = to_undirected(full_edge_index, full_edge_weight, reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0],
                                      col=new_edge_index[1],
                                      value=new_edge_weight.to(torch.float32))
            data.edge_index = new_edge_index

            if args.use_coalesce:
                full_edge_index, full_edge_weight = coalesce(full_edge_index, full_edge_weight, num_nodes, num_nodes)

            # edge weight normalization
            split_edge['train']['edge'] = full_edge_index.t()
            deg = data.adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            split_edge['train']['weight'] = deg_inv_sqrt[full_edge_index[0]] * full_edge_weight * deg_inv_sqrt[
                full_edge_index[1]]

#         # reindex node ids on sub-graph
#         if args.train_on_subgraph:
#             # extract involved nodes
#             row, col, edge_weight = data.adj_t.coo()
#             subset = set(row.tolist()).union(set(col.tolist()))
#             subset, _ = torch.sort(torch.tensor(list(subset)))
#             # For unseen node we set its index as -1
#             n_idx = torch.zeros(num_nodes, dtype=torch.long) - 1
#             n_idx[subset] = torch.arange(subset.size(0))
#             # Reindex edge_index, adj_t, num_nodes
#             data.edge_index = n_idx[data.edge_index]
#             data.adj_t = SparseTensor(row=n_idx[row], col=n_idx[col], value=edge_weight)
#             num_nodes = subset.size(0)
#             if hasattr(data, 'x'):
#                 if data.x is not None:
#                     data.x = data.x[subset]
#             # Reindex train valid test edges
#             split_edge['train']['edge'] = n_idx[split_edge['train']['edge']]
#             split_edge['valid']['edge'] = n_idx[split_edge['valid']['edge']]
#             split_edge['valid']['edge_neg'] = n_idx[split_edge['valid']['edge_neg']]
#             split_edge['test']['edge'] = n_idx[split_edge['test']['edge']]
#             split_edge['test']['edge_neg'] = n_idx[split_edge['test']['edge_neg']]

    data = data.to(device)

    if args.encoder.upper() == 'GCN':
        # Pre-compute GCN normalization.
        data.adj_t = gcn_normalization(data.adj_t)

    if args.encoder.upper() == 'WSAGE':
        data.adj_t = adj_normalization(data.adj_t)

    if args.encoder.upper() == 'TRANSFORMER':
        row, col, edge_weight = data.adj_t.coo()
        data.adj_t = SparseTensor(row=row, col=col)

    model = BaseModel(
        lr=args.lr,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        gnn_num_layers=args.gnn_num_layers,
        mlp_num_layers=args.mlp_num_layers,
        emb_hidden_channels=args.emb_hidden_channels,
        gnn_hidden_channels=args.gnn_hidden_channels,
        mlp_hidden_channels=args.mlp_hidden_channels,
        num_nodes=num_nodes,
        num_node_feats=num_node_feats,
        gnn_encoder_name=args.encoder,
        predictor_name=args.predictor,
        loss_func=args.loss_func,
        optimizer_name=args.optimizer,
        device=device,
        use_node_feats=args.use_node_feats,
        train_node_emb=args.train_node_emb,
        pretrain_emb=args.pretrain_emb
    )

    total_params = sum(p.numel() for param in model.para_list for p in param)
    total_params_print = f'Total number of model parameters is {total_params}'
    print(total_params_print)
    with open(log_file, 'a') as f:
        f.write(total_params_print + '\n')

    evaluator = Evaluator(name=args.data_name)

    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
        }

    if args.random_walk_augment:
        rw_row, rw_col, _ = data.adj_t.coo()
        if args.walk_start_type == 'edge':
            rw_start = torch.reshape(split_edge['train']['edge'], (-1,)).to(device)
        else:
            rw_start = torch.arange(0, num_nodes, dtype=torch.long).to(device)

    for run in range(args.runs):
        model.param_init()
        start_time = time.time()

        cur_lr = args.lr
        for epoch in range(1, 1 + args.epochs):
            if args.random_walk_augment:
                walk = random_walk(rw_row, rw_col, rw_start, walk_length=args.walk_length)
                pairs = []
                weights = []
                for j in range(args.walk_length):
                    pairs.append(walk[:, [0, j + 1]])
                    weights.append(torch.ones((walk.size(0),), dtype=torch.float) / (j + 1))
                pairs = torch.cat(pairs, dim=0)
                weights = torch.cat(weights, dim=0)
                # remove self-loop edges
                mask = ((pairs[:, 0] - pairs[:, 1]) != 0)
                split_edge['train']['edge'] = torch.masked_select(pairs, mask.view(-1, 1)).view(-1, 2)
                split_edge['train']['weight'] = torch.masked_select(weights, mask)

            loss = model.train(data, split_edge,
                               batch_size=args.batch_size,
                               neg_sampler_name=args.neg_sampler,
                               num_neg=args.num_neg)

            if epoch % args.eval_steps == 0:
                results = model.test(data, split_edge,
                                     batch_size=args.batch_size,
                                     evaluator=evaluator,
                                     eval_metric=args.eval_metric)
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                if epoch % args.log_steps == 0:
                    spent_time = time.time() - start_time
                    for key, result in results.items():
                        valid_res, test_res = result
                        to_print = (f'Run: {run + 1:02d}, '
                                    f'Epoch: {epoch:02d}, '
                                    f'Loss: {loss:.4f}, '
                                    f'Learning Rate: {cur_lr:.4f}, '
                                    f'Valid: {100 * valid_res:.2f}%, '
                                    f'Test: {100 * test_res:.2f}%')
                        print(key)
                        print(to_print)
                        with open(log_file, 'a') as f:
                            print(key, file=f)
                            print(to_print, file=f)
                    print('---')
                    print(
                        f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
                    print('---')
                    start_time = time.time()

            if args.use_lr_decay:
                cur_lr = adjust_lr(model.optimizer,
                                   epoch / args.epochs,
                                   args.lr)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run, last_best=args.eval_last_best)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f, last_best=args.eval_last_best)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(last_best=args.eval_last_best)
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f, last_best=args.eval_last_best)


if __name__ == "__main__":
    main()