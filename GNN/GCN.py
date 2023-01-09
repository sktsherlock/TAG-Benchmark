#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import time
import wandb
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from model.models import GCN
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


def gen_model(args):
    if args.use_labels:
        model = GCN(
            in_feats + n_classes,
            args.n_hidden,
            n_classes,
            args.n_layers,
            F.relu,
            args.dropout,
            args.use_linear,
            args.input_drop,
        )
    else:
        model = GCN(
            in_feats,
            args.n_hidden,
            n_classes,
            args.n_layers,
            F.relu,
            args.dropout,
            args.use_linear,
            args.input_drop,
        )
    return model


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]

def compute_rocauc(pred, labels):
    from sklearn.metrics import roc_auc_score
    y_pred = pred
    y_true = labels
    y_true, y_pred = _parse_and_check_input(y_pred, y_true)
    rocauc_list = []

    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            is_labeled = y_true[:,i] == y_true[:,i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i], multi_class='ovr'))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

def _parse_and_check_input(y_pred, y_true):
        # converting to torch.Tensor to numpy on cpu
        if th is not None and isinstance(y_true, th.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if th is not None and isinstance(y_pred, th.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        ## check type
        if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
            raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        if not y_true.shape == y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')

        if not y_true.ndim == 2:
            raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))


        return y_true, y_pred


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, feat, labels, train_idx, optimizer, use_labels):
    model.train()


    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(
    model, graph, feat, labels, one_hot_labels, train_idx, val_idx, test_idx, use_labels, evaluator
):
    model.eval()

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        # compute_rocauc(pred[train_idx], one_hot_labels[train_idx]),
        # compute_rocauc(pred[val_idx], one_hot_labels[val_idx]),
        # compute_rocauc(pred[test_idx], one_hot_labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
    )


def run(
    args, graph, feat, labels, one_hot_labels, train_idx, val_idx, test_idx, evaluator, n_running
):
    # define model and optimizer
    model = gen_model(args)
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=100,
        verbose=True,
        min_lr=1e-3,
    )

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_rocauc, final_test_rocauc, best_val_loss = 0, 0, 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        loss, pred = train(
            model, graph, feat, labels, train_idx, optimizer, args.use_labels
        )
        acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)
        # rocauc = compute_rocauc(pred[train_idx], one_hot_labels[train_idx])

        (
            train_acc,
            val_acc,
            test_acc,
            # train_rocauc,
            # val_rocauc,
            # test_rocauc,
            train_loss,
            val_loss,
            test_loss,
        ) = evaluate(
            model,
            graph,
            feat,
            labels,
            one_hot_labels,
            train_idx,
            val_idx,
            test_idx,
            args.use_labels,
            evaluator,
        )
        wandb.log({'Train_loss': train_loss, 'Val_loss': val_loss, 'Test_loss': test_loss})
        lr_scheduler.step(loss)

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            # best_val_rocauc = val_rocauc
            # final_test_rocauc = test_rocauc


        if epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

        for l, e in zip(
            [
                accs,
                train_accs,
                val_accs,
                test_accs,
                losses,
                train_losses,
                val_losses,
                test_losses,
            ],
            [
                acc,
                train_acc,
                val_acc,
                test_acc,
                loss,
                train_loss,
                val_loss,
                test_loss,
            ],
        ):
            l.append(e)

    print("*" * 50)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)

    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [accs, train_accs, val_accs, test_accs],
            ["acc", "train acc", "val acc", "test acc"],
        ):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gcn_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses],
            ["loss", "train loss", "val loss", "test loss"],
        ):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gcn_loss_{n_running}.png")

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )


def main():
    global device, in_feats, n_classes

    argparser = argparse.ArgumentParser(
        "GCN on OGBN-Arxiv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU mode. This option overrides --gpu.",
    )
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument(
        "--n-runs", type=int, default=10, help="running times"
    )
    argparser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of epochs"
    )
    argparser.add_argument(
        "--use-labels",
        type=bool,
        default=False,
        help="Use labels in the training set as input features.",
    )
    argparser.add_argument(
        "--use-linear", type=bool, default=True, help="Use linear layer."
    )
    argparser.add_argument(
        "--lr", type=float, default=0.005, help="learning rate"
    )
    argparser.add_argument(
        "--n-layers", type=int, default=3, help="number of layers"
    )
    argparser.add_argument(
        "--n-hidden", type=int, default=256, help="number of hidden units"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate"
    )
    argparser.add_argument(
        "--input-drop", type=float, default=0.1, help="input drop rate"
    )
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument(
        "--log-every", type=int, default=20, help="log every LOG_EVERY epochs"
    )
    argparser.add_argument(
        "--plot-curves", action="store_true", help="plot learning curves"
    )
    argparser.add_argument(
        "--use_PLM", type=str, default=None, help="Use LM embedding as feature"
    )
    args = argparser.parse_args()
    wandb.config = args
    wandb.init(config=args, reinit=True)

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)

    # load data
    data = DglNodePropPredDataset(name="ogbn-arxiv")
    evaluator = Evaluator(name="ogbn-arxiv")

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    if args.use_PLM:
        feat = th.from_numpy(np.load(args.use_PLM).astype(np.float32)).to(device)
        in_feats = feat.shape[1]
    else:
        feat = graph.ndata["feat"].to(device)
        in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels1 = np.array(labels.squeeze().tolist())
    graph = graph.to(device)
    one_hot_labels = th.sparse.torch.eye(40).index_select(0, th.tensor(labels1)).to(device)
    labels = labels.to(device)

    # run
    val_accs = []
    test_accs = []

    for i in range(args.n_runs):
        val_acc, test_acc = run(
            args, graph, feat, labels, one_hot_labels, train_idx, val_idx, test_idx, evaluator, i
        )
        wandb.log({'Val_Acc': val_acc, 'Test_Acc': test_acc})
        val_accs.append(val_acc)
        test_accs.append(test_acc)


    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters(args)}")
    wandb.log({'Mean_Val_Acc': np.mean(val_accs), 'Mean_Test_Acc': np.mean(test_accs)})


if __name__ == "__main__":
    main()