import argparse
import torch
import wandb


def args_init():
    argparser = argparse.ArgumentParser(
        "GNN(GCN,GIN,GraphSAGE,GAT) on OGBN-Arxiv/Amazon-Books/and so on. Node Classification tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU mode. This option overrides --gpu.",
    )
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument(
        "--n-runs", type=int, default=3, help="running times"
    )
    argparser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of epochs"
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
        "--input-drop", type=float, default=0.1, help="input drop rate"
    )
    argparser.add_argument(
        "--learning-eps", type=bool, default=True,
        help="If True, learn epsilon to distinguish center nodes from neighbors;"
             "If False, aggregate neighbors and center nodes altogether."
    )
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument(
        "--num-mlp-layers", type=int, default=2, help="number of mlp layers"
    )
    argparser.add_argument(
        "--neighbor-pooling-type", type=str, default='mean', help="how to aggregate neighbors (sum, mean, or max)"
    )
    # ! GAT
    argparser.add_argument(
        "--no-attn-dst", type=bool, default=True, help="Don't use attn_dst."
    )
    argparser.add_argument(
        "--n-heads", type=int, default=3, help="number of heads"
    )
    argparser.add_argument(
        "--attn-drop", type=float, default=0.0, help="attention drop rate"
    )
    argparser.add_argument(
        "--edge-drop", type=float, default=0.0, help="edge drop rate"
    )
    # ! RevGAT
    # ! SAGE
    argparser.add_argument("--aggregator-type", type=str, default="mean",
                           help="Aggregator type: mean/gcn/pool/lstm")
    # ! JKNET
    argparser.add_argument(
        "--mode", type=str, default='cat', help="the mode of aggregate the feature, 'cat', 'lstm'"
    )
    #! Monet
    argparser.add_argument(
        "--pseudo-dim",type=int,default=2,help="Pseudo coordinate dimensions in GMMConv, 2 and 3",
    )
    argparser.add_argument(
        "--n-kernels",type=int,default=3,help="Number of kernels in GMMConv layer",
    )
    #! APPNP
    argparser.add_argument(
        "--alpha", type=float, default=0.1, help="Teleport Probability"
    )
    argparser.add_argument(
        "--k", type=int, default=10, help="Number of propagation steps"
    )
    argparser.add_argument(
        "--hidden_sizes", type=int, nargs="+", default=[64], help="hidden unit sizes for appnp",
    )
    # ! default
    argparser.add_argument(
        "--log-every", type=int, default=20, help="log every LOG_EVERY epochs"
    )
    argparser.add_argument(
        "--eval_steps", type=int, default=5, help="eval in every epochs"
    )
    argparser.add_argument(
        "--use_PLM", type=str, default=None, help="Use LM embedding as feature"
    )
    argparser.add_argument(
        "--model_name", type=str, default='RevGAT', help="Which GNN be implemented"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate"
    )
    argparser.add_argument(
        "--data_name", type=str, default='ogbn-arxiv', help="The datasets to be implemented."
    )
    argparser.add_argument(
        "--metric", type=str, default='acc', help="The datasets to be implemented."
    )
    # ! Split datasets
    argparser.add_argument(
        "--train_ratio", type=float, default=0.6, help="training ratio"
    )
    argparser.add_argument(
        "--val_ratio", type=float, default=0.2, help="training ratio"
    )
    return argparser


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, key='Hits@10'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'{key} Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            wandb.log({f'{key} Highest Train Acc': float(f'{r.mean():.2f}'),
                       f'{key} Highest Train Std': float(f'{r.std():.2f}')})
            r = best_result[:, 1]
            print(f'{key} Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            wandb.log({f'{key} Highest Valid Acc': float(f'{r.mean():.2f}'),
                       f'{key} Highest Valid Std': float(f'{r.std():.2f}')})
            r = best_result[:, 2]
            print(f'{key} Final Train: {r.mean():.2f} ± {r.std():.2f}')
            wandb.log(
                {f'{key} Final Train Acc': float(f'{r.mean():.2f}'), f'{key} Final Train Std': float(f'{r.std():.2f}')})
            r = best_result[:, 3]
            print(f'{key} Final Test: {r.mean():.2f} ± {r.std():.2f}')
            wandb.log(
                {f'{key} Final Test Acc': float(f'{r.mean():.2f}'), f'{key} Final Test Std': float(f'{r.std():.2f}')})

