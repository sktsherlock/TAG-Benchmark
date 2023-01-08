## Usage & Options

### GCN

```
usage: GCN on OGBN-Arxiv [-h] [--cpu] [--gpu GPU] [--n-runs N_RUNS] [--n-epochs N_EPOCHS] [--use-labels] [--use-linear] [--lr LR] [--n-layers N_LAYERS] [--n-hidden N_HIDDEN]
                         [--dropout DROPOUT] [--wd WD] [--log-every LOG_EVERY] [--plot-curves]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 CPU mode. This option overrides --gpu. (default: False)
  --gpu GPU             GPU device ID. (default: 0)
  --n-runs N_RUNS       running times (default: 10)
  --n-epochs N_EPOCHS   number of epochs (default: 1000)
  --use-labels          Use labels in the training set as input features. (default: False)
  --use-linear          Use linear layer. (default: False)
  --lr LR               learning rate (default: 0.005)
  --n-layers N_LAYERS   number of layers (default: 3)
  --n-hidden N_HIDDEN   number of hidden units (default: 256)
  --dropout DROPOUT     dropout rate (default: 0.75)
  --wd WD               weight decay (default: 0)
  --log-every LOG_EVERY
                        log every LOG_EVERY epochs (default: 20)
  --plot-curves         plot learning curves (default: False)
  --use_PLM             Use LM embedding as feature (Input the root of the embedding npy file)
```

### GAT

```
usage: GAT on OGBN-Arxiv [-h] [--cpu] [--gpu GPU] [--n-runs N_RUNS] [--n-epochs N_EPOCHS] [--use-labels] [--n-label-iters N_LABEL_ITERS] [--no-attn-dst]
                         [--use-norm] [--lr LR] [--n-layers N_LAYERS] [--n-heads N_HEADS] [--n-hidden N_HIDDEN] [--dropout DROPOUT] [--input-drop INPUT_DROP]
                         [--attn-drop ATTN_DROP] [--edge-drop EDGE_DROP] [--wd WD] [--log-every LOG_EVERY] [--plot-curves]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 CPU mode. This option overrides --gpu. (default: False)
  --gpu GPU             GPU device ID. (default: 0)
  --n-runs N_RUNS       running times (default: 10)
  --n-epochs N_EPOCHS   number of epochs (default: 2000)
  --use-labels          Use labels in the training set as input features. (default: False)
  --n-label-iters N_LABEL_ITERS
                        number of label iterations (default: 0)
  --no-attn-dst         Don't use attn_dst. (default: False)
  --use-norm            Use symmetrically normalized adjacency matrix. (default: False)
  --lr LR               learning rate (default: 0.002)
  --n-layers N_LAYERS   number of layers (default: 3)
  --n-heads N_HEADS     number of heads (default: 3)
  --n-hidden N_HIDDEN   number of hidden units (default: 250)
  --dropout DROPOUT     dropout rate (default: 0.75)
  --input-drop INPUT_DROP
                        input drop rate (default: 0.1)
  --attn-drop ATTN_DROP
                        attention dropout rate (default: 0.0)
  --edge-drop EDGE_DROP
                        edge drop rate (default: 0.0)
  --wd WD               weight decay (default: 0)
  --residual            use residual connection (default: False)
  --log-every LOG_EVERY
                        log every LOG_EVERY epochs (default: 20)
  --plot-curves         plot learning curves (default: False)
```