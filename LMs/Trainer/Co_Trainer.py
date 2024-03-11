import dgl.dataloading
import wandb
from datasets import load_metric
from transformers import AutoModel, EvalPrediction, TrainingArguments, Trainer, AutoTokenizer
import utils as uf
from model import *
from utils.data.datasets import *
import torch as th
from dgl.nn.pytorch import SAGEConv
import torch.nn as nn
from utils.data.preprocess import load_ogb_graph_structure_only, load_amazon_graph_structure_only, load_webkb_graph_structure_only
import dgl.nn.pytorch as dglnn
from sklearn.metrics import f1_score, precision_score, recall_score

METRICS = {  # metric -> metric_path
    'accuracy': 'hf_accuracy.py',
    'f1score': 'hf_f1.py',
    'precision': 'hf_precision.py',
    'recall': 'hf_recall.py',
    'spearmanr': 'hf_spearmanr.py',
    'pearsonr': 'hf_pearsonr.py',

}


class CoTModel(PreTrainedModel):
    def __init__(self, model, GNN, n_labels, dropout=0.0, seed=0, cla_bias=True):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.gnn = GNN
        hidden_dim = GNN.n_hidden
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)

    def forward(self, g, input_text):
        # Extract outputs from the model
        outputs = self.bert_encoder(**input_text, output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        text_emb = emb.permute(1, 0, 2)[0]
        x = self.gnn(g, text_emb)
        logits = self.classifier(x)

        return logits


def load_subtensor(ndata, seeds, labels, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes.
    """
    # ! To Tensor

    _load = lambda k: th.IntTensor(np.array(ndata[k][input_nodes]))
    input_text = {}
    for k in ndata.keys():
        if k != 'labels':
            input_text[k] = _load(k).to(device)

    return input_text, labels[seeds].to(device)


class CoT_Trainer():
    def __init__(self, cf):
        self.cf = cf
        # logging.set_verbosity_warning()
        from transformers import logging as trfm_logging
        self.logger = cf.logger
        self.log = cf.logger.log
        trfm_logging.set_verbosity_error()

    @uf.time_logger
    def train_notrainer(self):
        import math
        import json
        import wandb
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        from tqdm.auto import tqdm
        import os.path as osp
        from transformers import (
            get_scheduler,
            AutoModel,
        )

        self.d = d = Sequence(cf := self.cf).init()
        self.device = th.device("cuda")
        self.train_x, self.val_x, self.test_x = [
            th.tensor(getattr(d, f'{_}_x')).to(self.device) for _ in ['train', 'valid', 'test']]
        if d.md['type'] == 'ogb':
            self.g = load_ogb_graph_structure_only(cf)[0]
            self.labels = load_ogb_graph_structure_only(cf)[1]
            self.labels = th.from_numpy(self.labels).to(th.int64).to(self.device)
        elif d.md['type'] == 'amazon':
            self.g, self.labels, _ = load_amazon_graph_structure_only(cf)
            self.labels = th.from_numpy(self.labels).to(th.int64).to(self.device)
        elif d.md['type'] == 'webkb':
            self.g, self.labels, _ = load_webkb_graph_structure_only(cf)
            self.labels = th.from_numpy(self.labels).to(th.int64).to(self.device)

        print(f"Total edges before adding self-loop {self.g.number_of_edges()}")
        self.g = self.g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {self.g.number_of_edges()}")
        self.g = dgl.to_bidirected(self.g)
        self.g = self.g.to(self.device)
        # ! Pick fanouts and the dataloader
        # ! Pick fanouts and the dataloader
        init_random_state(cf.seed)
        dgl.random.seed(cf.seed)
        fanouts = [cf.fanouts] * cf.n_layers
        if cf.sampler_way == 'default':
            sampler = dgl.dataloading.NeighborSampler(fanouts)
        elif cf.sampler_way == 'shadow':
            sampler = dgl.dataloading.ShaDowKHopSampler(fanouts)
        elif cf.sampler_way == 'full':
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(cf.n_layers)
        else:
            raise NotImplementedError

        # ! Prepare your Dataloader
        per_device_eval_batch_size = cf.per_device_bsz * 6 if cf.hf_model in {'distilbert-base-uncased',
                                                                              'google/electra-base-discriminator'} else cf.per_eval_bsz

        self.train_dataloader = dgl.dataloading.DataLoader(self.g,
                                                           self.train_x,
                                                           sampler,
                                                           batch_size=cf.per_device_bsz,
                                                           shuffle=True,
                                                           drop_last=False)

        self.val_dataloader = dgl.dataloading.DataLoader(self.g,
                                                         self.val_x,
                                                         sampler,
                                                         batch_size=cf.per_eval_bsz,
                                                         shuffle=True,
                                                         drop_last=False)

        self.test_dataloader = dgl.dataloading.DataLoader(self.g,
                                                          self.test_x,
                                                          sampler,
                                                          batch_size=cf.per_eval_bsz,
                                                          shuffle=True,
                                                          drop_last=False)

        # ! Load Model for NP with no trainer
        PLM = AutoModel.from_pretrained(cf.hf_model)
        if cf.gnn_name == 'SAGE':
            GNN = GraphSAGE(in_feats=PLM.config.hidden_size, n_hidden=cf.n_hidden, n_classes=cf.n_hidden,
                            n_layers=cf.n_layers, activation=F.relu, dropout=cf.dropout, aggregator_type='mean',
                            input_drop=0.1)
        elif cf.gnn_name == 'GCN':
            GNN = GCN(in_feats=PLM.config.hidden_size, n_hidden=cf.n_hidden, n_classes=cf.n_hidden,
                      n_layers=cf.n_layers, activation=F.relu, dropout=cf.dropout, input_drop=0.1)
        elif cf.gnn_name == 'JKNet':
            pass
        else:
            raise NotImplementedError

        self.model = CoTModel(
            PLM,
            GNN,
            n_labels=cf.data.n_labels,
            dropout=cf.cla_dropout,
            cla_bias=cf.cla_bias == 'T',
        ).to(self.device)
        self.model_path = os.path.dirname(
            f'{cf.out_dir}{cf.gnn_name}/{cf.n_hidden}/{cf.n_layers}/{cf.sampler_way}/{cf.fanouts}/')
        # ! Prepare the optimizer
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cf.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = th.optim.AdamW(optimizer_grouped_parameters, lr=cf.lr)

        # ! # Prepare some config
        train_steps = len(self.d.train_x) // cf.eq_batch_size + 1
        warmup_steps = int(cf.warmup_epochs * train_steps)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / cf.grad_steps)
        total_train_steps = self.cf.epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps * cf.grad_steps,
            num_training_steps=total_train_steps * cf.grad_steps,
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / cf.grad_steps)
        if overrode_max_train_steps:
            total_train_steps = self.cf.epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.cf.epochs = math.ceil(total_train_steps / num_update_steps_per_epoch)

        self.log(self.model.config)
        self.log("***** Running training *****")
        self.log(f"  Num examples = {len(self.d.train_x)}")
        self.log(f"  Num Epochs = {self.cf.epochs}")
        self.log(f"  Instantaneous batch size per device = {self.cf.per_device_bsz}")
        self.log(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.cf.eq_batch_size}")
        self.log(f"  Gradient Accumulation steps = {self.cf.grad_steps}")
        self.log(f"  Total optimization steps = {total_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(total_train_steps))
        completed_steps = 0
        starting_epoch = 0
        # update the progress_bar if load from checkpoint
        progress_bar.update(starting_epoch * num_update_steps_per_epoch)
        completed_steps = starting_epoch * num_update_steps_per_epoch
        self.loss_func = th.nn.CrossEntropyLoss(label_smoothing=cf.label_smoothing_factor, reduction=cf.ce_reduction)

        best_val = 0
        for epoch in range(self.cf.epochs):
            self.model.train()

            for batch, (input_nodes, output_nodes, block) in enumerate(self.train_dataloader):
                input_text, labels = load_subtensor(self.d.ndata, output_nodes, self.labels, input_nodes, self.device)
                block = [block_.to(self.device) for block_ in block]

                y_pre = self.model(block, input_text)
                if labels.shape[-1] == 1:
                    labels = labels.squeeze()
                loss = self.loss_func(y_pre, labels)
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                wandb.log({'CoT_Train_loss': loss})

                progress_bar.update(1)
                completed_steps += 1

                if completed_steps >= total_train_steps:
                    break

            # Validation
            self.model.eval()
            with th.no_grad():
                if cf.metric == 'acc':
                    val_acc, val_loss = self.eval_accuracy(self.val_dataloader)
                    wandb.log({'val_acc': val_acc, 'val_loss': val_loss})

                    if best_val < val_acc:
                        best_val = val_acc
                        wandb.log({'best_val_acc': best_val})
                        if not os.path.exists(self.model_path):
                            os.makedirs(self.model_path, exist_ok=True)
                        th.save(self.model, os.path.join(self.model_path, 'model.pt'))
                elif cf.metric == 'f1':
                    val_f1, val_loss = self.eval_macro_f1_score(self.val_dataloader)
                    wandb.log({'val_f1': val_f1, 'val_loss': val_loss})

                    if best_val < val_f1:
                        best_val = val_f1
                        wandb.log({'best_val_f1': best_val})
                        if not os.path.exists(self.model_path):
                            os.makedirs(self.model_path, exist_ok=True)
                        th.save(self.model, os.path.join(self.model_path, 'model.pt'))

    @uf.time_logger
    def test(self):
        init_random_state(self.cf.seed)
        dgl.random.seed(self.cf.seed)
        self.model = th.load(os.path.join(self.model_path, 'model.pt')).to(self.device)

        self.model.eval()
        with th.no_grad():
            if self.cf.metric == 'acc':
                test_acc, test_loss = self.eval_accuracy(self.test_dataloader)
                wandb.log({'test_acc': test_acc, 'test_loss': test_loss})
                print('Test Accuracy: ', test_acc)
            elif self.cf.metric == 'f1':
                test_f1, test_loss = self.eval_macro_f1_score(self.test_dataloader)
                wandb.log({'test_f1': test_f1, 'test_loss': test_loss})
                print('Test F1: ', test_f1)

        os.remove(os.path.join(self.model_path, 'model.pt'))

    def eval_accuracy(self, dataloader=None):
        loss_all = 0
        val_correct_sum = 0
        for batch, (input_nodes, output_nodes, block) in enumerate(dataloader):
            input_text, labels = load_subtensor(self.d.ndata, output_nodes, self.labels, input_nodes,
                                                self.device)
            block = [block_.to(self.device) for block_ in block]

            y_pre = self.model(block, input_text)
            loss = self.loss_func(y_pre, labels.squeeze())

            y_pre = y_pre.argmax(1).view(-1, 1)
            correct_pre = (y_pre == labels.view(-1, 1)).sum().item()
            val_correct_sum += correct_pre
            loss_all += loss.item()
        loss_all = loss_all / len(dataloader)
        eval_acc = val_correct_sum / len(dataloader)
        return eval_acc, loss_all

    def eval_macro_f1_score(self, dataloader=None):
        y_true = []
        y_pred = []
        loss_all = 0

        for batch, (input_nodes, output_nodes, block) in enumerate(dataloader):
            input_text, labels = load_subtensor(self.d.ndata, output_nodes, self.labels, input_nodes, self.device)
            block = [block_.to(self.device) for block_ in block]
            y_pre = self.model(block, input_text)
            loss = self.loss_func(y_pre, labels.squeeze())
            loss_all += loss.item()

            y_pre = y_pre.argmax(1).view(-1, 1)
            y_true.extend(labels.view(-1, 1).tolist())
            y_pred.extend(y_pre.tolist())

        loss_all /= len(dataloader)

        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

        macro_f1 = f1_score(y_true, y_pred, average='macro')

        return macro_f1, loss_all


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 input_drop=0.0):
        super(GraphSAGE, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # input layer
        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes

            self.layers.append(dglnn.SAGEConv(in_hidden, out_hidden, aggregator_type))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, feat):
        h = feat
        h = self.input_drop(h)

        for l, layer in enumerate(self.layers):
            conv = layer(blocks[l], h)
            h = conv

            if l != len(self.layers) - 1:
                h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h


class GCN(nn.Module):
    def __init__(
            self,
            in_feats,
            n_hidden,
            n_classes,
            n_layers,
            activation,
            dropout,
            input_drop=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(
                dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias)
            )

            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, block, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](block[i], h)
            h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h
