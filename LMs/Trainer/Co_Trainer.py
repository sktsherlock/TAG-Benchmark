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

METRICS = {  # metric -> metric_path
    'accuracy': 'hf_accuracy.py',
    'f1score': 'hf_f1.py',
    'precision': 'hf_precision.py',
    'recall': 'hf_recall.py',
    'spearmanr': 'hf_spearmanr.py',
    'pearsonr': 'hf_pearsonr.py',

}

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_layers, activation=F.relu,aggregator='mean'):
        super(GraphSAGE, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = hidden_dim
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        self.layer.append(SAGEConv(in_feats, self.n_hidden, aggregator))
        for i in range(1, self.n_layers - 1):
            self.layer.append(SAGEConv(self.n_hidden, self.n_hidden, aggregator))
        self.layer.append(SAGEConv(self.n_hidden, self.n_hidden, aggregator))
        self.dropout = nn.Dropout(cf.dropout)
        self.activation = activation
    def forward(self, blocks, feas):
        h = feas
        for i, (layer, block) in enumerate(zip(self.layer, blocks)):
            h = layer(block, h)
            if i != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

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
        labels = input_text.pop('labels')
        outputs = self.bert_encoder(**input_text, output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        text_emb = emb.permute(1, 0, 2)[0]
        x =  self.gnn(g, text_emb)
        x = self.classifier(x)
        logits = self.classifier(x)

        return logits

def load_subtensor(ndata, seeds, labels,input_nodes, device):
    """
    Extracts features and labels for a subset of nodes.
    """
    input_text = {}
    for k in ndata.keys():
        input_text[k] = ndata[k][input_nodes].to(device)

    label = labels[seeds]
    return input_text, label


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
        from accelerate import Accelerator, DistributedType
        from tqdm.auto import tqdm
        import os.path as osp
        from transformers import (
            get_scheduler,
            AutoModel,
        )

        self.d = d = Sequence(cf := self.cf).init()
        gold_data = SeqGraphDataset(self.d)
        subset_data = lambda sub_idx: th.utils.data.Subset(gold_data, sub_idx)
        self.datasets = {_: subset_data(getattr(d, f'{_}_x'))
                         for _ in ['train', 'valid', 'test']}
        #! Pick fanouts and the dataloader
        fanouts = [2,2]
        sampler = dgl.dataloading.NeighborSampler(fanouts)

        # ! Prepare the Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=cf.grad_acc_steps)

        # ! Prepare your Dataloader
        per_device_eval_batch_size = cf.batch_size * 6 if cf.hf_model in {'distilbert-base-uncased',
                                                                          'google/electra-base-discriminator'} else cf.batch_size * 10

        self.train_dataloader = dgl.dataloading.DataLoader(self.d.g,
                                                      self.datasets['train'],
                                                      sampler,
                                                      batch_size=cf.batch_size,
                                                      shuffle=True,
                                                      drop_last=False)

        self.val_dataloader = dgl.dataloading.DataLoader(self.d.g,
                                                    self.datasets['valid'],
                                                    sampler,
                                                    batch_size=cf.batch_size,
                                                    shuffle=True,
                                                    drop_last=False)

        self.test_dataloader = dgl.dataloading.DataLoader(self.d.g,
                                                     self.datasets['test'],
                                                     sampler,
                                                     batch_size=cf.batch_size,
                                                     shuffle=True,
                                                     drop_last=False)

        # ! Load Model for NP with no trainer
        PLM = AutoModel.from_pretrained(cf.hf_model)
        GNN = GraphSAGE(in_feats=PLM.config.hidden_size, hidden_dim=128, n_layers=2)
        self.model = CoTModel(
            PLM,
            GNN,
            dropout=cf.cla_dropout,
            cla_bias=cf.cla_bias == 'T',
        )

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
        train_steps = len(self.train_data) // cf.eq_batch_size + 1
        warmup_steps = int(cf.warmup_epochs * train_steps)
        eval_steps = cf.eval_patience // cf.eq_batch_size

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / cf.grad_acc_steps)
        total_train_steps = self.cf.epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps * cf.grad_acc_steps,
            num_training_steps=total_train_steps * cf.grad_acc_steps,
        )

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, _, _, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / cf.grad_acc_steps)
        if overrode_max_train_steps:
            total_train_steps = self.cf.epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.cf.epochs = math.ceil(total_train_steps / num_update_steps_per_epoch)

        self.log(self.model.config)
        self.log("***** Running training *****")
        self.log(f"  Num examples = {len(self.train_data)}")
        self.log(f"  Num Epochs = {self.cf.epochs}")
        self.log(f"  Instantaneous batch size per device = {self.cf.batch_size}")
        self.log(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.cf.eq_batch_size}")
        self.log(f"  Gradient Accumulation steps = {self.cf.grad_acc_steps}")
        self.log(f"  Total optimization steps = {total_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(total_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        # update the progress_bar if load from checkpoint
        progress_bar.update(starting_epoch * num_update_steps_per_epoch)
        completed_steps = starting_epoch * num_update_steps_per_epoch
        loss_func = th.nn.CrossEntropyLoss(label_smoothing=cf.label_smoothing_factor, reduction=cf.ce_reduction)

        for epoch in range(self.cf.epochs):
            self.model.train()
            correct_sum = 0
            for batch, (input_nodes, output_nodes, block) in enumerate(self.train_dataloader):
                with accelerator.accumulate(self.model):
                    input_text, labels = load_subtensor(self.data.ndata, output_nodes,self.data.labels, input_nodes, cf.device)
                    block = [block_.to(cf.device) for block_ in block]

                    y_pre = self.model(block,input_text)
                    loss = loss_func(y_pre, labels.squeeze())
                    accelerator.backward(loss)

                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()
                    wandb.log({'CoT_Train_loss': loss})

                    y_pre = y_pre.argmax(1).view(-1, 1)
                    correct_pre = (y_pre == labels).sum().item()
                    correct_sum += correct_pre

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= total_train_steps:
                    break

            # Validation
            self.model.eval()
            val_correct_sum = 0

            with th.no_grad():
                for batch, (input_nodes, output_nodes, block) in enumerate(self.val_dataloader):
                    input_text, labels = load_subtensor(self.data.ndata, output_nodes, self.data.labels, input_nodes,
                                                        cf.device)
                    block = [block_.to(cf.device) for block_ in block]

                    y_pre = self.model(block, input_text)
                    loss = loss_func(y_pre, labels.squeeze())
                    wandb.log({'val_cotrain_loss': loss})

                    y_pre = y_pre.argmax(1).view(-1, 1)
                    correct_pre = (y_pre == labels).sum().item()
                    val_correct_sum += correct_pre

            val_acc = val_correct_sum / self.datasets['valid'].shape[0]
            wandb.log({'val_acc': val_acc})

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                wandb.log({'best_val_acc': best_val_acc})
                th.save(self.model.state_dict(), cf.out_dir)

    def test(self):
        self.model = th.load(self.cf.out_dir)
        self.model.eval()
        test_correct_sum = 0
        with th.no_grad():
            for batch, (input_nodes, output_nodes, block) in enumerate(self.test_dataloader):
                input_text, labels = load_subtensor(self.data.ndata, output_nodes, self.data.labels, input_nodes,
                                                    self.cf.device)
                block = [block_.to(self.cf.device) for block_ in block]

                y_pre = self.model(block, input_text)
                y_pre = y_pre.argmax(1).view(-1, 1)
                correct_pre = (y_pre == labels).sum().item()
                test_correct_sum += correct_pre

        test_acc = test_correct_sum / self.datasets['test'].shape[0]






