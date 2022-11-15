import torch as th
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from utils.function.os_utils import init_random_state
import numpy as np
import torch.nn.functional as F

def compute_loss(logits, labels, loss_func, is_gold=None, pl_weight=0.5, is_augmented=False):
    """
    Combine two types of losses: (1-α)*MLE (CE loss on gold) + α*Pl_loss (CE loss on pseudo labels)
    """
    import torch as th

    if is_augmented and ((n_pseudo := sum(~is_gold)) > 0):
        deal_nan = lambda x: 0 if th.isnan(x) else x
        mle_loss = deal_nan(loss_func(logits[is_gold], labels[is_gold]))
        pl_loss = deal_nan(loss_func(logits[~is_gold], labels[~is_gold]))
        loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
    else:
        loss = loss_func(logits, labels)
    return loss


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, pseudo_label_weight=1, dropout=0.0, seed=0, cla_bias=True, is_augmented=False, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)
        self.pl_weight = pseudo_label_weight
        self.is_augmented = is_augmented

    def forward(self, **input):
        # Extract outputs from the model
        labels, is_gold = [input.pop(_) for _ in ["labels", 'is_gold']]
        outputs = self.bert_encoder(**input, output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)
