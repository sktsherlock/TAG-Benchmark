import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn.functional as F
from utils.function import init_random_state
from torch_geometric.nn import GINConv, global_add_pool



class BertLinker(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, pseudo_label_weight=1, dropout=0.0, seed=0, cla_bias=True,
                 feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        hidden_dim = model.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)
        self.pl_weight = pseudo_label_weight

    def infer(self, input_ids_node_and_neighbors_batch, attention_mask_node_and_neighbors_batch,
              mask_node_and_neighbors_batch):
        B, N, L = input_ids_node_and_neighbors_batch.shape
        D = self.config.hidden_size
        input_ids = input_ids_node_and_neighbors_batch.view(B * N, L)
        attention_mask = attention_mask_node_and_neighbors_batch.view(B * N, L)
        hidden_states = self.bert(input_ids, attention_mask, mask_node_and_neighbors_batch)
        last_hidden_states = hidden_states[0]
        cls_embeddings = last_hidden_states[:, 1].view(B, N, D)  # [B,N,D]
        node_embeddings = cls_embeddings[:, 0, :]  # [B,D]
        return node_embeddings

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # Extract outputs from the model
        outputs = self.bert_encoder(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
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


class BertEmbInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.bert_encoder = model

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Extract outputs from the model
        outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        emb = outputs['hidden_states'][-1]  # Last layer
        # Use CLS Emb as sentence emb.
        node_cls_emb = emb.permute(1, 0, 2)[0]
        return TokenClassifierOutput(logits=node_cls_emb)

