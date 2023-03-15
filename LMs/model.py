import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn.functional as F
from utils.function import init_random_state
from torch_geometric.nn import GINConv, global_add_pool


class TNPClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, dropout=0.0, seed=0, cla_bias=True):
        super().__init__(model.config)
        self.encoder, self.loss_func = model, loss_func
        self.dropout = nn.Dropout(dropout)
        hidden_dim = model.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # Extract outputs from the model
        outputs = self.encoder(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        emb = self.dropout(outputs['hidden_states'][-1])  # outputs[0]=last hidden state
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        logits = self.classifier(cls_token_emb)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)

class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, pseudo_label_weight=1, dropout=0.0, seed=0, cla_bias=True,
                 feat_shrink=''):
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


class DistilBertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, pseudo_label_weight=1, dropout=0.0, seed=0, cla_bias=True,
                 feat_shrink=''):
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

    def forward(self, input_ids, attention_mask, labels=None):
        # Extract outputs from the model
        outputs = self.bert_encoder(input_ids, attention_mask, output_hidden_states=True)
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


class CLModel(PreTrainedModel):
    def __init__(self, PLM, dropout=0.0):
        super().__init__(PLM.config)
        self.dropout = nn.Dropout(dropout)
        hidden_dim = PLM.config.hidden_size
        self.text_encoder = PLM

        self.Aggregate = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim))

        self.project = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, node_id=None,
                nb_input_ids=None, nb_attention_mask=None, nb_token_type_ids=None):
        # Getting Center Node text features and its neighbours feature
        center_node_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True
        )
        center_node_emb = self.dropout(center_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]

        toplogy_node_outputs = self.text_encoder(
            input_ids=nb_input_ids, attention_mask=nb_attention_mask, token_type_ids=nb_token_type_ids, output_hidden_states=True
        )

        toplogy_emb = self.dropout(toplogy_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]
        toplogy_emb = self.Aggregate(toplogy_emb) #! To Update
        # 10; 20->sum mean max -> 10 ->MLP -> batch id; 1-10; 20 *128 -> 20 * 128; 10 * 128;

        center_contrast_embeddings = self.project(center_node_emb)
        toplogy_contrast_embeddings = self.project(toplogy_emb)

        return center_contrast_embeddings, toplogy_contrast_embeddings

class CLFModel(PreTrainedModel):
    def __init__(self, model, n_labels, loss_func, dropout=0.0, alpha=0.5):
        super().__init__(model.config)
        self.dropout = nn.Dropout(dropout)
        #! 从model中加载PLM与Aggregate
        self.text_encoder = model.text_encoder
        self.Aggregate = model.Aggregate
        self.alpha = alpha # for mixup
        hidden_dim = self.text_encoder.config.hidden_size
        self.loss_func =  loss_func

        self.classifier = nn.Linear(hidden_dim, n_labels, bias=True)
        self.MLP = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, node_id=None,
                nb_input_ids=None, nb_attention_mask=None, nb_token_type_ids=None, labels=None):
        # Getting Center Node text features and its neighbours feature
        center_node_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True
        )
        center_node_emb = self.dropout(center_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]

        toplogy_node_outputs = self.text_encoder(
            input_ids=nb_input_ids, attention_mask=nb_attention_mask, token_type_ids=nb_token_type_ids, output_hidden_states=True
        )

        toplogy_emb = self.dropout(toplogy_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]
        toplogy_emb = self.Aggregate(toplogy_emb) #! To Update

        text_embedding = (1 - self.alpha) * toplogy_emb + self.alpha * center_node_emb # mixup
        text_embedding = self.MLP(text_embedding)
        logits = self.classifier(text_embedding)
        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)
        return TokenClassifierOutput(loss=loss, logits=logits)

class CL_Dis_Model(PreTrainedModel):
    def __init__(self, PLM, dropout=0.0):
        super().__init__(PLM.config)
        self.dropout = nn.Dropout(dropout)
        hidden_dim = PLM.config.hidden_size
        self.text_encoder = PLM

        self.project = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, node_id=None,
                nb_input_ids=None, nb_attention_mask=None, nb_token_type_ids=None):
        # Getting Center Node text features and its neighbours feature
        center_node_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        center_node_emb = self.dropout(center_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]

        toplogy_node_outputs = self.text_encoder(
            input_ids=nb_input_ids, attention_mask=nb_attention_mask, output_hidden_states=True
        )

        toplogy_emb = self.dropout(toplogy_node_outputs['hidden_states'][-1]).permute(1, 0, 2)[0]

        center_contrast_embeddings = self.project(center_node_emb)
        toplogy_contrast_embeddings = self.project(toplogy_emb)

        return center_contrast_embeddings, toplogy_contrast_embeddings

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

class DistillBertEmbInfModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)
        self.bert_encoder = model

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Extract outputs from the model
        outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        emb = outputs['hidden_states'][-1]  # Last layer
        # Use CLS Emb as sentence emb.
        node_cls_emb = emb.permute(1, 0, 2)[0]
        return TokenClassifierOutput(logits=node_cls_emb)

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def infonce(anchor, sample, tau=0.2):
    sim = _similarity(anchor, sample) / tau
    num_nodes = anchor.shape[0]
    device = anchor.device
    pos_mask = torch.eye(num_nodes, dtype=torch.float32).to(device)
    neg_mask = 1. - pos_mask
    assert sim.size() == pos_mask.size()  # sanity check
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    return -loss.mean()