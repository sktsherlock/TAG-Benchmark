import wandb
from datasets import load_metric
from transformers import AutoModel, EvalPrediction, TrainingArguments, Trainer, AutoTokenizer
import utils as uf
from model import *
from utils.data.datasets import *
import torch as th

METRICS = {  # metric -> metric_path
    'accuracy': 'hf_accuracy.py',
    'f1score': 'hf_f1.py',
    'precision': 'hf_precision.py',
    'recall': 'hf_recall.py',
    'spearmanr': 'hf_spearmanr.py',
    'pearsonr': 'hf_pearsonr.py',

}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        # forward pass
        center_contrast_embeddings, toplogy_contrast_embeddings = model(**inputs)
        # compute
        loss = infonce(center_contrast_embeddings, toplogy_contrast_embeddings)
        return  loss

class TCLTrainer():
    def __init__(self, cf):
        self.cf = cf
        # logging.set_verbosity_warning()
        from transformers import logging as trfm_logging
        self.logger = cf.logger
        self.log = cf.logger.log
        trfm_logging.set_verbosity_error()

    @uf.time_logger
    def train(self):
        # ! Prepare data
        self.d = d = Sequence(cf := self.cf).init()
        self.train_data = SeqCLDataset(self.d)

        # Finetune on dowstream tasks
        train_steps = len(d.train_x) // cf.eq_batch_size + 1
        warmup_steps = int(cf.warmup_epochs * train_steps)
        # ! Load Model for NP with no trainer
        PLM = AutoModel.from_pretrained(cf.hf_model)
        if cf.model == 'Distilbert':
            self.model = CL_Dis_Model(
                PLM,
                dropout=cf.cla_dropout,
            )
        else:
            self.model = CLModel(
                PLM,
                dropout=cf.cla_dropout,
            )
        if cf.local_rank <= 0:
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f" LM Model parameters are {trainable_params}")
        if cf.model == 'Distilbert':
            self.model.config.dropout = cf.dropout
            self.model.config.attention_dropout = cf.att_dropout
        elif cf.model == 'GPT2':
            self.model.config.attn_pdrop = cf.att_dropout
            self.model.config.embd_pdrop = cf.dropout
        else:
            self.model.config.hidden_dropout_prob = cf.dropout
            self.model.config.attention_probs_dropout_prob = cf.att_dropout
        self.log(self.model.config)

        if cf.grad_steps is not None:
            cf.grad_acc_steps = cf.grad_steps
            cf.batch_size = cf.per_device_bsz

        training_args = TrainingArguments(
            output_dir=cf.out_dir,
            learning_rate=cf.lr, weight_decay=cf.weight_decay,
            gradient_accumulation_steps=cf.grad_acc_steps,
            save_total_limit=None,
            report_to='wandb' if cf.wandb_on else None,
            per_device_train_batch_size=cf.batch_size,
            per_device_eval_batch_size=cf.batch_size * 6 if cf.hf_model in {'distilbert-base-uncased',
                                                                            'google/electra-base-discriminator'} else cf.batch_size * 10,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            dataloader_drop_last=True,
            num_train_epochs=cf.epochs,
            local_rank=cf.local_rank,
            dataloader_num_workers=1,
            fp16=True,
        )

        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
        )
        self.trainer.train()

        if cf.local_rank <= 0:
            if cf.cache_dir is not None:
                print(f'Save the finnal cl model in {cf.cache_dir}')
                PLM.save_pretrained(cf.cache_dir)
                ckpt = f'{cf.cache_dir}{cf.model}.ckpt'
                th.save(self.model.state_dict(), uf.init_path(ckpt))
            else:
                PLM.save_pretrained(cf.out_dir)
                th.save(self.model.state_dict(), uf.init_path(cf.lm.ckpt))
        else:
            print('Dont save the model in the local_rank:', cf.local_rank)