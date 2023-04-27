from datasets import load_metric
from transformers import AutoModel, EvalPrediction, TrainingArguments, Trainer, AutoTokenizer,BertModel
import utils as uf
from model import *
from utils.data.datasets import *
import torch as th
from torch.utils.data import random_split
import os
os.environ["WANDB_DISABLED"] = "False"
METRICS = {  # metric -> metric_path
    'accuracy': 'hf_accuracy.py',
    'f1score': 'hf_f1.py',
    'precision': 'hf_precision.py',
    'recall': 'hf_recall.py',
    'spearmanr': 'hf_spearmanr.py',
    'pearsonr': 'hf_pearsonr.py',

}


class LMTrainer():
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
        gold_data = SeqGraphDataset(self.d)
        subset_data = lambda sub_idx: th.utils.data.Subset(gold_data, sub_idx)
        self.datasets = {_: subset_data(getattr(d, f'{_}_x'))
                             for _ in ['train', 'valid', 'test']}
        train_steps = len(d.train_x) // cf.eq_batch_size + 1
        self.metrics = {m: load_metric(m_path) for m, m_path in METRICS.items()}

        # Finetune on dowstream tasks
        self.train_data = self.datasets['train']

        warmup_steps = int(cf.warmup_epochs * train_steps)
        eval_steps = cf.eval_patience // cf.eq_batch_size

        # ! Load bert and build classifier
        model = AutoModel.from_pretrained(cf.hf_model) if cf.pretrain_path is None else AutoModel.from_pretrained(f'{cf.pretrain_path}')
        #! Freeze the model.encoder layer if cf.freeze is not None
        if cf.freeze is not None:
            for param in model.parameters():
                param.requires_grad = False
            if cf.local_rank <= 0:
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                assert trainable_params == 0
            for param in model.encoder.layer[-cf.freeze:].parameters():
                param.requires_grad = True
            if cf.local_rank <= 0:
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                print(f" Pass the freeze layer, the LM Encoder  parameters are {trainable_params}")

        self.model = BertClassifier(
                model, cf.data.n_labels,
                dropout=cf.cla_dropout,
                loss_func=th.nn.CrossEntropyLoss(label_smoothing=cf.label_smoothing_factor, reduction=cf.ce_reduction),
                cla_bias=cf.cla_bias == 'T',
            )
        if cf.local_rank <= 0:
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f" LM Model parameters are {trainable_params}")

        load_best_model_at_end = True
        if cf.model == 'Distilbert':
            self.model.config.dropout = cf.dropout
            self.model.config.attention_dropout = cf.att_dropout
        else:
            self.model.config.hidden_dropout_prob = cf.dropout
            self.model.config.attention_probs_dropout_prob = cf.att_dropout
        self.log(self.model.config)

        training_args = TrainingArguments(
            output_dir=cf.out_dir,
            evaluation_strategy='steps',
            eval_steps=eval_steps,
            save_strategy='steps',
            save_steps=eval_steps,
            learning_rate=cf.lr, weight_decay=cf.weight_decay,
            load_best_model_at_end=load_best_model_at_end, gradient_accumulation_steps=cf.grad_steps,
            save_total_limit=None,
            report_to='wandb' if cf.wandb_on else None,
            per_device_train_batch_size=cf.per_device_bsz,
            per_device_eval_batch_size=cf.per_device_bsz * 6 if cf.hf_model in {'distilbert-base-uncased',
                                                                            'google/electra-base-discriminator'} else cf.per_eval_bsz,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            dataloader_drop_last=True,
            num_train_epochs=cf.epochs,
            local_rank=cf.local_rank,
            dataloader_num_workers=1,
            fp16=True,
        )

        def compute_metrics(pred: EvalPrediction):
            predictions, references = pred.predictions.argmax(1), pred.label_ids.argmax(1)
            return {m_name: metric.compute(predictions=predictions, references=references)
            if m_name in {'accuracy', 'pearsonr', 'spearmanr'} else metric.compute(predictions=predictions,
                                                                                   references=references,
                                                                                   average='macro')
                    for m_name, metric in self.metrics.items()}

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.datasets['valid'],
            compute_metrics=compute_metrics,
        )
        self.eval_phase = 'Eval'
        self.trainer.train()

        if cf.local_rank <= 0:
            if cf.cache_dir is not None:
                print(f'save the language models in {cf.cache_dir}')
                model.save_pretrained(cf.cache_dir)
            else:
                print(f'save the language models in {cf.out_dir}')
                model.save_pretrained(cf.out_dir)
        else:
            print('Dont save the model in the local_rank:', cf.local_rank)

        self.log(f'LM saved to {cf.out_dir}')

    def eval_and_save(self):
        def get_metric(split):
            self.eval_phase = 'Test' if split == 'test' else 'Eval'
            mtc_dict = self.trainer.predict(self.datasets[split]).metrics
            ret = {f'{split}_{_}': mtc_dict[m][_] for m in mtc_dict if (_ := m.split('_')[-1]) in METRICS}
            return ret

        cf = self.cf
        res = {**get_metric('valid'), **get_metric('test')}
        uf.pickle_save(res, cf.lm.result)
        cf.wandb_log({f'lm_finetune_{k}': v for k, v in res.items()})

        self.log(f'\nTrain seed{cf.seed} finished\nResults: {res}\n{cf}')