import math

from datasets import load_metric
from transformers import AutoModel, EvalPrediction, TrainingArguments, Trainer
import utils.function as uf
from LMs.model import *
from utils.data.datasets import *
import torch as th

METRICS = {  # metric -> metric_path
    'accuracy': 'src/utils/function/hf_accuracy.py',
    'f1score': 'src/utils/function/hf_f1.py',
    'precision': 'src/utils/function/hf_precision.py',
    'recall': 'src/utils/function/hf_recall.py',
    'spearmanr': 'src/utils/function/hf_spearmanr.py',
    'pearsonr': 'src/utils/function/hf_pearsonr.py',

}


class LMTrainer():
    """Convert textural graph to text list"""

    def __init__(self, cf):
        self.cf = cf
        # logging.set_verbosity_warning()
        from transformers import logging as trfm_logging
        trfm_logging.set_verbosity_error()
        self.logger = cf.logger
        self.log = cf.logger.log
        self.update_ratio = 1

    @uf.time_logger
    def train(self):
        # ! Prepare data
        self.d = d = SeqGraph(cf := self.cf).init()
        gold_data = SeqGraphDataset(self.d, mode='train_gold')
        subset_data = lambda sub_idx: th.utils.data.Subset(gold_data, sub_idx)
        self.datasets = {_: subset_data(getattr(d, f'{_}_x'))
                         for _ in ['train', 'valid', 'test']}
        self.metrics = {m: load_metric(m_path) for m, m_path in METRICS.items()}


        # Pretrain on gold data
        self.train_data = self.datasets['train']
        train_steps = len(d.train_x) // cf.eq_batch_size + 1
        warmup_steps = int(cf.warmup_epochs * train_steps)
        eval_steps = cf.eval_patience // cf.eq_batch_size

        # ! Load bert and build classifier
        bert_model = AutoModel.from_pretrained(cf.hf_model)
        self.model = BertClassifier(
            bert_model, cf.data.n_labels,
            pseudo_label_weight=cf.pl_weight if cf.is_augmented else 0,
            dropout=cf.cla_dropout,
            loss_func=th.nn.CrossEntropyLoss(label_smoothing=cf.label_smoothing_factor, reduction=cf.ce_reduction),
            cla_bias=cf.cla_bias == 'T',
            is_augmented=cf.is_augmented,
            feat_shrink=cf.feat_shrink
        )
        if cf.local_rank <= 0:
            trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
            print(f" LM Model parameters are {trainable_params}")

        load_best_model_at_end = True
        if cf.hf_model == 'distilbert-base-uncased':
            self.model.config.dropout = cf.dropout
            self.model.config.attention_dropout = cf.att_dropout
        else:
            print('default dropout and attention_dropout are:', self.model.config.hidden_dropout_prob, self.model.config.attention_probs_dropout_prob)
            self.model.config.hidden_dropout_prob = cf.dropout
            self.model.config.attention_probs_dropout_prob = cf.att_dropout

        training_args = TrainingArguments(
            output_dir=cf.out_dir,
            evaluation_strategy='steps',
            eval_steps=eval_steps,
            save_strategy='steps',
            save_steps=eval_steps,
            learning_rate=cf.lr, weight_decay=cf.weight_decay,
            load_best_model_at_end=load_best_model_at_end, gradient_accumulation_steps=cf.grad_acc_steps,
            save_total_limit=1,
            report_to='wandb' if cf.wandb_on else None,
            per_device_train_batch_size=cf.batch_size,
            per_device_eval_batch_size=cf.batch_size * 10,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            dataloader_drop_last=True,
            num_train_epochs=cf.epochs,
            local_rank=cf.local_rank,
            dataloader_num_workers=1,
            fp16=True,
        )

        # ! Get dataloader

        def compute_metrics(pred: EvalPrediction):
            predictions, references = pred.predictions.argmax(1), pred.label_ids.argmax(1)
            return {m_name: metric.compute(predictions=predictions, references=references)
            if m_name in {'accuracy', 'pearsonr', 'spearmanr'} else metric.compute(predictions=predictions, references=references, average='macro')
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
        # ! Save bert
        # self.model.save_pretrained(cf.out_ckpt, self.model.state_dict())
        # ! Save BertClassifer Save model parameters
        if cf.local_rank <= 0:
            th.save(self.model.state_dict(), uf.init_path(cf.lm.ckpt))
        self.log(f'LM saved to {cf.lm.ckpt}')

    def eval_and_save(self):
        def get_metric(split):
            self.eval_phase = 'Test' if split == 'test' else 'Eval'
            mtc_dict = self.trainer.predict(self.datasets[split]).metrics
            ret = {f'{split}_{_}': mtc_dict[m][_] for m in mtc_dict if (_ := m.split('_')[-1]) in METRICS}
            return ret

        cf = self.cf
        res = {**get_metric('valid'), **get_metric('test')}
        res = {'val_acc': res['valid_accuracy'], 'test_acc': res['test_accuracy']}
        uf.pickle_save(res, cf.lm.result)
        cf.wandb_log({f'lm_prt_{k}': v for k, v in res.items()})

        self.log(f'\nTrain seed{cf.seed} finished\nResults: {res}\n{cf}')
