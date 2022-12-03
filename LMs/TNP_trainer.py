from datasets import load_metric
from transformers import EvalPrediction, TrainingArguments, Trainer, AutoTokenizer, AutoModelForNextSentencePrediction
import utils as uf
from model import *
from utils.data.datasets import *
import torch as th
import evaluate
import wandb

class TNPTrainer():
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
        self.d = d = Sequence(cf := self.cf).NP_init()
        gold_data = NP_Dataset(self.d)
        subset_data = lambda sub_idx: th.utils.data.Subset(gold_data, sub_idx)
        self.datasets = {_: subset_data(getattr(d, f'{_}_x'))
                         for _ in ['train', 'valid']}
        self.metric = evaluate.load("accuracy")

        # Toplogical pretrain in the TNP tasks
        self.train_data = self.datasets['train']
        train_steps = len(self.train_data) // cf.eq_batch_size + 1
        warmup_steps = int(cf.warmup_epochs * train_steps)
        eval_steps = cf.eval_patience // cf.eq_batch_size

        # ! Load bert and build classifier
        self.model = AutoModelForNextSentencePrediction.from_pretrained(
            cf.hf_model)  # TinyBert NSP: 4386178; Pure TinyBERT: 4385920;
        if cf.local_rank <= 0:
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f" LM Model parameters are {trainable_params}")

        load_best_model_at_end = True
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

        training_args = TrainingArguments(
            output_dir=cf.out_dir,
            evaluation_strategy='steps',
            eval_steps=eval_steps,
            save_strategy='steps',
            save_steps=eval_steps,
            learning_rate=cf.lr, weight_decay=cf.weight_decay,
            load_best_model_at_end=load_best_model_at_end, gradient_accumulation_steps=cf.grad_acc_steps,
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

        def compute_metrics(pred: EvalPrediction):
            predictions, references = pred.predictions.argmax(1), pred.label_ids
            return self.metric.compute(predictions=predictions, references=references)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.datasets['valid'],
            compute_metrics=compute_metrics,
        )
        self.eval_phase = 'Eval'
        self.trainer.train()
        self.trainer.save_model()

        self.log(f'TNP LM saved finish.')

    def eval_and_save(self):
        def get_metric(split):
            self.eval_phase = 'Eval'
            mtc_dict = self.trainer.predict(self.datasets[split]).metrics
            return mtc_dict

        cf = self.cf
        res = {**get_metric('valid')}
        uf.pickle_save(res, cf.lm.result)
        cf.wandb_log({f'TNP_{k}': v for k, v in res.items()})

        self.log(f'\nTrain seed{cf.seed} finished\nResults: {res}\n{cf}')
