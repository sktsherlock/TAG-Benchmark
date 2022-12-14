from datasets import load_metric
from transformers import (
    EvalPrediction,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForNextSentencePrediction,
    AutoModel,
)
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
        model = AutoModel.from_pretrained(cf.hf_model)  # TinyBert NSP: 4386178; Pure TinyBERT: 4385920;
        self.model = TNPClassifier(
            model=model, n_labels=3,
            dropout=cf.cla_dropout,
            loss_func=th.nn.CrossEntropyLoss(label_smoothing=cf.label_smoothing_factor, reduction=cf.ce_reduction),
            cla_bias=cf.cla_bias == 'T',
        )
        load_best_model_at_end = True
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
            per_device_eval_batch_size= cf.batch_size * 6 if cf.hf_model in {'distilbert-base-uncased',
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
            predictions, references = pred.predictions.argmax(1), pred.label_ids.argmax(1)
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

    @uf.time_logger
    def train_notrainer(self):
        import math
        import json
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        from accelerate import Accelerator, DistributedType
        from tqdm.auto import tqdm
        import os.path as osp
        from transformers import (
            get_scheduler,
            AutoModelForMaskedLM,
            AutoModel,
            )

        # ! Prepare data
        self.d = d = Sequence(cf := self.cf).NP_init()
        gold_data = NP_Dataset(self.d)
        subset_data = lambda sub_idx: th.utils.data.Subset(gold_data, sub_idx)
        self.datasets = {_: subset_data(getattr(d, f'{_}_x'))
                         for _ in ['train', 'valid']}
        self.metric = evaluate.load("accuracy")

        # Toplogical pretrain in the TNP tasks
        self.train_data = self.datasets['train']

        # ! Prepare the Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=cf.grad_acc_steps)

        # ! Prepare your Dataloader
        per_device_eval_batch_size =  cf.batch_size * 6 if cf.hf_model in {'distilbert-base-uncased',
                                                                            'google/electra-base-discriminator'} else cf.batch_size * 10
        train_dataloader = DataLoader(self.train_data, shuffle=True,  batch_size=cf.batch_size )
        eval_dataloader = DataLoader(self.datasets['valid'], batch_size=per_device_eval_batch_size)
        # ! Load Model for NP with no trainer
        self.model = AutoModelForMaskedLM.from_pretrained(cf.hf_model, num_labels=3)

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
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cf.grad_acc_steps)
        total_train_steps = self.cf.epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=self.optimizer,
            num_warmup_steps= warmup_steps * cf.grad_acc_steps,
            num_training_steps= total_train_steps * cf.grad_acc_steps,
        )

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cf.grad_acc_steps)
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

        for epoch in range(starting_epoch, cf.batch_size):
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= total_train_steps:
                    break

            self.model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with th.no_grad():
                    outputs = self.model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))

            losses = th.cat(losses)
            try:
                eval_loss = th.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            self.log(f"epoch {epoch}: perplexity: {perplexity}")
            # ! Save the final model
            if cf.out_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                    cf.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    with open(osp.join(cf.out_dir, "all_results.json"), "w") as f:
                        json.dump({"perplexity": perplexity}, f)