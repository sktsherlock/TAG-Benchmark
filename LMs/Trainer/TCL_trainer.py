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
            AutoModelForMaskedLM,
            AutoModel,
        )

        # ! Prepare data
        self.d = d = Sequence(cf := self.cf).init()
        self.train_data = SeqCLDataset(self.d)

        # ! Prepare the Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=cf.grad_acc_steps)

        # ! Prepare your Dataloader
        per_device_eval_batch_size = cf.batch_size * 6 if cf.hf_model in {'distilbert-base-uncased',
                                                                          'google/electra-base-discriminator'} else cf.batch_size * 10
        train_dataloader = DataLoader(self.train_data, shuffle=True, batch_size=cf.batch_size)
        eval_dataloader = DataLoader(self.datasets['valid'], batch_size=per_device_eval_batch_size)
        # ! Load Model for NP with no trainer
        PLM = AutoModel.from_pretrained(cf.hf_model)

        self.model = CLIPModel(
            PLM,
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
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cf.grad_acc_steps)
        total_train_steps = self.cf.epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps * cf.grad_acc_steps,
            num_training_steps=total_train_steps * cf.grad_acc_steps,
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
                    loss = self.model(**batch)
                    accelerator.backward(loss)
                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()
                    wandb.log('CL_loss', loss)


                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= total_train_steps:
                    break

            # ! Save the final model
            if cf.out_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(PLM)
                unwrapped_model.save_pretrained(
                    cf.out_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )

    @uf.time_logger
    def train_trainer(self):
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
                th.save(self.model.state_dict(), uf.init_path(cf.cache_dir))
            else:
                PLM.save_pretrained(cf.out_dir)
                th.save(self.model.state_dict(), uf.init_path(cf.lm.ckpt))
        else:
            print('Dont save the model in the local_rank:', cf.local_rank)

