import os.path as osp
import sys
from tqdm import tqdm

from transformers import AutoModel, TrainingArguments, Trainer

from lm_utils import *
from utils.function.os_utils import mkdir_p
from model import *
import numpy as np
from utils.data.datasets import SeqGraphDataset
from transformers import logging as trfm_logging
from ogb.nodeproppred import Evaluator

class LmInfTrainer:
    """Convert textural graph to text list"""

    def __init__(self, cf):
        self.cf = cf
        self.logger = cf.logger
        self.log = cf.logger.log

    @torch.no_grad()
    def inference_emb(self):
        self.d = d = Sequence(cf := self.cf).tokenize_init()
        inference_dataset = SeqGraphDataset(self.d, mode='inference')
        # Save embedding and predictions
        self.model = AutoModel.from_pretrained(cf.hf_model) if cf.pretrain_path is None else AutoModel.from_pretrained(
            f'{cf.pretrain_path}')
        # The reduction should be sum in case unbalanced gold and pseudo data
        self.log(f'Performing inference using LM model: {cf.pretrain_path}')
        inf_model = BertEmbInfModel(self.model)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=f'{self.cf.out_dir}inf/',
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.cf.inf_batch_size,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            local_rank=self.cf.local_rank,
            fp16_full_eval=True,
        )
        self.trainer = Trainer(model=inf_model, args=inference_args)
        out_emb = self.trainer.predict(inference_dataset)
        if self.cf.inference_dir is None:
            raise ValueError('Please input the true inference dir.')
        else:
            if not osp.exists(self.cf.inference_dir):
                mkdir_p(self.cf.inference_dir)
                with open(osp.join(self.cf.inference_dir, 'emb.npy'), 'wb') as f:
                    np.save(f, out_emb.predictions)
            else:
                self.log('LM have inferenced before')
            self.log(f'LM inference completed in {self.cf.inference_dir})')
