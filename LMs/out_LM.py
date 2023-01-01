import os.path as osp
import sys
from tqdm import tqdm

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from transformers import AutoModel, TrainingArguments, Trainer

from lm_utils import *
from model import *
from utils.data.datasets import SeqGraphDataset
from transformers import logging as trfm_logging
from ogb.nodeproppred import Evaluator

METRIC_LIST = ['accuracy']


class LmInfTrainer:
    """Convert textural graph to text list"""

    def __init__(self, cf):
        self.cf = cf
        self.logger = cf.logger
        self.log = cf.logger.log
        self.d = d = Sequence(cf := self.cf).init()
        # self._evaluator = Evaluator(name=cf.data.ogb_name)
        # self.evaluator = lambda preds, labels: self._evaluator.eval({
        #     "y_true": th.tensor(labels).view(-1, 1),
        #     "y_pred": th.tensor(preds).view(-1, 1),
        # })["acc"]
        # ! memmap
        # self.emb = np.memmap(uf.init_path(self.cf.emi.lm.emb), dtype=np.float16, mode='w+',
        #                      shape=(self.d.n_nodes, self.cf.hidden_dim))
        # self.pred = np.memmap(uf.init_path(self.cf.emi.lm.pred), dtype=np.float16, mode='w+',
        #                       shape=(self.d.n_nodes, self.d.md['n_labels']))
        # ! Load BertConfigs Encoder + Decoder together and output emb + predictions
        self.model = AutoModel.from_pretrained(cf.hf_model) if cf.pretrain_path is None else AutoModel.from_pretrained(
            f'{cf.pretrain_path}')
        # The reduction should be sum in case unbalanced gold and pseudo data
        self.log(f'Performing inference using LM model: {cf.pretrain_path}')

    @torch.no_grad()
    def inference_emb(self):
        th.cuda.empty_cache()
        inference_dataset = SeqGraphDataset(self.d)
        # Save embedding and predictions
        inf_model = BertEmbInfModel(self.model)  # .to(self.cf.device)
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
        trainer = Trainer(model=inf_model, args=inference_args)
        out_emb = trainer.predict(inference_dataset)
        with open(osp.join(self.cf.out_dir, 'emb.npy'), 'wb') as f:
            np.save(f, out_emb)

        self.log(f'LM inference completed')


if __name__ == "__main__":
    # ! Init Arguments
    model = get_lm_model()
    Config = get_lm_config(model)
    args = Config().parse_args()
    cf = Config(args).init()

    # ! Load data and train
    trainer = LmInfTrainer(cf=cf)
    trainer.inference_emb()
