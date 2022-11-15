import os
import utils.function as uf
from utils.data import SeqGraph
from utils.modules import ModelConfig
from utils.settings import *
from importlib import import_module


class LMConfig(ModelConfig):
    def __init__(self, args=None):
        # ! INITIALIZE ARGS
        super(LMConfig, self).__init__('LMs')

        # ! LM Settings
        self.model = 'Bert'

        self.lr = 0.00002
        self.eq_batch_size = 36
        self.weight_decay = 0.01
        self.label_smoothing_factor = 0.1
        self.dropout = 0.1
        self.warmup_epochs = 0.2
        self.att_dropout = 0.1
        self.cla_dropout = 0.1
        self.cla_bias = 'T'
        self.grad_acc_steps = 2
        self.load_best_model_at_end = 'T'

        self.save_folder = ''
        self.emi_file = ''
        self.ce_reduction = 'mean'

        self.feat_shrink = '100'
        self.feat_shrink = ''
        self.eval_patience = 100000
        self.md = None  # Tobe initialized in sub module

    def _intermediate_args_init(self):
        """
        Parse intermediate settings that shan't be saved or printed.
        """
        self.mode = 'pre_train'
        self.lm_meta_data_init()
        self.hf_model = self.md.hf_model
        self.hidden_dim = int(self.feat_shrink) if self.feat_shrink else self.md.hidden_dim

        # * Init LM settings using pre-train folder
        self.lm = self.get_lm_info(self.save_folder, self.model)

    def get_lm_info(self, lm_folder, model):
        return SN(folder=lm_folder,
                  emb=f'{lm_folder}{model}.emb',
                  pred=f'{lm_folder}{model}.pred',
                  ckpt=f'{lm_folder}{model}.ckpt',
                  result=f'{lm_folder}{model}.result')

    def lm_meta_data_init(self):
        self.md = self.meta_data[self.model]

    def _exp_init(self):
        super()._exp_init()
        # ! Batch_size Setting
        max_bsz = self.md.max_bsz
        self.batch_size, self.grad_acc_steps = uf.calc_bsz_grad_acc(self.eq_batch_size, max_bsz.train, SV_INFO)
        self.inf_batch_size = uf.get_max_batch_size(SV_INFO.gpu_mem, max_bsz.inf)

    def _data_args_init(self):
        # Dataset
        self.lm_md = self.md
        self.data = SeqGraph(self)

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {
        'model': '', 'lr': 'lr', 'eq_batch_size': 'bsz',
        'weight_decay': 'wd', 'dropout': 'do', 'att_dropout': 'atdo', 'cla_dropout': 'cla_do', 'cla_bias': 'cla_bias',
        'epochs': 'e', 'warmup_epochs': 'we', 'eval_patience': 'ef',
        'load_best_model_at_end': 'load',
        'label_smoothing_factor': 'lsf', 'pl_weight': 'alpha', 'pl_ratio': '', 'ce_reduction': 'red', 'feat_shrink': ''}

    args_to_parse = list(para_prefix.keys())
    meta_data = None

    @property
    def parser(self):
        parser = super().parser
        parser.add_argument("-m", "--model", default='TinyBert')
        parser.add_argument("-I", "--is_inf", action="store_true")
        return parser

    @property
    def out_dir(self):
        return f'{TEMP_PATH}{self.model}/ckpts/{self.dataset}/{self.model_cf_str}/'

    @property
    def model_cf_str(self):
        return self.para_prefix


# ! LM Settings
LM_SETTINGS = {}
LM_MODEL_MAP = {
    'Deberta-large': 'Deberta',
    'TinyBert': 'Bert',
    'RoBerta-large': 'RoBerta'
}

#! Need
def get_lm_model():
    return LMConfig().parser.parse_known_args()[0].model


def get_lm_trainer(model):
    if model in ['GPT2','GPT2-large']:
        from LMs.GPT_trainer import GPTTrainer as LMTrainer
    else:
        from LMs.lm_trainer import LMTrainer as LMTrainer
    return LMTrainer


def get_lm_config(model):
    model = LM_MODEL_MAP[model] if model in LM_MODEL_MAP else model
    return import_module(f'LMs.{model}').Config


