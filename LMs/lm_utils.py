import os
import utils.function as uf
from utils.data import Sequence
from utils.modules import ModelConfig
from utils.settings import *
from importlib import import_module
from argparse import ArgumentParser
from utils.modules.logger import Logger

class LMConfig(ModelConfig):
    def __init__(self, args=None):
        # ! INITIALIZE ARGS
        super(LMConfig, self).__init__()

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
        self.ce_reduction = 'mean'

        self.feat_shrink = '100'
        self.feat_shrink = ''
        self.eval_patience = 100000
        self.md = None  # Tobe initialized in sub module

        # ! Experiments Settings
        self.seed = 0
        self.wandb_name = ''
        self.wandb_id = ''
        self.dataset = (d := DEFAULT_DATASET)
        self.epochs = 4
        self.verbose = 1
        self.device = None
        self.wandb_on = False
        self.birth_time = uf.get_cur_time(t_format='%m_%d-%H_%M_%S')
        self._wandb = None


    def init(self):
        """Initialize path, logger, experiment environment
        These environment variables should only be initialized in the actual training process. In other cases, where we only want the config parameters parser/res_file, the init function should not be called.
        """

        self._path_init()
        self.wandb_init()
        self.logger = Logger(self)
        self.log = self.logger.log
        self.wandb_log = self.logger.wandb_log
        self.log(self)
        self._exp_init()
        return self


    def _intermediate_args_init(self):
        """
        Parse intermediate settings that shan't be saved or printed.
        """
        self.mode = 'pre_train'
        self.md = self.meta_data[self.model]
        self.hf_model = self.md.hf_model
        self.father_model = self.md.father_model
        self.hidden_dim = int(self.feat_shrink) if self.feat_shrink else self.md.hidden_dim

        # * Init LM settings using pre-train folder
        self.lm = self.get_lm_info(self.save_folder, self.model)

    def get_lm_info(self, lm_folder, model):
        return SN(folder=lm_folder,
                  emb=f'{lm_folder}{model}.emb',
                  pred=f'{lm_folder}{model}.pred',
                  ckpt=f'{lm_folder}{model}.ckpt',
                  result=f'{lm_folder}{model}.result')

    def _exp_init(self):
        super()._exp_init() # will initialize the data
        # ! Batch_size Setting
        max_bsz = self.md.max_bsz
        self.batch_size, self.grad_acc_steps = uf.calc_bsz_grad_acc(self.eq_batch_size, max_bsz.train, SV_INFO)
        self.inf_batch_size = uf.get_max_batch_size(SV_INFO.gpu_mem, max_bsz.inf)

    def _data_args_init(self):
        # Dataset
        self.lm_md = self.md
        self.data = Sequence(self)

    meta_data = None

    @property
    def parser(self):
        parser = ArgumentParser("Experimental settings")
        parser.add_argument("-g", '--gpus', default='cpu', type=str,
                            help='a list of active gpu ids, separated by ",", "cpu" for cpu-only mode.')
        parser.add_argument("-d", "--dataset", type=str, default=DEFAULT_DATASET)
        parser.add_argument("-t", "--train_percentage", default=DEFAULT_D_INFO['train_ratio'], type=int)
        parser.add_argument("-v", "--verbose", default=1, type=int, help='Verbose level, higher level generates more log, -1 to shut down')
        parser.add_argument('--tqdm_on', action="store_true", help='show log by tqdm or not')
        parser.add_argument("-w", "--wandb_name", default='OFF', type=str, help='Wandb logger or not.')
        parser.add_argument("--epochs", default=3, type=int)
        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("-m", "--model", default='TinyBert')
        parser.add_argument("-I", "--is_inf", action="store_true")
        parser.add_argument("-lr", "--lr", default=0.00002, type=float, help='LM model learning rate')
        parser.add_argument("-bsz", "--eq_batch_size", default=36, type=int)
        parser.add_argument("-wd", "--weight_decay", default=0.01)
        parser.add_argument("-do", "--dropout", default=0.1, type=float)
        parser.add_argument("-atdo", "--att_dropout", default=0.1, type=float)
        parser.add_argument("-cla", "--cla_dropout", default=0.1, type=float)
        parser.add_argument("-cla_bias", "--cla_bias", default='T',help='Classification model bias')
        parser.add_argument("-wmp", "--warmup_epochs", default=0.2, type=float)
        parser.add_argument("-ef", "--eval_patience", default=100000, type=int)
        parser.add_argument("-lsf", "--label_smoothing_factor", default= 0.1, type=float)
        parser.add_argument("-ce", "--ce_reduction", default='mean')
        parser.add_argument("-feat_shrink", "--feat_shrink", default=None, type=str)
        parser.add_argument("-wid", "--wandb_id", default=None, type=str)
        parser.add_argument("--device", default=None, type=str)
        parser.add_argument("--wandb_on", default=False, type=bool)
        parser.add_argument("-prt", "--pretrain", default=False, type=bool)
        return parser


        return parser

    @property
    def out_dir(self):
        return f'{TEMP_PATH}{self.model}/ckpts/{self.dataset}/'

    @property
    def model_cf_str(self):
        return self.para_prefix


# ! LM Settings
LM_SETTINGS = {}
LM_MODEL_MAP = {
    'Deberta-large': 'Deberta',
    'TinyBert': 'Bert',
    'Roberta-large': 'RoBerta',
    'LinkBert-large': 'LinkBert',
    'Bert-large': 'Bert',
    'GPT2': 'GPT',
    'GPT2-large': 'GPT',
    'Electra-large': 'Electra',
    'Electra-base': 'Electra',
}

#! Need
def get_lm_model():
    return LMConfig().parser.parse_known_args()[0].model


def get_lm_trainer(model):
    if model in ['GPT2','GPT2-large']:
        from GPT_trainer import GPTTrainer as LMTrainer
    else:
        from lm_trainer import LMTrainer as LMTrainer
    return LMTrainer


def get_lm_config(model):
    model = LM_MODEL_MAP[model] if model in LM_MODEL_MAP else model
    return import_module(f'{model}').Config


