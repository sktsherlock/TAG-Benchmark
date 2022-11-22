from abc import ABCMeta

import utils as uf
from utils.settings import *
import os


class ModelConfig(metaclass=ABCMeta):
    def __init__(self):
        # Other attributes
        self._ignored_paras = ['verbose', 'important_paras', 'device']

    def _post_init(self, args):
        """Post initialize arguments
        Step 1: Overwrite default parameters by args.
        Step 2: Post process args (e.g. parse args to sub args)
        """
        if args is not None:
            self.__dict__.update(args.__dict__)
            self._post_process_args()

    def _exp_init(self):
        self._data_args_init()
        uf.exp_init(self)

    # *  <<<<<<<<<<<<<<<<<<<< POST INIT FUNCS >>>>>>>>>>>>>>>>>>>>
    def _post_process_args(self):
        prev_cf = list(self.__dict__.keys())

        # Register common intermediate settings
        self.local_rank = -1 if 'LOCAL_RANK' not in os.environ else int(os.environ['LOCAL_RANK'])
        self.verbose = -1 if self.local_rank > 0 else self.verbose

        # Ignore intermediate settings

        self._intermediate_args_init()
        intermediate_paras = [_ for _ in self.__dict__ if _ not in prev_cf]
        self._ignored_paras += intermediate_paras

    def wandb_init(self):
        # Turn off Wandb gradients loggings
        os.environ["WANDB_WATCH"] = "false"

        wandb_settings_given = self.wandb_name != 'OFF' or self.wandb_id != ''
        not_parallel = self.local_rank <= 0

        if wandb_settings_given and not_parallel:
            try:
                import wandb
                from private.exp_settings import WANDB_API_KEY, WANDB_DIR, WANDB_PROJ, WANDB_ENTITY
                os.environ['WANDB_API_KEY'] = WANDB_API_KEY
                print('self.wandb_id ==', self.wandb_id)
                # ! Create wandb session
                if self.wandb_id == '':
                    # First time running, create new wandb
                    wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, reinit=True, config=self.model_conf)
                    self.wandb_id = wandb.run.id
                else:
                    print(f'Resume from previous wandb run {self.wandb_id}')
                    wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, reinit=True,
                               resume='must', id=self.wandb_id)
                self.wandb_on = True
                print('self.wandb_on is ', self.wandb_on)
            except:
                print('self.wandb_on is ', self.wandb_on)
                return None
        else:
            os.environ["WANDB_DISABLED"] = "true"
            return None

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    # 对结果有影响的参数写在 para_prefix 里，结果没有影响的参数不要写在 para_prefix 里（e.g. shared configs like LM checkpoint）
    # Example: 'epochs' : 'e' 表示 epochs 参数要以 e 为 prefix 成为 f_prefix 的一部分
    _path_attrs = []

    def _path_init(self, ):
        path_dict = {_: getattr(self, _) for _ in self._path_attrs}
        self.path = SN(**path_dict)
        uf.mkdir_list(list(path_dict.values()))

    @property
    def f_prefix(self):
        return f"{self.model}/{self.dataset}/seed{self.seed}{self.model_cf_str}"

    @property
    def res_file(self):
        return f'{TEMP_RES_PATH}{self.f_prefix}.json'

    @property
    def model_cf_str(self):
        """Model config to str
        Default as model f_prefix
        If there are multiple submodules, this property should be overwritten
        """
        return self._model.f_prefix

    @property
    def model_conf(self):
        valid_types = [str, int, float, type(None)]
        judge_para = lambda p, v: p not in self._ignored_paras and p[0] != '_' and type(v) in valid_types
        # Print the model settings only.
        return {p: v for p, v in sorted(self.__dict__.items()) if judge_para(p, v)}

    def __str__(self):
        return f'{self.model} config: \n{self.model_conf}'

    def parse_args(self):
        # ! Parse defined args
        defined_args = (parser := self.parser).parse_known_args()[0]
        return parser.parse_args()


class SubConfig(SN):
    """
    SubConfig for parsing and generating file-prefixes that distinguishes results.
    """

    def __init__(self, conf: ModelConfig, para_prefix_dict, sub_cf_prefix=None, ignored_paras=[]):
        para_map = lambda x: f"{'' if sub_cf_prefix is None else f'{sub_cf_prefix}_'}{x}"
        # Init sub config by para_prefix_dict
        super().__init__(**{_: getattr(conf, para_map(_)) for _ in para_prefix_dict})
        cf_to_str = lambda c, f: '' if f is None else f'{f}{getattr(conf, para_map(c))}'
        self.f_prefix = '_'.join(cf_to_str(c, f) for c, f in para_prefix_dict.items())
        for c, f in para_prefix_dict.items():
            cf_to_str(c, f)
        # Defines intermediate variables that shan't be passed to args
        self._ignored_paras = ['train_cmds', 'f_prefix'] + ignored_paras

    @property
    def model_conf(self):
        is_para = lambda para: para[0] != '_' and para not in self._ignored_paras
        return SN(**{k: v for k, v in self.__dict__.items() if is_para(k)})

    def combine(self, new_conf):
        return SN(**{**new_conf.model_conf.__dict__, **self.model_conf.__dict__})