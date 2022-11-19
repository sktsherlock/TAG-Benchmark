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
            except:
                return None
        else:
            os.environ["WANDB_DISABLED"] = "true"
            return None

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    # 对结果有影响的参数写在 para_prefix 里，结果没有影响的参数不要写在 para_prefix 里（e.g. shared configs like LM checkpoint）
    # Example: 'epochs' : 'e' 表示 epochs 参数要以 e 为 prefix 成为 f_prefix 的一部分
    para_prefix = {_: '' for _ in ['gpus', 'dataset']}
    _path_attrs = []

    def _path_init(self, ):
        path_dict = {_: getattr(self, _) for _ in self._path_attrs}
        self.path = SN(**path_dict)
        uf.mkdir_list(list(path_dict.values()))

    @property
    def f_prefix(self):
        return f"{self.model}/{self.dataset}/l{self.train_percentage:02d}/seed{self.seed}{self.model_cf_str}"

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

    # @property
    # def parser(self) -> ArgumentParser:
    #     # parser = ArgumentParser("Experimental settings")
    #     # parser.add_argument("-g", '--gpus', default=DEFAULT_GPU, type=str,
    #     #                     help='a list of active gpu ids, separated by ",", "cpu" for cpu-only mode.')
    #     # parser.add_argument("-d", "--dataset", type=str, default=DEFAULT_DATASET)
    #     # parser.add_argument("-t", "--train_percentage", default=DEFAULT_D_INFO['train_ratio'], type=int)
    #     # parser.add_argument("-v", "--verbose", default=1, type=int, help='Verbose level, higher level generates more log, -1 to shut down')
    #     # parser.add_argument('--tqdm_on', action="store_true", help='show log by tqdm or not')
    #     # parser.add_argument("-w", "--wandb_name", default='OFF', type=str, help='Wandb logger or not.')
    #     # parser.add_argument("--epochs", default=MAX_EPOCHS, type=int)
    #     # parser.add_argument("--seed", default=0, type=int)
    #     return parser

    def __str__(self):
        return f'{self.model} config: \n{self.model_conf}'

    def parse_args(self):
        # ! Parse defined args
        defined_args = (parser := self.parser).parse_known_args()[0]
        return parser.parse_args()
