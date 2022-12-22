import os.path as osp
import sys

# sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from lm_utils import *

if __name__ == "__main__":
    # ! Init Arguments
    model = get_lm_model()
    Trainer = get_lm_trainer(model)
    Config = get_lm_config(model)

    args = Config().parse_args()
    cf = Config(args).init()
    # ! Load data and train
    trainer = Trainer(cf=cf)
    #trainer.train()
    trainer.train_notrainer()
    trainer.eval_and_save()
