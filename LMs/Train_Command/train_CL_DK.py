import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('LMs')[0]+'LMs'))
from lm_utils import *

if __name__ == "__main__":
    # ! Init Arguments
    model = get_lm_model()
    Trainer = get_lm_trainer(model, 'CL_DK')
    Config = get_lm_config(model)

    args = Config().parse_args()
    cf = Config(args).init()
    # ! Load data and train
    trainer = Trainer(cf=cf)
    trainer.train_trainer()

