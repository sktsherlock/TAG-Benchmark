from models.LMs.lm_utils import *


class GPTConfig(LMConfig):

    def __init__(self, args=None):
        super(GPTConfig, self).__init__(args)
        self.model = 'GTP'
        self._post_init(args)

    para_prefix = {**LMConfig.para_prefix}
    args_to_parse = list(para_prefix.keys())
    meta_data = {
        'GPT2':
            SN(
                hf_model='gpt2',
                hidden_dim=768,
                max_bsz=SN(  # Batch size for different device
                    train={12: 8, 16: 12, 24: 9, 32: 30, 40: 32, 70: 96},
                    inf={12: 150, 16: 200, 24: 150, 32: 512, 40: 120, 70: 1120},
                ),
                prt_lm={  # Initial LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=4 --eq_batch_size=36 --eval_patience=50000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=2e-05 --warmup_epochs=0.6',
                        max_n_gpus=4,
                    )
                },
            ),
        'GPT2-large':
            SN(
                hf_model='gpt2-large',
                hidden_dim=1280,
                max_bsz=SN(  # Batch size for different device
                    train={12: 8, 16: 12, 24: 9, 32: 3, 40: 32, 70: 96},
                    inf={12: 150, 16: 200, 24: 150, 32: 90, 40: 120, 70: 1120},
                ),
                prt_lm={  # Initial LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=4 --eq_batch_size=36 --eval_patience=50000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=2e-05 --warmup_epochs=0.6',
                        max_n_gpus=4,
                    )
                },
            ),

    }
