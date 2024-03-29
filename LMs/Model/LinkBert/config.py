from lm_utils import *


class LinkBertConfig(LMConfig):

    def __init__(self, args=None):
        super(LinkBertConfig, self).__init__(args)
        self.model = 'LinkBert'
        self._post_init(args)

    para_prefix = {**LMConfig.para_prefix}
    args_to_parse = list(para_prefix.keys())
    meta_data = {
        'LinkBert':
            SN(
                hf_model='michiyasunaga/LinkBERT-base',
                father_model='LinkBert',
                hidden_dim=768,
                max_bsz=SN(  # Batch size for different device
                    train={12: 8, 16: 12, 24: 9, 32: 40, 40: 32, 70: 96},
                    inf={12: 150, 16: 200, 24: 150, 32: 360, 40: 580, 70: 1120},
                ),
                prt_lm={  # Initial LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=4 --eq_batch_size=36 --eval_patience=50000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=2e-05 --warmup_epochs=0.6',
                        max_n_gpus=4,
                    )
                },
            ),
        'LinkBert-large':
            SN(
                hf_model='michiyasunaga/LinkBERT-large',
                father_model='LinkBert',
                hidden_dim=1024,
                max_bsz=SN(  # Batch size for different device
                    train={12: 6, 16: 10, 24: 16, 32: 12},
                    inf={12: 150, 16: 200, 24: 150, 32: 250},
                ),
                prt_lm={  # Initial LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--lr=1e-05 --eq_batch_size=36 --weight_decay=0.01 --dropout=0.1 --att_dropout=0.1 --cla_dropout=0.1 --cla_bias=T --epochs=4 --warmup_epochs=0.2 --eval_patience=50000',
                        max_n_gpus=4,
                    )
                }
            ),
    }
