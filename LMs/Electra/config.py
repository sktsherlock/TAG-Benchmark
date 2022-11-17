from models.LMs.lm_utils import *


class ElectraConfig(LMConfig):

    def __init__(self, args=None):
        super(ElectraConfig, self).__init__(args)
        self.model = 'Electra'
        self._post_init(args)

    para_prefix = {**LMConfig.para_prefix}
    args_to_parse = list(para_prefix.keys())
    meta_data = {
        'Electra':
            SN(
                hf_model='google/electra-small-discriminator',
                hidden_dim=256,
                max_bsz=SN(  # Batch size for different device
                    train={12: 8, 16: 40, 24: 9, 32: 80, 40: 18, 70: 48},
                    inf={12: 150, 16: 350, 24: 150, 32: 700, 40: 300, 70: 560},
                ),
                prt_lm={  # Initial LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=4 --eq_batch_size=36 --eval_patience=50000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=2e-05 --warmup_epochs=0.6',
                        max_n_gpus=4,
                    ),
                    'products': SN(
                        model='FtV1',
                        cmd='--lr=2e-05 --eq_batch_size=144 --weight_decay=0.01 --dropout=0.1 --att_dropout=0.3 --cla_dropout=0.2 --cla_bias=T --warmup_epochs=0.2 --eval_patience=65308 --epochs=4 --label_smoothing_factor=0.1 --warmup_epochs=0.6',
                        max_n_gpus=8,
                    ),
                    'paper': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=5 --eq_batch_size=288 --eval_patience=410000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=5e-05 --warmup_epochs=0.6',
                        max_n_gpus=16,
                    )
                },
            ),
        'Electra-base':
            SN(
                hf_model='google/electra-base-discriminator',
                hidden_dim=768,
                max_bsz=SN(  # Batch size for different device
                    train={12: 8, 16: 18, 24: 9, 32: 40, 40: 18, 70: 48},
                    inf={12: 150, 16: 150, 24: 150, 32: 512, 40: 300, 70: 560},
                ),
                prt_lm={  # Initial LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=4 --eq_batch_size=36 --eval_patience=50000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=2e-05 --warmup_epochs=0.6',
                        max_n_gpus=4,
                    ),
                    'products': SN(
                        model='FtV1',
                        cmd='--lr=2e-05 --eq_batch_size=144 --weight_decay=0.01 --dropout=0.1 --att_dropout=0.3 --cla_dropout=0.2 --cla_bias=T --warmup_epochs=0.2 --eval_patience=65308 --epochs=4 --label_smoothing_factor=0.1 --warmup_epochs=0.6',
                        max_n_gpus=8,
                    ),
                    'paper': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=5 --eq_batch_size=288 --eval_patience=410000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=5e-05 --warmup_epochs=0.6',
                        max_n_gpus=16,
                    )
                },
            ),
        'Electra-large':
            SN(
                hf_model='google/electra-large-discriminator',
                hidden_dim=768,
                max_bsz=SN(  # Batch size for different device
                    train={12: 8, 16: 12, 24: 9, 32: 12, 40: 18, 70: 48},
                    inf={12: 150, 16: 200, 24: 150, 32: 200, 40: 300, 70: 560},
                ),
                prt_lm={  # Initial LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=4 --eq_batch_size=36 --eval_patience=50000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=2e-05 --warmup_epochs=0.6',
                        max_n_gpus=4,
                    ),
                    'products': SN(
                        model='FtV1',
                        cmd='--lr=2e-05 --eq_batch_size=144 --weight_decay=0.01 --dropout=0.1 --att_dropout=0.3 --cla_dropout=0.2 --cla_bias=T --warmup_epochs=0.2 --eval_patience=65308 --epochs=4 --label_smoothing_factor=0.1 --warmup_epochs=0.6',
                        max_n_gpus=8,
                    ),
                    'paper': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=5 --eq_batch_size=288 --eval_patience=410000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=5e-05 --warmup_epochs=0.6',
                        max_n_gpus=16,
                    )
                },
            ),
    }

