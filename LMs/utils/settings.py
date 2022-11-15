import subprocess as sp
from pathlib import Path
from types import SimpleNamespace as SN

LINUX_HOME = str(Path.home())


class ServerInfo:
    def __init__(self):
        self.gpu_mem, self.gpus, self.n_gpus = 0, [], 0
        try:
            import numpy as np
            command = "nvidia-smi --query-gpu=memory.total --format=csv"
            gpus = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            self.gpus = np.array(range(len(gpus)))
            self.n_gpus = len(gpus)
            self.gpu_mem = round(int(gpus[0].split()[0]) / 1024)
            self.sv_type = f'{self.gpu_mem}Gx{self.n_gpus}'
        except:
            print('NVIDIA-GPU not found, set to CPU.')
            self.sv_type = f'CPU'

    def __str__(self):
        return f'SERVER INFO: {self.sv_type}'


SV_INFO = ServerInfo()

PROJ_NAME = 'TAG-Benchmark'
# ! Project Path Settings

GPU_CF = {
    'py_path': f'{str(Path.home())}/miniconda/envs/ct/bin/python',
    'mnt_dir': f'{LINUX_HOME}/{PROJ_NAME}/',
    'default_gpu': '0',
}
CPU_CF = {
    'py_path': f'python',
    'mnt_dir': '',
    'default_gpu': '-1',
}
get_info_by_sv_type = lambda attr, t: CPU_CF[attr] if t == 'CPU' else GPU_CF[attr]
DEFAULT_GPU = get_info_by_sv_type('default_gpu', SV_INFO)
PYTHON = get_info_by_sv_type('py_path', SV_INFO)
MNT_DIR = get_info_by_sv_type('mnt_dir', SV_INFO)

import os.path as osp

PROJ_DIR = osp.abspath(osp.dirname(__file__)).split('utils')[0]

# Temp paths: discarded when container is destroyed
TEMP_DIR = PROJ_DIR
TEMP_PATH = f'{TEMP_DIR}temp/'
LOG_PATH = f'{TEMP_DIR}log/'

MNT_TEMP_DIR = f'{PROJ_DIR}temp/'
TEMP_RES_PATH = f'{PROJ_DIR}temp_results/'
RES_PATH = f'{PROJ_DIR}results/'
DB_PATH = f'{PROJ_DIR}exp_db/'

# ! Data Settings
DATA_PATH = f'{PROJ_DIR}data/'
OGB_ROOT = f'{PROJ_DIR}data/ogb/'

DATA_INFO = {
    'arxiv': {
        'type': 'ogb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 40,
        'n_nodes': 169343,
        'ogb_name': 'ogbn-arxiv',
        'raw_data_path': OGB_ROOT,  # Place to save raw data
        'max_length': 512,  # Place to save raw data
        'data_root': f'{OGB_ROOT}ogbn_arxiv/',  # Default ogb download target path
        'raw_text_url': 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz',
    },
}
get_d_info = lambda x: DATA_INFO[x.split('_')[0]]

TR_RATIO_DICT = {_d: _['train_ratio'] for _d, _ in DATA_INFO.items()}
DATASETS = list(DATA_INFO.keys())
DEFAULT_DATASET = 'arxiv_TA'
DEFAULT_D_INFO = get_d_info(DEFAULT_DATASET)

METRIC = 'acc'

# ! Default Settings
EARLY_STOP = 30
MAX_EPOCHS = 300
