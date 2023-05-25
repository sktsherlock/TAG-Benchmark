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
# MNT_DIR = get_info_by_sv_type('mnt_dir', SV_INFO)

import os.path as osp

PROJ_DIR = osp.abspath(osp.dirname(__file__)).split('LMs')[0]
LM_PROJ_DIR = osp.join(PROJ_DIR, 'LMs/')

# Mount paths: kept in cloud and shared cross containers
MS_USER = 'v-wzhuang'
MNT_ROOT = f'/mnt/{MS_USER}/'
# Mount path: to be shared
MOUNTED = osp.exists(MNT_ROOT)
MNT_DIR = f'{MNT_ROOT}{PROJ_NAME}/' if MOUNTED else PROJ_DIR
# Temp paths: discarded when container is destroyed
TEMP_DIR = LM_PROJ_DIR
TEMP_PATH = f'{LM_PROJ_DIR}temp/'
LOG_PATH = f'{LM_PROJ_DIR}log/'

MNT_TEMP_DIR = f'{MNT_DIR}temp/'
TEMP_RES_PATH = f'{LM_PROJ_DIR}temp_results/'
RES_PATH = f'{LM_PROJ_DIR}results/'
DB_PATH = f'{LM_PROJ_DIR}exp_db/'

# ! Data Settings
DATA_PATH = f'{MNT_DIR}data/'
OGB_ROOT = f'{MNT_DIR}data/ogb/'
AMAZON_ROOT = f'{MNT_DIR}data/amazon/'
DBLP_ROOT = f'{MNT_DIR}data/dblp/'

DATA_INFO = {
    'arxiv': {
        'type': 'ogb',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 40,
        'n_nodes': 169343,
        'ogb_name': 'ogbn-arxiv',
        'raw_data_path': OGB_ROOT,  # Place to save raw data
        'max_length': 512,  # Place to save raw data ARXIV_ta 512； arxiv_T 64
        'data_root': f'{OGB_ROOT}ogbn_arxiv/',  # Default ogb download target path
        'raw_text_url': 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz',
    },
    'Children': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 24,
        'n_nodes': 76875,
        'data_name': 'Books-Children',
        'max_length': 256,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Books/Children/',  # Default ogb download target path
    },
    'History': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 13,
        'n_nodes': 41551,
        'data_name': 'Books-History',
        'max_length': 256,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Books/History/',  # Default ogb download target path
    },
    'Computers': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 10,
        'n_nodes': 87229,
        'data_name': 'Electronics-Computers',
        'max_length': 256,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Electronics/Computers/',  # Default ogb download target path
    },
    'Fitness': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 13,
        'n_nodes': 173091,
        'data_name': 'Sports-Fitness',
        'max_length': 64,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Sports/Fitness/',
    },
    'Photo': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 12,
        'n_nodes': 48362,
        'data_name': 'Electronics-Photo',
        'max_length': 512,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Electronics/Photo/',
    },
    'Music': {
        'type': 'amazon',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 0,
        'n_nodes': 4290,
        'data_name': 'Digital-Music',
        'max_length': 40,  # Place to save raw data
        'data_root': f'{AMAZON_ROOT}Digital/Music/',
    },
    'DBLP': {
        'type': 'dblp',
        'train_ratio': 0,  # Default (public) split
        'n_labels': 40,
        'n_nodes': 1106759,
        'data_name': 'Citation-2015',
        'max_length': 256,  # Place to save raw data
        'data_root': f'{DBLP_ROOT}Citation2015/',
    },
}

get_d_info = lambda x: DATA_INFO[x.split('_')[0]]

DATASETS = list(DATA_INFO.keys())
DEFAULT_DATASET =  'History_DT' #'Children_TB'#'arxiv_TA'
DEFAULT_D_INFO = get_d_info(DEFAULT_DATASET)

# Datasets Name
# arxiv_TA/ Children_DT / History_DT/ Fitness_T / Computers_RS / Photo_RS / Music_T/

