import torch.nn.functional as F

from utils.function import *
from utils.function.dgl_utils import *
from utils.settings import *
from utils.data.preprocess import tokenize_graph, load_graph_info
import numpy as np


class Sequence():
    def __init__(self, cf):
        # Process split settings, e.g. -1/2 means first split
        self.cf = cf
        self.hf_model = cf.lm_md.hf_model
        self.device = None
        self.lm_emb_dim = self.cf.lm_md.hidden_dim if not self.cf.feat_shrink else int(self.cf.feat_shrink)
        self.name, process_mode = (_ := cf.dataset.split('_'))[0], _[1]  # e.g. name of "arxiv_TA" is "arxiv"
        self.process_mode = process_mode
        self.md = info_dict = DATA_INFO[self.name]

        self.n_labels = info_dict['n_labels']
        self.__dict__.update(info_dict)
        self.subset_ratio = 1 if len(_) == 2 else float(_[2])
        self.cut_off = info_dict['cut_off'] if 'cut_off' in info_dict else 512

        self.label_keys = ['labels', 'is_gold']
        self.token_keys = ['attention_mask', 'input_ids', 'token_type_ids']
        self.ndata = {}

        # * GVF-related files information
        self._g_info_folder = init_path(f"{DATA_PATH}{cf.dataset}/")
        self._g_info_file = f"{self._g_info_folder}graph.info "
        self._token_folder = init_path(f"{DATA_PATH}{self.name}{process_mode}_{self.hf_model}/")
        self._processed_flag = {
            'g_info': f'{self._g_info_folder}processed.flag',
            'token': f'{self._token_folder}processed.flag',
        }
        self.g, self.split = None, None

        self.info = {
            'input_ids': SN(shape=(self.md['n_nodes'], self.md['max_length']), type=np.uint16),
            'attention_mask': SN(shape=(self.md['n_nodes'], self.md['max_length']), type=bool),
            'token_type_ids': SN(shape=(self.md['n_nodes'], self.md['max_length']), type=bool)
        }
        for k, info in self.info.items():
            info.path = f'{self._token_folder}{k}.npy'
        return

    def init(self):
        # ! Load sequence graph info which is shared by GNN and LMs
        cf = self.cf
        self.gi = g_info = load_graph_info(cf)
        self.__dict__.update(g_info.splits)
        self.n_nodes = g_info.n_nodes
        self.ndata.update({_: getattr(g_info, _) for _ in self.label_keys})
        self.labeled_nodes = np.arange(self.n_nodes)[g_info.is_gold]
        # ! LM phase
        tokenize_graph(self.cf)
        self._load_data_fields(self.token_keys)
        self.device = cf.device  # if cf.local_rank<0 else th.device(cf.local_rank)

        return self

    def _load_data_fields(self, k_list):
        for k in k_list:
            i = self.info[k]
            try:
                self.ndata[k] = np.memmap(i.path, mode='r', dtype=i.type, shape=i.shape)
            except:
                raise ValueError(f'Shape not match {i.shape}')

    def save_g_info(self, g_info):
        pickle_save(g_info, self._g_info_file)
        pickle_save('processed', self._processed_flag['g_info'])
        return

    def is_processed(self, field):
        return os.path.exists(self._processed_flag[field])

    def _from_numpy(self, x, on_cpu=False):
        return th.from_numpy(np.array(x)) if on_cpu else th.from_numpy(np.array(x)).to(self.device)


    def is_gold(self, nodes, on_cpu=False):
        return self._from_numpy(self.ndata['is_gold'][nodes], on_cpu)

    def y_gold(self, nodes, on_cpu=False):
        labels = self._from_numpy(self.ndata['labels'][nodes], on_cpu).to(th.int64)
        return F.one_hot(labels, num_classes=self.n_labels).type(th.FloatTensor) if on_cpu \
            else F.one_hot(labels, num_classes=self.n_labels).type(th.FloatTensor).to(self.device)

    def __getitem__(self, k):
        return self.ndata[k]

    def get_tokens(self, node_id):
        # node_id = self.gi.IDs[node_id] if hasattr(self.gi, 'IDs') else node_id
        _load = lambda k: th.IntTensor(np.array(self.ndata[k][node_id]))
        item = {k: _load(k) for k in self.token_keys if k != 'input_ids'}
        item['input_ids'] = th.IntTensor(np.array(self['input_ids'][node_id]).astype(np.int32))
        return item


class SeqGraphDataset(th.utils.data.Dataset):  # Map style
    def __init__(self, data: Sequence, mode):
        super().__init__()
        self.d, self.mode = data, mode

    def __getitem__(self, node_id):
        item = self.d.get_tokens(node_id)
        item['is_gold'] = self.d.is_gold(node_id)
        item['labels'] = self.d.y_gold(node_id)
        return item

    def __len__(self):
        return self.d.n_nodes
