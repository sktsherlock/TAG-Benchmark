import torch.nn.functional as F

from utils.function import *
from utils.function.dgl_utils import *
from utils.settings import *
from utils.data.preprocess import tokenize_graph, load_TAG_info
import numpy as np

from scipy.sparse import coo_matrix
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import time


class Sequence():
    def __init__(self, cf):
        # Process split settings, e.g. -1/2 means first split
        self.cf = cf
        self.hf_model = cf.lm_md.hf_model
        self.father_model = cf.lm_md.father_model
        self.device = None
        self.lm_emb_dim = self.cf.lm_md.hidden_dim if not self.cf.feat_shrink else int(self.cf.feat_shrink)
        self.name, process_mode = (_ := cf.dataset.split('_'))[0], _[1]  # e.g. name of "arxiv_TA" is "arxiv"
        self.process_mode = process_mode
        self.md = info_dict = DATA_INFO[self.name]

        self.n_labels = info_dict['n_labels']
        self.__dict__.update(info_dict)
        self.subset_ratio = 1 if len(_) == 2 else float(_[2])
        self.cut_off = info_dict['cut_off'] if 'cut_off' in info_dict else 512
        self.ndata = {}

        # * TAG-related files information
        self._g_info_folder = init_path(f"{DATA_PATH}{cf.dataset}/")
        self._g_info_file = f"{self._g_info_folder}graph.info "
        self._token_folder = init_path(f"{DATA_PATH}{cf.dataset}/{self.father_model}/{cf.model}/")
        self._NP_token_folder = init_path(f"{DATA_PATH}{cf.dataset}/TNP/{self.father_model}/{cf.model}/")
        self._TRP_token_folder = init_path(f"{DATA_PATH}{cf.dataset}/TRP/{self.father_model}/{cf.model}/")
        self._processed_flag = {
            'g_info': f'{self._g_info_folder}processed.flag',
            'token': f'{self._token_folder}processed.flag',
            'TNP_token': f'{self._NP_token_folder}processed.flag',
            'TRP_token': f'{self._TRP_token_folder}processed.flag',
        }
        self.g, self.split = None, None

        self.info = {
            'input_ids': SN(shape=(self.md['n_nodes'], self.md['max_length']), type=np.uint16),
            'attention_mask': SN(shape=(self.md['n_nodes'], self.md['max_length']), type=bool),
            'token_type_ids': SN(shape=(self.md['n_nodes'], self.md['max_length']), type=bool)
        }
        for k, info in self.info.items():
            info.path = f'{self._token_folder}/{k}.npy'
        return

    def init(self):
        # ! Load sequence graph info which is shared by GNN and LMs
        cf = self.cf
        self.gi = g_info = load_TAG_info(cf)
        self.__dict__.update(g_info.splits)
        self.n_nodes = g_info.n_nodes
        self.ndata.update({_: getattr(g_info, _) for _ in ['labels']})
        # ! LM phase
        tokenize_graph(self.cf)
        self._load_data_fields()
        self.device = cf.device  # if cf.local_rank<0 else th.device(cf.local_rank)
        self.neighbours = self.get_neighbours(self.n_nodes)

        return self

    def NP_init(self):
        from utils.data.preprocess import tokenize_NP_graph
        cf = self.cf
        self.gi = g_info = load_TAG_info(cf)
        # ! LM phase
        tokenize_NP_graph(self.cf)
        self._load_NP_data_fields()
        self.device = cf.device  # if cf.local_rank<0 else th.device(cf.local_rank)
        # 划分训练集 验证集
        from sklearn.model_selection import train_test_split
        train_x, valid_x, _, _ = train_test_split(np.arange(1354744), self.ndata['labels'], test_size=0.1)
        dic = {'train_x': train_x, 'valid_x': valid_x}
        self.__dict__.update(dic)

        return self

    def _load_NP_data_fields(self):
        for k, info in self.info.items():
            info.NP_path = f'{self._NP_token_folder}/{k}.npy'
            try:
                self.ndata[k] = np.load(info.NP_path)
            except:
                raise ValueError(f'There is no file')
        self.ndata['labels'] = np.load(f'{self._NP_token_folder}/labels.npy')

    # TRP
    def TRP_init(self):
        from utils.data.preprocess import tokenize_TRP_graph
        cf = self.cf
        self.gi = g_info = load_TAG_info(cf)
        # ! LM phase
        tokenize_TRP_graph(self.cf)
        self._load_TRP_data_fields()
        self.device = cf.device
        # 划分训练集 验证集
        from sklearn.model_selection import train_test_split
        train_x, valid_x, _, _ = train_test_split(np.arange(self.n_nodes), self.ndata['labels'], test_size=0.05)
        dic = {'train_x': train_x, 'valid_x': valid_x}
        self.__dict__.update(dic)

        return self

    def _load_TRP_data_fields(self):
        for k, info in self.info.items():
            info.TRP_path = f'{self._TRP_token_folder}/{k}.npy'
            try:
                self.ndata[k] = np.load(info.TRP_path)
            except:
                raise ValueError(f'There is no file')
        self.ndata['labels'] = np.load(f'{self._TRP_token_folder}/labels.npy')

    def _load_data_fields(self):
        for k in self.info:
            i = self.info[k]
            try:
                self.ndata[k] = np.load(i.path)  # np.memmap(i.path, mode='r', dtype=i.type, shape=i.shape)
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

    def _th_float(self, x, on_cpu=False):
        return self._from_numpy(x, on_cpu).to(th.float32)

    def y_gold(self, nodes, on_cpu=False):
        labels = self._from_numpy(self.ndata['labels'][nodes], on_cpu).to(th.int64)
        return F.one_hot(labels, num_classes=self.n_labels).type(th.FloatTensor) if on_cpu \
            else F.one_hot(labels, num_classes=self.n_labels).type(th.FloatTensor).to(self.device)

    def __getitem__(self, k):
        return self.ndata[k]

    def get_neighbours(self, n_nodes):

        dataset = DglNodePropPredDataset('ogbn-arxiv', root=self.raw_data_path)
        g, _ = dataset[0]
        g = dgl.to_bidirected(g)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([999])
        collator = dgl.dataloading.NodeCollator(g, np.arange(g.num_nodes()), sampler)
        _, _, blocks = collator.collate(np.arange(g.num_nodes()))
        edge0 = np.array(blocks[0].edges()[1])
        edge1 = np.array(blocks[0].edges()[0])
        assert len(edge0) == len(edge1)

        adj1 = coo_matrix((np.ones(edge0.shape), (edge0, edge1)), shape=(n_nodes, n_nodes))
        neighbours_1 = list(adj1.tolil().rows)
        return neighbours_1

    def get_tokens(self, node_id):
        _load = lambda k: th.IntTensor(np.array(self.ndata[k][node_id]))
        item = {}
        item['attention_mask'] = _load('attention_mask')
        item['input_ids'] = th.IntTensor(np.array(self['input_ids'][node_id]).astype(np.int32))
        if self.hf_model not in ['distilbert-base-uncased', 'roberta-base']:
            item['token_type_ids'] = _load('token_type_ids')
        return item

    def get_NP_tokens(self, node_id):
        _load = lambda k: th.IntTensor(np.array(self.ndata[k][node_id]))
        item = {}
        item['attention_mask'] = _load('attention_mask')
        item['input_ids'] = th.IntTensor(np.array(self['input_ids'][node_id]).astype(np.int32))
        item['labels'] = self._from_numpy(self.ndata['labels'][node_id]).type(th.FloatTensor)
        if self.hf_model not in ['distilbert-base-uncased', 'roberta-base']:
            item['token_type_ids'] = _load('token_type_ids')
        return item


class SeqGraphDataset(th.utils.data.Dataset):  # Map style
    def __init__(self, data: Sequence):
        super().__init__()
        self.d = data

    def __getitem__(self, node_id):
        item = self.d.get_tokens(node_id)
        item['labels'] = self.d.y_gold(node_id)
        # neighbours = self.d.neighbours[node_id]
        # item['neighbours'] = self.d.get_tokens(neighbours[0])
        return item

    def __len__(self):
        return self.d.n_nodes

class SeqCLDataset(th.utils.data.Dataset):
    def __init__(self, data: Sequence):
        super().__init__()
        self.d = data

    def __getitem__(self, node_id):
        item = self.d.get_tokens(node_id)
        neighbours = self.d.neighbours[node_id]
        k = np.random.choice(neighbours, 1)
        item['neighbours'] = self.d.get_tokens(k[0])
        return item

    def __len__(self):
        return self.d.n_nodes


class NP_Dataset(th.utils.data.Dataset):  # Map style
    def __init__(self, data: Sequence):
        super().__init__()
        self.d = data

    def __getitem__(self, node_id):
        item = self.d.get_NP_tokens(node_id)
        return item

    def __len__(self):
        return self.d.n_nodes