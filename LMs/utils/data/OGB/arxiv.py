import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer
import utils.function as uf
from utils.settings import *
from tqdm import tqdm
from utils.function.os_utils import mkdir_p


def top_Augmentation(d, nums=1):
    from scipy.sparse import coo_matrix
    from ogb.nodeproppred import DglNodePropPredDataset
    import dgl
    import time
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=d.raw_data_path)
    g, _ = dataset[0]
    g = dgl.to_bidirected(g)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([999])
    collator = dgl.dataloading.NodeCollator(g, np.arange(g.num_nodes()), sampler)
    _, _, blocks = collator.collate(np.arange(g.num_nodes()))
    edge0 = np.array(blocks[0].edges()[1])
    edge1 = np.array(blocks[0].edges()[0])
    assert len(edge0) == len(edge1)

    adj1 = coo_matrix((np.ones(edge0.shape), (edge0, edge1)), shape=(d.n_nodes, d.n_nodes))
    adj2 = adj1 @ adj1
    print('Start adj3')
    a = time.time()
    adj3 = adj1 @ adj2
    print('waste time in adj3:', time.time() - a)
    print('Start adj4')
    # adj4 = adj1@adj3
    print('waste time:', time.time() - a)
    a = time.time()
    neighbours_1 = adj1.tolil().rows
    print('waste time:', time.time() - a)
    neighbours_2 = adj2.tolil().rows
    print('waste time in neighbours2:', time.time() - a)
    neighbours_3 = adj3.tolil().rows
    print('waste time in neighbours3:', time.time() - a)
    # 去重 neighbours_2 中不包括 neighbours_1 的元素
    neighbours_2 = [list(set(neighbours_2[i]) - set(neighbours_1[i])) for i in range(len(neighbours_2))]
    # 去重 neighbours_3 中不包括 neighbours_1 和 neighbours_2 的元素
    neighbours_3 = [list(set(neighbours_3[i]) - set(neighbours_1[i]) - set(neighbours_2[i])) for i in range(len(neighbours_3))]

    return neighbours_1, neighbours_2, neighbours_3


def _tokenize_ogb_arxiv_datasets(d):
    def merge_by_ids(meta_data, node_ids, categories):
        meta_data.columns = ["ID", "Title", "Abstract"]
        # meta_data.drop([0, meta_data.shape[0] - 1], axis=0, inplace=True)  # Drop first and last in Arxiv full dataset processing
        meta_data["ID"] = meta_data["ID"].astype(np.int64)
        meta_data.columns = ["mag_id", "title", "abstract"]
        data = pd.merge(node_ids, meta_data, how="left", on="mag_id")
        data = pd.merge(data, categories, how="left", on="label_id")
        return data

    def read_ids_and_labels(data_root):
        category_path_csv = f"{data_root}/mapping/labelidx2arxivcategeory.csv.gz"
        paper_id_path_csv = f"{data_root}/mapping/nodeidx2paperid.csv.gz"  #
        paper_ids = pd.read_csv(paper_id_path_csv)
        categories = pd.read_csv(category_path_csv)
        categories.columns = ["ID", "category"]  # 指定ID 和 category列写进去
        paper_ids.columns = ["ID", "mag_id"]
        categories.columns = ["label_id", "category"]

        return categories, paper_ids  # 返回类别和论文ID

    def process_raw_text_df(meta_data, node_ids, categories):
        data = merge_by_ids(meta_data.dropna(), node_ids, categories)
        data = data[~data['title'].isnull()]
        text_func = {
            'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}",
            'T': lambda x: x['title'],
        }
        # Merge title and abstract
        data['text'] = data.apply(text_func[d.process_mode], axis=1)
        data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:d.max_length]), axis=1)
        data['len'] = data.apply(lambda x: len(x['text'].split(' ')), axis=1)

        return data['text'], data['len']

    from ogb.utils.url import download_url
    # Get Raw text path
    assert d.ogb_name in ['ogbn-arxiv']
    raw_text_path = download_url(d.raw_text_url, d.data_root)
    if not osp.exists(osp.join(d.data_root, 'ogbn-arxiv.txt')):
        categories, node_ids = read_ids_and_labels(d.data_root)
        text = pd.read_table(raw_text_path, header=None, skiprows=[0])
        text, length = process_raw_text_df(text, node_ids, categories)
        # 保存text，text_len 文件
        text.to_csv(osp.join(d.data_root, 'ogbn-arxiv.txt'), sep='\t', header=None, index=False)
        # 保存文本统计信息
        text_stat = pd.DataFrame(length.describe())
        text_stat.index.rename('Statics', inplace=True)
        text_stat.columns = ["Length"]
        text_stat.to_csv(osp.join(d.data_root, 'ogbn-arxiv_stat.txt'))
    else:
        text = pd.read_csv(osp.join(d.data_root, 'ogbn-arxiv.txt'), sep='\t', header=None)
        text = text[0]

    # Tokenize
    if d.hf_model in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        from transformers import GPT2TokenizerFast
        tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
        tokenizer.padding_side = 'right'
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenized = tokenizer(text.tolist(), padding='max_length', truncation=True, max_length=512,
                              return_token_type_ids=True).data
    else:
        tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
        tokenized = tokenizer(text.tolist(), padding='max_length', truncation=True, max_length=64,
                              return_token_type_ids=True).data
    mkdir_p(d._token_folder)
    for k in tokenized:
        with open(osp.join(d._token_folder, f'{k}.npy'), 'wb') as f:
            np.save(f, tokenized[k])
    uf.pickle_save('processed', d._processed_flag['token'])
    return


def _tokenize_NP_ogb_arxiv_datasets(d, labels, NP=False):
    def NP_make_corpus(d, text, neighbours):
        print('start NP_make_corpus')
        neighbours_1, neighbours_2, neighbours_3 = neighbours
        Document_a = []
        Document_b = []
        label = []
        corpus = text.to_list()
        print('start NP_make_corpus!!!!!!!')
        for i in range(d.n_nodes):
            # this is IsNeighbour
            j = np.random.choice(neighbours_1[i], 4)
            for k in range(4):
                Document_a.append(' '.join(corpus[i].split(' ')[0:256]))
                Document_b.append(corpus[j[k]])
                label.append(0)
            # this is NotNeighbour
            j = np.random.choice(neighbours_3[i], 4)
            for k in range(4):
                Document_a.append(' '.join(corpus[i].split(' ')[0:256]))
                Document_b.append(corpus[j[k]])
                label.append(1)

        return Document_a, Document_b, label

    from ogb.utils.url import download_url
    # Get Raw text path
    assert d.ogb_name in ['ogbn-arxiv']
    raw_text_path = download_url(d.raw_text_url, d.data_root)
    # 先加载原始数据
    if not osp.exists(osp.join(d.data_root, 'ogbn-arxiv.txt')):
        _tokenize_ogb_arxiv_datasets(d, labels)
    text = pd.read_csv(osp.join(d.data_root, 'ogbn-arxiv.txt'), sep='\t', header=None)
    text = text[0]

    # 处理数据
    if not osp.exists(osp.join(d.data_root, 'ogbn-arxiv_TNP.txt')):
        neighbours = top_Augmentation(d)
        Document_a, Document_b, label = NP_make_corpus(d, text, neighbours)
        # 保存数据
        data = pd.DataFrame({'Document_a': Document_a, 'Document_b': Document_b, 'label': label})
        data.to_csv(osp.join(d.data_root, 'ogbn-arxiv_TNP.txt'), sep='\t', header=None, index=False)
    else:
        data = pd.read_csv(osp.join(d.data_root, 'ogbn-arxiv_TNP.txt'), sep='\t', header=None)
        data.columns = ['Document_a', 'Document_b', 'label']
        label = data['label'].tolist()
        Document_a = data['Document_a'].tolist()
        Document_b = data['Document_b'].tolist()
    print('start tokenizer!!!')
    # Tokenize
    if d.hf_model in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        from transformers import GPT2TokenizerFast
        tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
        tokenizer.padding_side = 'right'
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenized = tokenizer(text.tolist(), padding='max_length', truncation=True, max_length=512,
                              return_token_type_ids=True).data
    else:
        tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
        tokenized = tokenizer(Document_a, Document_b, padding='max_length', truncation=True, max_length=512,
                              return_token_type_ids=True).data
    label = np.array(label).T
    label = th.sparse.torch.eye(2).index_select(0, th.tensor(label))
    tokenized['labels'] = np.array(label)
    mkdir_p(d._NP_token_folder)
    for k in tokenized:
        with open(osp.join(d._NP_token_folder, f'{k}.npy'), 'wb') as f:
            np.save(f, tokenized[k])
    uf.pickle_save('processed', d._processed_flag['TNP_token'])
    return

# def _tokenize_TRP_ogb_arxiv_datasets
def _tokenize_TRP_ogb_arxiv_datasets(d, labels):
    def TRP_make_corpus(d, text, neighbours):
        neighbours_1, neighbours_2, neighbours_3 = neighbours
        import random
        Document_a = []
        Document_b = []
        Document_c = []
        Document_d = []
        label = []
        corpus = text.to_list()
        print('start TRP_make_corpus!')
        for i in range(d.n_nodes):
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() < 1 / 6:
                # this is A->B->C->D
                Document_a.append(' '.join(corpus[i].split(' ')[0:128]))
                j = np.random.choice(neighbours_1[i], 1)
                Document_b.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_2[i], 1)
                while j in neighbours_1[i]:
                    j = np.random.choice(neighbours_2[i], 1)
                Document_c.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_3[i], 1)
                while j in neighbours_1[i] or j in neighbours_2[i]:
                    j = np.random.choice(neighbours_3[i], 1)
                Document_d.append(corpus[j[0]])
                label.append(0)
            elif random.random() < 2 / 6:
                # this is A->B->D->C
                Document_a.append(' '.join(corpus[i].split(' ')[0:128]))
                j = np.random.choice(neighbours_1[i], 1)
                Document_b.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_2[i], 1)
                while j in neighbours_1[i]:
                    j = np.random.choice(neighbours_2[i], 1)
                Document_d.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_3[i], 1)
                while j in neighbours_1[i] or j in neighbours_2[i]:
                    j = np.random.choice(neighbours_3[i], 1)
                Document_c.append(corpus[j[0]])
                label.append(1)
            elif random.random() < 3 / 6:
                # this is A->C->B->D
                Document_a.append(' '.join(corpus[i].split(' ')[0:128]))
                j = np.random.choice(neighbours_1[i], 1)
                Document_c.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_2[i], 1)
                while j in neighbours_1[i]:
                    j = np.random.choice(neighbours_2[i], 1)
                Document_b.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_3[i], 1)
                while j in neighbours_1[i] or j in neighbours_2[i]:
                    j = np.random.choice(neighbours_3[i], 1)
                Document_d.append(corpus[j[0]])
                label.append(2)
            elif random.random() < 4 / 6:
                # this is A->C->D->B
                Document_a.append(' '.join(corpus[i].split(' ')[0:128]))
                j = np.random.choice(neighbours_1[i], 1)
                Document_c.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_2[i], 1)
                while j in neighbours_1[i]:
                    j = np.random.choice(neighbours_2[i], 1)
                Document_d.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_3[i], 1)
                while j in neighbours_1[i] or j in neighbours_2[i]:
                    j = np.random.choice(neighbours_3[i], 1)
                Document_b.append(corpus[j[0]])
                label.append(3)
            elif random.random() < 5 / 6:
                # this is A->D->B->C
                Document_a.append(' '.join(corpus[i].split(' ')[0:128]))
                j = np.random.choice(neighbours_1[i], 1)
                Document_d.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_2[i], 1)
                while j in neighbours_1[i]:
                    j = np.random.choice(neighbours_2[i], 1)
                Document_b.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_3[i], 1)
                while j in neighbours_1[i] or j in neighbours_2[i]:
                    j = np.random.choice(neighbours_3[i], 1)
                Document_c.append(corpus[j[0]])
                label.append(4)
            else:

                # this is A->D->C->B
                Document_a.append(' '.join(corpus[i].split(' ')[0:128]))
                j = np.random.choice(neighbours_1[i], 1)
                Document_d.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_2[i], 1)
                while j in neighbours_1[i]:
                    j = np.random.choice(neighbours_2[i], 1)
                Document_c.append(' '.join(corpus[j[0]].split(' ')[0:128]))
                j = np.random.choice(neighbours_3[i], 1)
                while j in neighbours_1[i] or j in neighbours_2[i]:
                    j = np.random.choice(neighbours_3[i], 1)
                Document_b.append(corpus[j[0]])
                label.append(5)

        return Document_a, Document_b, Document_c, Document_d, label

    from ogb.utils.url import download_url
    # Get Raw text path
    assert d.ogb_name in ['ogbn-arxiv']
    raw_text_path = download_url(d.raw_text_url, d.data_root)
    # 先加载原始数据
    if not osp.exists(osp.join(d.data_root, 'ogbn-arxiv.txt')):
        _tokenize_ogb_arxiv_datasets(d, labels)
    text = pd.read_csv(osp.join(d.data_root, 'ogbn-arxiv.txt'), sep='\t', header=None)
    text = text[0]
    # 处理数据
    if not osp.exists(osp.join(d.data_root, 'ogbn-arxiv_TTRP.txt')):
        neighbours = top_Augmentation(d)
        Document_a, Document_b, Document_c, Document_d, label = TRP_make_corpus(d, text, neighbours)
        # 保存数据
        data = pd.DataFrame(
            {'Document_a': Document_a, 'Document_b': Document_b, 'Document_c': Document_c, 'Document_d': Document_d, 'label': label})
        data.to_csv(osp.join(d.data_root, 'ogbn-arxiv_TTRP.txt'), sep='\t', header=None, index=False)
    else:
        data = pd.read_csv(osp.join(d.data_root, 'ogbn-arxiv_TTRP.txt'), sep='\t', header=None)
        data.columns = ['Document_a', 'Document_b', 'Document_c', 'Document_d', 'label']
        label = data['label'].tolist()
        Document_a = data['Document_a'].tolist()
        Document_b = data['Document_b'].tolist()
        Document_c = data['Document_c'].tolist()
        Document_d = data['Document_d'].tolist()
    print('strart tokenizer of the TRP Task datasets')

    tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
    tokenized = tokenizer(Document_a, Document_b, Document_c, Document_d, padding='max_length', truncation=True,
                          max_length=512,
                          return_token_type_ids=True).data
    label = np.array(label).T
    label = th.sparse.torch.eye(6).index_select(0, th.tensor(label))
    tokenized['labels'] = np.array(label)
    mkdir_p(d._TRP_token_folder)
    for k in tokenized:
        with open(osp.join(d._TRP_token_folder, f'{k}.npy'), 'wb') as f:
            np.save(f, tokenized[k])
    uf.pickle_save('processed', d._processed_flag['TRP_token'])
    return