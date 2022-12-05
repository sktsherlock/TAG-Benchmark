import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer
import utils.function as uf
from utils.settings import *
from tqdm import tqdm
from utils.function.os_utils import mkdir_p


def _tokenize_ogb_arxiv_datasets(d, labels):
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
        paper_ids["label_id"] = labels
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
        data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:d.cut_off]), axis=1)
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
        tokenized = tokenizer(text.tolist(), padding='max_length', truncation=True, max_length=512,
                              return_token_type_ids=True).data
    mkdir_p(d._token_folder)
    for k in tokenized:
        with open(osp.join(d._token_folder, f'{k}.npy'), 'wb') as f:
            np.save(f, tokenized[k])
    uf.pickle_save('processed', d._processed_flag['token'])
    return


def _tokenize_NP_ogb_arxiv_datasets(d, labels, NP=False):
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
        paper_ids["label_id"] = labels
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
        data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:d.cut_off]), axis=1)
        data['len'] = data.apply(lambda x: len(x['text'].split(' ')), axis=1)

        return data['text'], data['len']

    def top_Augmentation(d):
        from scipy.sparse import coo_matrix
        from ogb.nodeproppred import DglNodePropPredDataset
        import dgl
        import os
        dataset = DglNodePropPredDataset('ogbn-arxiv', root=d.raw_data_path)
        g, _ = dataset[0]
        g = dgl.to_bidirected(g)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([999])
        collator = dgl.dataloading.NodeCollator(g, np.arange(g.num_nodes()), sampler)
        _, _, blocks = collator.collate(np.arange(g.num_nodes()))
        edge0 = np.array(blocks[0].edges()[1])
        edge1 = np.array(blocks[0].edges()[0])

        assert len(edge0) == len(edge1)
        adj0 = coo_matrix((np.ones(edge0.shape), (edge0, edge1)), shape=(169343, 169343))
        adj1 = adj0@adj0
        print('Start adj2')
        adj2 = adj0@adj1
        print('Start adj3')
        adj3 = adj0@adj2
        neighbours_0 = [row for row in adj0.tolil().rows]
        neighbours_1 = [row for row in adj1.tolil().rows]
        neighbours_2 = [row for row in adj2.tolil().rows]
        neighbours_3 = [row for row in adj3.tolil().rows]

        return neighbours_0, neighbours_1, neighbours_2, neighbours_3

    def NP_make_corpus(d, text, neighbours):
        n1, n2, n3, n4 = neighbours
        import random
        Document_a = []
        Document_b = []
        label = []
        corpus = text.to_list()
        print('start NP_make_corpus!!!!!!!')
        for i in range(d.n_nodes):
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNeighbour
                Document_a.append(' '.join(corpus[i].split(' ')[0:256]))
                j = np.random.choice(n1[i], 1)
                Document_b.append(corpus[j[0]])
                label.append(0)
            else:
                # this is NotNeighbour
                Document_a.append(' '.join(corpus[i].split(' ')[0:256]))
                j = np.random.choice(n4[i], 1)
                while j in n3[i]:
                    j = np.random.choice(n4[i], 1)
                Document_b.append(corpus[j[0]])
                label.append(1)

        return Document_a, Document_b, label

    from ogb.utils.url import download_url
    # Get Raw text path
    assert d.ogb_name in ['ogbn-arxiv']
    raw_text_path = download_url(d.raw_text_url, d.data_root)
    # 先加载原始数据
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

    # 处理数据
    if not osp.exists(osp.join(d.data_root, 'ogbn-arxiv_TNP.txt')):
        neighbours = top_Augmentation(d)
        Document_a, Document_b, label = NP_make_corpus(d, text, neighbours)
        # 保存数据
        data = pd.DataFrame({'Document_a': Document_a, 'Document_b': Document_b, 'label': label})
        data.to_csv(osp.join(d.data_root, 'ogbn-arxiv_NP.txt'), sep='\t', header=None, index=False)
    else:
        data = pd.read_csv(osp.join(d.data_root, 'ogbn-arxiv_NP.txt'), sep='\t', header=None)
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
    uf.pickle_save('processed', d._processed_flag['NP_token'])
    return
