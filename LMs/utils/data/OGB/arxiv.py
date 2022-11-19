import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer
import utils.function as uf
from utils.settings import *
from tqdm import tqdm
from utils.function.os_utils import mkdir_p


def _tokenize_ogb_arxiv_datasets(d, labels, chunk_size=50000):
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
            'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}.",
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
    tokenizer = AutoTokenizer.from_pretrained(d.hf_model)

    if d.hf_model in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        print('Adding pad token')
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenizer(text.tolist(), padding='max_length', truncation=True, max_length=512,
                          return_token_type_ids=True).data
    mkdir_p(d._token_folder)
    for k in tokenized:
        with open(osp.join(d._token_folder, f'{k}.npy'), 'wb') as f:
            np.save(f, tokenized[k])
    uf.pickle_save('processed', d._processed_flag['token'])
    return
