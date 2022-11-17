import pandas as pd
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.utils.url import download_url


def read_data(filename):
    df = pd.read_csv(filename)
    return df


def chaoshen():
    # load dgl dataset
    dataset = DglNodePropPredDataset('ogbn-arxiv', root='data/ogb/')
    g, labels = dataset[0]
    split_idx = dataset.get_idx_split()
    labels = labels.squeeze().numpy()
    labels.unique().shape[0]
    # make a label.txt
    f = open('label.txt', 'w')
    split = [None] * len(labels)
    for s, ids in split_idx.items():
        for i in ids:
            split[i] = s

    df = pd.DataFrame({'split': np.array(split), 'label': labels})
    df.to_csv('a.txt', sep='\t', header=False)


def read_ids_and_labels(data_root, labels):
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
    text_func = {
        'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}.",
        'T': lambda x: x['title'],
    }
    # Merge title and abstract
    data['text'] = data.apply(text_func['TA'], axis=1)
    data['len'] = data.apply(lambda x: len(x['text'].split(' ')), axis=1)
    # data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:512]), axis=1) # 截断
    return data['text'], data['len']


def merge_by_ids(meta_data, node_ids, categories):
    meta_data.columns = ["ID", "Title", "Abstract"]
    # meta_data.drop([0, meta_data.shape[0] - 1], axis=0, inplace=True)  # Drop first and last in Arxiv full dataset processing
    meta_data["ID"] = meta_data["ID"].astype(np.int64)
    meta_data.columns = ["mag_id", "title", "abstract"]
    data = pd.merge(node_ids, meta_data, how="left", on="mag_id")
    data = pd.merge(data, categories, how="left", on="label_id")
    return data


def main():
    # ! 统计每个item的文本信息的 min max mean
    text = args.text_root
    text_len = text.apply(lambda x: len(text.split(' ')), axis=1)
    # 保存text，text_len 文件
    text_len.to_csv('ogbn-arxiv_len.txt', sep='\t', header=None, index=False)
    # 保存文本统计信息
    text_stat = pd.DataFrame(text_len.describe())
    text_stat.index.rename('Statics', inplace=True)
    text_stat.columns = ["Length"]
    text_stat.to_csv('ogbn-arxiv_stat.txt')


if __name__ == '__main__':
    main()