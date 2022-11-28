import pandas as pd
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.utils.url import download_url
import dgl
import copy

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
        'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}",
        'T': lambda x: x['title'],
    }
    # Merge title and abstract
    data['text'] = data.apply(text_func['TA'], axis=1)
    data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:512]), axis=1)
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

def Toplogy_Augment(text, edge0, edge1):
    edge0 = np.array(edge0)
    edge1 = np.array(edge1)
    assert len(edge0) == len(edge1)
    ans = copy.copy(text)
    for i, j in zip(edge0, edge1):
        ans[i] += ' Neighbour: ' + text[j]
    text = ans
    return text
    #169343 个点；107337个边，0-5 +

def main():
    dataset = DglNodePropPredDataset('ogbn-arxiv', root='/mnt/v-haoyan1/CirTraining/data/ogb/')
    g, labels = dataset[0]
    g.remove_self_loop()
    sampler = dgl.dataloading.MultiLayerNeighborSampler([2])
    collator = dgl.dataloading.NodeCollator(g, np.arange(g.num_nodes()), sampler)
    _, _, blocks = collator.collate(np.arange(g.num_nodes()))
    # zipped = zip(blocks[0].edges()[0], blocks[0].edges()[1])
    # sort_zipped = sorted(zipped,key=lambda x:(x[0],x[1]))

    raw_text_path = '/mnt/v-haoyan1/CirTraining/data/ogb/ogbn_arxiv/titleabs.tsv.gz'
    categories, node_ids = read_ids_and_labels('/mnt/v-haoyan1/CirTraining/data/ogb/ogbn_arxiv', labels)
    text = pd.read_table(raw_text_path, header=None, skiprows=[0])
    # ! 统计每个item的文本信息的 min max mean
    text, _ = process_raw_text_df(text, node_ids, categories)
    # Augmentation
    text = Toplogy_Augment(text, blocks[0].edges()[1], blocks[0].edges()[0])
    # 保存text，text_len 文件
    text.to_csv('ogbn-arxiv.txt', sep='\t', header=None, index=False)


if __name__ == '__main__':
    main()
