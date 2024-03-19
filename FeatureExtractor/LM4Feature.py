import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel, TrainingArguments, PreTrainedModel, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Dataset




def reduce_dimension(features, n_components):
    # features: 节点表征矩阵，大小为 (num_nodes, 4096)
    # n_components: 降维后的维度数

    # 创建PCA模型
    pca = PCA(n_components=n_components)

    # 对节点表征进行降维
    reduced_features = pca.fit_transform(features)

    return reduced_features


def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(
        description='Process text data and save the overall representation as an NPY file.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--text_column', type=str, default='text', help='Name of the column containing text data')
    parser.add_argument('--model_name', type=str, default='prajjwal1/bert-tiny', required=True,
                        help='Name or path of the Huggingface model')
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--name', type=str, default='Movies', help='Prefix name for the  NPY file')
    parser.add_argument('--path', type=str, default='./', help='Path to the NPY File')
    parser.add_argument('--pretrain_path', type=str, default=None, help='Path to the NPY File')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the text for language models')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of batch size for inference')
    parser.add_argument('--fp16', type=bool, default=True, help='if fp16')
    parser.add_argument('--mean', action='store_true', help='whether use mean pooling to represent the whole text')
    parser.add_argument('--nomask', action='store_true', help='whether do not use mask to claculate the mean pooling')
    parser.add_argument('--norm', type=bool, default=False, help='nomic use True')
    parser.add_argument('--access_token', type=str, default=None, help='If you want to use some LLM, please ensure your huggingface token can visit these models hub')


    # 解析命令行参数
    args = parser.parse_args()
    csv_file = args.csv_file
    text_column = args.text_column
    model_name = args.model_name
    name = args.name
    max_length = args.max_length
    batch_size = args.batch_size


    tokenizer_name = args.tokenizer_name

    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(root_dir.rstrip('/'))
    print(base_dir)
    Feature_path = os.path.join(base_dir, args.path)
    cache_path = f"{Feature_path}cache/"
    print(Feature_path)
    print(model_name)

    if not os.path.exists(Feature_path):
        os.makedirs(Feature_path)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    if args.pretrain_path is not None:
        output_file = Feature_path + name + '_' + model_name.split('/')[-1].replace("-", "_") + '_' + str(max_length) + '_' + 'Tuned'
    else:
        output_file = Feature_path + name + '_' + model_name.split('/')[-1].replace("-", "_") + '_' + str(max_length)

    class CLSEmbInfModel(PreTrainedModel):
        def __init__(self, model):
            super().__init__(model.config)
            self.encoder = model

        @torch.no_grad()
        def forward(self, input_ids, attention_mask):
            # Extract outputs from the model3
            outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
            # Use CLS Emb as sentence emb.
            # Use  pooler_output ? Dont use pooler_output
            #node_cls_emb = outputs.pooler_output
            node_cls_emb = outputs.last_hidden_state[:, 0, :]  # Last layer
            return TokenClassifierOutput(logits=node_cls_emb)


    class AttentionMeanEmbInfModel(PreTrainedModel):
        def __init__(self, model, norm=False):
            super().__init__(model.config)
            self.encoder = model
            self.norm = norm

        @torch.no_grad()
        def mean_pooling(self, token_embeddings, attention_mask):
            # Mask out padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            masked_token_embeddings = token_embeddings * input_mask_expanded
            # Calculate mean pooling
            mean_embeddings = masked_token_embeddings.sum(dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return mean_embeddings

        @torch.no_grad()
        def forward(self, input_ids, attention_mask):
            # Extract outputs from the model
            outputs = self.encoder(input_ids, attention_mask) if self.encoder.config._name_or_path in {'nomic-ai/nomic-embed-text-v1'} else self.encoder(input_ids, attention_mask, output_hidden_states=True)
            node_mean_emb = self.mean_pooling(outputs.last_hidden_state, attention_mask)
            node_mean_emb = F.normalize(node_mean_emb, p=2, dim=1) if self.norm is True else node_mean_emb
            return TokenClassifierOutput(logits=node_mean_emb)

    # read the csv file from the local
    df = pd.read_csv(os.path.join(base_dir, csv_file))
    text_data = df[text_column].tolist()

    # 加载模型和分词器
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=args.access_token,  trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=args.access_token,  trust_remote_code=True)

    # 编码文本数据并转为数据集
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_inputs = tokenizer(text_data, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    dataset = Dataset.from_dict(encoded_inputs)

    if args.pretrain_path is not None:
        model = AutoModel.from_pretrained(f'{args.pretrain_path}')
        print('Loading model from the path: {}'.format(args.pretrain_path))
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=args.access_token)


    CLS_Feateres_Extractor = CLSEmbInfModel(model)
    Mask_Mean_Features_Extractor = AttentionMeanEmbInfModel(model, norm=args.norm)
    CLS_Feateres_Extractor.eval()
    Mask_Mean_Features_Extractor.eval()

    inference_args = TrainingArguments(
        output_dir=cache_path,
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        dataloader_num_workers=1,
        fp16_full_eval=args.fp16,
    )

    # CLS representatoin
    if not os.path.exists(output_file + "_cls.npy"):
        trainer = Trainer(model=CLS_Feateres_Extractor, args=inference_args)
        cls_emb = trainer.predict(dataset)
        np.save(output_file + "_cls.npy", cls_emb.predictions)
        print('Existing saved to the {}'.format(output_file))

    else:
        print('Existing saved CLS')

    if args.mean:
        if not os.path.exists(output_file + "_mean.npy"):
            trainer = Trainer(model=Mask_Mean_Features_Extractor, args=inference_args)
            mean_emb = trainer.predict(dataset)

            # 保存平均特征表示为NPY文件
            np.save(output_file + "_mean.npy", mean_emb.predictions)
            print('Existing saved to the {}'.format(output_file))

        else:
            print('Existing saved MEAN')






if __name__ == "__main__":
    main()
