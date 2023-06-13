# CS-TAG 
Text-attributed Graph Datasets and Benchmark
## Datasets
If you want to process the data from scratch, you can see the data_process.py to do it.
If you just want to download the datasets we process, you can open the https://drive.google.com/drive/folders/1rArc3kIDaJlIhOLj_pTaWxACbM6H_iE1?usp=sharing and then download from it.

For example, you can use such code to download the file:
gdown -c https://drive.google.com/uc?id= 
## Baseline
Representation learning on the TAGs often depend on the two type models: Graph Neural Networks and Language Models.
For the latter, we often use the Pretrained Language Models(PLMs) to encode the text.

For the GNNs, we follow the DGL toolkit and implement them in the GNN library.
For the PLMs, we follow the huggingface trainer to implement the PLMs in a same pipeline.
We know that there are no absolute fair between the two type baselines.

