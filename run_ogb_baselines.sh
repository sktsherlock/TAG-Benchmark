
dataset=$1
gnn_model=$2
graph_task=$3
lm_type=$4
lm_model=$5
RUNS=5

if [ ${dataset} == "ogbn-arxiv" ]; then
    if [ ${gnn_model} == "mlp" ]; then
        python -u OGB/${graph_task}/${dataset}/mlp.py \
            --runs ${RUNS} \
            --data_root_dir ../dataset/OGB/ \
            --node_emb_path ../Features/OGB/${lm_type}/${lm_model}/X_emb.npy \
    elif [ ${gnn_model} == "graph-sage" ]; then
        python -u OGB/${dataset}/gnn.py \
    else
        echo "gnn_model=${gnn_model} is not yet supported for ogbn-arxiv!"
    fi
elif [ ${dataset} == "ogbn-products" ]; then
    if [ ${gnn_model} == "mlp" ]; then
        python -u OGB/${dataset}/mlp.py \
            --runs ${RUNS} \
            --data_root_dir ./dataset \
            --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.npy \
    elif [ ${gnn_model} == "graph-saint" ]; then
        CUDA_VISIBLE_DEVICES=1 python -u OGB/${dataset}/graph_saint.py \
            --runs ${RUNS} \
            --data_root_dir ./dataset \
            --eval_steps 10 \
            --epochs 50 \
            --num_layers 1 \
            --walk_length 1 \
            --hidden_channels 192 \
            --lr 1e-3 \
            --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.v2.npy \
    else
        echo "gnn_model=${gnn_model} is not supported for ogbn-arxiv!"
    fi
else
    echo "dataset=${dataset} is not yet supported!"
fi
