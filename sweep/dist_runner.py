import os
import sys


def find_free_port():
    from contextlib import closing
    import socket
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


cmd = ' '.join(sys.argv[1:])

N_GPUS = len((gpus := os.environ['CUDA_VISIBLE_DEVICES']).split(','))
if cmd.split()[0] in {'train_MLM.py','train_CLM.py', 'LMs/train_MLM.py'}:
    cmd = f"CUDA_VISIBLE_DEVICES={gpus} torchrun --master_port={find_free_port()} --nproc_per_node={N_GPUS} {cmd}"
else:
    cmd = f"CUDA_VISIBLE_DEVICES={gpus} torchrun --master_port={find_free_port()} --nproc_per_node={N_GPUS} {cmd} --gpus={gpus}"
print(f"Sweep training command to run: {cmd}")
os.system(cmd)
