import os
import sys

cmd = ' '.join(sys.argv[1:])
N_GPUS = len((gpus := os.environ['CUDA_VISIBLE_DEVICES']).split(','))
cmd = f"python {cmd} --gpus={gpus}"
print(f"Sweep training command to run: {cmd}")
os.system(cmd)
