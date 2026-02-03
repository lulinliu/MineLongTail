#!/bin/bash
#SBATCH -J qwen_filterFeb
#SBATCH -A CGAI24022
#SBATCH -p gh-dev
#SBATCH -N 2                 # 节点数；按需改
#SBATCH --ntasks-per-node=1   # 每节点只启动一个脚本，torchrun 再在节点内拉起多进程
#SBATCH --gpus-per-node=1     # 每节点 GPU 数；与 qwen_filter.sh 里的 GPUS_PER_NODE 保持一致
#SBATCH -t 1:00:00
#SBATCH -o /scratch/10102/hh29499/MineLongTail/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/MineLongTail/%x_%j.err

srun --export=ALL \
  bash /scratch/10102/hh29499/MineLongTail/qwen_filter.sh
