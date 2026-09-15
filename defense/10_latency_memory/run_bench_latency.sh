#!/bin/bash
#SBATCH --job-name=vitguard_bench
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=defense/10_latency_memory/results/10_bench_run_%j.txt
cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python defense/10_latency_memory/bench_latency_memory.py --n_warmup 20 --n_iters 100
