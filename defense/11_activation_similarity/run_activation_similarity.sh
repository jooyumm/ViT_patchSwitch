#!/bin/bash
# [탐색적, 롤백 가능] P16/P8 체크포인트 중간 레이어(6,9) activation 유사도(CKA/cosine) 확인
# PatchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_actsim
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=defense/11_activation_similarity/results/11_actsim_run_%j.txt

cd /home/jooyumm/ViT_robust/PatchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python defense/11_activation_similarity/activation_similarity.py --num_samples 200 --seed 42 --layers 6 9
