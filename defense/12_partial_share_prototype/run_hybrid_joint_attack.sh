#!/bin/bash
# [탐색적, 롤백 가능] 부분 공유 backbone 시제품 + joint attack stress test (핵심 성패 테스트)
# PatchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_hybridjoint
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=defense/12_partial_share_prototype/results/12_hybridjoint_run_%j.txt

cd /home/jooyumm/ViT_robust/PatchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python defense/12_partial_share_prototype/hybrid_joint_attack_test.py --seed 123 --num_samples 50 --chunk 20 --split_layer 5
