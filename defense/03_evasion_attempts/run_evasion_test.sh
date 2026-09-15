#!/bin/bash
# [탐색적, 롤백 가능] PatchFool을 저saliency 위치로 강제했을 때 탐지기 회피 여부 확인
# PatchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_evasion
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=defense/03_evasion_attempts/results/03_evasion_run_%j.txt

cd /home/jooyumm/ViT_robust/PatchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python defense/03_evasion_attempts/vitguard_evasion_test.py --patch_size 16 --num_samples 30 --seed 42
