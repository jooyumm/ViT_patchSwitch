#!/bin/bash
# [탐색적, 롤백 가능] calibration/evaluation 분리 + 확대 표본으로 탐지/위치특정/복원율/오탐률 통합 검증
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_finalval
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=results/full_reclassification/final_validation/06_finalval_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python defense/full_reclassification/final_validation/vitguard_final_validation.py --seed 42 --num_samples 200
