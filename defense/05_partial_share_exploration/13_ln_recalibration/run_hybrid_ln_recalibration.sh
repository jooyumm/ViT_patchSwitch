#!/bin/bash
# [탐색적, 아주 저렴, 롤백 가능] 부분 공유 시제품 branch8 0% 붕괴가 LN 통계 재보정으로 회복되는지
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_lnrecal
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=results/05_partial_share_exploration/13_ln_recalibration/13_lnrecal_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python defense/05_partial_share_exploration/13_ln_recalibration/hybrid_ln_recalibration.py --calib_samples 100 --calib_seed 555 --eval_samples 50 --eval_seed 123 --split_layer 5
