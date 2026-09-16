#!/bin/bash
# [탐색적, 롤백 가능] §5/§12 "P8/P16 임베딩 공간 불일치"를 confound 제거하고 재검증
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_incompat_rigor
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=results/05_partial_share_exploration/15_incompatibility_rigor/15_incompatibility_rigor_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python defense/05_partial_share_exploration/15_incompatibility_rigor/rigorous_incompatibility_test.py \
  --num_calib 100 --num_eval 50 --split_layer 5 --seed 42 --chunk 20
