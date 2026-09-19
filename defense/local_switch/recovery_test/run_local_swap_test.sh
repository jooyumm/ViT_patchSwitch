#!/bin/bash
# [탐색적, 롤백 가능] §16. 국소 토큰 세분화(Local Token Subdivision) 1단계 실현성 테스트
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_local_swap
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/local_switch/recovery_test/16_local_swap_l12_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python defense/local_switch/recovery_test/local_swap_test.py \
  --num_calib 100 --num_eval 150 --seed 42 --attn_layer_idx 4 --chunk 20
