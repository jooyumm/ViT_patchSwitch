#!/bin/bash
# [탐색적, 롤백 가능] "의심되면 P16->P8 통째 재분류" 방어의 복원율/부작용률 검증
# PatchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_p8rescue
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=archive/06_p8_rescue_test/results/06_p8rescue_run_%j.txt

cd /home/jooyumm/ViT_robust/PatchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python archive/06_p8_rescue_test/vitguard_p8_rescue_test.py --seed 42 --num_samples 30
