"""
probes/vitguard_p8_rescue_test.py — [탐색적 검증, 롤백 가능] "의심되면 P16→P8 전체 재분류"
방어의 핵심 검증.

배경
----
"의심 영역만 P8로 국소 재분할"은 P8/P16이 완전히 독립적으로 학습된 별개 체크포인트라서
(임베딩 공간이 안 섞임) 지금은 불가능함을 확인했다. 대신 같은 핵심 아이디어("P16이 못 미더우면
상대적으로 강건한 P8을 대신 쓴다")를 "의심되면 이미지 전체를 P8로 통째로 재분류"로 단순화해서
검증한다.

방법
----
1) P16에 표준 PatchFool 공격(attn_layer_idx=4, patch_select='Attn') 적용
2) 탐지 임계값: probes/results/layer_sweep_P16_raw.npz에 저장된 L=12 raw top-4 mass 값
   (clean 30개, PatchFool 30개, 이전 vitguard_layer_sweep.py 실행 결과)으로 Youden's J를
   최대화하는 임계값을 구함.
   *** 주의: 같은 30개 표본으로 임계값을 정하고 이번 평가도 하는 거라 약간 낙관적으로 편향될
   수 있다 — 제대로 하려면 표본을 나누거나(train/holdout) 확대해야 하며, 이건 다음 단계
   (표본 수 확대)에서 반드시 짚어야 할 한계다. ***
3) (1) "공격 성공" 이미지(원래 P16이 clean에서 맞혔는데 공격으로 틀려진 것)를 전부 P8로
   재분류 -> clean label 복원 비율
4) (2) clean 이미지 중 원래 P16이 맞혔는데 탐지기가 오탐(score>threshold)한 것들만 골라서
   P8로 재분류 -> 오히려 틀리게 되는(부작용) 비율

주의: src/attacks/patch_fool.py는 import만(수정 없음). 이 파일도 지우면 원상복구.

사용법:
  python probes/vitguard_p8_rescue_test.py --seed 42 --num_samples 30
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack


def _attn_hook(weights_list):
    def hook(module, input, output):
        with torch.no_grad():
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            weights_list.append(attn.mean(dim=1).detach())
    return hook


def collect_layer_attn(model, images):
    weights = []
    hooks = [blk.attn.register_forward_hook(_attn_hook(weights)) for blk in model.blocks]
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    return weights


def raw_at_layer(layer_weights, L):
    attn = layer_weights[L - 1]
    cls_to_patch = attn[:, 0, 1:]
    return cls_to_patch / cls_to_patch.sum(dim=1, keepdim=True)


def top4_mass(v):
    return v.topk(4, dim=1).values.sum(dim=1)


def youden_threshold(pos_scores, neg_scores):
    """pos=PatchFool, neg=Clean. TPR-FPR(J)이 최대인 임계값 반환."""
    candidates = np.unique(np.concatenate([pos_scores, neg_scores]))
    best_j, best_t = -1.0, candidates[0]
    for t in candidates:
        tpr = (pos_scores > t).mean()
        fpr = (neg_scores > t).mean()
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--detect_layer', type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        pred16_clean = model16(images).argmax(dim=1)
        pred8_clean = model8(images).argmax(dim=1)

    print(f"[참고] P16 clean acc: {(pred16_clean == labels).float().mean().item():.3f}  "
          f"P8 clean acc: {(pred8_clean == labels).float().mean().item():.3f}")

    print("\n[PatchFool on P16] 공격 생성 중 (attn_layer_idx=4, 표준 attention 선택)...")
    adv16, _ = patch_fool_attack(
        model16, images, labels, device, patch_size_model=16,
        attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
        attn_layer_idx=args.attn_layer_idx)
    with torch.no_grad():
        pred16_adv = model16(adv16).argmax(dim=1)

    # 임계값: 기존 layer_sweep 결과 재사용 (같은 30개 표본이라 다소 낙관적 편향 가능성 있음)
    # [재구성 주의] 01_layer_sweep_P16_raw.npz는 ViT_patchSwitch/defense/01_detection_feasibility/results/에
    # 있음(vitguard_layer_sweep.py 산출물) -- 이 스크립트(vitguard_p8_rescue_test.py)는
    # ViT_patchSwitch/archive/06_p8_rescue_test/에 있지만 원래 01_detection_feasibility의 산출물을
    # 재사용하던 관계라 경로가 폴더 경계를 넘어감 (ViT_robust/probes/ 시절부터 있던 관계, 원래도 그랬음)
    npz_path = os.path.join(ROOT, 'defense', '01_detection_feasibility', 'results',
                             '01_layer_sweep_P16_raw.npz')
    data = np.load(npz_path)
    clean_top4 = data['12_raw_clean_top4']
    pf_top4 = data['12_raw_pf_top4']
    thr, j = youden_threshold(pf_top4, clean_top4)
    print(f"\n탐지 임계값(top-4 mass, Youden's J 최적): {thr:.4f} (J={j:.3f})  "
          f"[layer_sweep 결과 재사용 — 같은 30개 표본이라 다소 낙관적 편향 가능]")

    lw_clean = collect_layer_attn(model16, images)
    score_clean = top4_mass(raw_at_layer(lw_clean, args.detect_layer)).cpu().numpy()
    lw_adv = collect_layer_attn(model16, adv16)
    score_adv = top4_mass(raw_at_layer(lw_adv, args.detect_layer)).cpu().numpy()

    flagged_clean = torch.tensor(score_clean > thr, device=device)
    flagged_adv = torch.tensor(score_adv > thr, device=device)
    print(f"clean {args.num_samples}개 중 오탐(flag) 비율: {flagged_clean.float().mean().item():.3f} "
          f"({flagged_clean.sum().item()}/{args.num_samples})")
    print(f"공격 {args.num_samples}개 중 정탐(flag) 비율: {flagged_adv.float().mean().item():.3f} "
          f"({flagged_adv.sum().item()}/{args.num_samples})")

    # (1) 공격 성공 이미지 -> P8 재분류 -> clean label 복원 비율
    with torch.no_grad():
        pred8_on_adv = model8(adv16).argmax(dim=1)

    attack_succeeded = (pred16_clean == labels) & (pred16_adv != labels)  # 원래 맞았는데 공격으로 틀려짐
    n_attacked = attack_succeeded.sum().item()
    recovered = attack_succeeded & (pred8_on_adv == labels)
    n_recovered = recovered.sum().item()
    recovery_rate = n_recovered / max(n_attacked, 1)

    print(f"\n=== (1) 공격 성공 이미지 -> P8 통째 재분류 시 clean label 복원율 ===")
    print(f"  공격 성공 이미지(원래 맞았는데 P16이 틀림): {n_attacked}/{args.num_samples}")
    print(f"  P8로 재분류 시 복원: {n_recovered}/{n_attacked} = {recovery_rate:.3f}")
    caught = (attack_succeeded & flagged_adv).sum().item()
    caught_and_recovered = (attack_succeeded & flagged_adv & (pred8_on_adv == labels)).sum().item()
    print(f"  (참고: 그 중 탐지기가 실제로 flag한 것 {caught}/{n_attacked}, "
          f"flag된 것 중 복원: {caught_and_recovered}/{max(caught,1)})")

    # (2) clean 이미지 오탐 -> P8 전환 -> 부작용(오히려 틀림) 비율
    orig_correct = (pred16_clean == labels)
    false_positive = orig_correct & flagged_clean
    n_fp = false_positive.sum().item()
    harmed = false_positive & (pred8_clean != labels)
    n_harmed = harmed.sum().item()
    harm_rate = n_harmed / max(n_fp, 1)

    print(f"\n=== (2) clean 이미지 오탐 -> P8 전환 시 부작용(오히려 틀림) 비율 ===")
    print(f"  원래 P16이 맞힌 이미지: {orig_correct.sum().item()}/{args.num_samples}")
    print(f"  그 중 오탐(flag)된 것: {n_fp}/{orig_correct.sum().item()}")
    print(f"  오탐 -> P8 전환 시 오히려 틀림: {n_harmed}/{max(n_fp,1)} = {harm_rate:.3f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, '06_p8_rescue_test.npz'),
             threshold=thr, score_clean=score_clean, score_adv=score_adv,
             pred16_clean=pred16_clean.cpu().numpy(), pred16_adv=pred16_adv.cpu().numpy(),
             pred8_clean=pred8_clean.cpu().numpy(), pred8_on_adv=pred8_on_adv.cpu().numpy(),
             labels=labels.cpu().numpy(),
             flagged_clean=flagged_clean.cpu().numpy(), flagged_adv=flagged_adv.cpu().numpy())
    print(f"\nSaved: {os.path.join(out_dir, '06_p8_rescue_test.npz')}")


if __name__ == '__main__':
    main()
