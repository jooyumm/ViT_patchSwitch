"""
probes/vitguard_final_validation.py — [탐색적 검증, 롤백 가능] calibration/evaluation을 분리한
확대 표본으로 탐지·위치특정·복원율·오탐률을 한 번에 재측정.

배경
----
vitguard_p8_rescue_test.py(n=30)는 임계값을 정한 표본과 평가한 표본이 같아서 낙관적으로
편향됐을 수 있다는 한계가 있었다. 이번엔:
  1) calibration set과 evaluation set을 인덱스로 완전히 분리 (겹치지 않음)
  2) 임계값(threshold)은 오직 calibration set의 clean/PatchFool 점수로만 정함
  3) recall / FPR / 복원율 / 부작용률은 오직 evaluation set에서만 계산 (진짜 held-out 평가)
  4) localization(recall@K, 그리드 거리)은 임계값과 무관(순위 기반)하므로 순환성 문제가
     없어 calibration+evaluation 전체 표본으로 계산해서 통계적 안정성을 높임

방법
----
- num_samples장 로드 -> 앞 절반 calibration, 뒤 절반 evaluation으로 분리
- attn_layer_idx=4 표준 PatchFool 공격을 전체 배치에 한 번에 적용 (원본 함수 그대로)
- L=12 raw attention top-4 mass 점수로 Youden's J 임계값을 calibration에서만 도출
- evaluation에서: 탐지 recall/FPR, localization recall@K/그리드거리, P8 복원율, P8 부작용률

P16->P8 전이 저항성에 대한 adaptive attack 스트레스 테스트는 이번 스코프에서 제외(다음 논의).

주의: src/attacks/patch_fool.py는 import만(수정 없음). 이 파일도 지우면 원상복구.

사용법:
  python probes/vitguard_final_validation.py --seed 42 --num_samples 200
  (num_samples를 절반씩 calibration/evaluation으로 나눔)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, PROBES)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack, _collect_attn, _select_patch_attn
from eval_utils import calibration_eval_split, youden_threshold, wilson_ci


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=200, help='calibration+evaluation 합계')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--detect_layer', type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cal, ev = calibration_eval_split(args.num_samples, frac=0.5)
    n_cal = cal.stop - cal.start
    n_eval = ev.stop - ev.start

    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()
    ppl = 224 // 16

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    print(f"calibration: {n_cal}개 (인덱스 0~{n_cal-1}), evaluation: {n_eval}개 (인덱스 {n_cal}~{n_cal+n_eval-1})")

    with torch.no_grad():
        pred16_clean = model16(images).argmax(dim=1)
        pred8_clean = model8(images).argmax(dim=1)
    print(f"[참고] P16 clean acc: {(pred16_clean == labels).float().mean().item():.3f}  "
          f"P8 clean acc: {(pred8_clean == labels).float().mean().item():.3f}")

    # 정답 토큰 (localization용, 전체 배치 한 번에)
    attn_weights, hooks = _collect_attn(model16)
    with torch.no_grad():
        model16(images)
    for h in hooks:
        h.remove()
    true_idx = _select_patch_attn(attn_weights, args.attn_layer_idx, 1, device)[:, 0]

    # PatchFool 공격은 250iter 동안 gradient/optimizer state를 유지해야 해서 메모리를 많이 씀 —
    # 배치 200개를 한 번에 돌리면 OOM 나므로(실측됨), CHUNK 단위로 나눠서 돌리고 이어붙인다.
    # (참고: 순수 forward-only pass는 200개 배치로도 문제없었음 — 여기만 나눔)
    CHUNK = 50
    print(f"\n[PatchFool on P16] 공격 생성 중 (총 {args.num_samples}개, {CHUNK}개씩 나눠서)...")
    adv_chunks = []
    for s in range(0, args.num_samples, CHUNK):
        e = min(s + CHUNK, args.num_samples)
        print(f"  [{s}:{e}] 처리 중...")
        adv_chunk, _ = patch_fool_attack(
            model16, images[s:e], labels[s:e], device, patch_size_model=16,
            attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
            attn_layer_idx=args.attn_layer_idx)
        adv_chunks.append(adv_chunk)
        torch.cuda.empty_cache()
    adv16 = torch.cat(adv_chunks, dim=0)

    with torch.no_grad():
        pred16_adv = model16(adv16).argmax(dim=1)
        pred8_on_adv = model8(adv16).argmax(dim=1)

    lw_clean = collect_layer_attn(model16, images)
    score_clean = top4_mass(raw_at_layer(lw_clean, args.detect_layer)).cpu().numpy()
    lw_adv = collect_layer_attn(model16, adv16)
    v_adv = raw_at_layer(lw_adv, args.detect_layer)
    score_adv = top4_mass(v_adv).cpu().numpy()
    sorted_adv = v_adv.argsort(dim=1, descending=True)

    # ── 1) 임계값: calibration에서만 ────────────────────────────────
    thr, j = youden_threshold(score_adv[cal], score_clean[cal])
    print(f"\n=== 임계값 (calibration {n_cal}개로만 도출) ===")
    print(f"  threshold={thr:.4f}  Youden's J={j:.3f}")
    print(f"  calibration clean top4 mean={score_clean[cal].mean():.3f}  "
          f"pf top4 mean={score_adv[cal].mean():.3f}")

    # ── 2) 탐지 recall/FPR: evaluation에서만 ─────────────────────────
    flagged_clean_ev = score_clean[ev] > thr
    flagged_adv_ev = score_adv[ev] > thr
    fpr = flagged_clean_ev.mean()
    recall = flagged_adv_ev.mean()
    fpr_ci = wilson_ci(flagged_clean_ev.sum(), n_eval)
    recall_ci = wilson_ci(flagged_adv_ev.sum(), n_eval)
    print(f"\n=== 탐지 (evaluation {n_eval}개, held-out) ===")
    print(f"  FPR    = {fpr:.3f} ({flagged_clean_ev.sum()}/{n_eval})  95% CI [{fpr_ci[0]:.3f}, {fpr_ci[1]:.3f}]")
    print(f"  recall = {recall:.3f} ({flagged_adv_ev.sum()}/{n_eval})  95% CI [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]")

    # ── 3) localization: 임계값 무관, 전체 표본(calibration+evaluation) ──
    hit1 = (sorted_adv[:, :1] == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    hit4 = (sorted_adv[:, :4] == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    hit8 = (sorted_adv[:, :8] == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    top1 = sorted_adv[:, 0]
    true_r, true_c = true_idx // ppl, true_idx % ppl
    top1_r, top1_c = top1 // ppl, top1 % ppl
    cheby = torch.maximum((true_r - top1_r).abs(), (true_c - top1_c).abs()).float()
    print(f"\n=== Localization (전체 {args.num_samples}개, 임계값 무관) ===")
    print(f"  recall@1={hit1:.3f}  recall@4={hit4:.3f}  recall@8={hit8:.3f}")
    print(f"  grid dist mean={cheby.mean().item():.2f}  median={cheby.median().item():.1f}  "
          f"exact={float((cheby==0).float().mean()):.3f}")

    # ── 4) 복원율: evaluation에서만 ──────────────────────────────────
    pred16_clean_ev = pred16_clean[ev]; pred16_adv_ev = pred16_adv[ev]
    pred8_on_adv_ev = pred8_on_adv[ev]; labels_ev = labels[ev]

    attack_succeeded = (pred16_clean_ev == labels_ev) & (pred16_adv_ev != labels_ev)
    n_attacked = attack_succeeded.sum().item()
    recovered = attack_succeeded & (pred8_on_adv_ev == labels_ev)
    n_recovered = recovered.sum().item()
    recovery_rate = n_recovered / max(n_attacked, 1)
    recovery_ci = wilson_ci(n_recovered, max(n_attacked, 1))
    print(f"\n=== (1) 복원율 (evaluation {n_eval}개, held-out) ===")
    print(f"  공격 성공: {n_attacked}/{n_eval}")
    print(f"  P8 재분류 시 복원: {n_recovered}/{n_attacked} = {recovery_rate:.3f}  "
          f"95% CI [{recovery_ci[0]:.3f}, {recovery_ci[1]:.3f}]")

    # ── 5) 부작용률: evaluation에서만 ────────────────────────────────
    pred8_clean_ev = pred8_clean[ev]
    orig_correct = (pred16_clean_ev == labels_ev)
    false_positive = orig_correct & torch.tensor(flagged_clean_ev, device=device)
    n_fp = false_positive.sum().item()
    harmed = false_positive & (pred8_clean_ev != labels_ev)
    n_harmed = harmed.sum().item()
    harm_rate = n_harmed / max(n_fp, 1)
    print(f"\n=== (2) 부작용률 (evaluation {n_eval}개, held-out) ===")
    print(f"  원래 P16이 맞힌 것: {orig_correct.sum().item()}/{n_eval}, 그 중 오탐: {n_fp}")
    print(f"  오탐 -> P8 전환 시 오히려 틀림: {n_harmed}/{max(n_fp,1)} = {harm_rate:.3f}")
    if n_fp == 0:
        print(f"  (오탐 자체가 0건 — 부작용률은 정의 불가. 대신 오탐률 자체의 상한을 참고: "
              f"FPR 95% CI 상단 {fpr_ci[1]:.3f})")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'06_final_validation_n{args.num_samples}.npz'),
             n_cal=n_cal, n_eval=n_eval, threshold=thr,
             score_clean=score_clean, score_adv=score_adv,
             true_idx=true_idx.cpu().numpy(), sorted_adv=sorted_adv.cpu().numpy(),
             pred16_clean=pred16_clean.cpu().numpy(), pred16_adv=pred16_adv.cpu().numpy(),
             pred8_clean=pred8_clean.cpu().numpy(), pred8_on_adv=pred8_on_adv.cpu().numpy(),
             labels=labels.cpu().numpy())

    # 요약 그림
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['FPR', 'Detect recall', 'Localize recall@1', 'Recovery rate']
    values = [fpr, recall, hit1, recovery_rate]
    cis = [fpr_ci, recall_ci, (None, None), recovery_ci]
    colors = ['#E94B3C', '#16A34A', '#2563EB', '#F59E0B']
    bars = ax.bar(metrics, values, color=colors, alpha=0.85)
    for i, (v, ci) in enumerate(zip(values, cis)):
        if ci[0] is not None:
            ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt='none', ecolor='black', capsize=5)
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Final validation (calibration/evaluation split, n_eval={n_eval})')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'06_final_validation_n{args.num_samples}.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
