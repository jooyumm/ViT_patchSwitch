"""
probes/hybrid_joint_attack_test.py — [탐색적 검증, 롤백 가능] 부분 공유 backbone 시제품에
대한 joint attack(P16+P8 동시 최적화) stress test — 이번 부분 공유 아이디어의 핵심 성패 테스트.

배경
----
hybrid_partial_share.py의 HybridViT: patch_embed+block[0:5]("layer 1~5")는 P16/P8 각자
독립, block[5:]("layer 6~12")+norm+head는 P16 것을 공유. activation_similarity.py에서
이 공유 구간(layer 6/9)의 표현이 이미 상당히 정렬돼 있음(CKA 0.87~0.96)을 확인했다.

이 스크립트가 답하는 질문: 이 구조에 대해 지금까지와 "완전히 동일한" joint attack
(P16 경로 + P8 경로를 하나의 delta로 동시에 속이려는 adaptive attacker)을 돌렸을 때,
두 경로가 동시에 뚫리는 비율(완전 무력화율)이
  - 기존 P16≠P8 독립 아키텍처: 18.4%  (probes/vitguard_joint_attack_test.py)
  - 같은 patch size, 다른 seed 학습(P16-A/B): 52.3%  (probes/vitguard_diversity_test.py)
중 어느 쪽에 더 가까운가. 18.4% 근처를 유지하면 "후반부 공유해도 방어 기제는 안 죽는다"는
신호, 52.3%에 가까워지면 "공유가 사실상 같은 모델을 만들어버려서 위험하다"는 신호.

중요: joint_patch_fool_attack 자체는 patch_fool_joint.py에서 100% 그대로 재사용한다
(공격 로직을 바꾸지 않고, model16/model8 자리에 HybridBranch 래퍼만 끼워 넣음).

주의: patch_fool_joint.py, src/attacks/patch_fool.py, src/models.py, src/dataset.py는
import만(수정 없음). 이 파일 지우면 원상복구.

사용법:
  python probes/hybrid_joint_attack_test.py --seed 123 --num_samples 50 --chunk 20
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

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack
from patch_fool_joint import joint_patch_fool_attack
from hybrid_partial_share import build_hybrid, HybridBranch, sanity_check_branch16_matches_model16

# 기존 n=200 최종 검증(calibration 100개)에서 확정한 임계값 재사용 — 여기서 새로 안 정함
CALIBRATED_THRESHOLD = 0.5567

# 대조를 위한 기존 결과 (ViT_patchSwitch/README.md)
PREV_HETERO_FOOL_BOTH = 0.184    # P16 != P8 (기존 구조, 완전 독립)
PREV_HOMO_FOOL_BOTH = 0.523      # 같은 patch size(P16), 다른 seed 학습


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
    """model.blocks (HybridBranch도 .blocks 속성 제공)에 hook 걸어 attention 수집."""
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
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)  # 기존 joint attack test와 같은 seed -> 표본 비교 용이
    parser.add_argument('--split_layer', type=int, default=5)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--detect_layer', type=int, default=12)
    parser.add_argument('--chunk', type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()
    ppl = 224 // 16

    print(f"\n[구조] split_layer={args.split_layer} -> "
          f"P16/P8 각자 독립: block[0:{args.split_layer}] (layer 1~{args.split_layer}), "
          f"공유(P16 가중치): block[{args.split_layer}:12]+norm+head (layer {args.split_layer+1}~12)")

    hybrid = build_hybrid(model16, model8, split_layer=args.split_layer).to(device)
    branch16 = HybridBranch(hybrid, '16').to(device).eval()
    branch8 = HybridBranch(hybrid, '8').to(device).eval()

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    # ── (0) 정합성 검증: branch16은 model16과 100% 동일해야 함 (wiring 버그 있으면 여기서 걸림) ──
    ok, max_diff = sanity_check_branch16_matches_model16(hybrid, model16, images[:8])
    print(f"\n[정합성 검증] branch16 == model16 여부: {ok} (max_diff={max_diff:.2e})")
    assert ok, "HybridBranch16이 원본 model16과 다름 -- wiring 버그, 결과 신뢰 불가"

    # ── (1) clean accuracy 먼저 정직하게 측정 (P8 경로는 재학습 없이 낀 것이라 낮을 수 있음) ──
    with torch.no_grad():
        pred16_clean = branch16(images).argmax(dim=1)
        pred8_clean = branch8(images).argmax(dim=1)
        pred16_orig_clean = model16(images).argmax(dim=1)
        pred8_orig_clean = model8(images).argmax(dim=1)
    acc16 = (pred16_clean == labels).float().mean().item()
    acc8 = (pred8_clean == labels).float().mean().item()
    acc16_orig = (pred16_orig_clean == labels).float().mean().item()
    acc8_orig = (pred8_orig_clean == labels).float().mean().item()
    both_orig_correct = (pred16_clean == labels) & (pred8_clean == labels)
    print(f"\n=== Clean accuracy ===")
    print(f"  branch16(hybrid) clean acc: {acc16:.3f}   (참고: 원본 model16 단독: {acc16_orig:.3f})")
    print(f"  branch8(hybrid)  clean acc: {acc8:.3f}   (참고: 원본 model8 단독: {acc8_orig:.3f})")
    print(f"  둘 다 원래 맞춘 이미지: {both_orig_correct.sum().item()}/{args.num_samples}")
    if acc8 < 0.3:
        print(f"  [경고] branch8 clean accuracy가 매우 낮음 ({acc8:.3f}) -- 후반부를 미세조정 없이 "
              f"P8 초반부에 그대로 이어붙인 대가. 아래 공격 결과는 '이미 상당히 망가진 분류기'를 "
              f"공격하는 것이므로 기존 18.4%/52.3%와 직접 비교 시 주의 필요.")

    if both_orig_correct.sum().item() < 5:
        print("\n[중단] 둘 다 원래 맞춘 표본이 너무 적어 공격 실험 의미 없음. 여기서 종료.")
        out_dir = os.path.dirname(os.path.abspath(__file__)).replace('/defense/', '/results/', 1)
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, f'12_hybrid_joint_attack_n{args.num_samples}.npz'),
                 acc16=acc16, acc8=acc8, acc16_orig=acc16_orig, acc8_orig=acc8_orig,
                 both_orig_correct=both_orig_correct.cpu().numpy(), aborted=True)
        return

    # ── (A) 참고용: P16(branch16)만 공격했을 때 P8(branch8)까지 같이 속는 비율 ──
    print(f"\n[참고 baseline] branch16 단독 공격 생성 중 (전이 여부 확인용)...")
    adv_single_chunks = []
    for s in range(0, args.num_samples, args.chunk):
        e = min(s + args.chunk, args.num_samples)
        c, _ = patch_fool_attack(
            branch16, images[s:e], labels[s:e], device, patch_size_model=16,
            attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
            attn_layer_idx=args.attn_layer_idx)
        adv_single_chunks.append(c)
        torch.cuda.empty_cache()
    adv_single = torch.cat(adv_single_chunks, dim=0)
    with torch.no_grad():
        pred16_single = branch16(adv_single).argmax(dim=1)
        pred8_single = branch8(adv_single).argmax(dim=1)
    fool16_single = both_orig_correct & (pred16_single != labels)
    fool_both_single = fool16_single & (pred8_single != labels)
    n16f = max(fool16_single.sum().item(), 1)
    print(f"  branch16만 공격: branch16 속음 {fool16_single.sum().item()}/{both_orig_correct.sum().item()}, "
          f"그 중 branch8까지 같이 속음(전이) {fool_both_single.sum().item()}/{n16f} "
          f"= {fool_both_single.sum().item()/n16f:.3f}")

    # ── (B) joint attack: branch16+branch8 동시 공격 (기존 함수 100% 재사용) ──────
    print(f"\n[Joint attack] branch16+branch8 동시 공격 생성 중 (총 {args.num_samples}개, {args.chunk}개씩)...")
    adv_joint_chunks = []
    true_idx_chunks = []
    for s in range(0, args.num_samples, args.chunk):
        e = min(s + args.chunk, args.num_samples)
        print(f"  [{s}:{e}] 처리 중...")
        c, ti = joint_patch_fool_attack(
            branch16, branch8, images[s:e], labels[s:e], device,
            attn_layer_idx=args.attn_layer_idx, num_patch=1, train_attack_iters=250)
        adv_joint_chunks.append(c)
        true_idx_chunks.append(ti)
        torch.cuda.empty_cache()
    adv_joint = torch.cat(adv_joint_chunks, dim=0)
    true_idx = torch.cat(true_idx_chunks, dim=0)

    with torch.no_grad():
        pred16_joint = branch16(adv_joint).argmax(dim=1)
        pred8_joint = branch8(adv_joint).argmax(dim=1)

    fool16_joint = both_orig_correct & (pred16_joint != labels)
    fool8_joint = both_orig_correct & (pred8_joint != labels)
    fool_both_joint = both_orig_correct & (pred16_joint != labels) & (pred8_joint != labels)
    n_base = both_orig_correct.sum().item()
    rate_both = fool_both_joint.sum().item() / max(n_base, 1)
    print(f"\n=== Joint attack 결과 (원래 둘 다 맞춘 {n_base}개 기준) ===")
    print(f"  branch16만 속음(branch8은 여전히 맞음): {(fool16_joint & ~fool8_joint).sum().item()}/{n_base}")
    print(f"  branch8만 속음(branch16은 여전히 맞음): {(fool8_joint & ~fool16_joint).sum().item()}/{n_base}")
    print(f"  둘 다 속음(방어 완전 무력화): {fool_both_joint.sum().item()}/{n_base} = {rate_both:.3f}")
    print(f"\n=== 기존 결과와 비교 ===")
    print(f"  이 시제품 (부분 공유, layer {args.split_layer+1}~12 공유):        {rate_both:.3f}")
    print(f"  기존 P16≠P8 독립 구조 (probes/vitguard_joint_attack_test.py):     {PREV_HETERO_FOOL_BOTH:.3f}")
    print(f"  같은 patch size, 다른 seed 학습 (probes/vitguard_diversity_test.py): {PREV_HOMO_FOOL_BOTH:.3f}")

    # ── (C) 탐지기: branch16 경로에 기존 calibration 임계값 그대로 적용 (일관성 체크 겸) ──
    lw_joint = collect_layer_attn(branch16, adv_joint)
    v_joint = raw_at_layer(lw_joint, args.detect_layer)
    score_joint = top4_mass(v_joint).cpu().numpy()
    flagged_joint = score_joint > CALIBRATED_THRESHOLD
    sorted_joint = v_joint.argsort(dim=1, descending=True)

    print(f"\n=== 탐지기 (branch16 경로, 기존 calibration 임계값 {CALIBRATED_THRESHOLD} 그대로) ===")
    print(f"  joint attack {args.num_samples}개 중 flag된 비율: {flagged_joint.mean():.3f} "
          f"({flagged_joint.sum()}/{args.num_samples})")
    flagged_joint_t = torch.tensor(flagged_joint, device=device)
    caught_both_fooled = fool_both_joint & flagged_joint_t
    print(f"  '둘 다 속은'(방어 무력화) 사례 중 그래도 탐지는 된 것: "
          f"{caught_both_fooled.sum().item()}/{max(fool_both_joint.sum().item(),1)}")

    hit1 = (sorted_joint[:, :1] == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    hit4 = (sorted_joint[:, :4] == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    top1 = sorted_joint[:, 0]
    true_r, true_c = true_idx // ppl, true_idx % ppl
    top1_r, top1_c = top1 // ppl, top1 % ppl
    cheby = torch.maximum((true_r - top1_r).abs(), (true_c - top1_c).abs()).float()
    print(f"\n=== Localization (branch16 경로) ===")
    print(f"  recall@1={hit1:.3f}  recall@4={hit4:.3f}  grid_dist_mean={cheby.mean().item():.2f}")

    out_dir = os.path.dirname(os.path.abspath(__file__)).replace('/defense/', '/results/', 1)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'12_hybrid_joint_attack_n{args.num_samples}.npz')
    np.savez(save_path,
             split_layer=args.split_layer, threshold=CALIBRATED_THRESHOLD,
             acc16=acc16, acc8=acc8, acc16_orig=acc16_orig, acc8_orig=acc8_orig,
             score_joint=score_joint, rate_both=rate_both,
             pred16_clean=pred16_clean.cpu().numpy(), pred8_clean=pred8_clean.cpu().numpy(),
             pred16_single=pred16_single.cpu().numpy(), pred8_single=pred8_single.cpu().numpy(),
             pred16_joint=pred16_joint.cpu().numpy(), pred8_joint=pred8_joint.cpu().numpy(),
             labels=labels.cpu().numpy(), true_idx=true_idx.cpu().numpy(),
             sorted_joint=sorted_joint.cpu().numpy())
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
