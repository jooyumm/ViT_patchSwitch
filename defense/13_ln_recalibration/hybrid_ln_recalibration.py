"""
probes/hybrid_ln_recalibration.py — [탐색적, 아주 저렴, 롤백 가능] 부분 공유 시제품의
branch8 clean accuracy 0% 붕괴가 "통계 재보정"만으로 조금이라도 살아나는지 확인.

중요한 정정
-----------
사용자가 요청한 원래 아이디어("공유되는 late block의 LayerNorm running statistics를
train mode로 P8 배치를 통과시켜 재보정")는 이 아키텍처에는 문자 그대로 적용할 수 없다:
timm ViT는 BatchNorm을 전혀 쓰지 않고 LayerNorm만 쓰는데(25개 전부 LayerNorm, 확인함),
LayerNorm은 BatchNorm과 달리 running_mean/running_var 같은 누적 버퍼가 없다 — 매 forward마다
그 순간의 입력에 대해 토큰별로 즉석 정규화한다. 즉 model.train()으로 놓고 P8 배치를 통과시켜도
아무 상태도 안 바뀐다(진짜 no-op). 그래서 정신은 같지만 실제로 뭔가 바뀌는 가장 저렴한 대안으로
치환했다:

  LayerNorm은 토큰 내부(채널 축)만 정규화하고 채널별(배치·토큰에 걸친) 분포는 그대로
  통과시킨 뒤 학습된 gamma/beta(채널별 affine)를 곱한다. 그 gamma/beta는 P16 분포에서 나온
  채널별 평균/분산을 P16 후반부가 기대하는 값으로 맞추도록 학습된 값이다. P8 초반부가 만드는
  채널별 분포가 다르면 이 gamma/beta가 안 맞을 수 있다. 그래서:
    1. branch16(자기 집에 있는 상태)에서 각 공유 LayerNorm의 "정상 출력" 채널별 평균/표준편차
       (target_mean, target_std)를 구하고,
    2. branch8이 만드는 해당 LayerNorm 직전 입력을 가져와 LayerNorm의 토큰별 정규화까지만
       직접 재현한 뒤(gamma/beta 적용 전) 그 채널별 평균/표준편차(src_mean, src_std)를 구해서,
    3. new_gamma = target_std/src_std, new_beta = target_mean - new_gamma*src_mean 로
       branch8 전용 affine을 closed-form(그래디언트 없이)으로 새로 계산한다.
  이건 branch8이 forward할 때만 임시로 적용하고(원본 파라미터는 건드리지 않고 forward
  직전/직후에 교체), branch16은 원래 파라미터 그대로 쓴다 — 즉 "무거운 Q/K/V/MLP 가중치는
  그대로 공유하되, 가벼운 LayerNorm affine만 경로별로 따로 두면 살아나는가"를 보는 테스트.

방법
----
- 보정용 배치(seed=555, 별도 표본)로 target/src 통계 계산
- 평가용 배치는 job 2147400과 동일(seed=123, n=50)로 재사용 -> branch8 원래 0.000과 직접 비교
- 그래디언트/옵티마이저 전혀 없음, forward 몇 번 + 텐서 산술만 -> "아주 싼" 요청에 부합

주의: src/models.py, src/dataset.py, hybrid_partial_share.py는 import만(수정 없음).
이 파일 지우면 원상복구.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, PROBES)

import torch
import torch.nn as nn

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from hybrid_partial_share import build_hybrid, HybridBranch


def get_shared_layernorms(hybrid):
    """공유 후반부(blocks_late)에 있는 모든 LayerNorm + 최종 norm. fc_norm은 Identity라 제외."""
    lns = []
    for blk in hybrid.blocks_late:
        lns.append(blk.norm1)
        lns.append(blk.norm2)
    lns.append(hybrid.norm)
    return lns


@torch.no_grad()
def manual_ln_prenorm(x, ln):
    """nn.LayerNorm의 affine 적용 전, 토큰별(채널축) 정규화만 직접 재현."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + ln.eps)


@torch.no_grad()
def collect_ln_output_stats(model, images, ln_modules, chunk=50):
    """model 통과시키며 각 ln의 '출력'(post-affine) 채널별 평균/표준편차 계산."""
    captured = {ln: [] for ln in ln_modules}
    hooks = [ln.register_forward_hook(
        (lambda l: (lambda m, i, o: captured[l].append(o.detach())))(ln))
        for ln in ln_modules]
    for s in range(0, images.shape[0], chunk):
        e = min(s + chunk, images.shape[0])
        for ln in ln_modules:
            captured[ln].clear()
        model(images[s:e])
        for ln in ln_modules:
            out = captured[ln][0]  # (b, N, C)
            captured[ln] = [out]
    for h in hooks:
        h.remove()
    stats = {}
    for ln in ln_modules:
        out = captured[ln][0]
        flat = out.reshape(-1, out.shape[-1])
        stats[ln] = (flat.mean(dim=0), flat.std(dim=0))
    return stats


@torch.no_grad()
def collect_ln_prenorm_stats(model, images, ln_modules, chunk=50):
    """model 통과시키며 각 ln '직전 입력'을 받아, affine 적용 전 정규화(z)의 채널별 평균/표준편차."""
    captured = {ln: [] for ln in ln_modules}
    hooks = [ln.register_forward_hook(
        (lambda l: (lambda m, i, o: captured[l].append(i[0].detach())))(ln))
        for ln in ln_modules]
    for s in range(0, images.shape[0], chunk):
        e = min(s + chunk, images.shape[0])
        for ln in ln_modules:
            captured[ln].clear()
        model(images[s:e])
        for ln in ln_modules:
            x = captured[ln][0]
            captured[ln] = [x]
    for h in hooks:
        h.remove()
    stats = {}
    for ln in ln_modules:
        x = captured[ln][0]
        z = manual_ln_prenorm(x, ln)
        flat = z.reshape(-1, z.shape[-1])
        stats[ln] = (flat.mean(dim=0), flat.std(dim=0))
    return stats


class TempAffineOverride:
    """branch8 forward 동안만 공유 LN들의 weight/bias를 재보정 값으로 임시 교체."""
    def __init__(self, new_params):
        self.new_params = new_params  # {ln: (new_weight, new_bias)}
        self._orig = {}

    def __enter__(self):
        for ln, (w, b) in self.new_params.items():
            self._orig[ln] = (ln.weight.data.clone(), ln.bias.data.clone())
            ln.weight.data.copy_(w)
            ln.bias.data.copy_(b)
        return self

    def __exit__(self, *exc):
        for ln, (w, b) in self._orig.items():
            ln.weight.data.copy_(w)
            ln.bias.data.copy_(b)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_samples', type=int, default=100)
    parser.add_argument('--calib_seed', type=int, default=555)
    parser.add_argument('--eval_samples', type=int, default=50)
    parser.add_argument('--eval_seed', type=int, default=123)  # job 2147400과 동일 -> 직접 비교
    parser.add_argument('--split_layer', type=int, default=5)
    parser.add_argument('--chunk', type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    hybrid = build_hybrid(model16, model8, split_layer=args.split_layer).to(device)
    branch16 = HybridBranch(hybrid, '16').to(device).eval()
    branch8 = HybridBranch(hybrid, '8').to(device).eval()
    ln_modules = get_shared_layernorms(hybrid)
    print(f"[참고] 공유 LayerNorm 개수: {len(ln_modules)} (block당 2개 x {12-args.split_layer}개 block + 최종 norm 1개)")

    # ── 보정용 배치로 통계 계산 (평가 배치와 분리) ──
    calib_loader, _ = get_dataloader(batch_size=args.calib_samples,
                                      num_samples=args.calib_samples, seed=args.calib_seed)
    calib_images, _ = next(iter(calib_loader))
    calib_images = calib_images.to(device)

    print("[1/3] branch16(집)에서 목표 통계(target) 수집 중...")
    target_stats = collect_ln_output_stats(branch16, calib_images, ln_modules, chunk=args.chunk)

    print("[2/3] branch8(원본 gamma/beta 그대로)에서 현재 통계(src) 수집 중...")
    src_stats = collect_ln_prenorm_stats(branch8, calib_images, ln_modules, chunk=args.chunk)

    new_params = {}
    for ln in ln_modules:
        t_mean, t_std = target_stats[ln]
        s_mean, s_std = src_stats[ln]
        new_gamma = t_std / (s_std + 1e-6)
        new_beta = t_mean - new_gamma * s_mean
        new_params[ln] = (new_gamma, new_beta)

    # ── 평가 배치 (job 2147400과 동일 seed/샘플 수 -> 0.000과 직접 비교) ──
    eval_loader, _ = get_dataloader(batch_size=args.eval_samples,
                                     num_samples=args.eval_samples, seed=args.eval_seed)
    eval_images, eval_labels = next(iter(eval_loader))
    eval_images, eval_labels = eval_images.to(device), eval_labels.to(device)

    print("[3/3] branch8 재평가: 재보정 전 vs 후...")
    with torch.no_grad():
        pred_before = branch8(eval_images).argmax(dim=1)
    acc_before = (pred_before == eval_labels).float().mean().item()

    with torch.no_grad(), TempAffineOverride(new_params):
        pred_after = branch8(eval_images).argmax(dim=1)
    acc_after = (pred_after == eval_labels).float().mean().item()

    # branch16은 원본 파라미터 그대로이므로 재보정과 무관 -- 안 건드렸는지 확인용
    with torch.no_grad():
        pred16 = branch16(eval_images).argmax(dim=1)
    acc16 = (pred16 == eval_labels).float().mean().item()

    print(f"\n=== 결과 (n={args.eval_samples}, seed={args.eval_seed}, job 2147400과 동일 배치) ===")
    print(f"  branch8 clean acc, 재보정 전: {acc_before:.3f}  (참고: job 2147400에서는 0.000)")
    print(f"  branch8 clean acc, 재보정 후: {acc_after:.3f}")
    print(f"  branch16 clean acc (비교용, 영향 없어야 함): {acc16:.3f}  (참고: job 2147400에서는 0.840)")
    if acc_after < 0.05:
        print("\n  [결론] 재보정해도 사실상 그대로 0% 근처 -> LayerNorm 채널별 통계 불일치가 "
              "원인이 아니었다. 통계 재보정으로는 못 살림. 이 라인 접는 게 맞음.")
    elif acc_after < 0.3:
        print("\n  [결론] 0%에서는 벗어났지만 여전히 낮음 -> 통계보다 더 깊은(semantic) 불일치가 "
              "지배적. 재학습 없이는 실용성 낮음.")
    else:
        print("\n  [결론] 재보정만으로 꽤 회복됨 -> 재학습 없이도 살릴 여지 있음, 추가 조사 가치 있음.")


if __name__ == '__main__':
    main()
