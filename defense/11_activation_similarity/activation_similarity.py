"""
probes/activation_similarity.py — [탐색적, 롤백 가능] P8/P16 체크포인트의 중간 레이어
activation이 서로 얼마나 유사한지 가볍게 확인 (부분 공유 backbone 설계 이전 사전 점검).

배경
----
diversity diagnostic 결과(같은 patch size, 다른 seed 학습 = 52.3% 취약 vs 다른 patch
size = 18.4% 취약)로 "토큰화 격자 불일치"가 방어의 핵심 기제라는 해석에 무게가 실렸다.
그렇다면 부분 공유 backbone(중간 레이어 이후를 P8/P16이 공유)을 설계하기 전에, 지금
독립적으로 학습된 두 체크포인트의 중간 레이어 activation이 애초에 얼마나 다른지부터
봐야 한다. 이미 꽤 유사하다면 공유해도 (그 유사한 레이어까지는) 구조적 방어가 크게
훼손되지 않을 가능성이 있다는 신호이고, 많이 다르다면 공유가 위험하다는 신호다.

방법
----
- P16/P8 각각 clean 이미지에 대해 forward, block 6/9의 출력(hidden state)을 hook으로 수집.
- P16과 P8은 토큰 개수가 다르므로(196 vs 784 patch tokens) 토큰 단위로 직접 비교 불가.
  대신 이미지당 하나의 벡터로 요약하는 두 가지 방식을 모두 본다:
    (a) CLS 토큰 그 자체 (분류에 쓰이는 벡터, 토큰화와 무관하게 정의됨)
    (b) patch 토큰 평균 (mean pooling, 토큰 개수 차이를 흡수)
- 두 요약 벡터 집합(P16 vs P8, 같은 이미지 순서로 정렬됨) 사이의:
    - linear CKA (Kornblith et al. 2019; 대표적인 activation 유사도 지표)
    - 평균 cosine similarity (샘플별로 계산 후 평균)
  을 계산.
- 해석 기준선(null baseline)으로, P8 쪽 이미지 순서를 무작위로 섞은 뒤(mismatched pairing)
  같은 지표를 계산 — "우연히 이 정도는 비슷해 보인다"는 하한선을 제공한다.
  진짜(매칭된) 유사도가 이 baseline보다 뚜렷이 높아야 "구조적으로 유사하다"고 말할 수 있다.

주의: src/models.py, src/dataset.py는 import만(수정 없음). 이 파일 지우면 원상복구.

사용법:
  python probes/activation_similarity.py --num_samples 200 --layers 6 9
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


def linear_cka(X, Y):
    """Kornblith et al. 2019, linear kernel. X:(N,Dx), Y:(N,Dy), 행=샘플(이미지)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
    normX = np.linalg.norm(X.T @ X, 'fro')
    normY = np.linalg.norm(Y.T @ Y, 'fro')
    return hsic / (normX * normY + 1e-12)


def mean_cosine(X, Y):
    """샘플(행)별 cosine similarity를 구해 평균."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return float((Xn * Yn).sum(axis=1).mean())


def collect_block_outputs(model, images, layer_indices, chunk=50):
    """layer_indices: 1-based block 번호 리스트 (예: [6,9] -> blocks[5], blocks[8]).
    반환: {layer: {'cls': (N,C) ndarray, 'mean_patch': (N,C) ndarray}}"""
    captured = {L: [] for L in layer_indices}

    hooks = []
    for L in layer_indices:
        def make_hook(Lc):
            def hook(module, inp, out):
                captured[Lc].append(out.detach())
            return hook
        hooks.append(model.blocks[L - 1].register_forward_hook(make_hook(L)))

    results = {L: {'cls': [], 'mean_patch': []} for L in layer_indices}
    with torch.no_grad():
        for s in range(0, images.shape[0], chunk):
            e = min(s + chunk, images.shape[0])
            for L in layer_indices:
                captured[L].clear()
            model(images[s:e])
            for L in layer_indices:
                out = captured[L][0]  # (b, N+1, C)
                results[L]['cls'].append(out[:, 0, :].cpu().numpy())
                results[L]['mean_patch'].append(out[:, 1:, :].mean(dim=1).cpu().numpy())

    for h in hooks:
        h.remove()

    for L in layer_indices:
        results[L]['cls'] = np.concatenate(results[L]['cls'], axis=0)
        results[L]['mean_patch'] = np.concatenate(results[L]['mean_patch'], axis=0)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 9])
    parser.add_argument('--chunk', type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images = images.to(device)
    print(f"[참고] 이미지 {images.shape[0]}개, 레이어 {args.layers}에서 activation 비교")

    r16 = collect_block_outputs(model16, images, args.layers, chunk=args.chunk)
    r8 = collect_block_outputs(model8, images, args.layers, chunk=args.chunk)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(images.shape[0])

    out_summary = {}
    print(f"\n{'layer':>6} {'repr':>10} {'CKA(matched)':>13} {'CKA(shuffled)':>14} "
          f"{'cos(matched)':>13} {'cos(shuffled)':>14}")
    for L in args.layers:
        for repr_name in ('cls', 'mean_patch'):
            X = r16[L][repr_name]
            Y = r8[L][repr_name]
            cka_matched = linear_cka(X, Y)
            cka_shuffled = linear_cka(X, Y[perm])
            cos_matched = mean_cosine(X, Y)
            cos_shuffled = mean_cosine(X, Y[perm])
            print(f"{L:>6} {repr_name:>10} {cka_matched:>13.4f} {cka_shuffled:>14.4f} "
                  f"{cos_matched:>13.4f} {cos_shuffled:>14.4f}")
            out_summary[f'L{L}_{repr_name}'] = dict(
                cka_matched=cka_matched, cka_shuffled=cka_shuffled,
                cos_matched=cos_matched, cos_shuffled=cos_shuffled)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'11_activation_similarity_n{args.num_samples}.npz')
    np.savez(save_path, layers=args.layers, summary=out_summary)
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
