"""
§15. [탐색적, 롤백 가능] "P8/P16 임베딩 공간 불일치" 가설을 §12보다 엄밀하게 재검증.

배경 — §12/§13이 왜 "확실한 근거"가 아니었나
--------------------------------------------
§12(hybrid_partial_share.py)는 P8 초반부(patch_embed8+block[0:5]) 출력을 그대로
P16 후반부(block[5:12]+norm+head)에 이어붙였다. 근데 P8 초반부 출력은 785토큰
(784패치+CLS)이고, P16 후반부는 원래 학습 때 197토큰(196패치+CLS)만 본 적이 있다.
그래서 branch8의 clean accuracy 0% 붕괴가 "두 모델의 표현이 의미적으로 안 맞아서"인지,
그냥 "본 적 없는 길이(4배)의 시퀀스를 넣어서 기계적으로 깨진 것"인지 이 실험만으로는
분리가 안 됐다. 이 스크립트는 그 confound를 두 가지 방법으로 제거한다.

방법
----
(A) 시퀀스 길이를 맞춘다: P16의 패치 1개 = P8의 패치 2x2(4개)이므로, P8 초반부가 낸 784개
    패치 토큰을 28x28 그리드로 복원해 2x2 average pooling하면 정확히 P16이 기대하는
    14x14=196개가 된다(CLS 토큰은 그대로 둠, 합쳐서 197개). 이 상태에서도 붕괴하면
    시퀀스 길이 문제가 아니라는 훨씬 강한 증거가 된다. (조건: pooled_p8_to_p16)

(B) 양성 대조군: "독립 학습된 두 네트워크는 어차피 뭐든 못 이어붙인다"는 일반적 현상과,
    "patch size(토큰화 격자) 차이가 특별히 더 문제"라는 주장을 분리하기 위해, §8
    diversity diagnostic에서 쓴 두 번째 P16 체크포인트(model B, 같은 patch size·다른 학습)를
    가져와 P16-A 초반부 + P16-B 후반부를 이어붙인다. 여긴 애초에 토큰 수가 둘 다 197개라
    풀링이 필요 없다. 이것도 0%로 무너지면 patch size와 무관하게 "임의의 두 독립 학습
    네트워크는 못 이어붙인다"는 뜻이 되고, 이게 안 무너지는데 (A)만 무너지면 patch size
    차이가 특별히 더 치명적이라는 뜻이 된다. (조건: p16a_to_p16b)

(C) 최소 보정: (A)가 무너진다면, 완전히 안 맞는 건지 아니면 값싼 선형 변환 하나로 보정되는
    수준인지 확인한다. 접합부에서 "P8-초반부(풀링됨) 활성값 -> P16-초반부(model16 자기
    자신) 활성값"을 매핑하는 아핀 변환을 calibration 세트에서 최소제곱으로 닫힌 형태로
    구해서(gradient descent 아님, 1회 계산) 적용해보고 accuracy가 회복되는지 본다.
    (조건: pooled_p8_to_p16 + linear_adapter)
    §13(LayerNorm 재보정)은 "스케일/분포만" 맞췄는데, 이건 "방향(회전/사상)까지" 맞추는
    거라 §13보다 더 관대한 보정이다. 이것도 안 되면 표현이 정말 다른 정보를 담고 있다는
    뜻이지, 좌표계만 다른 게 아니라는 뜻이다.

기존 대비 추가/변경
------------------
- baseline(no_pool) 조건도 같이 돌려서 §12 재현 숫자를 이 스크립트 안에서 다시 확인한다
  (같은 결론이 나오는지 cross-check).
- sanity_p16 조건: branch16(초반부=P16, 후반부=P16 자기 자신)이 model16(images)과 100%
  일치하는지 확인(wiring 버그 방지, §12의 sanity_check_branch16_matches_model16과 동일 취지).

주의: src/models.py, src/attacks/patch_fool.py는 import만(수정 없음). model B 로딩은
§8(vitguard_diversity_test.py)의 MODEL_B_NAME을 그대로 재사용. 이 파일(과 이 폴더) 지우면
원상복구.

사용법:
  python rigorous_incompatibility_test.py --num_calib 50 --num_eval 30 --split_layer 5 --seed 42
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader

MODEL_B_NAME = 'vit_base_patch16_224.augreg_in21k_ft_in1k'  # §8과 동일 — "다른 학습" P16 대리


def load_model_b(device):
    model = timm.create_model(MODEL_B_NAME, pretrained=True)
    model = model.to(device)
    model.eval()
    return model


class GeneralHybridViT(nn.Module):
    """early_model의 patch_embed+초반 block으로 시작해서 late_model의 후반 block+norm+head로
    끝나는 스플라이스. early/late의 토큰 수가 다르면 pool_fn으로, 표현 공간이 다르면
    adapter로 접합부를 보정할 수 있다(둘 다 None이면 §12와 동일한 naive 이어붙이기)."""

    def __init__(self, early_model, late_model, split_layer, pool_fn=None, adapter=None):
        super().__init__()
        self.patch_embed = early_model.patch_embed
        self.cls_token = early_model.cls_token
        self.pos_embed = early_model.pos_embed
        self.pos_drop = early_model.pos_drop
        self.blocks_early = early_model.blocks[:split_layer]

        self.blocks_late = late_model.blocks[split_layer:]
        self.norm = late_model.norm
        self.fc_norm = late_model.fc_norm
        self.head = late_model.head

        self.pool_fn = pool_fn
        self.adapter = adapter

    def early_forward(self, images):
        """접합부 활성값까지만 계산 (adapter 학습용 feature 수집에도 재사용)."""
        x = self.patch_embed(images)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks_early:
            x = blk(x)
        if self.pool_fn is not None:
            cls_tok, patch_tok = x[:, :1], x[:, 1:]
            patch_tok = self.pool_fn(patch_tok)
            x = torch.cat([cls_tok, patch_tok], dim=1)
        return x

    def forward(self, images):
        x = self.early_forward(images)
        if self.adapter is not None:
            x = self.adapter(x)
        for blk in self.blocks_late:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.fc_norm(x)
        return self.head(x)


def pool_2x2(patch_tokens, grid=28, target=14):
    """(B, grid*grid, C) 패치 토큰을 2x2 average pooling해서 (B, target*target, C)로.
    P8(28x28)->P16(14x14) 격자 대응에 씀. timm PatchEmbed는 Conv2d(B,C,H,W)를
    flatten(2).transpose(1,2)로 펴기 때문에 row-major(래스터) 순서가 보장된다."""
    B, N, C = patch_tokens.shape
    assert N == grid * grid, f"기대한 토큰 수({grid * grid})와 다름({N})"
    x = patch_tokens.transpose(1, 2).reshape(B, C, grid, grid)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    x = x.reshape(B, C, target * target).transpose(1, 2)
    return x


class AffineAdapter(nn.Module):
    """토큰별 아핀 변환 y = xW + b. 최소제곱으로 닫힌 형태로 1회 피팅(gradient descent 아님)."""

    def __init__(self, dim=768):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, x):
        return x @ self.W + self.b

    @torch.no_grad()
    def fit(self, x_src, y_tgt):
        """x_src, y_tgt: (N_tokens_total, C). 편향 항을 위해 x에 1열 추가 후 최소제곱."""
        C = x_src.shape[1]
        ones = torch.ones(x_src.shape[0], 1, dtype=x_src.dtype)
        x_aug = torch.cat([x_src, ones], dim=1)          # (N, C+1)
        sol = torch.linalg.lstsq(x_aug, y_tgt).solution    # (C+1, C)
        self.W.copy_(sol[:C])
        self.b.copy_(sol[C])


@torch.no_grad()
def eval_clean_acc(model, images, labels, chunk=20):
    preds = []
    for i in range(0, images.shape[0], chunk):
        out = model(images[i:i + chunk])
        preds.append(out.argmax(dim=1))
    preds = torch.cat(preds)
    return (preds == labels).float().mean().item(), preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_calib', type=int, default=50,
                         help='선형 adapter 피팅용 calibration 이미지 수 (accuracy 평가와 분리)')
    parser.add_argument('--num_eval', type=int, default=30,
                         help='clean accuracy 평가용 held-out 이미지 수')
    parser.add_argument('--split_layer', type=int, default=5, help='§12/§13과 동일 기본값')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk', type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    model16 = load_vit_model(16, device)
    model8 = load_vit_model(8, device)
    model16b = load_model_b(device)
    model16.eval(); model8.eval(); model16b.eval()

    n_total = args.num_calib + args.num_eval
    loader, _ = get_dataloader(batch_size=n_total, num_samples=n_total, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    calib_images, calib_labels = images[:args.num_calib], labels[:args.num_calib]
    eval_images, eval_labels = images[args.num_calib:], labels[args.num_calib:]

    print(f"calibration={args.num_calib}, eval={args.num_eval} (겹치지 않는 분리된 표본)")

    results = {}

    # ── 0) sanity check: early=P16, late=P16(자기 자신) 이어붙이면 model16과 100% 일치해야 함 ──
    hybrid_sanity = GeneralHybridViT(model16, model16, args.split_layer)
    acc, _ = eval_clean_acc(hybrid_sanity, eval_images, eval_labels, args.chunk)
    ref_acc, _ = eval_clean_acc(model16, eval_images, eval_labels, args.chunk)
    print(f"\n[sanity] branch(P16->P16) acc={acc:.3f}  vs  model16 직접 acc={ref_acc:.3f}  "
          f"(정확히 같아야 wiring 정상)")
    results['sanity_p16_acc'] = acc
    results['model16_ref_acc'] = ref_acc

    # ── 1) §12 재현: P8 초반부(785토큰, 풀링 없음) + P16 후반부 — confound 있는 원래 버전 ──
    hybrid_naive = GeneralHybridViT(model8, model16, args.split_layer, pool_fn=None)
    acc_naive, _ = eval_clean_acc(hybrid_naive, eval_images, eval_labels, args.chunk)
    print(f"\n[1. naive, §12 재현] P8초반(785토큰,풀링없음)+P16후반: clean acc={acc_naive:.3f}")
    results['naive_p8_to_p16_acc'] = acc_naive

    # ── 2) 핵심: P8 초반부를 196토큰으로 풀링해서 시퀀스 길이 맞춘 뒤 P16 후반부 ──
    pool_fn = pool_2x2
    hybrid_pooled = GeneralHybridViT(model8, model16, args.split_layer, pool_fn=pool_fn)
    acc_pooled, _ = eval_clean_acc(hybrid_pooled, eval_images, eval_labels, args.chunk)
    print(f"\n[2. pooled, 핵심] P8초반(2x2 풀링->196토큰)+P16후반: clean acc={acc_pooled:.3f}  "
          f"(naive={acc_naive:.3f}보다 확실히 높아지면 시퀀스 길이가 일부 원인이었다는 뜻,"
          f" 그래도 낮으면 진짜 표현 불일치)")
    results['pooled_p8_to_p16_acc'] = acc_pooled

    # ── 3) 양성 대조군: P16-A 초반부 + P16-B(다른 학습) 후반부 — 토큰 수는 이미 같음(197) ──
    hybrid_ab = GeneralHybridViT(model16, model16b, args.split_layer, pool_fn=None)
    acc_ab, _ = eval_clean_acc(hybrid_ab, eval_images, eval_labels, args.chunk)
    print(f"\n[3. 양성 대조군] P16-A초반+P16-B(다른학습)후반(풀링 불필요): clean acc={acc_ab:.3f}  "
          f"(이것도 무너지면 '독립학습 네트워크는 뭐든 못 붙임'이 원인, "
          f"이건 안 무너지고 (2)만 무너지면 patch size 차이가 특별히 치명적)")
    results['p16a_to_p16b_acc'] = acc_ab

    # ── 4) 최소 보정: (2)의 접합부에 선형 adapter를 최소제곱으로 피팅해서 재평가 ──
    with torch.no_grad():
        src_feat = hybrid_pooled.early_forward(calib_images)          # (num_calib, 197, 768), P8 풀링됨
        tgt_hybrid = GeneralHybridViT(model16, model16, args.split_layer)
        tgt_feat = tgt_hybrid.early_forward(calib_images)              # (num_calib, 197, 768), P16 자기 자신
    N, T, C = src_feat.shape
    adapter = AffineAdapter(dim=C)
    adapter.fit(src_feat.reshape(N * T, C).cpu(), tgt_feat.reshape(N * T, C).cpu())
    adapter = adapter.to(device)

    hybrid_adapted = GeneralHybridViT(model8, model16, args.split_layer, pool_fn=pool_fn, adapter=adapter)
    acc_adapted, _ = eval_clean_acc(hybrid_adapted, eval_images, eval_labels, args.chunk)
    print(f"\n[4. 선형 보정] (2)+선형 adapter(calibration {args.num_calib}장으로 최소제곱 피팅): "
          f"clean acc={acc_adapted:.3f}  (pooled={acc_pooled:.3f}보다 확실히 높아지면 '좌표계만 다름',"
          f" 그대로면 선형으로 안 풀리는 진짜 불일치)")
    results['pooled_plus_linear_adapter_acc'] = acc_adapted

    # ── 참고용: 각 원본 모델의 개별 clean accuracy ──
    acc_model8, _ = eval_clean_acc(model8, eval_images, eval_labels, args.chunk)
    acc_model16b, _ = eval_clean_acc(model16b, eval_images, eval_labels, args.chunk)
    print(f"\n[참고] model8 단독 acc={acc_model8:.3f}  model16b 단독 acc={acc_model16b:.3f}")
    results['model8_ref_acc'] = acc_model8
    results['model16b_ref_acc'] = acc_model16b

    print("\n=== 요약 ===")
    print(f"  0) sanity (P16->P16, 대조용)          : {results['sanity_p16_acc']:.3f}")
    print(f"  1) naive  (P8->P16, 풀링 없음, §12 재현): {results['naive_p8_to_p16_acc']:.3f}")
    print(f"  2) pooled (P8->P16, 토큰수 맞춤)        : {results['pooled_p8_to_p16_acc']:.3f}")
    print(f"  3) P16-A->P16-B (양성 대조군)           : {results['p16a_to_p16b_acc']:.3f}")
    print(f"  4) pooled+선형보정                      : {results['pooled_plus_linear_adapter_acc']:.3f}")

    out_dir = os.path.dirname(os.path.abspath(__file__)).replace('/defense/', '/results/', 1)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'15_incompatibility_rigor_n{args.num_eval}.npz'),
             split_layer=args.split_layer, num_calib=args.num_calib, num_eval=args.num_eval,
             **results)
    print(f"\nSaved: {os.path.join(out_dir, f'15_incompatibility_rigor_n{args.num_eval}.npz')}")


if __name__ == '__main__':
    main()
