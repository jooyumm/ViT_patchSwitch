"""
§16. [탐색적, 롤백 가능] "국소 토큰 세분화(Local Token Subdivision)" 1단계 실현성 테스트.

배경
----
지금까지의 방어(§6)는 공격이 의심되면 이미지 **전체**를 P8로 통째 재분류했다. 사용자가
제안한 새 설계는: 의심되는 P16 패치 1개만 국소적으로 2×2=4개의 P8 서브패치로 쪼개서
그 자리에 끼워 넣고(196→199토큰), 나머지는 그대로 P16으로 계속 처리하자는 것.

§15(archive/15_incompatibility_rigor_global_splice/)는 이걸 **전역**으로(모든 784토큰을
P8→풀링→P16후반부) 테스트해서 "선형 변환 하나로 완전히 복구됨(84%)"을 확인했었다. 이 실험은
그 결론을 **국소** 버전으로 이어서 테스트한다 — 단, 이번엔 중간 레이어(block 5)가 아니라
**입력(patch_embed) 레벨**에서 교체한다: 그래야 라우터를 어느 레이어에 두든(지금은 L=12,
나중에 L=6으로 당기는 것도 가능) "이미 학습된 P16 12개 레이어를 통째로, 전혀 안 건드리고"
그대로 재사용할 수 있다.

방법
----
1) Calibration(clean 이미지, 재학습 없이 최소제곱 1회): P16의 patch_embed+pos_embed 출력
   196개와, P8의 patch_embed+pos_embed 출력을 2×2 spatial 대응으로 묶어서(각 P16 패치 1개당
   대응하는 P8 서브패치 4개, 타깃은 그 P16 패치 임베딩을 4번 복제) 아핀 변환(768×768+bias)을
   §15와 동일한 방식(닫힌 형태, gradient descent 없음)으로 피팅한다. §15는 block-5 활성값을
   맞췄지만 이번엔 **입력 임베딩 자체**를 맞춘다 — 그래야 P16 12개 레이어를 통째로 안전하게
   재사용 가능.
2) 공격 이미지에서 §2와 동일한 방식(L=12 raw attention top-1)으로 의심 패치 위치를 찾는다
   (탐지기 자체는 이미 검증됨, recall@1 96.7%).
3) 그 위치의 P16 패치 임베딩을, 같은 이미지의 그 영역에 해당하는 P8 서브패치 4개(패치임베딩+
   자기 pos_embed, 보정 어댑터 적용)로 **in-place 교체**한다 — 나머지 195개 패치는 원래 P16
   임베딩(공격 이미지 것) 그대로 둔다. 결과: 1 CLS + 195 P16패치 + 4 P8서브패치 = 200토큰.
4) 이 200토큰 시퀀스를 P16의 12개 레이어(가중치 전혀 안 바꿈) + norm + head에 그대로 통과.

조건
----
0) sanity   : 교체 없이 그대로(model16과 100% 동일해야 함 — wiring 확인)
1) adv_p16  : 방어 없는 P16의 공격 이미지 정확도 (공격이 실제로 먹히는지 확인용, 낮아야 정상)
2) local_swap_adv : **핵심** — 공격 성공한 이미지에 국소 교체 적용 시 정답으로 복원되는 비율
3) local_swap_clean_fp : clean 이미지에 (실제로는 공격이 없는데도) 같은 방식으로 국소 교체를
   적용했을 때 정확도가 얼마나 떨어지는지 — "탐지기가 잘못 발동했을 때의 부작용" 비용

주의: src/models.py, src/attacks/patch_fool.py는 import만(수정 없음). 이 파일 지우면 원상복구.

사용법:
  python local_swap_test.py --num_calib 50 --num_eval 30 --seed 42
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack, _collect_attn, _select_patch_attn


class AffineAdapter(nn.Module):
    """토큰별 아핀 변환 y = xW + b. 최소제곱으로 닫힌 형태 1회 피팅 (gradient descent 아님)."""

    def __init__(self, dim=768):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, x):
        return x @ self.W + self.b

    @torch.no_grad()
    def fit(self, x_src, y_tgt):
        C = x_src.shape[1]
        ones = torch.ones(x_src.shape[0], 1, dtype=x_src.dtype)
        x_aug = torch.cat([x_src, ones], dim=1)
        sol = torch.linalg.lstsq(x_aug, y_tgt).solution
        self.W.copy_(sol[:C])
        self.b.copy_(sol[C])


def patch_embed_with_pos(model, images):
    """patch_embed(x) + pos_embed[patch part]. CLS 제외 (B, N, D)."""
    x = model.patch_embed(images)
    return x + model.pos_embed[:, 1:, :]


def p16_to_p8_subpatch_indices(idx16, ppl16=14, ppl8=28):
    """P16 flat index -> 대응하는 P8 flat index 4개 (2x2 블록, row-major)."""
    r16, c16 = idx16 // ppl16, idx16 % ppl16
    r8, c8 = r16 * 2, c16 * 2
    return [r8 * ppl8 + c8, r8 * ppl8 + c8 + 1, (r8 + 1) * ppl8 + c8, (r8 + 1) * ppl8 + c8 + 1]


@torch.no_grad()
def full_forward_from_tokens(model16, cls_plus_patches):
    """CLS+패치 임베딩 시퀀스(이미 pos_embed 포함)를 P16의 12개 레이어+head에 그대로 통과."""
    x = model16.pos_drop(cls_plus_patches)
    for blk in model16.blocks:
        x = blk(x)
    x = model16.norm(x)
    x = x[:, 0]
    x = model16.fc_norm(x)
    return model16.head(x)


@torch.no_grad()
def localize_top1(model16, images, detect_layer=12):
    """§2와 동일: L=detect_layer raw attention top-1 (실제 배포에서 탐지기가 쓰는 방식)."""
    weights = []

    def hook(module, inp, out):
        x = inp[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        weights.append(attn.mean(dim=1).detach())

    hooks = [blk.attn.register_forward_hook(hook) for blk in model16.blocks]
    model16(images)
    for h in hooks:
        h.remove()
    attn = weights[detect_layer - 1]
    cls_to_patch = attn[:, 0, 1:]
    cls_to_patch = cls_to_patch / cls_to_patch.sum(dim=1, keepdim=True)
    return cls_to_patch.argmax(dim=1)  # (B,)


def build_local_swap_batch(model16, model8, adapter, images, flag_idx, ppl16=14):
    """images의 각 샘플에서 flag_idx[i] 위치의 P16 패치를, 대응 P8 서브패치 4개(어댑터 보정)로
    in-place 교체한 (B, 200, D) 시퀀스를 만든다. 나머지 195개는 원래 P16 임베딩 그대로."""
    B = images.shape[0]
    p16_full = patch_embed_with_pos(model16, images)          # (B, 196, D)
    p8_full = patch_embed_with_pos(model8, images)             # (B, 784, D)
    p8_adapted = adapter(p8_full.reshape(-1, p8_full.shape[-1])).reshape(p8_full.shape)

    cls = model16.cls_token.expand(B, -1, -1) + model16.pos_embed[:, :1, :]

    seqs = []
    for i in range(B):
        idx16 = int(flag_idx[i])
        keep_mask = torch.ones(p16_full.shape[1], dtype=torch.bool)
        keep_mask[idx16] = False
        kept = p16_full[i, keep_mask]                         # (195, D)
        sub_idx = p16_to_p8_subpatch_indices(idx16, ppl16=ppl16)
        replacement = p8_adapted[i, sub_idx]                   # (4, D)
        seq = torch.cat([cls[i], kept, replacement], dim=0)    # (1+195+4=200, D)
        seqs.append(seq)
    return torch.stack(seqs, dim=0)


@torch.no_grad()
def eval_acc(preds, labels):
    return (preds == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_calib', type=int, default=50, help='입력 레벨 어댑터 피팅용')
    parser.add_argument('--num_eval', type=int, default=30, help='공격/복원 평가용 held-out')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attn_layer_idx', type=int, default=4, help='PatchFool 공격 타겟 레이어')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    n_total = args.num_calib + args.num_eval
    loader, _ = get_dataloader(batch_size=n_total, num_samples=n_total, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    calib_images = images[:args.num_calib]
    eval_images, eval_labels = images[args.num_calib:], labels[args.num_calib:]
    print(f"calibration={args.num_calib}, eval={args.num_eval} (겹치지 않는 분리된 표본)")

    # ── 1) 입력 레벨 어댑터 피팅 (닫힌 형태, calibration 이미지의 P16<->P8 패치 임베딩 쌍) ──
    with torch.no_grad():
        p16_full = patch_embed_with_pos(model16, calib_images)   # (N,196,D)
        p8_full = patch_embed_with_pos(model8, calib_images)     # (N,784,D)
    N, n16, D = p16_full.shape
    ppl16 = 14
    src_list, tgt_list = [], []
    for idx16 in range(n16):
        sub_idx = p16_to_p8_subpatch_indices(idx16, ppl16=ppl16)
        for si in sub_idx:
            src_list.append(p8_full[:, si, :])
            tgt_list.append(p16_full[:, idx16, :])
    x_src = torch.cat(src_list, dim=0)   # (N*196*4, D)
    y_tgt = torch.cat(tgt_list, dim=0)
    adapter = AffineAdapter(dim=D)
    adapter.fit(x_src.cpu(), y_tgt.cpu())
    adapter = adapter.to(device)
    print(f"어댑터 피팅: {x_src.shape[0]}쌍으로 {D}x{D} 아핀 변환 1회 계산 완료")

    # ── 0) sanity: 교체 없이 그대로 -> model16과 100% 동일해야 함 ──
    cls0 = model16.cls_token.expand(eval_images.shape[0], -1, -1) + model16.pos_embed[:, :1, :]
    p16_eval_full = patch_embed_with_pos(model16, eval_images)
    seq0 = torch.cat([cls0, p16_eval_full], dim=1)
    out_sanity = full_forward_from_tokens(model16, seq0)
    out_ref = model16(eval_images)
    max_diff = (out_sanity - out_ref).abs().max().item()
    print(f"\n[0. sanity] full_forward_from_tokens vs model16 직접: max diff={max_diff:.2e} "
          f"(0에 가까워야 wiring 정상)")

    # ── clean 기준 ──
    pred16_clean = model16(eval_images).argmax(dim=1)
    acc_clean = eval_acc(pred16_clean, eval_labels)
    print(f"[참고] P16 clean acc={acc_clean:.3f}")

    # ── 1) PatchFool 공격, 방어 없는 P16 정확도 ──
    print("\n[PatchFool 공격 생성 중...]")
    adv_images, _ = patch_fool_attack(
        model16, eval_images, eval_labels, device, patch_size_model=16,
        attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
        attn_layer_idx=args.attn_layer_idx)
    pred16_adv = model16(adv_images).argmax(dim=1)
    acc_adv = eval_acc(pred16_adv, eval_labels)
    attack_succeeded = (pred16_clean == eval_labels) & (pred16_adv != eval_labels)
    n_attacked = int(attack_succeeded.sum().item())
    print(f"[1. 방어 없음] adv acc={acc_adv:.3f}  (공격 성공 {n_attacked}/{len(eval_labels)})")

    # ── 2) 핵심: 국소 교체로 복원되는가 (공격 성공한 이미지만) ──
    flag_idx_adv = localize_top1(model16, adv_images, detect_layer=12)
    seq_adv_swap = build_local_swap_batch(model16, model8, adapter, adv_images, flag_idx_adv, ppl16=ppl16)
    pred_local_swap_adv = full_forward_from_tokens(model16, seq_adv_swap).argmax(dim=1)
    if n_attacked > 0:
        recovered = attack_succeeded & (pred_local_swap_adv == eval_labels)
        recovery_rate = recovered.sum().item() / n_attacked
    else:
        recovery_rate = float('nan')
    acc_local_swap_adv_all = eval_acc(pred_local_swap_adv, eval_labels)
    print(f"[2. 국소 교체(공격 이미지)] 공격 성공 이미지 중 복원율={recovery_rate:.3f} "
          f"({int((attack_succeeded & (pred_local_swap_adv == eval_labels)).sum().item())}/{n_attacked})  "
          f"전체 acc={acc_local_swap_adv_all:.3f}")

    # ── 3) 부작용 비용: clean 이미지에 (불필요하게) 같은 교체를 적용하면 정확도가 얼마나 깎이나 ──
    flag_idx_clean = localize_top1(model16, eval_images, detect_layer=12)
    seq_clean_swap = build_local_swap_batch(model16, model8, adapter, eval_images, flag_idx_clean, ppl16=ppl16)
    pred_local_swap_clean = full_forward_from_tokens(model16, seq_clean_swap).argmax(dim=1)
    acc_local_swap_clean = eval_acc(pred_local_swap_clean, eval_labels)
    print(f"[3. 국소 교체(clean, 오탐 시뮬레이션)] acc={acc_local_swap_clean:.3f} "
          f"(clean 기준 {acc_clean:.3f} 대비 {acc_clean - acc_local_swap_clean:+.3f}p 손실)")

    print("\n=== 요약 ===")
    print(f"  0) sanity max diff              : {max_diff:.2e}")
    print(f"  1) 방어 없음 (adv acc)            : {acc_adv:.3f}")
    print(f"  2) 국소 교체 복원율 (핵심)         : {recovery_rate:.3f}")
    print(f"  3) 국소 교체 clean 부작용 비용     : -{acc_clean - acc_local_swap_clean:.3f}p")

    out_dir = os.path.dirname(os.path.abspath(__file__)).replace('/defense/', '/results/', 1)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'16_local_swap_l12_n{args.num_eval}.npz'),
             sanity_max_diff=max_diff, acc_clean=acc_clean, acc_adv=acc_adv,
             n_attacked=n_attacked, recovery_rate=recovery_rate,
             acc_local_swap_adv_all=acc_local_swap_adv_all,
             acc_local_swap_clean=acc_local_swap_clean)
    print(f"\nSaved: {os.path.join(out_dir, f'16_local_swap_l12_n{args.num_eval}.npz')}")


if __name__ == '__main__':
    main()
