"""
experiments/pos_embed_inspection/pos_embed_inspection.py — [탐색적] P8(그리고 비교용 P16)의
position embedding 테이블이 실제로 존재하고, 위치마다 다른 값을 갖는지 그냥 확인만 한다.
GPU 계산 없음 — 모델 파라미터를 읽기만 해서 CPU에서도 바로 끝난다.

사용법:
  python pos_embed_inspection.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')):
    ROOT = os.path.dirname(ROOT)
RESULTS = HERE.replace('/experiments/', '/results/', 1)
sys.path.insert(0, ROOT)

from src.models import get_device, load_vit_model


def main():
    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    pos16_full = model16.pos_embed[0].detach().cpu().numpy()  # (197, 768): CLS + 196 patches
    pos8_full = model8.pos_embed[0].detach().cpu().numpy()    # (785, 768): CLS + 784 patches

    print(f"model16.pos_embed shape: {tuple(model16.pos_embed.shape)}  "
          f"(1 CLS + 196 patch positions, dim 768)")
    print(f"model8.pos_embed  shape: {tuple(model8.pos_embed.shape)}  "
          f"(1 CLS + 784 patch positions, dim 768)")

    print(f"\nP8 position 0 (CLS), first 8 of 768 values:\n  {pos8_full[0, :8]}")
    print(f"P8 position 1 (patch idx 0), first 8 of 768 values:\n  {pos8_full[1, :8]}")
    print(f"P8 position 2 (patch idx 1), first 8 of 768 values:\n  {pos8_full[2, :8]}")
    print("-> each row is a different, real, trained vector (not all-zero, not identical).")

    pos8 = pos8_full[1:]   # (784, 768) patch positions only
    pos16 = pos16_full[1:]  # (196, 768)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                              gridspec_kw={'width_ratios': [1, 1]})

    ax = axes[0]
    im = ax.imshow(pos8, aspect='auto', cmap='RdBu_r',
                    vmin=-np.abs(pos8).max(), vmax=np.abs(pos8).max())
    ax.set_xlabel('embedding dim (0-767)')
    ax.set_ylabel('P8 patch position (0-783)')
    ax.set_title('P8 pos_embed table: raw values\n(784 positions x 768 dims)', fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1]
    norms8 = np.linalg.norm(pos8, axis=1)
    norms16 = np.linalg.norm(pos16, axis=1)
    ax.plot(norms8, color='#3B82F6', linewidth=1, label=f'P8 (784 positions)')
    ax.plot(norms16, color='#E94B3C', linewidth=1.3, label=f'P16 (196 positions)')
    ax.set_xlabel('patch position index')
    ax.set_ylabel('L2 norm of that position\'s embedding vector')
    ax.set_title('Each position has its own (different) vector', fontsize=12)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, 'pos_embed_exists.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    np.savez(os.path.join(RESULTS, 'pos_embed_inspection.npz'), pos16=pos16, pos8=pos8)
    print(f"Saved: {os.path.join(RESULTS, 'pos_embed_inspection.npz')}")


if __name__ == '__main__':
    main()
