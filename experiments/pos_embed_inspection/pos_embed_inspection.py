"""
experiments/pos_embed_inspection/pos_embed_inspection.py — [탐색적] P16/P8의 학습된 position
embedding 테이블을 직접 눈으로 확인한다. GPU 계산(공격 등) 없이 모델 파라미터만 읽는다 —
아주 가벼운 스크립트라 CPU에서도 빠르게 끝난다.

배경
----
local_switch의 4개 P8 서브패치가 "정말로 서로 다른 위치 임베딩"을 쓰는지에 대한 대화에서
나온 구체적 예시(P16 idx=15 <-> P8 idx=[58,59,86,87])를 실제 학습된 벡터로 확인한다.

만드는 그림
----
1) pos_embed_grid_P16.png / pos_embed_grid_P8.png
   ViT 논문(Fig.7) 스타일 — 그리드 위 여러 "기준 위치"를 골라서, 그 위치의 임베딩과 나머지
   전체 위치 임베딩의 코사인 유사도를 원래 그리드 모양(14x14 / 28x28)으로 펼쳐 그린다.
   기준 위치 자기 자신에서 유사도가 가장 높고(진한 색), 멀어질수록 낮아지는 공간적 구조가
   보이면 "위치마다 실제로 다른, 그러면서도 의미 있게 구성된 벡터"라는 뜻이다.
2) pos_embed_worked_example.png
   대화에서 다룬 구체적 예시 — P16 idx=15와 그에 대응하는 P8 서브패치 4개(58,59,86,87)를
   각 그리드 위에 표시하고, 이 5개 벡터끼리의 코사인 유사도 5x5 행렬을 히트맵으로 보여준다.
   (P16과 P8은 완전히 별도로 학습된 테이블이라 서로 비교할 절대적 기준은 없지만, "P8 안에서
   4개가 서로 얼마나 다른지"와 "P16 것과 P8 것들이 얼마나 무관한지"는 확인할 수 있다.)

사용법:
  python pos_embed_inspection.py
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')):
    ROOT = os.path.dirname(ROOT)
RESULTS = HERE.replace('/experiments/', '/results/', 1)

import sys
sys.path.insert(0, ROOT)
from src.models import get_device, load_vit_model
from defense.local_switch import p16_to_p8_subpatch_indices

PPL16, PPL8 = 14, 28


def cosine_sim_matrix(v, table):
    """v: (D,), table: (N,D) -> (N,) cosine similarity of v against every row of table."""
    v = v / (np.linalg.norm(v) + 1e-8)
    t = table / (np.linalg.norm(table, axis=1, keepdims=True) + 1e-8)
    return t @ v


def plot_grid_of_grids(pos_table, ppl, anchors, title, out_path):
    """pos_table: (ppl*ppl, D). anchors: list of flat indices to use as reference positions."""
    n = len(anchors)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.1 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for i, a in enumerate(anchors):
        sim = cosine_sim_matrix(pos_table[a], pos_table).reshape(ppl, ppl)
        ax = axes[i]
        im = ax.imshow(sim, cmap='viridis', vmin=-0.2, vmax=1.0)
        ar, ac = a // ppl, a % ppl
        ax.plot(ac, ar, marker='x', color='red', markersize=8, markeredgewidth=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'pos {a} = ({ar},{ac})', fontsize=8)
    for i in range(n, len(axes)):
        axes[i].axis('off')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label='cosine similarity to anchor (red x)')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    pos16 = model16.pos_embed[0, 1:, :].detach().cpu().numpy()  # (196, 768)
    pos8 = model8.pos_embed[0, 1:, :].detach().cpu().numpy()    # (784, 768)

    print(f"P16 pos_embed table: {pos16.shape}  (grid {PPL16}x{PPL16})")
    print(f"  per-position L2 norm: mean={np.linalg.norm(pos16, axis=1).mean():.3f}  "
          f"std={np.linalg.norm(pos16, axis=1).std():.3f}")
    print(f"P8  pos_embed table: {pos8.shape}  (grid {PPL8}x{PPL8})")
    print(f"  per-position L2 norm: mean={np.linalg.norm(pos8, axis=1).mean():.3f}  "
          f"std={np.linalg.norm(pos8, axis=1).std():.3f}")

    # ---- 1) ViT-paper-style grid-of-grids: 위치마다 임베딩이 실제로 공간 구조를 갖는지 ----
    anchors16 = [r * PPL16 + c for r in (1, 6, 12) for c in (1, 6, 12)]
    plot_grid_of_grids(pos16, PPL16, anchors16,
                        'P16 pos_embed: cosine similarity of each anchor (red x) to all 196 positions',
                        os.path.join(RESULTS, 'pos_embed_grid_P16.png'))

    anchors8 = [r * PPL8 + c for r in (2, 13, 25) for c in (2, 13, 25)]
    plot_grid_of_grids(pos8, PPL8, anchors8,
                        'P8 pos_embed: cosine similarity of each anchor (red x) to all 784 positions',
                        os.path.join(RESULTS, 'pos_embed_grid_P8.png'))

    # ---- 2) 대화에서 다룬 구체적 예시: P16 idx=15 <-> P8 idx=[58,59,86,87] ----
    idx16 = 15
    sub_idx = p16_to_p8_subpatch_indices(idx16, ppl16=PPL16, ppl8=PPL8)
    print(f"\nWorked example: P16 idx={idx16} -> P8 sub-patch indices {sub_idx}")

    labels = [f'P16[{idx16}]'] + [f'P8[{s}]' for s in sub_idx]
    vecs = np.stack([pos16[idx16]] + [pos8[s] for s in sub_idx], axis=0)  # (5, 768)
    norm_vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    sim5 = norm_vecs @ norm_vecs.T

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # P16 grid with idx16 marked
    ax = axes[0]
    grid16 = np.zeros((PPL16, PPL16))
    grid16[idx16 // PPL16, idx16 % PPL16] = 1
    ax.imshow(grid16, cmap='Reds', vmin=0, vmax=1)
    ax.set_title(f'P16 grid (14x14)\nidx={idx16} at ({idx16 // PPL16},{idx16 % PPL16})', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # P8 grid with the 4 sub-patch indices marked
    ax = axes[1]
    grid8 = np.zeros((PPL8, PPL8))
    for s in sub_idx:
        grid8[s // PPL8, s % PPL8] = 1
    ax.imshow(grid8, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'P8 grid (28x28)\n4 sub-patches = {sub_idx}', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    r0, c0 = sub_idx[0] // PPL8, sub_idx[0] % PPL8
    ax.set_xlim(c0 - 6, c0 + 8)
    ax.set_ylim(r0 + 8, r0 - 6)

    # 5x5 cosine similarity matrix among {P16[15], P8[58,59,86,87]}
    ax = axes[2]
    im = ax.imshow(sim5, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{sim5[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(sim5[i, j]) > 0.5 else 'black')
    ax.set_title('cosine similarity\n(these 5 position vectors)', fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Worked example from the discussion: P16 patch 15 -> its 4 P8 sub-patches',
                 fontsize=13, fontweight='bold', y=1.03)
    fig.tight_layout()
    out_path = os.path.join(RESULTS, 'pos_embed_worked_example.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"Saved: {out_path}")

    print("\ncosine similarity matrix:")
    print("            " + "  ".join(f'{l:>9}' for l in labels))
    for i, l in enumerate(labels):
        print(f'{l:>10}  ' + "  ".join(f'{sim5[i, j]:9.3f}' for j in range(5)))

    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, 'pos_embed_inspection.npz'),
             pos16=pos16, pos8=pos8, idx16=idx16, sub_idx=np.array(sub_idx),
             worked_example_labels=np.array(labels), worked_example_sim=sim5)
    print(f"Saved: {os.path.join(RESULTS, 'pos_embed_inspection.npz')}")


if __name__ == '__main__':
    main()
