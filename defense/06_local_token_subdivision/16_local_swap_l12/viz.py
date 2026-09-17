"""
16_local_swap_l12/viz.py — [탐색적, 롤백 가능] 16_local_swap_l12_n*.npz를 읽어서 그림만
다시 그린다 (GPU/재실험 불필요).

사용법:
  python viz.py [--n 50]
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE.replace('/defense/', '/results/', 1)

RED, GREEN, GRAY, DARK = '#E94B3C', '#22A559', '#6B7280', '#1F2937'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    if args.n is not None:
        npz_path = os.path.join(RESULTS, f'16_local_swap_l12_n{args.n}.npz')
    else:
        candidates = sorted(glob.glob(os.path.join(RESULTS, '16_local_swap_l12_n*.npz')))
        assert candidates, f"결과 npz를 못 찾음: {RESULTS}"
        npz_path = candidates[-1]
    d = np.load(npz_path)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle('§16. Local token subdivision — first feasibility test (router at L=12)',
                 fontsize=14, fontweight='bold', y=1.0)

    labels = ['clean\n(no attack)', 'adv, no defense\n(PatchFool)',
              'adv + local swap\n(recovery)', 'clean + local swap\n(false-positive cost)']
    heights = [float(d['acc_clean']) * 100, float(d['acc_adv']) * 100,
               float(d['recovery_rate']) * 100, float(d['acc_local_swap_clean']) * 100]
    colors = [GRAY, RED, GREEN, GRAY]
    xs = list(range(4))

    ax.bar(xs, heights, color=colors, edgecolor='white', linewidth=1.2, width=0.6)
    for xi, h in zip(xs, heights):
        ax.text(xi, h + 2, f'{h:.1f}%', ha='center', fontsize=12, fontweight='bold', color=DARK)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel('Accuracy / recovery rate (%)'); ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, -0.32, f'sanity wiring check: max diff = {float(d["sanity_max_diff"]):.1e} '
                        f'(should be ~0)', transform=ax.transAxes, ha='center', fontsize=9, color=GRAY)

    out = os.path.join(RESULTS, '16_local_swap_l12_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
