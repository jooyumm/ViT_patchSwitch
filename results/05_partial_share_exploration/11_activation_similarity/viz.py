"""
probes/results/05_partial_share_exploration/11_activation_similarity/viz.py — [탐색적, 롤백 가능] 11_activation_similarity_n200.npz를
읽어서 그림만 다시 그린다 (GPU/재실험 불필요).

사용법:
  python probes/results/05_partial_share_exploration/11_activation_similarity/viz.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

RED, GRAY, DARK = '#E94B3C', '#6B7280', '#1F2937'
FS_TITLE, FS_SUB, FS_TICK, FS_VAL = 14, 12, 10.5, 11


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)


def main():
    d = np.load(os.path.join(RESULTS, '11_activation_similarity_n200.npz'), allow_pickle=True)
    summary = d['summary'].item()
    keys = list(summary.keys())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle('§11. P16/P8 intermediate-layer activation similarity (linear CKA, n=200 clean)',
                 fontsize=FS_TITLE, fontweight='bold')
    x = np.arange(len(keys))
    width = 0.35
    matched = [summary[k]['cka_matched'] for k in keys]
    shuffled = [summary[k]['cka_shuffled'] for k in keys]
    ax.bar(x - width/2, matched, width, color=RED, edgecolor='white', linewidth=1.2, label='matched (same image)')
    ax.bar(x + width/2, shuffled, width, color=GRAY, edgecolor='white', linewidth=1.2, label='shuffled (chance level)')
    for xi, v in zip(x - width/2, matched):
        ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=FS_VAL, fontweight='bold', color=DARK)
    for xi, v in zip(x + width/2, shuffled):
        ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=FS_VAL, fontweight='bold', color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=FS_TICK)
    ax.set_ylabel('linear CKA'); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=FS_SUB, frameon=False)
    _style_ax(ax)

    out = os.path.join(RESULTS, '11_activation_similarity_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
