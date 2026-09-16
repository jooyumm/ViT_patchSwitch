"""
probes/results/05_partial_share_exploration/12_partial_share_prototype/viz.py — [탐색적, 롤백 가능] 12_hybrid_joint_attack_n50.npz를
읽어서 그림만 다시 그린다 (GPU/재실험 불필요).

사용법:
  python probes/results/05_partial_share_exploration/12_partial_share_prototype/viz.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

RED, GRAY, DARK = '#E94B3C', '#6B7280', '#1F2937'
FS_TITLE, FS_TICK, FS_VAL = 14, 10.5, 11


def main():
    d = np.load(os.path.join(RESULTS, '12_hybrid_joint_attack_n50.npz'))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle('§12. Partial-share prototype — P8 branch collapses without fine-tuning',
                 fontsize=FS_TITLE, fontweight='bold')
    xs = [0, 1, 2, 3]
    heights = [float(d['acc16_orig']) * 100, float(d['acc16']) * 100,
               float(d['acc8_orig']) * 100, float(d['acc8']) * 100]
    colors = [GRAY, RED, GRAY, RED]
    labels = ['original model16\n(reference)', 'branch16\n(hybrid)',
              'original model8\n(reference)', 'branch8\n(hybrid, shared late layers)']
    ax.bar(xs, heights, color=colors, edgecolor='white', linewidth=1.2, width=0.6)
    for xi, h in zip(xs, heights):
        ax.text(xi, h + 2, f'{h:.1f}%', ha='center', fontsize=FS_VAL, fontweight='bold', color=DARK)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel('Clean accuracy %'); ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    out = os.path.join(RESULTS, '12_hybrid_partial_share_collapse.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
