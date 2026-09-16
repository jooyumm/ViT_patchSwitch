"""
probes/results/02_system_validation/10_latency_memory/viz.py — [탐색적, 롤백 가능] 10_bench_latency_memory.npz를
읽어서 그림만 다시 그린다 (GPU/재실험 불필요).

사용법:
  python probes/results/02_system_validation/10_latency_memory/viz.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

RED, DARK = '#E94B3C', '#1F2937'
FS_TITLE, FS_SUB, FS_VAL = 14, 12, 11


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)


def main():
    d = np.load(os.path.join(RESULTS, '10_bench_latency_memory.npz'))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle('§10. P8 vs P16 deployment cost (batch=1, post warm-up, n=100)',
                 fontsize=FS_TITLE, fontweight='bold')

    metrics = [
        ('latency (ms)', d['P8_mean_ms'], d['P16_mean_ms'], d['P8_std_ms'], d['P16_std_ms']),
        ('peak memory (MB)', d['P8_peak_mem_mb'], d['P16_peak_mem_mb'], None, None),
        ('params (M)', d['P8_params'] / 1e6, d['P16_params'] / 1e6, None, None),
        ('FLOPs (G)', d['P8_flops'] / 1e9, d['P16_flops'] / 1e9, None, None),
    ]
    for ax, (name, v8, v16, e8, e16) in zip(axes, metrics):
        yerr = [e8, e16] if e8 is not None else None
        heights = [float(v8), float(v16)]
        ax.bar([0, 1], heights, color=RED, edgecolor='white', linewidth=1.2, yerr=yerr, capsize=4)
        for xi, h in zip([0, 1], heights):
            ax.text(xi, h + max(heights) * 0.02, f'{h:.1f}', ha='center', fontsize=FS_VAL,
                    fontweight='bold', color=DARK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['P8', 'P16'])
        ax.set_title(name, fontsize=FS_SUB)
        _style_ax(ax)

    out = os.path.join(RESULTS, '10_bench_latency_memory_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
