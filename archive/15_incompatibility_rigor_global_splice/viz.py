"""
15_incompatibility_rigor/viz.py — [탐색적, 롤백 가능] 15_incompatibility_rigor_n*.npz를
읽어서 그림만 다시 그린다 (GPU/재실험 불필요). num_eval 값은 실행 시 --num_eval과 맞춰서
파일명 뒤 n숫자를 바꿔줘야 함(기본 run_incompatibility_test.sh는 n50).

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
RESULTS = os.path.join(HERE, 'results')

RED, BLUE, GRAY, DARK = '#E94B3C', '#3B82F6', '#6B7280', '#1F2937'
FS_TITLE, FS_SUB, FS_TICK, FS_VAL = 14, 11, 10, 11


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None, help='npz 파일명의 num_eval 값. 생략 시 자동 탐색')
    args = parser.parse_args()

    if args.n is not None:
        npz_path = os.path.join(RESULTS, f'15_incompatibility_rigor_n{args.n}.npz')
    else:
        candidates = sorted(glob.glob(os.path.join(RESULTS, '15_incompatibility_rigor_n*.npz')))
        assert candidates, f"결과 npz를 못 찾음: {RESULTS}"
        npz_path = candidates[-1]
    d = np.load(npz_path)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle('§15. Re-testing §12 with the sequence-length confound removed',
                 fontsize=FS_TITLE, fontweight='bold')
    ax.set_title(f'calibration={int(d["num_calib"])}, eval={int(d["num_eval"])}, '
                 f'split_layer={int(d["split_layer"])}', fontsize=FS_SUB, color=GRAY)

    xs = list(range(5))
    heights = [float(d['sanity_p16_acc']) * 100,
               float(d['naive_p8_to_p16_acc']) * 100,
               float(d['pooled_p8_to_p16_acc']) * 100,
               float(d['p16a_to_p16b_acc']) * 100,
               float(d['pooled_plus_linear_adapter_acc']) * 100]
    colors = [GRAY, RED, RED, BLUE, RED]
    labels = ['0) sanity\nP16->P16\n(wiring control)',
              '1) naive\nP8->P16\n(sec.12 redo, 785 tok)',
              '2) pooled\nP8->P16\n(matched to 196 tok)',
              '3) positive control\nP16-A->P16-B\n(same patch size)',
              '4) pooled+linear fix\n(fit on calibration set)']

    ax.bar(xs, heights, color=colors, edgecolor='white', linewidth=1.2, width=0.62)
    for xi, h in zip(xs, heights):
        ax.text(xi, h + 2, f'{h:.1f}%', ha='center', fontsize=FS_VAL, fontweight='bold', color=DARK)

    ref = float(d['model16_ref_acc']) * 100
    ax.axhline(ref, color=GRAY, linestyle='--', linewidth=1, alpha=0.6)
    ax.text(1.5, ref + 3, f'model16 alone: {ref:.1f}%', fontsize=9, color=GRAY, ha='center')

    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylabel('Clean accuracy %'); ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    out = os.path.join(RESULTS, '15_incompatibility_rigor_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
