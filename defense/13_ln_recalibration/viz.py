"""
probes/defense/13_ln_recalibration/viz.py — [탐색적, 롤백 가능] 이 실험은 .npz를 저장하지
않았다(hybrid_ln_recalibration.py에 savez 호출이 없음) — 유일한 원자료는 실행 로그
results/13_ln_recalibration_run_2147532.txt 뿐이라, 거기 적힌 숫자를 직접 상수로 써서 그린다.

사용법:
  python probes/defense/13_ln_recalibration/viz.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

RED, GRAY, DARK = '#E94B3C', '#6B7280', '#1F2937'

# 값 출처: results/13_ln_recalibration_run_2147532.txt (job 2147532) -- npz 미저장이라 유일한 근거
ACC_BEFORE, ACC_AFTER, ACC16_REF = 0.000, 0.000, 0.840


def main():
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.suptitle('§13. LayerNorm recalibration — branch8 accuracy still 0% after\n'
                '(source: 13_ln_recalibration_run_2147532.txt, no npz saved)',
                 fontsize=14, fontweight='bold')
    xs = [0, 1, 2]
    heights = [ACC16_REF * 100, ACC_BEFORE * 100, ACC_AFTER * 100]
    colors = [GRAY, RED, RED]
    labels = ['branch16\n(reference, unaffected)', 'branch8\nbefore recalibration', 'branch8\nafter recalibration']
    ax.bar(xs, heights, color=colors, edgecolor='white', linewidth=1.2, width=0.55)
    for xi, h in zip(xs, heights):
        ax.text(xi, h + 2, f'{h:.1f}%', ha='center', fontsize=11, fontweight='bold', color=DARK)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel('Clean accuracy %'); ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    out = os.path.join(RESULTS, '13_hybrid_ln_recalibration_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
