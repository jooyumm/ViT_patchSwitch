"""
expected_cost_analysis.py — §6(탐지 FPR/recall) + §10(P8/P16 개별 비용)을 합쳐서,
"공격 비율(prevalence) π가 얼마든 상관없이 우리 시스템의 실제 기대 비용이 얼마인가"를 계산한다.

배경
----
지금까지 §10은 "P8이 P16보다 latency 2.7배/FLOPs 4.5배 비싸다"는 개별 비용만 보여줬는데,
실제 배포 시 매 이미지마다 P8을 도는 게 아니라 §6의 탐지기가 flag한 이미지에서만 P8을
추가로 돈다. 그래서 시스템 전체의 "기대 비용"은 공격 이미지 비율 π(prevalence)에 따라
달라진다:

  E[cost(π)] = P16_cost + [(1-π)·FPR + π·recall] · P8_cost

FPR/recall은 §6의 held-out evaluation(n=100)에서 나온 값을 그대로 쓴다(5.0%/64.0%).
비교 대상으로 "naive: 모든 이미지에 P16+P8을 항상 둘 다 돌린다"(π와 무관하게 상수 비용)도
같이 그려서, 우리 시스템이 항상-이득인지 아니면 어느 π 이상에서는 naive보다 비싼지 확인한다.

주의: 이건 새 GPU 실험이 아니라 기존 §6/§10 npz를 읽어서 계산만 하는 분석 스크립트다.

사용법:
  python expected_cost_analysis.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE.replace('/defense/', '/results/', 1)
VALID_DIR = os.path.join(ROOT, '06_final_validation')
LAT_DIR = os.path.join(ROOT, '10_latency_memory')

RED, GRAY, DARK, BLUE = '#E94B3C', '#6B7280', '#1F2937', '#3B82F6'


def main():
    val = np.load(os.path.join(VALID_DIR, '06_final_validation_n200.npz'))
    thr = float(val['threshold'])
    sc, sa = val['score_clean'], val['score_adv']
    n_cal = int(val['n_cal'])
    # eval(held-out) 슬라이스에서만 FPR/recall 계산 — calibration에 쓴 표본은 제외(순환평가 방지)
    fpr = float((sc[n_cal:] > thr).mean())
    recall = float((sa[n_cal:] > thr).mean())
    print(f"FPR={fpr:.3f}  recall={recall:.3f}  (n_eval={len(sc) - n_cal}, held-out)")

    lat = np.load(os.path.join(LAT_DIR, '10_bench_latency_memory.npz'))
    p16_ms, p8_ms = float(lat['P16_mean_ms']), float(lat['P8_mean_ms'])
    p16_flops, p8_flops = float(lat['P16_flops']), float(lat['P8_flops'])
    print(f"P16={p16_ms:.2f}ms/{p16_flops/1e9:.1f}GFLOPs  P8={p8_ms:.2f}ms/{p8_flops/1e9:.1f}GFLOPs")

    pis = np.linspace(0, 0.5, 200)  # 공격 이미지 비율 0~50%
    p_escalate = (1 - pis) * fpr + pis * recall

    def expected(base, extra, p_esc):
        return base + p_esc * extra

    ours_ms = expected(p16_ms, p8_ms, p_escalate)
    ours_flops = expected(p16_flops, p8_flops, p_escalate)
    naive_ms = p16_ms + p8_ms          # 매번 둘 다 돎 (π 무관 상수)
    naive_flops = p16_flops + p8_flops
    p8only_ms = np.full_like(pis, p8_ms)     # 참고: 아예 P8만 항상 쓰는 경우
    p8only_flops = np.full_like(pis, p8_flops)

    fig, (ax_ms, ax_flops) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle('PatchSwitch expected cost vs. attack prevalence (π)',
                 fontsize=17, fontweight='bold', y=1.0)

    for ax, ours, naive, p8only, unit, base in [
        (ax_ms, ours_ms, naive_ms, p8only_ms, 'ms/image (batch=1)', p16_ms),
        (ax_flops, ours_flops / 1e9, naive_flops / 1e9, p8only_flops / 1e9, 'GFLOPs/image', p16_flops / 1e9),
    ]:
        ax.plot(pis * 100, ours, color=RED, linewidth=2.5, label='PatchSwitch (detect + escalate)')
        ax.axhline(naive if unit.startswith('ms') else naive_flops / 1e9,
                   color=GRAY, linestyle='--', linewidth=1.5,
                   label='naive: always run P16+P8')
        ax.axhline(base, color=BLUE, linestyle=':', linewidth=1.5, label='P16 alone (no defense)')
        ax.fill_between(pis * 100, ours, base, color=RED, alpha=0.08)
        ax.set_xlabel('attack prevalence π (%)', fontsize=12)
        ax.set_ylabel(unit, fontsize=12)
        ax.set_title(unit, fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10, loc='upper left')

    crossover_ms = None
    if p8_ms > 0:
        # ours_ms(π) = naive_ms 가 되는 π: p16+p_esc(π)*p8 = p16+p8 -> p_esc(π)=1 -> 항상 naive보다 쌈
        pass

    out = os.path.join(HERE.replace('/defense/', '/results/', 1), 'expected_cost_vs_prevalence.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")

    print("\n=== 요약 (일부 π 지점) ===")
    for pi in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5):
        pe = (1 - pi) * fpr + pi * recall
        cms = p16_ms + pe * p8_ms
        print(f"  π={pi*100:5.1f}%  P(escalate)={pe*100:5.1f}%  "
              f"기대 latency={cms:.2f}ms ({cms/p16_ms:.2f}x P16, "
              f"naive={((p16_ms+p8_ms)/p16_ms):.2f}x)")


if __name__ == '__main__':
    main()
