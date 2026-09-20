"""
experiments/attack_type_comparison/attack_type_comparison_viz.py — 공격 유형별(PatchFool vs
LaVAN) 탐지 성능과 방어 효과를 한 그림에 비교한다. 새 GPU 실험이 아니라 기존 npz 2개를 읽어서
계산만 하는 분석 스크립트다(paper_summary.py와 동일한 성격):

  - results/detection_localization/signature/01_layer_sweep_P16_raw.npz
      (§1 원자료: clean/LaVAN/PatchFool 각 30장의 레이어별 top4_mass raw score)
      여기서 L=12 raw score로 AUROC(PatchFool vs Clean), AUROC(LaVAN vs Clean)를 직접
      재계산한다(README에 이미 인용된 0.879/0.363을 하드코딩하지 않고 원자료에서 재현 — 재현
      확인 결과 소수점까지 일치함).
  - results/system_comparison/system_comparison_n250.npz
      (PatchFool에 대한 무방어/all_switch/local_switch 시스템 정확도 — 8.7%/66.0%/65.3%)

LaVAN에 대해서는 이 프로젝트에서 "방어 적용 후" 정확도를 측정한 적이 없다(system_comparison은
PatchFool만 공격으로 씀) — AUROC가 chance 이하(0.363)라 탐지기가 원리적으로 LaVAN을 못 잡고,
그 결과 escalate 로직이 사실상 무작위로만 발동해 방어 효과가 없다는 것이 핵심 주장이므로, 없는
수치를 만들어내지 않고 "정량 측정 안 됨 / 무방어와 동일" 텍스트로만 표시한다.

사용법:
  python attack_type_comparison_viz.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'src')):
    ROOT = os.path.dirname(ROOT)
RESULTS = HERE.replace('/experiments/', '/results/', 1)

GREEN, RED_FAIL = '#16A34A', '#DC2626'
GRAY, RED_ALL, BLUE_LOCAL = '#6B7280', '#E94B3C', '#3B82F6'


def auroc(pos, neg):
    """Mann-Whitney U 기반 AUROC (sklearn 없이). pos/neg: 이상치 점수 1D 배열."""
    n1, n2 = len(pos), len(neg)
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    rank_pos = ranks[:n1].sum()
    u = rank_pos - n1 * (n1 + 1) / 2
    return u / (n1 * n2)


def main():
    sig = np.load(os.path.join(ROOT, 'results/detection_localization/signature/01_layer_sweep_P16_raw.npz'))
    pf, clean, lavan = sig['12_raw_pf_top4'], sig['12_raw_clean_top4'], sig['12_raw_lavan_top4']
    auroc_pf = auroc(pf, clean)
    auroc_lavan = auroc(lavan, clean)
    print(f"AUROC PatchFool vs Clean (L=12, raw, n={len(pf)}/{len(clean)}): {auroc_pf:.4f}")
    print(f"AUROC LaVAN vs Clean     (L=12, raw, n={len(lavan)}/{len(clean)}): {auroc_lavan:.4f}")

    sysd = np.load(os.path.join(ROOT, 'results/system_comparison/system_comparison_n250.npz'))
    p16_acc = float(sysd['p16_only_acc']) * 100
    all_acc = float(sysd['sys_acc_all']) * 100
    local_acc = float(sysd['sys_acc_local']) * 100
    print(f"PatchFool system accuracy: P16 {p16_acc:.1f}% -> all_switch {all_acc:.1f}% / "
          f"local_switch {local_acc:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- (a) Detection AUROC: PatchFool vs LaVAN, chance-level reference ----
    ax = axes[0]
    cats = ['PatchFool\nvs Clean', 'LaVAN\nvs Clean']
    vals = [auroc_pf, auroc_lavan]
    colors = [GREEN, RED_FAIL]
    bars = ax.bar(cats, vals, color=colors, alpha=0.88, width=0.55)
    ax.axhline(0.5, color=RED_FAIL, linestyle='--', linewidth=1.5, zorder=0)
    ax.text(1.48, 0.52, 'chance level (0.5)', color=RED_FAIL, fontsize=10, ha='right', style='italic')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f'{v:.3f}', ha='center',
                fontweight='bold', fontsize=13, color=bar.get_facecolor())
    ax.text(1, auroc_lavan - 0.06, 'below chance\nfundamentally\nundetectable', ha='center', va='top',
            fontsize=9.5, color='white', fontweight='bold', fontstyle='italic')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('AUROC (L=12 raw attention)')
    ax.set_title('(a) Detection performance by attack type', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25)

    # ---- (b) Defense effect: PatchFool measured, LaVAN 'not measured / same as undefended' ----
    ax = axes[1]
    x_pf = np.array([0, 1, 2])
    pf_vals = [p16_acc, all_acc, local_acc]
    pf_colors = [GRAY, RED_ALL, BLUE_LOCAL]
    pf_labels = ['Undefended\n(P16)', 'all_switch\napplied', 'local_switch\napplied']
    bars = ax.bar(x_pf, pf_vals, color=pf_colors, alpha=0.88, width=0.6)
    for bar, v in zip(bars, pf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f'{v:.1f}%', ha='center',
                fontweight='bold', fontsize=12, color=bar.get_facecolor())

    # LaVAN: hatched placeholder instead of a bar, no fabricated number
    x_lav = 3.3
    ax.bar([x_lav], [100], width=0.6, facecolor='none', edgecolor=RED_FAIL, hatch='//', linewidth=1.2)
    ax.text(x_lav, 50, 'not measured\n(detection inactive\n-> defense never\ntriggers)',
            ha='center', va='center', fontsize=9.5, color=RED_FAIL, fontweight='bold')

    ax.set_xticks(list(x_pf) + [x_lav])
    ax.set_xticklabels(pf_labels + ['LaVAN\n(defense inactive)'], fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel('System accuracy (%)')
    ax.set_title('(b) Defense effect: undefended vs. defended', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    ax.axvline(2.65, color='#D1D5DB', linewidth=1, linestyle=':')
    ax.text(1, 100, 'PatchFool (measured, system_comparison)', ha='center', fontsize=9, color=GRAY)

    fig.suptitle('Detection & defense by attack type — PatchFool is caught and defended;\n'
                  'LaVAN evades the shared detector entirely (same for all_switch and local_switch)',
                  fontsize=12.5, fontweight='bold', y=1.04)
    fig.tight_layout()

    os.makedirs(RESULTS, exist_ok=True)
    out_png = os.path.join(RESULTS, 'attack_type_comparison_viz.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_png}")

    np.savez(os.path.join(RESULTS, 'attack_type_comparison.npz'),
             auroc_patchfool=auroc_pf, auroc_lavan=auroc_lavan,
             p16_only_acc=p16_acc, all_switch_acc=all_acc, local_switch_acc=local_acc)
    print(f"Saved: {os.path.join(RESULTS, 'attack_type_comparison.npz')}")


if __name__ == '__main__':
    main()
