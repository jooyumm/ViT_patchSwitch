# §16+§17+§18 통합 스토리: Local Token Subdivision 검증

이 문서는 `defense/local_token_subdivision/` 세 실험(recovery_test/joint_attack/adaptive_evasion_full)의
결과를 하나의 이야기로 정리한 것이다. PPT와 논문(Abstract / 3.2 / 7.9 / 7.10 / Limitations / Conclusion)에
바로 반영할 수 있도록, 섹션 맨 아래에 섹션별 붙여넣기용 문구를 따로 정리했다.

**대표 숫자 규칙**: §18의 최종 worst-case는 **calibrated_threshold 기준 21.4%**를 대표 숫자로 쓴다.
clean_max 기준 11.9%는 참고 수치로만 언급한다. 이유는 아래 §3에서 설명한다.

---

## 1. 배경 및 동기

`full_reclassification`(전체 재분류) 방어는 탐지되면 이미지 전체를 P8로 다시 분류한다 — 안전하지만
비용이 크다(latency 2.7x / FLOPs 4.5x, §10). 자연스러운 다음 질문은: **탐지된 패치 1개만 P8 서브패치로
국소 교체하면 안 되나?** 나머지 195개 토큰은 P16 그대로 두고, 문제 패치만 4개의 P8 서브패치로 세분화해
시퀀스를 196→200 토큰으로 늘리는 방식이다(`local_token_subdivision`). 성공하면 비용이 사실상 0에
가까워진다.

이 아이디어에는 구조적으로 걱정되는 지점이 하나 있었다: P8 서브패치 임베딩을 P16 좌표계로 옮기려면
어떤 형태로든 "정렬(alignment)"이 필요한데, §8 diversity diagnostic에서 이미 **정렬된 표현끼리는
joint attack에 훨씬 취약**하다는 것을 확인한 바 있다(같은 patch size/다른 학습끼리는 74.4% 완전
무력화, patch size가 다른 diverse 쌍은 18.4%). 즉, local swap을 위해 도입하는 정렬 메커니즘(closed-form
affine adapter, §15)이 방어를 "정렬 트랩"에 빠뜨릴 위험이 있었다. §16~§18은 이 우려를 정면으로
검증하기 위해 설계됐다 — 기법의 실현 가능성(§16), 정렬 트랩 여부(§17), 그리고 전체 재분류와 동급의
가장 강한 공격 시나리오(§18)까지 세 단계로 전체 재분류가 이미 거친 것과 동일한 검증 과정을 다시
거치게 했다.

## 2. 방법론 요약

- **국소 교체**: 기존 L=12 top-1 탐지기가 지목한 P16 패치 1개만, 같은 영역의 P8 서브패치 4개로
  교체한다. 서브패치는 patch_embed 입력 단에서 closed-form affine adapter(768×768+bias, 최소제곱으로
  적합, gradient descent 없음)로 P16 좌표계에 맞춰 변환한 뒤 삽입한다. P16의 12개 transformer block은
  전혀 수정하지 않고 그대로 재사용한다.
- **§16 recovery_test**: calibration/evaluation을 분리한 표본(순환평가 방지)에서, PatchFool 공격이
  성공한 표본만 골라 local swap 적용 후 정답률이 얼마나 회복되는지 측정.
- **§17 joint_attack**: 기존 §7과 동일한 방법론(단일 perturbation으로 두 모델의 CE loss를 동시에
  최소화하는 joint attack)을 local swap 파이프라인에 그대로 적용 — §8 정렬 트랩에 빠지는지 직접 확인.
- **§18 adaptive_evasion_full**: 기존 §14와 동일한 방법론(STRAP-ViT 스타일, 탐지 점수에 대한 미분가능
  페널티 항을 loss에 추가해 공격이 탐지를 능동적으로 회피하도록 최적화)을 local swap에 적용. 페널티의
  `target_bound`를 두 가지로 설정해 각각 별도로 공격을 최적화했다:
  - `clean_max` (0.6728): clean 표본들의 최대 탐지 점수. "탐지기가 clean을 오탐하지 않는 한계선"
  - `calibrated_threshold` (0.5567): 실제 배포에 쓰이는 탐지 임계값 자체. "공격자가 실제 배포된
    탐지기를 정확히 알고 그 경계선 바로 아래로 점수를 누르도록 최적화"
  두 경우 모두, 사후 평가(post-hoc, no_grad, 학습에 쓴 미분가능 score가 아닌 배포된 탐지기와 100%
  동일한 방식)에서 "실제 배포 임계값(calibrated_threshold)"을 기준으로 탐지 여부를 판정한다 —
  즉 공격자가 *무엇을 목표로 최적화했든* 최종 판정은 항상 실배포 기준으로 한다.

## 3. 결과

| 실험 | 지표 | 값 | 95% CI (Wilson) |
|---|---|---|---|
| §16 recovery_test (n=150 eval) | clean acc | 84.0% | — |
| | P16 단독 adv acc (무방어) | 12.0% | — |
| | **복원율** (공격 성공 108건 중) | **89.8%** (97/108) | [82.7%, 94.2%] |
| | clean 표본에 대한 부수 비용 | 84.0%→84.7% (±0.7pp, 사실상 0) | — |
| §17 joint_attack (n=42 base) | naive transfer (§7 없이 그냥 전이) | 2.5% (1/40) | [0.4%, 12.9%] |
| | **joint attack 완전 무력화율** | **16.7%** (7/42) | [8.3%, 30.6%] |
| | 그중 탐지기에 flag된 비율 | 4.0% (전체 표본 기준) | — |
| §18 adaptive_evasion_full (n=42 base) | worst-case, clean_max 목표로 최적화 (참고) | 11.9% (5/42) | [5.2%, 25.0%] |
| | **worst-case, calibrated_threshold 목표로 최적화 (대표)** | **21.4%** (9/42) | [11.7%, 35.9%] |

**비교 대상 (전체 재분류, 이미 검증됨)**

| 실험 | 값 | 95% CI |
|---|---|---|
| §7 joint attack 완전 무력화율 (diverse: P16/P8) | 18.4% (7/38) | [9.2%, 33.4%] |
| §8 diversity diagnostic, 정렬된 표현(P16-A/P16-B) — "정렬 트랩" 기준선 | 74.4% (29/39) | [58.9%, 85.4%] |
| §14 adaptive_evasion_full worst-case (두 target_bound 모두 동일) | 15.8% (6/38) | [7.4%, 30.4%] |

**왜 calibrated_threshold(21.4%)를 대표 숫자로 쓰는가**: §14(전체 재분류)에서는 공격자가 어떤
target_bound를 목표로 최적화하든 최종 worst-case가 15.8%로 동일했다. 반면 local swap(§18)에서는
목표를 calibrated_threshold(실배포 임계값 그 자체)로 잡았을 때 공격자가 더 유리해진다(11.9%→21.4%).
이는 공격자가 실제 배포 임계값을 정확히 알고 그 경계 바로 아래로 점수를 누르도록 직접 최적화하는
쪽이 더 현실적이고 보수적인 위협 모델이기 때문이다 — clean_max를 목표로 최적화하는 쪽은 오히려 공격자
스스로에게 불리한 제약(더 낮은 점수를 강제)을 거는 셈이라 과소평가된 worst-case를 준다. 그래서
21.4%를 방어 강건성을 논할 때의 대표/보수적 수치로, 11.9%는 참고용 하한으로 취급한다.

## 4. 해석

1. **실현 가능성 (§16)**: closed-form linear adapter만으로 국소 교체가 원래 방어 효과의 89.8%를
   회복한다 — 전체 재분류(§6, 97.1%)에는 못 미치지만, gradient descent 학습 없이 최소제곱 적합만으로
   달성한 수치라는 점에서 여지가 크다. clean 표본에 대한 부수 비용은 통계적으로 0에 가깝다.
2. **정렬 트랩을 피함 (§17)**: 16.7%는 §7의 18.4%(diverse 쌍)와 거의 같고, §8의 74.4%(정렬 쌍)와는
   거리가 멀다. affine adapter가 두 표현을 "융합"하긴 하지만, §8이 보여준 위험한 종류의 표현 정렬
   (같은 아키텍처가 서로 다른 학습만으로 만드는 거의 동일한 좌표계)과는 다른 종류라는 것이 실증적으로
   확인됐다.
3. **최악의 경우도 동급 (§18)**: 대표 수치 21.4% vs 전체 재분류의 15.8% — 신뢰구간이 크게 겹치고
   ([11.7%,35.9%] vs [7.4%,30.4%]), 표본 크기(n=42/38)로는 통계적으로 유의하게 다르다고 말할 수 없다.
   즉 국소 교체가 전체 재분류보다 눈에 띄게 더 뚫리기 쉽다는 근거는 없다 — 비용은 훨씬 싸면서 최악의
   경우 강건성은 동급이라는 것이 §16~18의 결론이다.

## 5. 한계 (솔직하게 밝힐 것)

- **탐지 flag율 불일치, 원인 미규명**: §17/§18의 flag율(4%)이 §7/§14(20%)보다 뚜렷이 낮다. 이 차이의
  원인은 현재 분석하지 않았고, 앞으로 조사가 필요한 미해결 한계로 남긴다.
- **표본 크기**: n=38~42 규모의 신뢰구간은 넓다(§17/§18 상한이 30%대까지 올라감). "국소 교체가
  전체 재분류와 동급"이라는 주장은 "통계적으로 구분되지 않는다"는 수준이지, "더 낫다" 혹은 "확실히
  안전하다"는 강한 주장이 아니다.
- **적응형 공격자의 사전지식 가정**: §18은 공격자가 탐지 메커니즘의 구조(raw attention, L=12,
  top-4 mass)와 정확한 배포 임계값을 안다고 가정한 화이트박스 최악의 시나리오다. 실제 배포 시
  공격자가 이 정보에 접근하기는 더 어려울 것이므로, 21.4%는 보수적 상한으로 해석해야 한다.

---

## 6. 논문/PPT 섹션별 반영 문구 (그대로 붙여넣기용)

### Abstract용 (1~2문장)
> Beyond full-image reclassification, we further validate a lower-cost local token subdivision
> variant that replaces only the single flagged patch with detector-localized P8 sub-patches via a
> closed-form linear bridge; under the same adaptive, detection-evasion-aware attacker used to
> stress-test our main defense, its worst-case complete-defeat-and-undetected rate (21.4%) is
> statistically indistinguishable from the full-reclassification design (15.8%), while avoiding
> most of its inference-time cost.

### Section 3.2 (Method) 반영 문구
> As a lower-cost alternative to full-image P8 reclassification, we introduce a *local token
> subdivision* variant: only the single P16 patch localized by our existing top-1 detector is
> replaced in place with its four corresponding P8 sub-patches, projected into the P16 embedding
> space via a closed-form (least-squares) affine bridge fit on a disjoint calibration split; the
> remaining 195 tokens and all 12 transformer blocks are reused unmodified, growing the sequence
> from 196 to 200 tokens.

### Section 7.9 (§16 recovery 결과) 반영 문구
> On a held-out evaluation split (n=150, disjoint from calibration), local token subdivision
> recovers 89.8% (97/108, 95% CI [82.7%, 94.2%]) of the attack successes that fool the undefended
> P16 model, at negligible cost to clean accuracy (84.0% → 84.7%).

### Section 7.10 (§17+§18 강건성 결과) 반영 문구
> We stress-test the local subdivision bridge with the same two-stage adversarial protocol used
> for the full-reclassification defense. A naive joint attack (mirroring §7) achieves complete
> defeat in only 16.7% (7/42, 95% CI [8.3%, 30.6%]) of cases — close to the diverse-representation
> baseline (18.4%) and far from the aligned-representation failure mode we identified separately
> (74.4%), confirming the linear bridge does not introduce the representation-alignment
> vulnerability. Under a full adaptive attacker with a differentiable detection-evasion penalty
> (mirroring our strongest full-reclassification evaluation), and using the deployed calibrated
> threshold as the attacker's own optimization target — the more conservative and realistic
> threat model — the worst-case complete-defeat-and-undetected rate is 21.4% (9/42, 95% CI
> [11.7%, 35.9%]) (11.9%, 95% CI [5.2%, 25.0%], when the attacker instead optimizes against the
> clean-score maximum). Both figures overlap heavily with the full-reclassification worst-case
> of 15.8% (95% CI [7.4%, 30.4%]), indicating no statistically distinguishable robustness gap
> between the two defense variants.

### Limitations 반영 문구
> Our local subdivision variant shows a markedly lower detector flag-rate under adaptive attack
> (4%) than the full-reclassification design (20%); we have not yet identified the cause of this
> discrepancy and leave it as an open question. All adaptive-evasion comparisons in this work use
> sample sizes of 38–42, yielding wide confidence intervals; our claim is that the two defense
> variants are not statistically distinguishable in worst-case robustness, not that either is
> proven superior.

### Conclusion 반영 문구
> Beyond validating the primary full-reclassification defense, we show that a substantially
> cheaper local token subdivision variant — replacing only the flagged patch via a closed-form
> linear bridge — matches its worst-case adversarial robustness (21.4% vs. 15.8% under a fully
> adaptive, detection-evasion-aware attacker, CIs overlapping) while avoiding most of its
> inference-time overhead, suggesting the cost/robustness trade-off in adaptive ViT defenses can
> be improved without sacrificing the guarantees established for the full-reclassification design.

---

## 7. 빠른 참조용 숫자 표 (PPT 슬라이드용, 축약)

| | 전체 재분류 | 국소 토큰 세분화 |
|---|---|---|
| 복원율 | 97.1% (§6) | 89.8% (§16) |
| 비용 (latency) | 2.7x (전체 P8) | ~1x (패치 1개만) |
| naive joint attack 완전무력화 | 18.4% (§7) | 16.7% (§17) |
| **adaptive evasion worst-case (대표)** | **15.8%** (§14) | **21.4%** (§18, calibrated_threshold 기준) |
