# ViT_patchSwitch — ViT 적응형 방어(P16→P8 폴백) 프로젝트

**"P16으로 기본 추론하다가, 공격이 의심되는 이미지는 P8로 전환해 방어하는 적응형 ViT 방어"**를
설계·검증하는 독립 프로젝트다. 원래 [`ViT_tradeoff/`](../ViT_tradeoff/)(패치 크기 vs 강건성
7개 정식 실험 — PGD/LaVAN/PatchFool × P8/P16/P32)에서 나온 발견("PatchFool에 대해 P8이 P16보다
압도적으로 강건함")을 실제 방어로 발전시키기 위해 2026-09-15에 분리했다.

**범위 — ViT_tradeoff와 다른 점**: 이 프로젝트는 방어 메커니즘 자체(토큰화 격자 불일치)에
집중하므로 **PGD는 제외, patch size는 P8/P16만, 공격은 LaVAN·PatchFool만** 다룬다(P32는
방어 로직과 무관, PGD는 원래 실험에서도 전역 L∞라 RA가 전부 포화돼 분석 무의미했음).
모델은 ViT_tradeoff와 **완전히 동일한 방식으로** 가져오되([`src/models.py`](src/models.py)
— timm 체크포인트 이름까지 동일), 이 프로젝트 안에 **새로 복사된 파일**로 둬서 ViT_tradeoff에
전혀 의존하지 않는다.

## 두 가지 방어 메커니즘

탐지가 공격을 의심하면, P16 파이프라인을 어떻게 P8 강건성으로 전환할지에 대한 **두 가지 설계를
같은 세 단계 검증 프로토콜로 나란히 비교**한다:

| | **A. 전체 재분류** (`defense/full_reclassification/`) | **B. 국소 토큰 세분화** (`defense/local_token_subdivision/`) |
|---|---|---|
| 전환 방식 | 의심되면 이미지 **전체**를 P8로 다시 분류 | 의심되는 **패치 1개만** 4개의 P8 서브패치로 국소 교체 (196→200토큰), 나머지 195토큰·12개 block은 P16 그대로 재사용 |
| 비용 | latency 2.7배, FLOPs 4.5배 (P8 전체 재실행) | 사실상 추가 비용 없음 (패치 1개만 추가) |
| 정렬 메커니즘 | 없음 (완전히 별도 모델을 그대로 돌림) | closed-form 최소제곱 아핀 변환(768×768+bias)으로 patch_embed 레벨에서 P8 서브패치를 P16 좌표계에 투영 |

두 메커니즘 모두 동일한 3단계 검증을 거친다: **(1) 복원율 → (2) naive joint attack →
(3) 탐지 회피까지 포함한 완전판 adaptive evasion**. B는 A보다 훨씬 저렴하면서도 이 세 단계
전부에서 A와 통계적으로 구분되지 않는 결과를 보였다 — 상세 서사와 논문 반영용 문구는
[`NARRATIVE_16_17_18.md`](NARRATIVE_16_17_18.md) 참고.

## 핵심 성능 요약

| 지표 | A. 전체 재분류 | B. 국소 토큰 세분화 |
|---|---|---|
| 복원율 | **97.1%** (n=200, calibration/eval 분리) | **89.8%** (97/108, n=150 eval, 95% CI [82.7%, 94.2%]) |
| clean 정확도 손실 | 없음 | ~0% (오차범위 내) |
| 시스템 정확도 (naive attacker 기준) | 13.0%→64.0% (5배) | — (A와 같은 탐지기 재사용, §6 수치 그대로 적용됨) |
| naive joint attack 완전 무력화 | 18.4% (7/38, [9.2%, 33.4%]) | 16.7% (7/42, [8.3%, 30.6%]) |
| **완전판 adaptive evasion worst-case** | **15.8%** (6/38, [7.4%, 30.4%]) | **21.4%** (9/42, [11.7%, 35.9%], calibrated_threshold 기준; clean_max 기준 참고치 11.9%) |
| 배포 비용 | latency 2.7배 / FLOPs 4.5배; 기대비용 π=10%에서 1.30배 | 사실상 무시 가능 |

모든 adaptive evasion worst-case 신뢰구간이 서로 겹친다 — **국소 세분화가 전체 재분류보다
통계적으로 더 취약하다는 근거는 없다.** 두 메커니즘이 공유하는 탐지·위치특정 인프라(raw
attention L=12) 성능: AUROC 0.879(clean 대비)~0.891(LaVAN 대비), 위치특정 recall@1 96.7%.

## 디렉토리 구조

```
ViT_patchSwitch/
  src/                              ViT_tradeoff/src/에서 복사 + 범위에 맞게 정리
    models.py                         MODEL_NAMES에서 32 제외(P8/P16만)
    dataset.py                         원본과 동일
    attacks/{lavan.py, patch_fool.py}  원본과 동일 (pgd.py 없음 — 범위 밖)

  defense/                          실험 코드 (.py, .sh) — 결과물 없음
    full_reclassification/            A. 전체 재분류
      final_validation/                 §6  최종 시스템 검증 (FPR/recall/복원율)
      latency_memory/                   §10 배포 비용 (latency/FLOPs/memory)
      expected_cost_analysis.py         §6+§10 결합 기대비용(π) 분석
      joint_attack/                     §7  naive joint attack
      adaptive_evasion_full/            §14 완전판 adaptive evasion (탐지 회피 제약 포함)
    local_token_subdivision/          B. 국소 토큰 세분화
      recovery_test/                    §16 국소 교체 복원율
      joint_attack/                     §17 joint attack stress test (§8 정렬 트랩 회피 검증)
      adaptive_evasion_full/            §18 완전판 adaptive evasion (A와 동급 검증)
    experiments/                      두 메커니즘이 공유하는 인프라 + 보조 진단
      detection_localization/
        signature/                      §1  탐지 시그니처 + 레이어 스윕
        localization/                    §2  위치 특정
        evasion_robustness/               §3  탐지기 회피 시도 견고성
        layeridx_generalization/           §4  attn_layer_idx 일반화
      diversity_diagnostic/             §8  patch size 차이 vs "다른 모델" (정렬 트랩 진단)

  results/                           결과물(그림 .png, 원자료 .npz, 로그 .txt)만 — 코드 없음
                                      defense/와 완전히 같은 구조로 대응
```

`defense/.../*.py`가 결과를 저장할 때는 자기 파일 경로에서 `/defense/`를 `/results/`로 바꾼
경로에 쓴다(`HERE.replace('/defense/', '/results/', 1)`을 ROOT 탐색 기준으로 확장한 형태).
그래서 폴더를 옮겨도 결과 경로가 항상 자동으로 따라온다. 여러 실험이 같은 공격 로직(예: joint
attack)을 쓸 때는 모듈을 폴더마다 **복사**해서 넣었다 — 폴더 간 import를 없애서 폴더 하나만
통째로 옮기거나 지워도 다른 실험이 안 깨지게 하기 위함이다. 유일한 예외는 몇몇 `viz.py`가
다른 실험의 확정된 숫자를 (npz를 다시 열지 않고) 하드코딩된 상수로 인용하는 것
(예: `local_token_subdivision/*/`의 스크립트들이 `full_reclassification/`의 §6/§7/§14
수치를 참고용 상수로 가짐).

**결과 파일 이름은 `NN_설명.확장자`** 형식을 유지한다(예: `01_signature_P16.png`,
`16_local_swap_l12_viz.png`) — `NN`은 최초 설계 당시의 실험 번호이고, 이 문서와 코드 전체에서
그 번호로 실험을 지칭한다.

**2026-09-18 재구성**: 처음엔 순서대로 번호가 매겨진 하위 실험들의 나열로 구성됐던 프로젝트를,
**"전체 재분류 vs 국소 토큰 세분화"라는 하나의 비교축을 중심으로** 재편했다. 이 비교가 프로젝트의
핵심 기여이기 때문에 `defense/`/`results/` 바로 아래에는 이 두 메커니즘 폴더만 남기고, 둘이
공유하는 탐지 인프라와 부가 진단 실험(§1~4, §8)은 `experiments/`로 모았다. 예전에 탐색만
하고 종료된 방향(§5/§9/§11~13/§15의 부분 공유·전역 접합 실험)은 이미 삭제돼 있었고
(git 히스토리에서 복구 가능), 이번 재편에서도 다시 살리지 않았다.

## npz / 로그 정리 정책

**현재 살아있는 실험의 `.npz`는 전부 유지한다.** `viz.py`(또는 원본 실험 스크립트 자신)가
그림을 다시 그리는 데 쓰는 원자료라, 하나라도 지우면 그 실험의 그림을 재생성할 방법이 없어진다.

**`nohup_*.txt` 실행 로그는 대부분 삭제했다.** 숫자가 이미 README·npz·그림에 다 들어있어서
로그 자체는 중복이었던 것들은 지웠다. 예외 1개는 **그 실행 로그가 유일한 원자료라 보존**:
- `results/experiments/diversity_diagnostic/08_diversity_original_seed456_run_2143709.txt`
  — §8의 원래 seed=456 결과(52.3%)의 `.npz`가 나중에 seed=123 재실행 때 같은 파일명으로
  덮어써져서, 이 로그만 그 수치의 유일한 증거로 남음(아래 "샘플링 감사" 참고)

## A. 전체 재분류 (`defense/full_reclassification/`)

의심되면 이미지 전체를 P8로 다시 분류하는, 가장 단순하고 가장 강하게 검증된 설계.

- **§6 최종 검증** (`final_validation/`): 200장을 calibration 100/evaluation 100으로 분리,
  임계값은 calibration에서만, 성능(recall/FPR/복원율)은 evaluation에서만 계산 → **FPR 5.0%,
  탐지 recall 64.0%, 복원율 97.1%, 시스템 정확도 13.0%→64.0%(5배), clean 무손실**.
- **§10 배포 비용** (`latency_memory/`): batch=1, warm-up 이후 기준 latency/memory/FLOPs 측정
  → **P8이 latency 2.7배, FLOPs 4.5배 더 비쌈, 메모리는 거의 동일**.
- **기대 비용 분석** (`expected_cost_analysis.py`, §6+§10 결합, 새 GPU 실험 아님): 실제 배포
  시 P8은 탐지기가 flag한 이미지에서만 추가로 도니까, 시스템 전체의 기대 비용은 공격 이미지
  비율 π에 따라 `E[cost(π)] = P16_cost + [(1-π)·FPR + π·recall] · P8_cost`. π=1%에서 P16
  대비 **1.15배**, π=10%에서 **1.30배**, π=50%에서도 **1.95배**로, "매번 P16+P8 둘 다
  돈다"는 naive 대안(항상 3.74배)보다 항상 쌈.
- **§7 joint attack** (`joint_attack/`): P16+P8 손실을 합쳐 하나의 perturbation으로 동시
  최적화 → 나이브 전이(2.9%) 대비 joint attack은 **18.4%로 6배 위험**, 탐지기도 약해짐
  (joint attack의 20%만 flag).
- **§14 완전판 adaptive evasion** (`adaptive_evasion_full/`): joint attack 손실에 "탐지
  점수를 clean 범위 안으로 유지"하는 미분가능 페널티(STRAP-ViT류 설계)를 추가 → 회피 제약을
  걸어도 결과가 거의 그대로(18.4%) — 두 모델을 동시에 속이는 목표 자체가 이미 탐지 회피를
  "공짜로" 어느 정도 포함하고 있었다는 뜻. **최종 worst-case(무력화+미탐지) = 15.8%**
  (6/38, 95% CI [7.4%, 30.4%]), target_bound를 clean_max/calibrated_threshold 어느 쪽으로
  잡아도 동일하게 나옴.

## B. 국소 토큰 세분화 (`defense/local_token_subdivision/`)

의심되는 P16 패치 1개만 4개의 P8 서브패치로 국소 교체하는, 훨씬 저렴한 대안. closed-form
(최소제곱, 재학습 없음) 아핀 변환으로 patch_embed 레벨에서 P8 서브패치를 P16 좌표계로 투영하고,
기존 L=12 top-1 탐지기가 지목한 자리에 in-place로 끼워 넣는다. P16의 12개 transformer block은
전혀 수정하지 않는다.

전체 재분류가 거친 것과 **동일한 3단계 검증**을 거쳤다 — 상세 서사, 왜 이 세 단계가 각각
필요했는지, 논문/PPT용 문구는 [`NARRATIVE_16_17_18.md`](NARRATIVE_16_17_18.md)에 정리돼 있다.
요약:

- **§16 복원율** (`recovery_test/`, n=150 eval): clean acc 84.0%, 방어 없는 adv acc 12.0%
  → **복원율 89.8%**(97/108, 95% CI [82.7%, 94.2%]), clean 표본에 대한 부수 비용은 오차범위
  내(84.0%→84.7%).
- **§17 joint attack stress test** (`joint_attack/`): 정렬 메커니즘(아핀 변환)이 §8이 경고한
  "정렬된 표현=joint attack에 더 취약" 함정에 빠지는지 확인하기 위해 §7과 동일한 방법론을
  적용. 나이브 전이 2.5%(1/40) → **joint attack 완전 무력화율 16.7%**(7/42, 95% CI
  [8.3%, 30.6%]) — §7의 18.4%(다른 patch size)에 가깝고, §8의 74.4%(정렬된 표현)와는 한참
  떨어져 있다. **함정을 피했다.**
- **§18 완전판 adaptive evasion** (`adaptive_evasion_full/`): §14와 동일한 방법론(STRAP-ViT류
  제약 최적화)을 적용, target_bound를 clean_max/calibrated_threshold 두 가지로 각각 최적화.
  **calibrated_threshold 기준(공격자가 실배포 임계값을 직접 알고 최적화하는, 더 보수적인
  위협 모델) worst-case = 21.4%**(9/42, 95% CI [11.7%, 35.9%]) — 대표 수치.
  clean_max 기준으로는 11.9%(5/42, [5.2%, 25.0%], 참고치). §14의 15.8%와 신뢰구간이 크게
  겹쳐 **통계적으로 구분되지 않는다**.

**한계 (원인 미규명)**: 탐지기 flag율이 §17/§18(4%)에서 §7/§14(20%)보다 뚜렷이 낮다. 원인은
아직 분석하지 않았고, 열린 한계로 남겨둔다.

## 공유 인프라: 탐지 + 위치특정 (`defense/experiments/detection_localization/`)

A/B 두 메커니즘이 공통으로 의존하는 raw-attention 기반 탐지기의 신뢰성을 검증하는 섹션.

- **§1 탐지 가능성** (`signature/`): clean/LaVAN/PatchFool 각 30장, 층 1~12별 top-4 mass의
  AUROC 측정(rollout vs raw) → **L=12 raw attention이 최고, AUROC 0.879(clean 대비)~0.891
  (LaVAN 대비)**.
- **§2 위치 특정** (`localization/`): top-1(가장 attention 큰 토큰)이 실제 공격 토큰과
  일치하는 비율 → **recall@1 96.7%(29/30)**.
- **§3 회피 시도 3종** (`evasion_robustness/`): (1) 가장 낮은 saliency 위치를 강제 공격
  (2) attention/saliency 정규화한 새 탐지기 시도 (3) 실패 샘플들의 실제 saliency 범위로
  재타겟 → raw 탐지기는 전부 recall 0.900으로 안 뚫림, 대안(정규화) 탐지기만 recall 0.000으로
  자체 실패.
- **§4 attn_layer_idx 일반화** (`layeridx_generalization/`): 공격의 attn_layer_idx를
  1,2,4,6,8,10으로 바꿔가며 반복 → recall@1이 항상 0.900~0.967로 안정적.
- **탐지기는 LaVAN을 못 잡는다 (한계)**: §1의 npz를 재계산해 LaVAN vs Clean AUROC를 레이어별로
  따로 측정하면 L=1: 0.522, L=6: 0.434, **L=12(실제 방어가 쓰는 레이어): 0.363** —
  0.5(chance)보다 낮다. 즉 이 raw-attention 집중도 탐지기는 PatchFool에는 통하지만
  **LaVAN은 원리적으로 이 신호로 안 잡힌다**(§1의 "AUROC 0.891 LaVAN 대비"는 "PatchFool을
  LaVAN-or-clean 배경과 구분하는 능력"이지 "LaVAN 자체를 clean과 구분하는 능력"이 아니다 —
  혼동하기 쉬워서 명시해둠). LaVAN을 잡으려면 별도의 보완 게이트(후보: 패치별 patch_embed
  activation norm outlier)가 필요하며, 아직 구현·검증 안 됨.

## 보조 진단: Diversity diagnostic (`defense/experiments/diversity_diagnostic/`)

A/B 두 메커니즘의 방어력이 정확히 어디서 오는지("토큰화 구조가 다름" 자체인지, 그냥 "두 모델이
다름"인지)를 확인하고, B의 정렬 메커니즘이 밟을 수 있는 함정을 미리 특정해둔 §8 실험.

- **방법**: 같은 P16, 학습 레시피만 다른 두 번째 모델로 P16-A vs P16-B joint attack (§7과
  동일 이미지, seed=123으로 페어링 — 아래 "샘플링 감사" 참고).
- **결과**: 같은 patch size·다른 학습 = **74.4%**(29/39, [58.9%, 85.4%]) 뚫림 vs 다른 patch
  size(§7) = 18.4% → **방어력의 핵심은 "다른 patch size"이지 "다른 모델"이 아니다.** 동시에
  이 결과는 "표현을 정렬하면 joint attack에 취약해질 수 있다"는 구체적인 경고이기도 하다 —
  B(국소 토큰 세분화)가 도입하는 아핀 정렬이 바로 이 함정에 빠지는지를 검증한 것이 위 §17이다.

## 샘플링 감사

지금까지의 실험들이 매번 같은 고정 이미지 집합으로 평가됐는지(→ 비교가 공정한지), 아니면
실행마다 독립적으로 무작위 재샘플링했는지(→ 숫자 차이가 진짜 원인 때문인지 그냥 다른 표본
때문인지 불분명) 감사했다.

**메커니즘**: `src/dataset.py`의 `get_dataloader(seed, num_samples)`는 seed로 고정한
`torch.Generator`로 전체 데이터셋을 한 번 섞은 뒤 앞에서 `num_samples`개를 자른다. 그래서
**seed와 num_samples가 같으면 항상 같은 이미지가 같은 순서로 나온다**.

**감사 결과**:

| 비교 | 방식 | 판정 |
|---|---|---|
| `ViT_tradeoff`의 정식 실험 1~7, area-matched #4/#5 포함 | `experiments/main.py`가 loader를 P×attack 루프 밖에서 1회만 생성, 재사용 | ✅ 고정 이미지, 페어링됨 |
| §1 layer sweep, §4 attn_layer_idx 일반화 | 루프 밖에서 1회 호출 | ✅ 고정 이미지, 페어링됨 |
| **§8 vs §7** — "52.3% vs 18.4%" 반전 결과의 근거 | §8는 seed=456, §7는 seed=123 — **서로 다른 50장으로 비교되고 있었음** | ❌ 발견 → 재실행으로 수정 |

**수정** — §8을 §7과 같은 이미지(seed=123)로 재실행(job 2147607):

| 비교 (동일 50장, seed=123) | 둘 다 속음(완전 무력화) | 95% CI |
|---|---|---|
| P16-A vs P16-B (같은 patch size, 다른 학습) | **29/39 = 74.4%** | [58.9%, 85.4%] |
| P16 vs P8 (다른 patch size) | 7/38 = 18.4% | [9.2%, 33.4%] |

**결론: 숫자는 바뀌었지만(52.3%→74.4%) 결론은 안 바뀌었고 오히려 더 뚜렷해졌다.** 두 신뢰구간
겹침이 전혀 없어졌다. **논문/인용에는 페어링된 74.4%를 쓸 것.** 이후 §17/§18도 이 교훈을 따라
항상 동일 seed=123으로 §7/§8과 직접 비교 가능하게 실행했다.

## 한계 (논문 Limitations에 반영할 것)

- **탐지기 flag율 불일치, 원인 미규명**: B(§17/§18)의 탐지기 flag율(4%)이 A(§7/§14, 20%)보다
  뚜렷이 낮다. 원인은 분석하지 않았고, 열린 질문으로 남긴다.
- **LaVAN 비탐지**: raw-attention 탐지기는 LaVAN을 원리적으로 못 잡는다(AUROC 0.36~0.57,
  모든 레이어에서 chance 수준 이하). 별도 보완 게이트가 필요하며 아직 미구현.
- **표본 크기**: adaptive evasion 비교는 n=38~42 규모라 신뢰구간이 넓다(§17/§18 상한이 30%대).
  "A와 B가 통계적으로 구분되지 않는다"는 주장이지 "B가 더 낫다"는 주장이 아니다.
- **위협 모델 범위**: 검증한 공격은 LaVAN/PatchFool과 그 joint/adaptive 변형에 한정된 empirical
  보장이다. [PatchCleanser](https://www.usenix.org/conference/usenixsecurity22/presentation/xiang)
  같은 certified 방어와는 성격이 다르다는 점을 명시할 필요가 있다.

## 로드맵 / 다음 단계 후보

- B의 탐지기 flag율 4% vs A의 20% 차이 원인 분석 (우선순위 낮음, 의도적으로 보류 중)
- LaVAN용 보완 탐지 게이트 (후보: patch_embed activation norm outlier)
- 라우터를 L=6으로 당겼을 때의 recall/accuracy trade-off (L=6의 PatchFool vs Clean AUROC가
  0.639로 L=12의 0.879보다 낮아 recall 손실이 예상됨 — 아직 실행 안 함)
- 탐지 recall(64%)을 올리는 방법 — 현재 시스템의 실질적 병목
- 기존 baseline([PatchCleanser](https://www.usenix.org/conference/usenixsecurity22/presentation/xiang)
  등)을 같은 P8/16/32 세팅에 직접 돌려 비교 — 코드 공개돼 있고 아키텍처 무관이라 이식 쉬움
- (참고) [ViTGuard](https://arxiv.org/abs/2409.13828)는 attention+CLS token+MAE 재구성을
  결합한 탐지기를 7개 기존 detector·9개 attack과 비교해 검증한 바 있어, 이 프로젝트의
  raw-attention 탐지기는 그보다 단순한 버전이다.

## §9. Protocol C — ViT_tradeoff로 이동

면적 대신 토큰 개수를 P8/P16/**P32**에서 동일하게 고정하는 실험이라(P32 포함) 이 프로젝트
범위 밖으로 판단, [`ViT_tradeoff/09_protocol_c/`](../ViT_tradeoff/09_protocol_c/)로 옮겼다.
결과(P8 RA 22.0% vs P16/P32 0.0%)는 그쪽 README 참고.
