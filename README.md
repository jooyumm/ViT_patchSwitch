# ViT_patchSwitch — ViT 적응형 방어(P16→P8 폴백) 프로젝트

**"P16으로 기본 추론하다가, 공격이 의심되는 이미지는 P8로 통째 재분류하는 적응형 방어"**를
설계·검증하는 독립 프로젝트다. 원래 [`ViT_tradeoff/`](../ViT_tradeoff/)(패치 크기 vs 강건성
7개 정식 실험 — PGD/LaVAN/PatchFool × P8/P16/P32)에서 나온 발견("PatchFool에 대해 P8이 P16보다
압도적으로 강건함")을 실제 방어로 발전시키는 후속 연구를 위해 2026-09-15에 분리했다.

**범위 — ViT_tradeoff와 다른 점**: 이 프로젝트는 방어 메커니즘 자체(토큰화 격자 불일치)에
집중하므로 **PGD는 제외, patch size는 P8/P16만, 공격은 LaVAN·PatchFool만** 다룬다(P32는
방어 로직과 무관, PGD는 원래 실험에서도 전역 L∞라 RA가 전부 포화돼 분석 무의미했음).
모델은 ViT_tradeoff와 **완전히 동일한 방식으로** 가져오되([`src/models.py`](src/models.py)
— timm 체크포인트 이름까지 동일), 이 프로젝트 안에 **새로 복사된 파일**로 둬서 ViT_tradeoff에
전혀 의존하지 않는다(그쪽 코드가 바뀌어도 여기는 안 깨짐, 여기를 통째로 지워도 그쪽은 안 깨짐).

**2026-09-16 재구성**: 원래 14개(§1~§14, 그 중 §5는 실험 아님·§9는 ViT_tradeoff로 이동이라
실질 12개)로 나뉘어 있던 실험을 8페이지 논문 분량에 맞춰 **5개 결과 섹션**으로 물리적으로
재편했다(2026-09-17에 §16을 추가하면서 6번째 섹션이 생겼다 — 아래 "6. 국소 토큰 세분화"
참고). 그리고 **코드(`defense/`)와 결과물(`results/`)을 최상위에서 분리**했다 — 전에는
실험 폴더 하나에 `.py`/`.sh`와 그림/npz가 섞여 있었는데, 이제 `defense/`에는 코드만,
`results/`에는 그림·npz·로그만 있고 두 트리는 `GG_그룹/NN_실험/` 경로가 완전히 같은 모양으로
대응한다(예: 코드는 `defense/01_detection_localization/02_localization/`, 그 결과물은
`results/01_detection_localization/02_localization/`). 그림은 억지로 하나로 합치지 않고
원래대로 실험별 개별 그림을 유지한다 — "합친다"는 여러 정보를 한 그래프 안에 같이 보여줄 수
있을 때만 의미가 있지, 이미 완성된 그림을 이어붙이는 건 별 의미가 없다고 판단해서 되돌렸다.
아래 "실험 목록"의 §번호는 재구성 이전 번호를 그대로 쓴다(원본 파일명·npz 키가 이 번호를
쓰고 있어서 바꾸지 않았다).

## 한줄 요약 + 핵심 성능

P16으로 효율적으로 추론하다가, attention 기반 탐지로 공격이 의심되는 이미지만 강건한 P8로
재분류하는 적응형 ViT 방어.

| 지표 | 값 |
|---|---|
| 탐지 (raw attention, L=12) | AUROC 0.879(clean 대비)~0.891(LaVAN 대비) |
| 위치 특정 | recall@1 96.7% (29/30) |
| 최종 시스템 (n=200, calibration/eval 분리) | FPR 5.0%, 탐지 recall 64.0%, 복원율 97.1%, **시스템 정확도 13.0%→64.0%(5배)**, clean 정확도 손실 없음 |
| Joint attack (방어를 아는 공격자) | 나이브 전이 2.9% → joint attack 18.4% (6배 위험) |
| 완전판 adaptive (탐지 회피 제약까지) | 18.4%로 동일, 최종 worst-case(무력화+미탐지) 15.8% |
| ⭐ Diversity diagnostic | 같은 patch size·다른 학습 74.4% 뚫림 vs 다른 patch size 18.4% → **방어력의 원천은 "다른 patch size"** |
| 배포 비용 | P8이 latency 2.7배, FLOPs 4.5배 (메모리는 거의 동일); 기대비용은 π=10%에서 1.30배 |
| 🧩 국소 토큰 세분화 (§16, 진행 중) | 의심 패치 1개만 P8 서브패치로 교체(196→200토큰, P8 전체 재실행 없음) → 복원율 84.6%, clean 오탐 비용 0% |

## 디렉토리 구조

```
ViT_patchSwitch/
  src/                            ViT_tradeoff/src/에서 복사 + 범위에 맞게 정리
    models.py                       MODEL_NAMES에서 32 제외(P8/P16만)
    dataset.py                       원본과 동일(전처리 상수는 P16 기준 그대로 유효)
    attacks/
      lavan.py                        원본과 동일
      patch_fool.py                    원본과 동일
      (pgd.py 없음 — 이 프로젝트 범위 밖)
  defense/                         실험 코드 (.py, .sh) — 결과물은 하나도 없음
    01_detection_localization/      §1+§2+§3+§4 — 탐지·위치특정·회피견고성·레이어일반화
      01_signature/                   §1
      02_localization/                 §2
      03_evasion_robustness/            §3
      04_layeridx_generalization/        §4
    02_system_validation/           §6+§10 — 최종 시스템 검증 + 배포 비용
      06_final_validation/            §6
      10_latency_memory/               §10
    03_adaptive_attack/             §7+§14 — naive/완전판 adaptive attacker
      07_joint_attack/                 §7
      14_adaptive_evasion_full/          §14
    04_diversity_diagnostic/        §8 — 핵심 반전 결과
      08_diversity_diagnostic/
    06_local_token_subdivision/     §16 — 국소 토큰 세분화 (진행 중, 2026-09-17 시작)
      16_local_swap_l12/                §16 1단계
  results/                         결과물(그림 .png, 원자료 .npz, 로그 .txt)만 — 코드 없음
                                    defense/와 완전히 같은 GG_그룹/NN_실험/ 구조로 대응
    01_detection_localization/{01_signature,02_localization,03_evasion_robustness,04_layeridx_generalization}/
    02_system_validation/{06_final_validation,10_latency_memory}/
    03_adaptive_attack/{07_joint_attack,14_adaptive_evasion_full}/
    04_diversity_diagnostic/08_diversity_diagnostic/
    06_local_token_subdivision/16_local_swap_l12/
```

**2026-09-17: §11~13(부분 공유 탐색)과 archive/(06_p8_rescue_test, 15_incompatibility_rigor_global_splice)를
전부 삭제했다** — 전부 이미 종료·대체된 방향이라 코드로서는 더 이상 필요 없다고 판단. 실험
자체의 결론(수치)은 아래 로드맵/§5 요약에 프로즈로 남겨뒀고, 원본 코드·npz·그림이 필요하면
git 히스토리(`git log --diff-filter=D -- defense/05_partial_share_exploration/`)에서 복구 가능.

`defense/GG_그룹/NN_실험/` 안의 스크립트가 결과를 저장할 때는 자기 파일 경로에서
`/defense/`를 `/results/`로 바꾼 경로에 쓴다(`HERE.replace('/defense/', '/results/', 1)`).
그래서 코드를 옮기면 결과가 자동으로 대응하는 `results/` 위치에 쓰이고, `results/` 트리에는
코드가 전혀 섞이지 않는다. `run_*.sh`의 `python defense/...` 호출 경로와 `#SBATCH --output=`
로그 경로도 이 분리에 맞춰 갱신했다.

여러 실험이 같은 공격 로직(예: joint attack)을 쓸 때는 모듈을 폴더마다 **복사**해서 넣었다
(`patch_fool_joint.py`가 `07_joint_attack/`과 `08_diversity_diagnostic/`에 각각 한 부씩) —
폴더 간 import를 없애서 폴더 하나만 통째로 옮기거나 지워도 다른 실험이 안 깨지게 하기 위함이다.
유일한 예외: `defense/04_diversity_diagnostic/`·`defense/03_adaptive_attack/14_adaptive_evasion_full/`의
`viz.py`가 `03_adaptive_attack/07_joint_attack`의 확정된 숫자를 (npz를 다시 열지 않고)
상수로 인용하는 것. (2026-09-17 이전엔 `archive/06_p8_rescue_test/`가 §1 산출물을 읽는
것도 예외였는데, 그 폴더 자체를 삭제하면서 없어짐 — 아래 참고)

**결과 파일 이름은 전부 `NN_설명.확장자`** 형식이다(예: `01_signature_P16.png`,
`08_diversity_diagnostic_headline.png`) — `NN`은 통합 이전의 원래 실험 번호와 정확히
대응한다. 실험 스크립트를 재실행해도 이 이름 그대로 저장된다.

§5, §9는 실험 번호가 비어있다 — §5는 실험이 아니라 설계 판단(국소 재분할 불가 확인)이고,
§9(Protocol C)는 P32를 포함하는 실험이라 이 프로젝트 범위 밖으로 판단해서
[`ViT_tradeoff/09_protocol_c/`](../ViT_tradeoff/09_protocol_c/)로 옮겼다(아래 §9 항목 참고).

이전 위치(`ViT_robust/probes/`, 재구성 전) 전체 백업: 재구성 전 probes/ 백업은
`/home/jooyumm/backups/probes_backup_20260903.tar.gz`, CISC-W'26 투고 끝날 때까지 보관 예정.
`ViT_robust→ViT_tradeoff+ViT_patchSwitch` 분리(2026-09-15) 시점엔 git 히스토리가 없었지만,
그 직후 이 프로젝트를 별도 git 저장소로 초기화했고([`github.com/jooyumm/ViT_patchSwitch`](https://github.com/jooyumm/ViT_patchSwitch)),
2026-09-16의 5섹션 재구성·`defense`/`results` 분리는 그 저장소 안에서 `git mv`로 진행해
히스토리가 보존돼 있다(문제 생기면 이전 커밋으로 되돌릴 수 있음).

## npz / 로그 정리 정책

**현재 살아있는(활성) 실험의 `.npz`는 전부 유지한다.** `defense/`의 `viz.py`(또는 원본 실험
스크립트 자신)가 그림을 다시 그리는 데 쓰는 원자료라, 하나라도 지우면 그 실험의 그림을
재생성할 방법이 없어진다. 5섹션 재구성·`defense`/`results` 분리 때도 npz는 전혀 손대지 않고
폴더만 옮겼다. **단, 2026-09-17에 이미 종료·대체된 방향(§11~13, archive/06_p8_rescue_test,
archive/15_incompatibility_rigor_global_splice)은 코드와 함께 npz/로그도 통째로 삭제했다** —
필요하면 git 히스토리에서 복구 가능(아래 "삭제된 방향" 참고).

**`nohup_*.txt` 실행 로그는 대부분 삭제했다.** 숫자가 이미 README·npz·그림에 다 들어있어서
로그 자체는 중복이었던 것들은 지웠다. 남아있는 예외 1개는 **그 실행 로그가 유일한 원자료라 보존**:
- `results/04_diversity_diagnostic/08_diversity_diagnostic/08_diversity_original_seed456_run_2143709.txt`
  — §8의 원래 seed=456 결과(52.3%)의 `.npz`가 나중에 seed=123 재실행 때 같은 파일명으로
  덮어써져서, 이 로그만 그 수치의 유일한 증거로 남음

## 배경 — 왜 이 조사를 시작했나

`ViT_tradeoff`의 정식 연구(실험 1~7)에서 PatchFool 공격에 대해 P8이 P16보다 압도적으로
강건하다는 걸 확인한 뒤(RA 67.6% vs 9.1%), 이걸 실제로 쓸 수 있는 방어로 만들 수 있는지
탐색했다. 아이디어: **"평소엔 효율 좋은 P16으로 추론하다가, attention 시그니처로 공격이
의심되면 그 이미지를 강건한 P8로 다시 분류한다."**

## 실험 목록 (5개 결과 섹션)

### 1. 탐지 + 위치특정 + 견고성 (§1+§2+§3+§4)
탐지기가 "공격 여부를 구분하고(§1) → 정확한 위치까지 찾아내고(§2) → 공격자가 일부러 피하려
해도 안 뚫리고(§3) → 특정 레이어에 과적합된 게 아닌지(§4)"를 하나의 질문("이 raw-attention
탐지기가 얼마나 견고한가")으로 묶은 섹션.

- **§1 탐지 가능성**: clean/LaVAN/PatchFool 각 30장, 층 1~12별 top-4 mass의 AUROC 측정(rollout
  vs raw) → **L=12 raw attention이 최고, AUROC 0.879(clean 대비)~0.891(LaVAN 대비)**
- **§2 위치 특정**: top-1(가장 attention 큰 토큰)이 실제 공격 토큰과 일치하는 비율 →
  **recall@1 96.7%(29/30)**, 놓친 1개는 11칸 떨어진 완전 실패
- **§3 회피 시도 3종**: (1) 가장 낮은 saliency 위치를 강제 공격 (2) attention/saliency 정규화한
  새 탐지기 시도 (3) 실패 샘플들의 실제 saliency 범위로 재타겟 → raw 탐지기는 전부 recall
  0.900으로 안 뚫림, 대안(정규화) 탐지기만 recall 0.000으로 자체 실패
- **§4 attn_layer_idx 일반화**: 공격의 attn_layer_idx를 1,2,4,6,8,10으로 바꿔가며 반복 →
  recall@1이 항상 0.900~0.967로 안정적
- **⚠️ 2026-09-17 추가 분석 — 탐지기는 LaVAN을 못 잡는다**: §1의 기존 npz(`01_layer_sweep_P16_raw.npz`,
  재실행 없이 재계산만)로 **LaVAN vs Clean** AUROC를 레이어별로 처음 계산해봤다. 결과가
  L=1에서 0.522, L=6 0.434, **L=12(지금 방어가 실제로 쓰는 레이어) 0.363** — 0.5(chance)보다
  낮다. 즉 이 raw-attention 집중도 탐지기는 PatchFool(모든 레이어가 공격 패치에 집중하도록
  만드는 공격)에만 통하고, **LaVAN은 원리적으로 이 신호로 안 잡힌다**(§1의 "AUROC 0.891
  LaVAN 대비"는 "PatchFool을 LaVAN-or-clean 배경과 구분하는 능력"이지 "LaVAN 자체를 clean과
  구분하는 능력"이 아니었음 — 이 둘을 혼동하기 쉬워서 명시해둠). 로드맵의 "아직 LaVAN
  테스트 안 함" 항목이 "테스트해보니 이 메커니즘으로는 안 됨, 별도 신호 필요"로 확정됨.
- **코드**: [`defense/01_detection_localization/`](defense/01_detection_localization/)
  (하위에 `01_signature/`, `02_localization/`, `03_evasion_robustness/`,
  `04_layeridx_generalization/`)
- **결과**: [`results/01_detection_localization/`](results/01_detection_localization/) —
  `01_signature/01_signature_P16.png`+`01_layer_sweep_P16.png`,
  `02_localization/02_localization_viz.png`,
  `03_evasion_robustness/03_evasion_attempts_summary.png`(+샘플 그림 3장),
  `04_layeridx_generalization/04_layeridx_generalization_P16.png`

### 2. 시스템 검증 + 배포 비용 (§6+§10)
"이 방어를 실제로 쓰면 정확도와 비용이 어떻게 되나"를 하나의 표/그림으로 묶은 섹션.

- **§6 최종 검증**: 200장을 calibration 100/evaluation 100으로 분리, 임계값은 calibration에서만,
  성능(recall/FPR/복원율)은 evaluation에서만 계산 → **FPR 5.0%, 탐지 recall 64.0%, 복원율
  97.1%, 시스템 정확도 13.0%→64.0%(5배), clean 무손실**. 초기 순환평가 편향 버전(임계값을
  정한 표본으로 그대로 평가해서 낙관적으로 편향됐던 파일럿, recall 76.7%로 더 높게 나왔었음)은
  `archive/06_p8_rescue_test/`에 보존해뒀었는데 2026-09-17에 삭제 — git 히스토리에서 복구 가능
- **§10 배포 비용**: batch=1, warm-up 이후 기준 latency/memory/FLOPs 측정 → **P8이 latency
  2.7배, FLOPs 4.5배 더 비쌈, 메모리는 거의 동일**
- **기대 비용 분석 (§6+§10 결합, 새 GPU 실험 아님)**: P8은 개별로는 2.7배/4.5배 비싸지만,
  실제로는 탐지기가 flag한 이미지에서만 추가로 도니까 시스템 전체의 **기대 비용**은
  공격 이미지 비율 π에 따라 `E[cost(π)] = P16_cost + [(1-π)·FPR + π·recall] · P8_cost`
  (FPR=5.0%, recall=64.0%, §6의 held-out eval 값 그대로). 계산해보면 π=1%에서 P16 대비
  **1.15배**, π=10%에서 **1.30배**, π=50%(비현실적으로 높음)에서도 **1.95배**로, "매번
  P16+P8 둘 다 돈다"는 naive 대안(항상 3.74배)보다 어떤 공격 비율에서도 확실히 쌈.
- **코드**: [`defense/02_system_validation/`](defense/02_system_validation/) (하위에
  `06_final_validation/`, `10_latency_memory/`, `expected_cost_analysis.py`)
- **결과**: [`results/02_system_validation/`](results/02_system_validation/) —
  `06_final_validation/06_final_validation_n200.png`,
  `10_latency_memory/10_bench_latency_memory_viz.png`,
  `expected_cost_vs_prevalence.png`

### 3. Adaptive attacker — naive와 완전판 (§7+§14)
"방어 구조를 아는 공격자"를 naive(§7)와 탐지 회피까지 명시적으로 노리는 완전판(§14)
두 단계로 나란히 보여주는 섹션.

- **§7 Joint attack**: P16+P8 손실을 합쳐 하나의 perturbation으로 동시 최적화 → 나이브
  전이(2.9%) 대비 joint attack은 **18.4%로 6배 위험**, 탐지기도 약해짐(joint attack의
  20%만 flag)
- **§14 완전판 adaptive attack**: joint attack 손실에 "탐지 점수를 clean 범위 안으로
  유지"하는 제약(STRAP-ViT류 설계) 추가 → **회피 제약을 걸어도 §7과 결과가 거의 동일
  (18.4%=18.4%)** — 두 모델을 동시에 속이는 목표 자체가 이미 탐지 회피를 "공짜로" 어느 정도
  포함. 최종 worst-case(무력화+미탐지) = **15.8%**
- **코드**: [`defense/03_adaptive_attack/`](defense/03_adaptive_attack/) (하위에
  `07_joint_attack/`, `14_adaptive_evasion_full/`)
- **결과**: [`results/03_adaptive_attack/`](results/03_adaptive_attack/) —
  `07_joint_attack/07_joint_attack_test_viz.png`,
  `14_adaptive_evasion_full/14_adaptive_evasion_full_viz.png`

### 4. Diversity diagnostic — patch size 차이 vs 그냥 "다른 모델" ⭐ 핵심 반전 결과
- **질문**: 방어력의 원천이 토큰화 구조 차이인지, 그냥 두 모델이 달라서인지?
- **방법**: 같은 P16, 학습 레시피만 다른 두 번째 모델로 P16-A vs P16-B joint attack (§7과
  동일 이미지, seed=123으로 페어링 — 아래 "샘플링 감사" 참고)
- **결과**: 같은 patch size, 다른 학습 = **74.4%** 뚫림 vs 다른 patch size(§7) = 18.4% →
  **방어력의 핵심은 "다른 patch size"이지 "다른 모델"이 아니다**
- **코드**: [`defense/04_diversity_diagnostic/08_diversity_diagnostic/`](defense/04_diversity_diagnostic/08_diversity_diagnostic/)
- **결과**: [`results/04_diversity_diagnostic/08_diversity_diagnostic/08_diversity_diagnostic_headline.png`](results/04_diversity_diagnostic/08_diversity_diagnostic/08_diversity_diagnostic_headline.png) ⭐

### 5. 부분 공유 탐색 — 실패, 종료된 방향 (§11+§12+§13+§15) — **코드 삭제됨 (2026-09-17)**
"뒷부분 layer를 P16/P8이 공유하면 체크포인트 2벌 문제를 풀 수 있지 않을까"를 검증하고
**완전히 접은** 섹션. §16(아래, 국소 토큰 세분화)으로 방향이 완전히 바뀌면서 코드·npz·그림을
전부 지웠다 — 아래는 결론만 남긴 요약이고, 원본이 필요하면 git 히스토리에서 복구 가능
(`git log --diff-filter=D --summary -- defense/05_partial_share_exploration/`).

- **§11 Activation 유사도**: CKA로 6·9번째 층의 표현 유사도 비교 → CKA 0.87~0.96(우연 수준
  0.11~0.38보다 훨씬 높음) — 공유해도 될 것 같다는 신호였음
- **§12 부분 공유 시제품**: patch_embed+앞 5층 독립, 뒤 7층+head 공유 하이브리드 제작 →
  branch8 clean accuracy **0%로 완전 붕괴**
- **§13 LayerNorm 재보정**: 재학습 없이 gamma/beta만 closed-form 재계산 → **재보정 전후
  모두 0%**
- **§15 §12 재검증(confound 제거, 2026-09-16, job 2263768, n=50)**: §12의 0% 붕괴가
  시퀀스 길이 문제(785 vs 197토큰) 때문일 수 있다는 의심을 확인 — 풀링으로 길이를 맞춰도
  여전히 0%(진짜 표현 불일치가 맞았음), **그런데 최소제곱 선형 변환 하나(768×768 아핀)만
  접합부에 끼우면 84.0%로 완전 회복**(sanity와 동일). §13(스케일/이동만 보정)은 실패, 이
  풀랭크 변환(회전/혼합 포함)은 성공 — 두 표현은 비선형이 아니라 **선형적으로 재배열된
  같은 정보**였다는 뜻. **§5/§12/§13의 "물리적으로 불가능" 결론을 재검토하게 만든 발견.**
  이게 실제 국소 전환 메커니즘(§16)으로 이어짐 — 다만 §8이 경고한 "정렬된 표현=joint
  attack에 더 취약" 함정에 §16이 빠지는지는 아직 별도 검증 필요(§16 항목 참고).

### 6. 국소 토큰 세분화 (§16, 진행 중) — 2026-09-17 시작

§15가 "선형 변환 하나면 P8/P16 표현이 완전히 이어진다"를 **전역**(이미지 전체)으로 보여준
직후 나온 후속 아이디어: 의심되는 **P16 패치 1~2개만** 국소적으로 2×2=4개의 P8 서브패치로
쪼개서 그 자리에 끼워 넣고(196→199~200토큰), 나머지는 그대로 P16으로 처리하면 어떨까 —
§6(전체 재분류)보다 훨씬 싸게 같은 효과를 낼 수 있을지도 모른다는 제안.

**§16 — 1단계: 국소 교체가 원리적으로 되는가 (라우터는 일단 L=12 고정)**
- **방법**: §15와 같은 닫힌 형태(최소제곱, 재학습 없음) 아핀 변환을, 이번엔 block-5 활성값이
  아니라 **입력(patch_embed) 레벨**에서 P8 서브패치 → 대응 P16 패치로 피팅. §2와 동일한
  L=12 raw attention top-1로 의심 패치를 찾은 뒤, 그 자리의 P16 임베딩을 어댑터로 보정한
  P8 서브패치 4개로 in-place 교체하고, **P16의 12개 레이어를 전혀 안 건드리고 그대로** 통과.
- **결과 (n=50, 2026-09-17 클러스터 실행, job 2275197)** ⭐: sanity max diff=0.0(배선 정상),
  clean acc=84.0%, 방어 없는 adv acc=6.0%(공격 성공 39/50) → **국소 교체 복원율 84.6%(33/39)**,
  **clean 오탐 비용 = 0.000p(전혀 없음)**. §6(전체 재분류, 복원율 97.1%)보다는 낮지만,
  이쪽은 **P8을 통째로 안 돌리고 196→200토큰만 바꿔서** 얻은 수치라 훨씬 싸다 — 그리고
  clean 이미지에 잘못 발동해도 정확도 손실이 전혀 없다는 게 특히 고무적(§6의 전체 재분류는
  오탐 시에도 어쨌든 다른 모델로 재분류하니 이론적으로 약간의 부작용 여지가 있었음).
  **1단계 결론: 국소 교체는 원리적으로 확실히 작동한다.**
- **코드**: [`defense/06_local_token_subdivision/16_local_swap_l12/`](defense/06_local_token_subdivision/16_local_swap_l12/)
- **결과**: [`results/06_local_token_subdivision/16_local_swap_l12/16_local_swap_l12_viz.png`](results/06_local_token_subdivision/16_local_swap_l12/16_local_swap_l12_viz.png) ⭐

**아직 다루지 않은 것 (다음 단계 후보)**:
- 라우터를 L=6으로 당겨서(잔여 레이어 7~12로 "치유"할 시간을 줌) 정확도가 얼마나 떨어지는지
  — L=6의 PF vs Clean AUROC가 0.639(L=12의 0.879보다 낮음)라 recall 손실이 예상됨
- LaVAN용 보완 게이트 (위 §1 항목 참고, patch_embedding norm outlier 후보)
- (§15와 동일 경고) 이 로컬 브릿지가 §8이 보여준 "정렬된 표현=joint attack에 취약" 함정에
  빠지는지 adaptive attacker로 스트레스 테스트 필요

### §9. Protocol C — ViT_tradeoff로 이동
면적 대신 토큰 개수를 P8/P16/**P32**에서 동일하게 고정하는 실험이라(P32 포함) 이 프로젝트
범위 밖으로 판단, [`ViT_tradeoff/09_protocol_c/`](../ViT_tradeoff/09_protocol_c/)로 옮겼다.
결과(P8 RA 22.0% vs P16/P32 0.0%)는 그쪽 README 참고.

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
| `ViT_tradeoff`의 정식 실험 1~7, area-matched #4/#5 포함 | `experiments/main.py`가 loader를 P×attack 루프 밖에서 1회만 생성, 재사용 | ✅ 고정 이미지, 페어링됨 (3 seed는 분산 추정 목적, 문제 아님) |
| §1 layer sweep, §4 attn_layer_idx 일반화 | 루프 밖에서 1회 호출 | ✅ 고정 이미지, 페어링됨 |
| **§8 vs §7** — "52.3% vs 18.4%" 반전 결과의 근거 | §8는 seed=456, §7는 seed=123 — **서로 다른 50장으로 비교되고 있었음** | ❌ 발견 → 재실행으로 수정 |

**수정** — §8을 §7과 같은 이미지(seed=123)로 재실행(`run_diversity_test_paired.sh`, job 2147607):

| 비교 (동일 50장, seed=123) | 둘 다 속음(완전 무력화) | 95% CI |
|---|---|---|
| P16-A vs P16-B (같은 patch size, 다른 학습) | **29/39 = 74.4%** | [58.9%, 85.4%] |
| P16 vs P8 (다른 patch size) | 7/38 = 18.4% | [9.2%, 33.4%] |

**결론: 숫자는 바뀌었지만(52.3%→74.4%) 결론은 안 바뀌었고 오히려 더 뚜렷해졌다.** 두 신뢰구간
겹침이 전혀 없어졌다. **논문/인용에는 페어링된 74.4%를 쓸 것.**

## 로드맵

### 재검토 중 (REOPENED, 2026-09-16): 공유 backbone + fusion 설계
CrossViT류로 "backbone 하나 공유 + patch_embed만 P별로 따로" 만들어서 리소스 문제(체크포인트
2벌=메모리 2배)와 국소 재분할 불가 문제를 동시에 풀려던 계획이었음.

**Diversity diagnostic(§8) 결과와 정면 충돌해서 처음엔 보류했다** — 정확히 그런 정렬 상태
(같은 토큰 그리드, 다른 가중치)를 흉내낸 실험(P16-A vs P16-B)이 오히려 joint attack에 훨씬
취약했다(74.4% vs 18.4%, 페어링된 값).

그 뒤 §11에서 "layer 6+는 어차피 표현이 정렬돼 있다(CKA 0.87~0.96)"는 신호가 나와서, "초반은
독립, 후반만 공유"하는 절충안을 실제로 작게 구현해서 시험했다(§12, §13). **결과: 재학습
없이는 완전히 망가진다** — branch8 clean accuracy 0.000%, LayerNorm 재보정해도 그대로 0.000%.
**→ 이때는 이 방향을 종료로 판단했다.**

**2026-09-16, §15에서 뒤집힘**: §12의 "0% 붕괴"가 시퀀스 길이 confound(785 vs 197토큰) 때문일
수도 있다는 지적이 있어서 재검증했더니 — confound를 제거해도(pooled 조건) 여전히 0%로
붕괴해서 "진짜 표현 불일치"라는 원래 결론은 맞았지만, **접합부에 최소제곱으로 구한 선형
변환(768×768 아핀) 하나만 끼우면 84.0%(sanity와 완전 동일)로 완전히 회복됐다**(n=50,
job 2263768). 즉 §13이 시도한 "스케일/이동만 보정"(LayerNorm)은 부족했고, "회전/채널
혼합까지 포함한 선형 변환"은 충분했다 — **두 표현은 비선형적으로 다른 게 아니라 선형적으로
재배열된 같은 정보를 담고 있었다는 뜻**. 이건 "재학습 없이는 완전히 망가진다"는 §12/§13의
결론을 정면으로 재검토해야 한다는 강한 신호다.

**그래서 "종료"를 "재검토 중"으로 되돌린다.** 다만 실제로 이걸 방어 메커니즘(예: 공격받은
부위만 국소 전환)으로 발전시키기 전에 반드시 확인해야 할 위험이 있다: **§8이 경고한 바로
그 함정**(같은 토큰 그리드로 정렬된 두 표현은 joint attack에 오히려 더 취약, 74.4% vs
18.4%)이 여기서도 재현될 수 있다 — 지금 이 선형 adapter는 정확히 "두 모델을 하나의 공유
좌표계로 이어주는 다리"라서, adaptive attacker가 그 다리(선형 변환)를 알면 두 모델을 동시에
속이기가 오히려 **쉬워질 수도** 있다. 그래서 다음 단계는 곧바로 "국소 전환 방어 구현"이
아니라, **"선형 adapter로 이어붙인 하이브리드 모델에 joint attack을 걸어봤을 때 얼마나
버티는지"부터 확인**하는 것이어야 한다(§12가 원래 하려던 것과 같은 stress test, 이번엔
adapter가 있는 버전으로). 이건 아직 안 돌렸다.

### 확인된 것 (재확인 불필요)
- 탐지(L=12 raw attention) + localization(top-1) 메커니즘은 견고함, 레이어 선택에 안 흔들림
- naive attacker 기준 방어는 강하게 작동(정확도 13%→64%, clean 무손실)
- adaptive(joint) attacker에게는 확실히 약해짐(2.9%→18.4%) — 그래도 patch size 차이가
  없었으면 훨씬 더 약했을 것(페어링된 비교 기준 74.4%)
- **탐지기 존재까지 아는 완전판 adaptive attacker(§14)도 무력화율을 못 올림** — 18.4%가
  naive/완전판 공격 둘 다에서 사실상 상한. "완전 무력화+미탐지" = 15.8%가 최종 worst-case

### 아직 열려있는 질문 (다음에 논의)
- 탐지 recall(64%)을 어떻게 올릴지 — 현재 시스템의 실질적 병목
- 체크포인트 2벌(메모리 2배) 문제는 여전히 미해결 — 부분 공유는 막혔으니 다른 방식(예: P8
  쪽을 더 작은/경량 모델로 바꾸는 것)이 필요하면 처음부터 새로 찾아야 함
- joint attack의 15.8%(완전 무력화+미탐지)를 더 낮출 수 있는 탐지/전환 전략이 있는지 — §14에서
  명시적 회피 목표를 추가해도 안 낮아졌으니, 단순 임계값 튜닝보다 근본적인 변화(예: L=12 하나가
  아니라 여러 레이어 앙상블 탐지)가 필요할 수 있음
- **[해결(부분): LaVAN은 안 잡힌다는 게 확인됨]** §1 raw-attention 탐지기는 LaVAN vs Clean
  AUROC가 모든 레이어에서 0.36~0.57(§1 항목 참고) — 재학습 없이 신호를 바꿀 수도 없으니,
  LaVAN을 잡으려면 **별도의 보완 게이트**가 필요하다(후보: 패치별 patch_embed activation
  norm outlier — 추가 forward pass 없이 이미 계산되는 값이라 값싸다). 아직 구현·검증 안 됨
- (참고, 다른 최신 연구와 비교) [ViTGuard](https://arxiv.org/abs/2409.13828)가 이미
  attention+CLS token+MAE 재구성을 결합한 탐지기를 7개 기존 detector·9개 attack과 비교해
  검증한 바 있어, §1의 raw-attention 탐지기는 이것보다 단순한 버전이다. [PatchCleanser](https://www.usenix.org/conference/usenixsecurity22/presentation/xiang)
  같은 certified 방어와 달리 이 프로젝트의 보장은 테스트한 공격(LaVAN/PatchFool)과 §7/§14의
  adaptive attacker에 한정된 empirical 보장이라는 점도 논문에 명시할 필요 있음. 다음 단계로는
  기존 방법(특히 PatchCleanser, 코드 공개돼 있고 아키텍처 무관이라 이식 쉬움) 하나를 같은
  P8/16/32 세팅에 baseline으로 돌려 직접 비교하는 게 가장 값싸게 설득력을 올리는 방법
