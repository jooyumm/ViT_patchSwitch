# ViT_patchSwitch — ViT 적응형 방어(P16→P8 폴백) 프로젝트

**"P16으로 기본 추론하다가, 공격이 의심되는 이미지는 P8로 통째 재분류하는 적응형 방어"**를
설계·검증하는 독립 프로젝트다. 원래 [`ViT_tradeoff/`](../ViT_tradeoff/)(패치 크기 vs 강건성
8개 정식 실험 — PGD/LaVAN/PatchFool × P8/P16/P32)에서 나온 발견("PatchFool에 대해 P8이 P16보다
압도적으로 강건함")을 실제 방어로 발전시키는 후속 연구를 위해 2026-09-15에 분리했다.

**범위 — ViT_tradeoff와 다른 점**: 이 프로젝트는 방어 메커니즘 자체(토큰화 격자 불일치)에
집중하므로 **PGD는 제외, patch size는 P8/P16만, 공격은 LaVAN·PatchFool만** 다룬다(P32는
방어 로직과 무관, PGD는 원래 실험에서도 전역 L∞라 RA가 전부 포화돼 분석 무의미했음).
모델은 ViT_tradeoff와 **완전히 동일한 방식으로** 가져오되([`src/models.py`](src/models.py)
— timm 체크포인트 이름까지 동일), 이 프로젝트 안에 **새로 복사된 파일**로 둬서 ViT_tradeoff에
전혀 의존하지 않는다(그쪽 코드가 바뀌어도 여기는 안 깨짐, 여기를 통째로 지워도 그쪽은 안 깨짐).

**2026-09-16 재구성**: 원래 14개(§1~§14, 그 중 §5는 실험 아님·§9는 ViT_tradeoff로 이동이라
실질 12개)로 나뉘어 있던 실험을 8페이지 논문 분량에 맞춰 **5개 결과 섹션**으로 물리적으로
재편했다. 그리고 **코드(`defense/`)와 결과물(`results/`)을 최상위에서 분리**했다 — 전에는
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
| 배포 비용 | P8이 latency 2.7배, FLOPs 4.5배 (메모리는 거의 동일) |

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
    05_partial_share_exploration/   §11+§12+§13 — 부분 공유 탐색 (실패, 종료된 방향)
      11_activation_similarity/         §11
      12_partial_share_prototype/        §12
      13_ln_recalibration/                §13
  results/                         결과물(그림 .png, 원자료 .npz, 로그 .txt)만 — 코드 없음
                                    defense/와 완전히 같은 GG_그룹/NN_실험/ 구조로 대응
    01_detection_localization/{01_signature,02_localization,03_evasion_robustness,04_layeridx_generalization}/
    02_system_validation/{06_final_validation,10_latency_memory}/
    03_adaptive_attack/{07_joint_attack,14_adaptive_evasion_full}/
    04_diversity_diagnostic/08_diversity_diagnostic/
    05_partial_share_exploration/{11_activation_similarity,12_partial_share_prototype,13_ln_recalibration}/
  archive/
    06_p8_rescue_test/              §6의 superseded 초기 버전, 재현 가능하게 보존
                                     (예외적으로 코드+결과물이 한 폴더에 그대로 있음 — 이미
                                     동결된 archive라 defense/results 분리 대상에서 제외)
```

`defense/GG_그룹/NN_실험/` 안의 스크립트가 결과를 저장할 때는 자기 파일 경로에서
`/defense/`를 `/results/`로 바꾼 경로에 쓴다(`HERE.replace('/defense/', '/results/', 1)`).
그래서 코드를 옮기면 결과가 자동으로 대응하는 `results/` 위치에 쓰이고, `results/` 트리에는
코드가 전혀 섞이지 않는다. `run_*.sh`의 `python defense/...` 호출 경로와 `#SBATCH --output=`
로그 경로도 이 분리에 맞춰 갱신했다.

여러 실험이 같은 공격 로직(예: joint attack)을 쓸 때는 모듈을 폴더마다 **복사**해서 넣었다
(`patch_fool_joint.py`가 `07_joint_attack/`과 `08_diversity_diagnostic/`에 각각 한 부씩) —
폴더 간 import를 없애서 폴더 하나만 통째로 옮기거나 지워도 다른 실험이 안 깨지게 하기 위함이다.
유일한 예외 둘(둘 다 코드 주석으로 명시돼 있음): `archive/06_p8_rescue_test/`가
`results/01_detection_localization/01_signature/`의 산출물을 읽는 것, 그리고
`defense/04_diversity_diagnostic/`·`defense/03_adaptive_attack/14_adaptive_evasion_full/`의
`viz.py`가 `03_adaptive_attack/07_joint_attack`의 확정된 숫자를 (npz를 다시 열지 않고)
상수로 인용하는 것.

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

**`.npz`는 전부 유지했다.** 지금 있는 npz는 전부 `defense/`의 `viz.py`(또는 원본 실험 스크립트
자신)가 그림을 다시 그리는 데 쓰는 원자료라, 하나라도 지우면 그 실험의 그림을 재생성할 방법이
없어진다. 5섹션 재구성·`defense`/`results` 분리 때도 npz는 전혀 손대지 않고 폴더만 옮겼다.

**`nohup_*.txt` 실행 로그는 대부분 삭제했다.** 숫자가 이미 README·npz·그림에 다 들어있어서
로그 자체는 중복이었던 것들은 지웠다. 예외 3개는 **그 실행 로그가 유일한 원자료라 보존**:
- `results/04_diversity_diagnostic/08_diversity_diagnostic/08_diversity_original_seed456_run_2143709.txt`
  — §8의 원래 seed=456 결과(52.3%)의 `.npz`가 나중에 seed=123 재실행 때 같은 파일명으로
  덮어써져서, 이 로그만 그 수치의 유일한 증거로 남음
- `results/05_partial_share_exploration/13_ln_recalibration/13_ln_recalibration_run_2147532.txt`
  — 이 실험은 애초에 `.npz`를 저장하지 않는 스크립트라 로그가 유일한 원자료
- `archive/06_p8_rescue_test/results/06_p8_rescue_test_run_2136804.txt`
  — archive 자체가 "재현 가능성 증거 보존" 목적이라 로그도 같이 둠 (코드+결과물 분리 대상 제외)

## 배경 — 왜 이 조사를 시작했나

`ViT_tradeoff`의 정식 연구(실험 1~8)에서 PatchFool 공격에 대해 P8이 P16보다 압도적으로
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
  97.1%, 시스템 정확도 13.0%→64.0%(5배), clean 무손실**. 초기 순환평가 편향 버전은
  [`archive/06_p8_rescue_test/`](archive/06_p8_rescue_test/)에 보존
- **§10 배포 비용**: batch=1, warm-up 이후 기준 latency/memory/FLOPs 측정 → **P8이 latency
  2.7배, FLOPs 4.5배 더 비쌈, 메모리는 거의 동일**
- **코드**: [`defense/02_system_validation/`](defense/02_system_validation/) (하위에
  `06_final_validation/`, `10_latency_memory/`)
- **결과**: [`results/02_system_validation/`](results/02_system_validation/) —
  `06_final_validation/06_final_validation_n200.png`,
  `10_latency_memory/10_bench_latency_memory_viz.png`

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

### 5. 부분 공유 탐색 — 실패, 종료된 방향 (§11+§12+§13)
"뒷부분 layer를 P16/P8이 공유하면 체크포인트 2벌 문제를 풀 수 있지 않을까"를 사전 점검(§11)
→ 실제 프로토타입(§12) → 재학습 없는 값싼 보정 시도(§13) 순으로 검증하고 **완전히 접은**
섹션. 셋 다 "부분 공유해봤는데 안 됐다"는 하나의 서사로 이어진다.

- **§11 Activation 유사도**: CKA로 6·9번째 층의 표현 유사도를 matched/shuffled(우연 수준)
  비교 → CKA 0.87~0.96(우연 수준 0.11~0.38보다 훨씬 높음), 공유해도 될 것 같다는 신호
- **§12 부분 공유 시제품**: patch_embed+앞 5층은 독립, 뒤 7층+head는 P16 것을 공유하는
  하이브리드 모델 제작 → branch8(P8 초반부+공유 후반부) clean accuracy가 **0%로 완전 붕괴**
- **§13 LayerNorm 재보정**: 재학습 없이 공유 LayerNorm 15개의 gamma/beta만 branch8 실제
  통계에 맞춰 closed-form 재계산 → **재보정 전후 모두 0%**, 이 방향 완전 종료
- **코드**: [`defense/05_partial_share_exploration/`](defense/05_partial_share_exploration/)
  (하위에 `11_activation_similarity/`, `12_partial_share_prototype/`, `13_ln_recalibration/`)
- **결과**: [`results/05_partial_share_exploration/`](results/05_partial_share_exploration/) —
  `11_activation_similarity/11_activation_similarity_viz.png`,
  `12_partial_share_prototype/12_hybrid_partial_share_collapse.png`,
  `13_ln_recalibration/13_hybrid_ln_recalibration_viz.png`

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
| `ViT_tradeoff`의 정식 실험 1~8, area-matched #4/#5 포함 | `experiments/main.py`가 loader를 P×attack 루프 밖에서 1회만 생성, 재사용 | ✅ 고정 이미지, 페어링됨 (3 seed는 분산 추정 목적, 문제 아님) |
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

### 종료 (CLOSED): 공유 backbone + fusion 설계
CrossViT류로 "backbone 하나 공유 + patch_embed만 P별로 따로" 만들어서 리소스 문제(체크포인트
2벌=메모리 2배)와 국소 재분할 불가 문제를 동시에 풀려던 계획이었음.

**Diversity diagnostic(§8) 결과와 정면 충돌해서 처음엔 보류했다** — 정확히 그런 정렬 상태
(같은 토큰 그리드, 다른 가중치)를 흉내낸 실험(P16-A vs P16-B)이 오히려 joint attack에 훨씬
취약했다(74.4% vs 18.4%, 페어링된 값).

그 뒤 §11에서 "layer 6+는 어차피 표현이 정렬돼 있다(CKA 0.87~0.96)"는 신호가 나와서, "초반은
독립, 후반만 공유"하는 절충안을 실제로 작게 구현해서 시험했다(§12, §13). **결과: 재학습
없이는 완전히 망가진다** — branch8 clean accuracy 0.000%, LayerNorm 재보정해도 그대로 0.000%.

**그래서 이 방향은 보류가 아니라 종료한다.** 저렴하게 시도할 수 있는 우회로는 다 막혔고, 남은
선택지는 불확실한 재학습뿐인데, 지금 독립 P8/P16 구조가 이미 검증된 실제 작동하는 방어라서
그걸 유지하는 쪽이 낫다고 판단.

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
- **아직 LaVAN에 대해서는 이 방어(탐지기+P8 폴백)를 한 번도 테스트 안 했다** — 지금까지
  §1(탐지 가능성 비교 기준선)에서만 LaVAN을 썼고, §6~14의 방어 검증은 전부 PatchFool
  기준이었음. 이 프로젝트가 LaVAN도 범위에 넣기로 한 만큼, LaVAN에 대한 탐지율/복원율을
  추가로 확인할 필요 있음(다음 실험 후보)
- (참고, 다른 최신 연구와 비교) [ViTGuard](https://arxiv.org/abs/2409.13828)가 이미
  attention+CLS token+MAE 재구성을 결합한 탐지기를 7개 기존 detector·9개 attack과 비교해
  검증한 바 있어, §1의 raw-attention 탐지기는 이것보다 단순한 버전이다. [PatchCleanser](https://www.usenix.org/conference/usenixsecurity22/presentation/xiang)
  같은 certified 방어와 달리 이 프로젝트의 보장은 테스트한 공격(LaVAN/PatchFool)과 §7/§14의
  adaptive attacker에 한정된 empirical 보장이라는 점도 논문에 명시할 필요 있음. 다음 단계로는
  기존 방법(특히 PatchCleanser, 코드 공개돼 있고 아키텍처 무관이라 이식 쉬움) 하나를 같은
  P8/16/32 세팅에 baseline으로 돌려 직접 비교하는 게 가장 값싸게 설득력을 올리는 방법
