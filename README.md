# PatchSwitch — ViT 적응형 방어(P16→P8 폴백) 프로젝트

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

## 디렉토리 구조

```
PatchSwitch/
  src/                      ViT_tradeoff/src/에서 복사 + 범위에 맞게 정리
    models.py                 MODEL_NAMES에서 32 제외(P8/P16만)
    dataset.py                 원본과 동일(전처리 상수는 P16 기준 그대로 유효)
    attacks/
      lavan.py                 원본과 동일
      patch_fool.py            원본과 동일
      (pgd.py 없음 — 이 프로젝트 범위 밖)
  defense/
    01_detection_feasibility/  ~  14_adaptive_evasion_full/   (아래 실험 목록 참고)
  archive/
    06_p8_rescue_test/        (§6의 superseded 초기 버전, 재현 가능하게 보존)
```

각 실험 폴더는 **완전히 자기 완결적**이다: 실험 스크립트(`.py`), 실행 스크립트(`run_*.sh`),
결과+그림(`results/`)이 한 폴더 안에 다 있다. 여러 실험이 같은 공격 로직(예: joint attack)을
쓸 때는 모듈을 폴더마다 **복사**해서 넣었다(`patch_fool_joint.py`가 `07_joint_attack/`과
`08_diversity_diagnostic/`에 각각 한 부씩) — 폴더 간 import를 없애서 폴더 하나만 통째로 옮기거나
지워도 다른 실험이 안 깨지게 하기 위함이다. 유일한 예외 둘(둘 다 코드 주석으로 명시돼 있음):
`archive/06_p8_rescue_test/`가 `defense/01_detection_feasibility/results/`의 산출물을 읽는
것, 그리고 `08_diversity_diagnostic/viz.py`·`14_adaptive_evasion_full/viz.py`가
`07_joint_attack`의 확정된 숫자를 (npz를 다시 열지 않고) 상수로 인용하는 것.

**결과 파일 이름은 전부 `NN_설명.확장자`** 형식이다(예: `01_signature_P16.png`,
`08_diversity_diagnostic_headline.png`) — `NN`은 아래 실험 번호와 정확히 대응한다. 실험
스크립트를 재실행해도 이 이름 그대로 저장된다.

§5, §9는 실험 번호가 비어있다 — §5는 실험이 아니라 설계 판단(국소 재분할 불가 확인)이고,
§9(Protocol C)는 P32를 포함하는 실험이라 이 프로젝트 범위 밖으로 판단해서
[`ViT_tradeoff/09_protocol_c/`](../ViT_tradeoff/09_protocol_c/)로 옮겼다(아래 §9 항목 참고).

이전 위치(`ViT_robust/probes/`, 재구성 전) 전체 백업: 아직 이 이동 자체에 대한 백업은 없음
(재구성 전 probes/ 백업은 `/home/jooyumm/backups/probes_backup_20260903.tar.gz`, CISC-W'26
투고 끝날 때까지 보관 예정 — 이번 ViT_robust→ViT_tradeoff+PatchSwitch 분리는 아직 별도
백업이 없으니 필요하면 말씀해주세요). 참고로 이 프로젝트의 git 히스토리는 사실상 없다 —
`ViT_robust`가 지금까지 커밋된 적이 없어서(마지막 커밋은 훨씬 이전의 완전히 다른 구조),
`ViT_tradeoff`로의 이동도 "git mv"가 아니라 그냥 파일 이동으로 처리했다(보존할 히스토리
자체가 없었음).

## npz / 로그 정리 정책

**`.npz`는 전부 유지했다.** 지금 있는 npz는 전부 각 폴더의 `viz.py`(또는 원본 실험 스크립트
자신)가 그림을 다시 그리는 데 쓰는 원자료라, 하나라도 지우면 그 실험의 그림을 재생성할 방법이
없어진다.

**`nohup_*.txt` 실행 로그는 대부분 삭제했다.** 숫자가 이미 README·npz·그림에 다 들어있어서
로그 자체는 중복이었던 것들은 지웠다. 예외 3개는 **그 실행 로그가 유일한 원자료라 보존**:
- `defense/08_diversity_diagnostic/results/08_diversity_original_seed456_run_2143709.txt`
  — §8의 원래 seed=456 결과(52.3%)의 `.npz`가 나중에 seed=123 재실행 때 같은 파일명으로
  덮어써져서, 이 로그만 그 수치의 유일한 증거로 남음
- `defense/13_ln_recalibration/results/13_ln_recalibration_run_2147532.txt`
  — 이 실험은 애초에 `.npz`를 저장하지 않는 스크립트라 로그가 유일한 원자료
- `archive/06_p8_rescue_test/results/06_p8_rescue_test_run_2136804.txt`
  — archive 자체가 "재현 가능성 증거 보존" 목적이라 로그도 같이 둠

## 배경 — 왜 이 조사를 시작했나

`ViT_tradeoff`의 정식 연구(실험 1~8)에서 PatchFool 공격에 대해 P8이 P16보다 압도적으로
강건하다는 걸 확인한 뒤(RA 67.6% vs 9.1%), 이걸 실제로 쓸 수 있는 방어로 만들 수 있는지
탐색했다. 아이디어: **"평소엔 효율 좋은 P16으로 추론하다가, attention 시그니처로 공격이
의심되면 그 이미지를 강건한 P8로 다시 분류한다."**

## 실험 목록

### §1. 탐지 가능성 — attention rollout으로 공격 여부를 구분할 수 있는가
- **질문**: clean 이미지와 PatchFool 공격 이미지를 attention만 보고 구분할 수 있나?
- **방법**: clean/LaVAN/PatchFool 각 30장, 층 1~12별 top-4 mass의 AUROC 측정(rollout vs raw)
- **파일**: [`defense/01_detection_feasibility/`](defense/01_detection_feasibility/) —
  `vitguard_signature.py`+`run_signature.sh`, `vitguard_layer_sweep.py`+`run_layer_sweep.sh`
- **결과**: L=12 raw attention이 최고, AUROC 0.879(clean 대비)~0.891(LaVAN 대비)
- **시각화**: `results/01_signature_P16.png`, `results/01_layer_sweep_P16.png`

### §2. 위치 특정 — 어디를 공격했는지도 맞히는가
- **질문**: 공격당한 정확한 토큰 위치도 찾아내나?
- **방법**: top-1(가장 attention 큰 토큰)이 실제 공격 토큰과 일치하는 비율
- **파일**: [`defense/02_localization/`](defense/02_localization/) — `vitguard_localization.py`+`run_localization.sh`, `viz.py`
- **결과**: recall@1 96.7%(29/30), 놓친 1개는 11칸 떨어진 완전 실패
- **시각화**: `results/02_localization_viz.png`

### §3. 실패 원인 진단 + 회피 시도 3종 — 전부 탐지기를 못 뚫음
- **질문**: §2에서 놓친 이유는? 공격자가 일부러 탐지를 피할 수 있나?
- **방법**: (1) 실패 샘플 이미지 직접 확인 (2) 가장 낮은 saliency 위치를 강제 공격
  (3) attention/saliency 정규화한 새 탐지기 시도 (4) 실패 샘플들의 실제 saliency 범위를
  측정해서 그 범위로 재타겟
- **파일**: [`defense/03_evasion_attempts/`](defense/03_evasion_attempts/) —
  `vitguard_failure_case.py`+`run_failure_case.sh`(1),
  `vitguard_evasion_test.py`+`run_evasion_test.sh`(2, (3)의 실패한 정규화 탐지기 B는 132~137행),
  `vitguard_range_evasion.py`+`run_range_evasion.sh`(4), 공격 로직 모듈 `patch_fool_lowsal.py`, `viz.py`
- **결과**: (2) recall 0.900로 안 뚫림, (3) recall 0.000으로 탐지기 자체가 실패,
  (4)도 recall 0.900로 안 뚫림 — raw 탐지기는 견고, 대안 탐지기(정규화)만 실패
- **시각화**: `results/03_failure_case_sample1.png` 외 3장, `results/03_evasion_attempts_summary.png`

### §4. attn_layer_idx 일반화 — 특정 레이어(4)에 과적합된 결과가 아님을 확인
- **질문**: 공격자가 다른 층 기준으로 토큰을 고르면 탐지가 무너지나?
- **방법**: 공격의 attn_layer_idx를 1,2,4,6,8,10으로 바꿔가며 반복
- **파일**: [`defense/04_layeridx_generalization/`](defense/04_layeridx_generalization/) — `vitguard_layeridx_generalization.py`+`run_layeridx_generalization.sh`
- **결과**: recall@1이 항상 0.900~0.967 — 안정적
- **시각화**: `results/04_layeridx_generalization_P16.png`

### §5. (실험 아님) "국소 재분할"은 불가능
원래 계획한 "의심 영역만 P8로 국소 재분할"은 P8/P16의 임베딩 공간이 호환되지 않아 물리적으로
불가능 — 대안으로 "의심되면 이미지 전체를 P8로 통째 재분류"로 단순화해서 계속 검증.

### §6. 최종 검증 — calibration/evaluation 분리 (n=200, held-out)
- **질문**: 탐지→P8 전환 파이프라인의 실제 시스템 정확도는? 이 숫자를 믿을 수 있나?
- **방법**: 200장을 calibration 100/evaluation 100으로 분리, 임계값은 calibration에서만,
  성능(recall/FPR/복원율)은 evaluation에서만 계산
- **파일**: [`defense/06_final_validation/`](defense/06_final_validation/) — `vitguard_final_validation.py`+`run_final_validation.sh`(정정판, 지금 쓰는 것), 모듈 사본 `eval_utils.py`.
  초기 순환평가 편향 버전은 [`archive/06_p8_rescue_test/`](archive/06_p8_rescue_test/)에 보존
- **결과**: FPR 5.0%, 탐지 recall 64.0%, 복원율 97.1%, **시스템 정확도 13.0%→64.0%(5배), clean 무손실**
- **시각화**: `results/06_final_validation_n200.png`

### §7. Joint attack — 방어 구조를 아는 adaptive attacker
- **질문**: "P16 걸리면 P8로 넘어간다"는 걸 아는 공격자라면?
- **방법**: P16+P8 손실을 합쳐서 하나의 perturbation으로 동시 최적화
- **파일**: [`defense/07_joint_attack/`](defense/07_joint_attack/) — `vitguard_joint_attack_test.py`+`run_joint_attack_test.sh`, 공격 로직 모듈 `patch_fool_joint.py`, `viz.py`
- **결과**: 나이브 전이(2.9%) 대비 joint attack은 **18.4%**로 6배 위험, 탐지기도 약해짐(joint attack의 20%만 flag)
- **시각화**: `results/07_joint_attack_test_viz.png`

### §8. Diversity diagnostic — patch size 차이 vs 그냥 "다른 모델" ⭐ 핵심 반전 결과
- **질문**: 방어력의 원천이 토큰화 구조 차이인지, 그냥 두 모델이 달라서인지?
- **방법**: 같은 P16, 학습 레시피만 다른 두 번째 모델로 P16-A vs P16-B joint attack (§7과
  동일 이미지, seed=123으로 페어링 — 아래 "샘플링 감사" 참고)
- **파일**: [`defense/08_diversity_diagnostic/`](defense/08_diversity_diagnostic/) —
  `vitguard_diversity_test.py`+`run_diversity_test.sh`(원본 seed=456)+`run_diversity_test_paired.sh`(페어링 재실행 seed=123),
  모듈 사본 `patch_fool_joint.py`, `viz.py`
- **결과**: 같은 patch size, 다른 학습 = **74.4%** 뚫림 vs 다른 patch size(§7) = 18.4% →
  **방어력의 핵심은 "다른 patch size"이지 "다른 모델"이 아니다**
- **시각화**: `results/08_diversity_diagnostic_headline.png` ⭐

### §9. Protocol C — ViT_tradeoff로 이동
면적 대신 토큰 개수를 P8/P16/**P32**에서 동일하게 고정하는 실험이라(P32 포함) 이 프로젝트
범위 밖으로 판단, [`ViT_tradeoff/09_protocol_c/`](../ViT_tradeoff/09_protocol_c/)로 옮겼다.
결과(P8 RA 22.0% vs P16/P32 0.0%)는 그쪽 README 참고.

### §10. Latency/Memory/Params/FLOPs
- **질문**: P8로 전환하는 실제 배포 비용(속도/메모리)은?
- **방법**: batch=1, warm-up 이후 기준 측정
- **파일**: [`defense/10_latency_memory/`](defense/10_latency_memory/) — `bench_latency_memory.py`+`run_bench_latency.sh`, `viz.py`
- **결과**: P8이 latency 2.7배, FLOPs 4.5배 더 비쌈, 메모리는 거의 동일
- **시각화**: `results/10_bench_latency_memory_viz.png`

### §11. Activation 유사도 — 부분 공유 착수 전 사전 점검
- **질문**: P16/P8 중간층이 원래 얼마나 비슷한가?
- **방법**: CKA로 6·9번째 층의 표현 유사도를 matched/shuffled(우연 수준) 비교
- **파일**: [`defense/11_activation_similarity/`](defense/11_activation_similarity/) — `activation_similarity.py`+`run_activation_similarity.sh`, `viz.py`
- **결과**: CKA 0.87~0.96(우연 수준 0.11~0.38보다 훨씬 높음)
- **시각화**: `results/11_activation_similarity_viz.png`

### §12. 부분 공유 시제품 + joint attack — 핵심 성패 테스트, **실패**
- **질문**: 뒷부분(6~12층)을 공유하는 모델을 실제로 만들면 작동하나?
- **방법**: patch_embed+앞 5층은 독립, 뒤 7층+head는 P16 것을 공유하는 하이브리드 모델 제작
- **파일**: [`defense/12_partial_share_prototype/`](defense/12_partial_share_prototype/) —
  `hybrid_joint_attack_test.py`+`run_hybrid_joint_attack.sh`, 모듈 사본 `hybrid_partial_share.py`+`patch_fool_joint.py`, `viz.py`
- **결과**: branch8(P8 초반부+공유 후반부) clean accuracy가 **0%로 완전 붕괴** — joint attack 자체를 못 함
- **시각화**: `results/12_hybrid_partial_share_collapse.png`

### §13. LayerNorm 재보정 시도 — 그래도 저렴하게 한 번 더, **역시 실패**
- **질문**: 재학습 없이 LayerNorm 통계만 재보정해도 §12가 회복되나?
- **방법**: 공유 LayerNorm 15개의 gamma/beta를 branch8 실제 통계에 맞춰 closed-form 재계산
- **파일**: [`defense/13_ln_recalibration/`](defense/13_ln_recalibration/) — `hybrid_ln_recalibration.py`+`run_hybrid_ln_recalibration.sh`, 모듈 사본 `hybrid_partial_share.py`, `viz.py`
- **결과**: 재보정 전후 모두 0% — **이 방향(§11~13, 부분 공유) 완전 종료**
- **시각화**: `results/13_hybrid_ln_recalibration_viz.png`

### §14. 완전판 adaptive attack — joint attack + 탐지 회피 제약, 진짜 worst-case
- **질문**: §7 공격자는 탐지기 존재를 몰랐는데, 탐지기까지 알면?
- **방법**: joint attack 손실에 "탐지 점수를 clean 범위 안으로 유지"하는 제약(STRAP-ViT류
  설계) 추가, 위반 시 페널티 가중치 자동 증가
- **파일**: [`defense/14_adaptive_evasion_full/`](defense/14_adaptive_evasion_full/) —
  `vitguard_adaptive_evasion_full_test.py`+`run_adaptive_evasion_full.sh`, 모듈 사본 `patch_fool_joint_evasive.py`+`eval_utils.py`, `viz.py`
- **결과**: 회피 제약을 걸어도 §7과 결과가 거의 동일(18.4%=18.4%) — 두 모델을 동시에 속이는
  목표 자체가 이미 탐지 회피를 "공짜로" 어느 정도 포함. **최종 worst-case(무력화+미탐지) = 15.8%**
- **시각화**: `results/14_adaptive_evasion_full_viz.png` ⭐

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
