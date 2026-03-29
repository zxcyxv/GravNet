# URM 원본 vs AMK-PD train.py 차이 분석

## 1. "Epoch"의 의미 차이

URM의 epoch과 우리 epoch은 규모가 완전히 다르다.

### URM 데이터 구조
- IterableDataset, 1 example = 1 group, `mean_puzzle_examples = 1`
- `subsample_size=1000, num_aug=0` 기준: **total_groups = 1,000**
- 1 epoch = 1,000 groups 순회 = `1000 / 128(gbs)` ≈ **7.8 batches**

### 우리 데이터 구조
- MapDataset, `subsample_size=1000, num_aug=1000`
- train_size = **1,001,000 examples**
- 1 epoch = `1,001,000 / 384(bs)` = **2,606 batches**

### total_steps 환산

| | URM | 우리 |
|---|---|---|
| epochs 설정 | 50,000 | 10 |
| batches/epoch | ~7.8 | 2,606 |
| **total_steps** | **~390,625** | **~26,060** |
| **비율** | **15x** | **1x** |

URM의 50,000 epochs ≈ 우리 기준 **~150 epochs**.
현재 우리 10 epochs는 URM 대비 **1/15 수준**.

## 2. 연산 깊이 (Computational Depth)

1회 forward call에서 gradient가 추적되는 블록 통과 횟수.

| | URM | 우리 |
|---|---|---|
| H_cycles | 2 | 2 |
| L_cycles | **6** | **1** |
| num_layers (K) | 4 | 4 |
| burn-in passes (no_grad) | 1 × 6 × 4 = 24 | 1 × 1 × 4 = 4 |
| grad passes | 1 × 6 × 4 = **24** | 1 × 1 × 4 = **4** |
| **gradient depth** | **24 blocks** | **4 blocks** |

URM은 1 forward에서 gradient graph가 **6배 깊다**.
이것이 "iterative refinement"의 핵심 — 블록을 6번 반복 통과하면서 Q가 점진적으로 정제된다.

> 주의: L_cycles를 늘리면 activation 메모리가 비례 증가하므로 batch_size를 줄이거나 grad_accum을 사용해야 함.

## 3. 옵티마이저 & LR

| | URM | 우리 |
|---|---|---|
| optimizer | AdamATan2 | AdamW (fused) |
| lr | 1e-4 | 3e-4 |
| betas | (불명) | (0.9, 0.95) |
| weight_decay | **1.0** | **0.1** |
| LR schedule | cosine, **min_ratio=1.0** (= 상수) | cosine → 0 |
| warmup | 2000 steps | 500 steps |

핵심 차이:
- URM의 LR은 warmup 후 **decay 없이 상수 유지**. 우리는 cosine으로 0까지 떨어짐.
- URM의 weight_decay가 10배 강함 (1.0 vs 0.1).
- URM의 LR이 3배 낮음 (1e-4 vs 3e-4).

## 4. Loss 계산

| | URM | 우리 |
|---|---|---|
| LM loss | stablemax_cross_entropy, **sum reduction** | F.cross_entropy, **mean reduction** |
| loss scaling | `1.0 / (global_batch_size * accum_steps)` 수동 | 없음 (mean이 자동 정규화) |
| halt loss | `0.5 * (BCE_halt + BCE_continue)` | `0.01 * MSE(margin, target)` |
| halt target | `seq_is_correct` (bool) | `2*acc - 1` (continuous) |

URM은 sum reduction + 수동 스케일링으로 gradient 크기를 정밀 제어.

## 5. 평가 (Eval)

| | URM | 우리 |
|---|---|---|
| eval 주기 | 2,000 epochs마다 (≈ 15,625 steps) | 2,500 steps마다 |
| eval carry | 배치마다 fresh | 배치마다 fresh |
| halting | **adaptive** (while not all_finish) | **고정** N loops |
| eval 범위 | 전체 test set | 200 samples |

URM은 adaptive halting으로 각 퍼즐이 스스로 "다 풀었다"고 판단할 때까지 루프를 돌림.
우리는 고정된 loops 횟수만큼 돌리고 끝.

## 6. 기타

| | URM | 우리 |
|---|---|---|
| EMA | 있음 (ema_rate) | 없음 |
| multi-GPU | 8 GPU (DDP) | 1 GPU |
| batch_size | global 128 (GPU당 16) | 384 |
| puzzle embedding | 있음 (별도 optimizer) | 없음 |
| data augmentation | 없음 (num_aug=0, 온라인 셔플) | 오프라인 1000x aug |

## 7. 권장 수정사항 (우선순위순)

### 필수
1. **L_cycles: 1 → 6** (batch_size를 줄이고 grad_accum으로 보상)
2. **epochs: 10 → 150+** (URM 대비 동등한 total_steps 확보)
3. **LR schedule: cosine → constant** (min_ratio=1.0, warmup 후 decay 없음)
4. **lr: 3e-4 → 1e-4**
5. **weight_decay: 0.1 → 1.0**

### 권장
6. eval을 adaptive halting으로 변경
7. EMA 추가
8. warmup_steps: 500 → 2000

### 메모리 대응 (L_cycles=6 적용 시)
- L_cycles=6이면 grad activation 메모리 6배 증가
- `batch_size=64, grad_accum=6` (effective batch = 384) 또는
- `batch_size=64, grad_accum=2` (effective batch = 128, URM과 동일)
