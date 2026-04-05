# URM vs AMK-PD 아키텍처 차이 완전 비교

`ref/URM/models/urm/urm.py` (URM) vs `amkpd_model.py` (AMK-PD) 사이의 모든 아키텍처 차이를 열거한다.
동일한 부분은 생략하고, **다른 것만** 기술한다.

> **범례**
> - ✅ 의도적 차이 (AMK-PD 고유 설계)
> - ⚠️ 일치시켜야 할 후보 (공정 비교를 위해)
> - 🔧 이미 수정 완료

---

## 동일한 부분 (확인 완료)

- Carry 구조체 (current_hidden, steps, halted, current_inputs, current_labels)
- 3중 루프 제어 흐름 (loops × H_cycles × L_cycles)
- Halting 로직 (steps >= loops, q_halt > 0, exploration)
- 입력 주입 방식: L_cycle 시작 시 `Q = Q + X` 1회 (URM과 동일)
- Optimizer: norm weight에 weight_decay 적용 (URM과 동일)
- 정규화: 파라미터 없는 `rms_norm` 함수 사용 (URM과 동일)
- 임베딩: `√d_model` 스케일링 적용 (URM과 동일)
- 가중치 초기화: Truncated LeCun Normal (URM과 동일)
- ConvSwiGLU intermediate 크기 계산: `round(expansion * d * 2/3)`, 256 정렬 (동일)
- ConvSwiGLU 연산 순서: `silu(gate) * up` → conv → `silu` → down (동일)
- 출력 헤드: post-norm 구조로 블록 마지막에 `rms_norm`이 적용되어 별도 `final_norm` 불필요 (양쪽 동일)
- Q head 초기화: weight=0, bias=-5 (동일)

---

## ✅ 1. 어텐션 메커니즘 — 핵심 차이 (의도적)

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **어텐션 종류** | Softmax 어텐션 (FlashAttention) | 커널 어텐션 (ELU+1 매핑 → 다항식 인력 행렬) |
| **어텐션 계산** | `flash_attn(Q, K, V)` | `Φ_Q = elu(Q)+1`, `Φ_K = elu(K)+1`, `W = relu(Φ_Q Φ_K^T · scale)^p`, `C = WV / sum(W)` |
| **출력 의미** | 가중 평균 (standard attention) | Mean Shift 벡터 `m = C − V` (입자 간 인력 방향) |
| **kernel_power** | 없음 | `p` 파라미터로 인력 행렬 비선형성 제어 (default 2) |
| **복잡도** | O(N²d_h) — FlashAttention 최적화 | O(N²d_h) — 명시적 N×N 행렬 생성 필수 |

---

## ✅ 2. 잔차 연결 — 어텐션 출력의 의미 차이 (의도적)

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **O_proj 입력** | FlashAttention의 가중합 출력 | Mean Shift 벡터 `m = C − V` (헤드 concat) |
| **O_proj 이름** | `o_proj` | `W_O_aux` |
| **잔차** | `hidden + attn_output` | `Q_in + W_O_aux(m)` |

URM: 원래 hidden에 attention 결과 자체를 가산.
AMK-PD: 원래 Q에 "이동해야 할 방향"을 가산. 어텐션이 아니라 입자 이동을 더하는 셈.

---

## ⚠️ 3. 위치 인코딩

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **방식** | RoPE (Rotary Positional Embedding) | 학습 가능한 위치 임베딩 (`nn.Embedding`) |
| **적용 위치** | Q, K에 회전 적용 (어텐션 내부) | 입력 임베딩에 합산 (어텐션 외부) |

URM은 어텐션 내부에서 Q/K에 상대 위치 정보를 주입한다.
AMK-PD는 입력 임베딩 단계에서 절대 위치를 더하며, 커널 어텐션 자체는 위치 구분 없이 동작한다.

AMK-PD에 `RotaryEmbedding` 클래스가 정의되어 있으나 **실제로 사용하지 않는다** (dead code).

**검토 필요**: AMK-PD의 커널 어텐션 구조에서 RoPE를 적용하려면
ELU+1 매핑 전 Q, K에 회전을 적용해야 한다.
다만 ELU+1 비선형 변환이 RoPE의 상대 위치 정보를 보존하는지 검토 필요.

---

## ⚠️ 4. Q-head 풀링 방식

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **Q-head 입력** | `hidden_states[:, 0]` (첫 토큰) | `Q.mean(dim=1)` (시퀀스 mean pooling) |
| **Q-head dtype** | `.to(torch.float32)` 명시 캐스팅 | 캐스팅 없음 |

URM은 첫 번째 토큰의 hidden state로 halt/continue 판단.
AMK-PD는 전체 시퀀스의 평균으로 판단.

float32 캐스팅: bfloat16 훈련 시 halting sigmoid 정밀도에 영향.

---

## ⚠️ 5. 혼합 정밀도 처리 (CastedLinear vs nn.Linear)

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **Linear** | `CastedLinear` — weight를 input dtype으로 수동 캐스팅 | `nn.Linear` — PyTorch autocast 의존 |
| **Embedding** | `CastedEmbedding` — bfloat16 명시 캐스팅 | `nn.Embedding` — 기본 dtype |
| **DWConv dtype** | `.to(dtype=torch.bfloat16)` 명시 | 기본 dtype (float32) |
| **forward_dtype** | config로 `bfloat16` 지정, 모델 전체에 적용 | autocast context manager 의존 |

URM은 수동으로 혼합 정밀도를 제어하여 어떤 연산이 어떤 정밀도로 실행되는지 명확하다.
AMK-PD는 PyTorch의 autocast에 의존하므로, autocast 없이 실행하면 전부 float32로 동작한다.

---

## ⚠️ 6. 임베딩 스케일링 범위

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **토큰 임베딩** | `embed_scale * embedding` (RoPE 사용, 절대 pos_emb 없음) | `embed_scale * token_emb + pos_emb` (pos_emb에는 스케일 미적용) |

URM은 RoPE를 사용하므로 절대 위치 임베딩이 없고 전체 embedding에 `√d` 스케일이 적용된다.
AMK-PD는 `√d` 스케일이 토큰 임베딩에만 적용되고, pos_emb는 스케일 없이 합산된다.
이 차이가 3번(위치 인코딩)과 연결된다.

---

## 기타 사소한 차이

| 항목 | URM | AMK-PD | 비고 |
|------|-----|--------|------|
| **DWConv 커널 기본값** | 2 | 3 (train.py에서 2 전달) | 실행 시 동일 |
| **Dead code** | 없음 | `RotaryEmbedding`, `rotate_half`, `apply_rotary_pos_emb` 미사용 정의 | 정리 권장 |
| **rms_norm eps** | config에서 `1e-5` (configurable) | 하드코딩 `1e-5` | 값 동일 |

---

## 요약 — 현재 상태

| # | 차이 | 상태 | 비고 |
|---|------|------|------|
| 1 | 어텐션 메커니즘 (커널 vs softmax) | ✅ 의도적 | AMK-PD 핵심 설계 |
| 2 | 잔차 연결 (mean shift vs attention) | ✅ 의도적 | 어텐션 메커니즘의 결과 |
| - | 입력 주입 빈도 | 🔧 수정 완료 | L_cycle당 1회로 URM과 동일 |
| - | Optimizer norm weight_decay | 🔧 수정 완료 | URM과 동일하게 적용 |
| - | 정규화 (파라미터 없는 rms_norm) | 🔧 수정 완료 | URM과 동일 |
| - | 임베딩 √d 스케일링 | 🔧 수정 완료 | URM과 동일 |
| - | 가중치 초기화 (trunc_normal_init_) | 🔧 수정 완료 | URM과 동일 |
| - | input_norm 제거 | 🔧 수정 완료 | URM과 동일 |
| - | final_norm 제거 | 🔧 수정 완료 | post-norm으로 불필요 |
| 3 | 위치 인코딩 (RoPE vs 절대) | ⚠️ 판단 필요 | 커널 어텐션과 RoPE 호환성 검토 필요 |
| 4 | Q-head 풀링 (첫 토큰 vs mean) + float32 | ⚠️ 판단 필요 | 간단 적용 가능 |
| 5 | 혼합 정밀도 (CastedLinear vs autocast) | ⚠️ 판단 필요 | bfloat16 훈련 인프라 관련 |
| 6 | 임베딩 스케일링 범위 (pos_emb) | ⚠️ 판단 필요 | 3번과 연결 |
