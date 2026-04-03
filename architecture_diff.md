# URM vs AMK-PD 아키텍처 차이 완전 비교

`urm_model.py` (URM) vs `amkpd_model.py` (AMK-PD) 사이의 모든 아키텍처 차이를 열거한다.
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
- 🔧 입력 주입 방식: L_cycle 시작 시 `Q = Q + X` 1회 (URM과 동일하게 수정됨)
- 🔧 Optimizer: norm weight에 weight_decay 적용 (URM과 동일하게 수정됨)

---

## ✅ 1. 어텐션 메커니즘 — 핵심 차이 (의도적)

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **어텐션 종류** | Softmax 어텐션 (`F.scaled_dot_product_attention`) | 커널 어텐션 (ELU+1 매핑 → 다항식 인력 행렬) |
| **어텐션 계산** | `softmax(QK^T/√d_h) · V` | `Φ_Q = elu(Q)+1`, `Φ_K = elu(K)+1`, `W = relu(Φ_Q Φ_K^T · scale)^p`, `C = WV / sum(W)` |
| **출력 의미** | 가중 평균 (standard attention) | Mean Shift 벡터 `m = C − V` (입자 간 인력 방향) |
| **kernel_power** | 없음 | `p` 파라미터로 인력 행렬 비선형성 제어 (default 2) |
| **복잡도** | O(N²d_h) — SDPA FlashAttention 최적화 가능 | O(N²d_h) — 명시적 N×N 행렬 생성 필수 |

AMK-PD의 핵심 가설: softmax attention을 입자 역학 기반의 Mean Shift로 대체하면
토큰 간 "인력"에 의한 응집 과정이 반복 루프와 시너지를 낸다.

---

## ✅ 2. 잔차 연결 — 어텐션 출력의 의미 차이 (의도적)

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **O_proj 입력** | SDPA의 가중합 출력 | Mean Shift 벡터 `m = C − V` (헤드 concat) |
| **O_proj 이름** | `o_proj` | `W_O_aux` |
| **잔차** | `hidden + Attn(hidden)` | `Q_in + W_O_aux(m)` |

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

**일치 필요 여부**: AMK-PD의 커널 어텐션 구조에서 RoPE를 적용하려면
ELU+1 매핑 전 Q, K에 회전을 적용해야 한다.
다만 ELU+1 비선형 변환이 RoPE의 상대 위치 정보를 보존하는지 검토 필요.

---

## ⚠️ 4. 정규화 (Normalization)

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **Norm 종류** | 파라미터 **없는** RMSNorm (함수) | 파라미터 **있는** `nn.RMSNorm` (모듈) |
| **학습 가능 스케일** | 없음 | 있음 (채널별 학습 가능한 `weight` 파라미터) |
| **입력 정규화** | 없음 (임베딩 스케일링만) | `self.input_norm = nn.RMSNorm(d_model)` |
| **최종 출력 정규화** | 없음 | `self.final_norm = nn.RMSNorm(d_model)` |
| **블록 내 위치** | post-norm (잔차 후) | post-norm (동일 위치이나 학습 가능 파라미터 포함) |

URM은 의도적으로 파라미터 없는 norm을 사용한다.
루프를 수십 회 반복하는 구조에서 norm weight가 매번 곱해지면
특정 차원이 지수적으로 폭발/소멸할 수 있기 때문으로 추정.

**일치 필요 여부**: 공정 비교를 위해 파라미터 없는 함수형 RMSNorm으로 통일하거나,
반대로 URM에도 학습 가능 norm을 주어 조건을 맞출 수 있다.
단, AMK-PD만 `input_norm`과 `final_norm`을 갖는 점은 별도로 판단 필요.

---

## ⚠️ 5. 임베딩 (Embedding)

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **토큰 임베딩** | `nn.Embedding` + 스케일링 `√d_model` | `nn.Embedding` (스케일링 없음) |
| **입력 정규화** | 없음 | `nn.RMSNorm` 적용 |
| **임베딩 계산** | `√d · embed(x)` | `embed(x) + pos_emb(pos)` → `RMSNorm` |

URM은 `√d` 스케일링으로 임베딩 벡터의 크기를 키워서 init_hidden 대비 충분한 신호를 보장한다.
AMK-PD는 스케일링 대신 RMSNorm으로 크기를 정규화한다.

**일치 필요 여부**: 둘 중 하나로 통일. `√d` 스케일링은 단순하고 추가 파라미터가 없으므로
AMK-PD에도 `√d` 스케일링을 적용하고 `input_norm`을 제거하는 것이 공정 비교에 유리.

---

## ⚠️ 6. 가중치 초기화 (Weight Initialization)

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **초기화 방식** | Truncated LeCun Normal (`1/√d` std) | PyTorch 기본값 + 일부 수동 |
| **QKV 사영** | `trunc_normal_(std=1/√hidden_size)` | Kaiming Uniform (기본값) |
| **O 사영** | `trunc_normal_(std=1/√hidden_size)` | `xavier_uniform_` |
| **MLP gate_up** | `trunc_normal_(std=1/√hidden_size)` | Kaiming Uniform (기본값) |
| **MLP down** | `trunc_normal_(std=1/√inter)` | Kaiming Uniform (기본값) |
| **임베딩** | `trunc_normal_(std=1/√d)` | `normal_(std=0.02)` |
| **LM head** | `trunc_normal_(std=1/√d)` | `normal_(std=0.02)` |
| **init_hidden** | `trunc_normal_(std=1)` | `randn() * 0.02` |

**일치 필요 여부**: 초기화는 학습 초기 수렴 속도와 안정성에 직접 영향.
공정 비교를 위해 AMK-PD도 URM과 동일한 truncated LeCun normal로 통일 권장.

---

## ⚠️ 7. 출력 헤드 (Output Head)

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **LM head 입력** | `hidden_states` (정규화 없음) | `self.final_norm(Q)` (RMSNorm 후) |
| **Q-head 입력** | `hidden_states[:, 0]` (첫 토큰) | `Q_norm.mean(dim=1)` (시퀀스 mean pooling) |
| **Q-head dtype** | `.to(torch.float32)` 명시 캐스팅 | 캐스팅 없음 |

**일치 필요 여부**:
- `final_norm`: AMK-PD 고유 선택일 수 있으나, 4번(정규화)과 함께 판단
- Q-head 풀링: 첫 토큰 vs mean — Sudoku에서는 첫 토큰이 특별한 의미 없으므로
  mean이 합리적이나, 공정 비교를 위해 통일 고려
- float32 캐스팅: bfloat16 훈련 시 halting 정밀도에 영향. AMK-PD에도 추가 권장

---

## ⚠️ 8. ConvSwiGLU 세부 차이

| 항목 | URM | AMK-PD |
|------|-----|--------|
| **DWConv 기본 커널** | 2 (arg default) | 3 (arg default, 실제 train.py에서 2 전달) |
| **MLP init** | `trunc_normal_` | PyTorch 기본값 |

intermediate 크기 계산 수식은 표현만 다르고 동일한 결과를 낸다.
DWConv 커널은 train.py에서 2를 전달하므로 실행 시 동일.

---

## 요약 — 현재 상태

| # | 차이 | 상태 | 비고 |
|---|------|------|------|
| 1 | 어텐션 메커니즘 | ✅ 의도적 | AMK-PD 핵심 설계 |
| 2 | 잔차 연결 (O_proj 의미) | ✅ 의도적 | 어텐션의 결과물 |
| - | 입력 주입 빈도 | 🔧 수정 완료 | L_cycle당 1회로 URM과 동일 |
| - | Optimizer norm weight_decay | 🔧 수정 완료 | URM과 동일하게 적용 |
| 3 | 위치 인코딩 (RoPE vs 절대) | ⚠️ 판단 필요 | 커널 어텐션과 RoPE 호환성 검토 필요 |
| 4 | 정규화 (파라미터 유무) | ⚠️ 판단 필요 | 루프 반복 시 안정성 영향 |
| 5 | 임베딩 (√d 스케일링 vs RMSNorm) | ⚠️ 판단 필요 | 신호 크기 보장 방식의 차이 |
| 6 | 가중치 초기화 | ⚠️ 판단 필요 | 수렴 속도/안정성에 영향 |
| 7 | 출력 헤드 (final_norm, 풀링, dtype) | ⚠️ 판단 필요 | 복수 항목 개별 판단 필요 |
| 8 | ConvSwiGLU init | ⚠️ 판단 필요 | 6번(초기화)과 함께 처리 |
