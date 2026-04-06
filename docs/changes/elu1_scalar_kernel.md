# ReLU^p → ELU+1 스칼라 커널 교체

## 변경 요약

커널 어텐션의 비선형 변환을 `ReLU(S)^p`에서 `(ELU(S) + 1)^p`로 교체하고,
이에 따라 불필요해진 `attn_bias` 파라미터를 제거했다.

```python
# 이전
W = F.relu(S)       # ~50% dead zone
W = W * W

# 이후
W = F.elu(S) + 1.0  # 항상 > 0, dense
W = W * W
```

---

## 배경: ELU+1을 벡터 vs 스칼라에 적용하는 차이

### backup2 방식 (벡터에 적용)

```
Φ_Q = ELU(Q_proj) + 1   [B, H, N, d_h]  ← 벡터 원소마다 적용
Φ_K = ELU(K_proj) + 1   [B, H, N, d_h]
W_ij = Φ_Q_i · Φ_K_j                    ← 피처 맵의 내적
```

양수 벡터끼리의 내적은 항상 ≥ 0이므로 W가 dense하고 안정적이었다.
그러나 RoPE와 양립 불가: RoPE는 내적 전 벡터에 회전을 가하는데,
ELU+1을 먼저 씌우면 회전 후 양수 보장이 깨진다.

### 현재 방식 (스칼라에 적용)

```
S_ij = (RoPE(Q_i) · RoPE(K_j)) / √d_h  ← 내적 결과 스칼라 (음수 가능)
W_ij = ELU(S_ij) + 1                    ← 스칼라에 단조 변환
```

RoPE가 벡터에 적용된 뒤 내적이 수행되어 상대 위치 정보가 스칼라 S_ij에
위상(phase) 형태로 보존된다. 이후 ELU+1은 단조 증가 함수이므로
S_ij의 순서 구조(상대적 크기 관계)를 파괴하지 않는다.

---

## ReLU → ELU+1으로 교체한 이유

### ReLU의 문제

`W = ReLU(S)^p`에서 S의 분포가 N(0, σ²)에 가깝기 때문에 W의 약 50%가
정확히 0이 된다 (hard dead zone). 이로 인해:

1. **Norm 불안정**: `Norm_i = Σ_j W_ij`가 배치마다 크게 달라짐
2. **Jacobian 불안정**: `∂L/∂S_ij ∝ 1/Norm_i` — Norm이 작으면 gradient 폭발
3. **L_cycles 증폭**: 6 L_cycles에서 Jacobian을 6번 곱하면 작은 불안정이
   지수적으로 증폭 (RNN gradient explosion과 동일한 구조)

실험(gnorm_per_loop.jsonl, 3000 스텝)에서 b3_QKV gradient가 최대 4784까지
폭발하는 것이 관측되었고, 이는 sparse W → 작은 Norm → 큰 Jacobian →
L_cycles 곱셈에 의한 증폭 메커니즘으로 설명된다.

### ELU+1의 해결

ELU(x) + 1은:
- x > 0: x + 1 > 1 (양수)
- x = 0: 1
- x → -∞: exp(x) → 0⁺ (0에 수렴하지만 정확히 0이 되지 않음)

따라서 W의 모든 entry가 항상 > 0이고, Norm_i = Σ_j W_ij에 하한이 생겨
Jacobian이 안정적으로 유지된다.

---

## attn_bias 제거

`attn_bias`는 ReLU의 dead zone 문제를 완화하기 위해 도입되었다.
score 분포를 양수 방향으로 shift하여 active entry 비율을 높이는 것이 목적이었다.

ELU+1으로 전환하면 모든 entry가 이미 양수이므로 attn_bias가 불필요하다.
헤드당 1개 스칼라 × 8헤드 = 8 파라미터를 제거한다.

---

## 파라미터 수 변화

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| `attn_bias` | `num_heads` 파라미터 (블록당) | 제거 |
| 그 외 | 동일 | 동일 |

6개 블록 기준 총 `6 × 8 = 48`개 파라미터 감소.

---

## 최종 커널 연산

```python
# QKV 사영
qkv = self.W_QKV(Q_in).view(B, N, 3, H, d_h)
Q_pre, K_pre = qkv[:,:,0], qkv[:,:,1]

# RoPE
Q_pre, K_pre = apply_rotary_pos_emb(Q_pre, K_pre, cos, sin)

# QK-RMSNorm (gradient 스케일 안정화)
Q_proj = rms_norm(Q_pre.transpose(1, 2))
K_proj = rms_norm(K_pre.transpose(1, 2))

# 내적 → ELU+1 → power (dense W, RoPE 정보 보존)
S = Q_proj @ K_proj.transpose(-1, -2) / sqrt(d_h)
W = (F.elu(S) + 1.0) ** kernel_power
```
