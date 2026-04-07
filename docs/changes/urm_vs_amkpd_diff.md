# URM vs AMK-PD: 전체 차이점 분석

> 목적: URM pretrain.py를 그대로 사용하여 AMK-PD 모델만 교체하는 공정한 실험을 위한 사전 분석.

---

## A. 의도적 아키텍처 차이 (유지)

이 차이들은 AMK-PD의 핵심 아이디어이므로 그대로 유지한다.

### A1. Attention Output → Mean Shift

| | URM | AMK-PD |
|---|---|---|
| 연산 | `attn_output = softmax(QK^T/√d)V` | `C = softmax(QK^T/√d)V`, `m = C - V` |
| Residual | `hidden + attn_output` | `Q + W_O(m)` |
| 의미 | Standard attention | Graph Laplacian diffusion: m = -L_rw V |

### A2. QK-RMSNorm

| | URM | AMK-PD |
|---|---|---|
| Q/K 정규화 | 없음 | `rms_norm(Q)`, `rms_norm(K)` (RoPE 후) |

AMK-PD에서 m = C - V 구조는 V의 스케일이 직접 출력에 영향을 주므로,
Q/K norm이 없으면 attention collapse가 발생할 수 있다 (docs/changes/qk_norm.md 참조).

---

## B. 인터페이스 차이 (pretrain.py 호환을 위해 통일 필수)

### B1. Forward 시그니처

```python
# URM
def forward(self, carry: URMCarry, batch: Dict[str, torch.Tensor],
            compute_target_q=False) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:

# AMK-PD (현재)
def forward(self, carry: AMKPDCarry, batch: tuple) -> Tuple[AMKPDCarry, Tensor, Tuple]:
```

**차이**: URM은 `batch`가 `Dict` (keys: "inputs", "labels", "puzzle_identifiers"),
AMK-PD는 `tuple(inputs, labels)`.

**통일 방향**: AMK-PD가 `Dict[str, Tensor]`를 받도록 변경. `compute_target_q` 인자 추가.

### B2. Return 시그니처

```python
# URM — outer wrapper (URM.forward)
return (URMCarry, {"logits": ..., "q_halt_logits": ..., "q_continue_logits": ...})

# AMK-PD (현재)
return (AMKPDCarry, logits, (q_halt_logits, q_continue_logits))
```

**통일 방향**: AMK-PD도 `(Carry, Dict)` 반환으로 변경.

### B3. Carry 구조

```python
# URM
@dataclass
class URMCarry:
    current_hidden: torch.Tensor              # [B, seq_len, hidden_size]
    steps: Optional[torch.Tensor] = None      # [B]
    halted: Optional[torch.Tensor] = None     # [B]
    current_data: Optional[Dict[str, torch.Tensor]] = None  # 범용 dict

# AMK-PD (현재)
@dataclass
class AMKPDCarry:
    current_hidden: torch.Tensor    # [B, N, d]
    steps: torch.Tensor             # [B]
    halted: torch.Tensor            # [B]
    current_inputs: torch.Tensor    # [B, N]
    current_labels: torch.Tensor    # [B, N]
```

**차이**: URM은 `current_data: Dict`로 inputs/labels/puzzle_identifiers를 범용 저장.
AMK-PD는 개별 필드.

**통일 방향**: AMK-PD를 `current_data: Dict` 방식으로 변경.

### B4. initial_carry 시그니처

```python
# URM
def initial_carry(self, batch: Dict[str, torch.Tensor]) -> URMCarry:
    batch_size = batch["inputs"].shape[0]
    # empty hidden + steps=0 + halted=True + current_data={empty_like}

# AMK-PD (현재)
def initial_carry(self, batch_size: int, seq_len: int, device: torch.device) -> AMKPDCarry:
```

**통일 방향**: AMK-PD를 `initial_carry(batch)` 시그니처로 변경.

### B5. Wrapper 구조 (URM: 2단 / AMK-PD: 1단)

```python
# URM: URM(outer) → URM_Inner(forward)
#   URM.forward: carry reset + halting 판정
#   URM_Inner.forward: embedding + 3중 루프 + output heads

# AMK-PD: AMKPDModel 하나에서 전부 처리
```

**통일 방향**: AMK-PD도 2단 분리 (`AMKPDModel` outer + `AMKPDModel_Inner` inner).
이유: pretrain.py의 `model.model.puzzle_emb` 같은 접근 패턴, `ACTLossHead(model)` 래핑 등이
outer wrapper 존재를 전제로 함.

### B6. Q Head Pooling

```python
# URM — 첫 번째 토큰 (CLS-like)
q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)

# AMK-PD — 시퀀스 평균
q_logits = self.halt_head(Q.mean(dim=1))
```

**차이**: URM은 puzzle_emb 토큰(첫 번째)을 CLS로 사용. AMK-PD는 puzzle_emb이 없으므로 mean pooling.

**결정 필요**: puzzle_emb 도입 여부에 따라 달라짐. puzzle_emb 있으면 첫 토큰, 없으면 mean.

### B7. Puzzle Embedding

```python
# URM
puzzle_emb = CastedSparseEmbedding(num_puzzle_identifiers, puzzle_emb_ndim=512, ...)
# → [B, puzzle_emb_len, hidden_size] 토큰을 시퀀스 앞에 prepend
# → lm_head 출력에서 puzzle 토큰 제거: output[:, puzzle_emb_len:]

# AMK-PD
# 없음
```

**차이**: URM은 퍼즐 종류별 학습 가능한 임베딩을 시퀀스 앞에 붙여서
모델이 "이 퍼즐이 어떤 종류인지" 인식할 수 있게 함.

**결정 필요**: 스도쿠 전용이면 puzzle_emb_ndim=0으로 비활성화 가능.
pretrain.py가 puzzle_identifiers를 batch에 항상 넘기므로, 모델이 이를 무시하면 됨.

### B8. ignore_index (라벨 마스킹)

```python
# URM
IGNORE_LABEL_ID = -100  # dataset.py에서 마스크할 위치를 -100으로 설정

# AMK-PD
ignore_index = 0  # PAD 토큰 = 0을 무시
```

**차이**: URM 데이터셋은 마스크할 위치에 -100, AMK-PD는 PAD=0.

**통일 방향**: URM pretrain.py + URM 데이터셋을 사용하면 자동으로 -100 방식.
AMK-PD 모델 내부에서 ignore_index를 직접 사용하는 곳은 없으므로 (loss는 pretrain.py가 처리)
모델 수정 불필요.

### B9. Logits 슬라이싱

```python
# URM — puzzle_emb 토큰 제거
output = lm_head(hidden_states)[:, self.puzzle_emb_len:]

# AMK-PD — 슬라이싱 없음
logits = self.lm_head(Q)
```

**통일 방향**: puzzle_emb_ndim=0이면 puzzle_emb_len=0이므로 슬라이싱이 no-op.
puzzle_emb_ndim>0이면 슬라이싱 필요.

---

## C. 구현 디테일 차이 (성능/안정성 영향)

### C1. Flash Attention vs F.scaled_dot_product_attention

```python
# URM
from flash_attn import flash_attn_func
attn_output = flash_attn_func(q=query, k=key, v=value, causal=False)

# AMK-PD
C = F.scaled_dot_product_attention(Q_proj, K_proj, V_proj, is_causal=False)
```

**차이**: URM은 flash_attn 패키지 직접 사용 (fa2/fa3 호환).
AMK-PD는 PyTorch 내장 SDPA (내부적으로 FlashAttention 백엔드 선택 가능).

**참고**: AMK-PD는 `m = C - V`를 계산해야 하므로 attention output (C)을 직접 사용.
URM의 flash_attn_func도 동일한 출력을 주므로 교체 가능하나,
AMK-PD는 V를 별도로 들고 있어야 하므로 URM의 Attention 클래스를 직접 재사용할 수 없음.

**결정**: AMK-PD의 SDPA 유지 (Mean Shift 구조상 필요). 성능 차이 미미.

### C2. CastedLinear/Embedding vs nn.Linear/Embedding

```python
# URM: CastedLinear — forward 시 weight를 forward_dtype으로 캐스팅
class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias else None)

# AMK-PD: 표준 nn.Linear (autocast가 dtype 관리)
```

**차이**: URM은 weight를 float32로 저장하고 forward 시 bfloat16으로 캐스팅.
AMK-PD는 torch.autocast 컨텍스트에 의존.

**영향**: pretrain.py가 autocast를 사용하지 않고 CastedLinear의 캐스팅에 의존하는지 확인 필요.
AMK-PD가 nn.Linear을 쓰면 weight가 float32로 연산될 수 있음 → 메모리/속도 차이.

**결정 필요**: AMK-PD도 CastedLinear로 교체할지, 아니면 autocast 래핑으로 충분한지.

### C3. ConvSwiGLU dwconv dtype

```python
# URM
self.dwconv = nn.Conv1d(...).to(dtype=torch.bfloat16)  # 하드코딩

# AMK-PD
self.dw_conv = nn.Conv1d(...)  # dtype 미지정 (기본 float32)
```

**영향**: Conv1d가 float32로 실행되면 mixed precision 경계에서 캐스팅 발생.

**결정 필요**: C2와 동일한 맥락. CastedLinear 채택 시 함께 통일.

### C4. RoPE max_position_embeddings

```python
# URM
max_position_embeddings = config.seq_len + self.puzzle_emb_len  # 81 + α

# AMK-PD
max_position_embeddings = 8192  # 하드코딩 (과대)
```

**영향**: 기능적 차이 없음 (더 긴 cos/sin 버퍼가 메모리만 약간 더 사용).

**통일 방향**: config에서 seq_len을 받아서 동적 설정.

### C5. CastedEmbedding vs nn.Embedding

```python
# URM: CastedEmbedding — cast_to=bfloat16
embedding = self.embed_tokens(input.to(torch.int32))  # int32로 캐스팅

# AMK-PD
X = self.embedding(new_inputs)  # 표준 nn.Embedding
```

**차이**: URM은 입력을 int32로 캐스팅, 출력을 bfloat16으로 캐스팅.

### C6. init_hidden dtype

```python
# URM
nn.Buffer(trunc_normal_init_(torch.empty(..., dtype=self.forward_dtype), std=1), persistent=True)
# forward_dtype = bfloat16

# AMK-PD
self.register_buffer('init_hidden', trunc_normal_init_(torch.empty(d_model), std=1.0))
# dtype = float32 (기본값)
```

**차이**: URM은 init_hidden을 bfloat16으로 저장. AMK-PD는 float32.

---

## D. 훈련 하이퍼파라미터 차이 (pretrain.py 사용 시 config로 통일)

pretrain.py를 사용하면 아래 항목들은 config yaml에서 통일된다.

| 항목 | URM 기본값 | AMK-PD 기본값 | pretrain.py 사용 시 |
|------|-----------|-------------|-------------------|
| **Loss** | ACTLossHead: CE/count + 0.5×BCE | CE(mean) + 0.01×MSE | ACTLossHead 사용 |
| **Optimizer** | AdamATan2 | AdamW | AdamATan2 사용 |
| **weight_decay** | 0.1 | 1.0 | config 값 (0.1) |
| **batch_size** | 768 (global) | 128 | config 값 |
| **hidden_size** | 512 | 384 | config에서 지정 |
| **num_layers** | 8 | 6 | config에서 지정 |
| **H_cycles** | 4 | 2 | config에서 지정 |
| **L_cycles** | 3 | 6 | config에서 지정 |
| **LR schedule** | cosine warmup (min_ratio=1.0) | 동일 | pretrain.py 사용 |
| **Grad clipping** | 없음 | 1.0 | pretrain.py 방식 (없음) |
| **EMA** | 지원 (0.999) | 미지원 | pretrain.py 사용 |
| **Eval 루프** | adaptive halt → halted만 메트릭 | 고정 loops → 전체 메트릭 | pretrain.py 방식 |
| **Mixed precision** | CastedLinear/Embedding | torch.autocast | pretrain.py 방식 |

---

## 요약: 수정이 필요한 항목 체크리스트

### 필수 (pretrain.py 호환)
- [ ] Forward 시그니처: `batch: Dict` 수용
- [ ] Return 시그니처: `(Carry, Dict)` 반환
- [ ] Carry 구조: `current_data: Dict` 방식
- [ ] initial_carry: `batch` dict 수용
- [ ] Wrapper 2단 분리 (outer + inner)
- [ ] Logits 슬라이싱 (puzzle_emb_len 지원)
- [ ] URMConfig 호환 (config_dict로 초기화)

### 결정 완료
- [x] Puzzle embedding: URM과 동일하게 도입 (puzzle_emb_ndim config로 제어)
- [x] Q head pooling: 첫 토큰 방식 (puzzle_emb 토큰을 CLS로 사용)
- [x] CastedLinear/CastedEmbedding: URM과 동일하게 채택
- [x] dwconv bfloat16 하드코딩: URM과 동일하게 채택
- [ ] 하이퍼파라미터 (hidden_size, num_layers, H/L_cycles): 나중에 결정
