# AMK-PD vs URM: Gradient Norm 비교 분석

## 1. 실험 설정

| 항목 | AMK-PD | URM |
|------|--------|-----|
| d_model | 384 | 384 |
| num_heads | 8 | 8 |
| num_layers | 2 | 2 |
| batch_size | 64 | 64 |
| loops | 8 | 8 |
| H_cycles | 2 | 2 |
| L_cycles | 1 | 1 |
| full_grad | Yes | Yes |
| Params | 6,704,644 | 3,554,306 |
| lr | 3e-4 | 3e-4 |
| optimizer | AdamW (β1=0.9, β2=0.95) | AdamW (β1=0.9, β2=0.95) |
| dataset | sudoku-extreme-1k-aug-1000 | sudoku-extreme-1k-aug-1000 |

---

## 2. 훈련 로그

### AMK-PD

```
[    50/156400]  loss=1.6574  tok=39.29%  gnorm=1.432
[   100/156400]  loss=1.5691  tok=39.70%  gnorm=1.904
[   150/156400]  loss=1.5030  tok=40.30%  gnorm=0.688
[   200/156400]  loss=1.5183  tok=38.93%  gnorm=0.698
[   250/156400]  loss=1.5075  tok=39.18%  gnorm=1.055
[   300/156400]  loss=1.5242  tok=39.49%  gnorm=1.067
[   350/156400]  loss=1.5159  tok=39.37%  gnorm=0.453
[   400/156400]  loss=1.5085  tok=39.66%  gnorm=0.767
[   450/156400]  loss=1.5048  tok=39.47%  gnorm=0.916
[   500/156400]  loss=1.5093  tok=39.27%  gnorm=0.762
[   550/156400]  loss=1.4781  tok=40.20%  gnorm=0.658
[   600/156400]  loss=1.4741  tok=40.24%  gnorm=0.586
[   650/156400]  loss=1.4673  tok=39.76%  gnorm=0.495
[   700/156400]  loss=1.4546  tok=40.39%  gnorm=0.386
[   750/156400]  loss=1.4403  tok=40.80%  gnorm=0.605
[   800/156400]  loss=1.4257  tok=42.52%  gnorm=0.426
[   850/156400]  loss=1.4352  tok=40.51%  gnorm=0.578
```

### URM

```
[    50/156400]  loss=1.9539  tok=34.65%  gnorm=3.757
[   100/156400]  loss=1.5440  tok=39.58%  gnorm=3.201
[   150/156400]  loss=1.4461  tok=43.15%  gnorm=1.225
[   200/156400]  loss=1.4073  tok=44.41%  gnorm=1.180
[   250/156400]  loss=1.3942  tok=44.04%  gnorm=1.274
[   300/156400]  loss=1.3644  tok=45.83%  gnorm=0.703
[   350/156400]  loss=1.3292  tok=46.76%  gnorm=0.671
[   400/156400]  loss=1.3011  tok=47.61%  gnorm=0.781
[   450/156400]  loss=1.3156  tok=45.79%  gnorm=1.322
[   500/156400]  loss=1.2866  tok=48.07%  gnorm=1.111
[   550/156400]  loss=1.2417  tok=48.57%  gnorm=0.781
[   600/156400]  loss=1.2118  tok=51.10%  gnorm=0.809
[   650/156400]  loss=1.2886  tok=47.01%  gnorm=1.005
[   700/156400]  loss=1.2370  tok=49.04%  gnorm=0.997
[   750/156400]  loss=1.1976  tok=51.43%  gnorm=0.674
[   800/156400]  loss=1.1582  tok=52.74%  gnorm=0.698
[   850/156400]  loss=1.2248  tok=49.04%  gnorm=1.196
```

---

## 3. 수치 비교

| Step | AMK-PD gnorm | URM gnorm | AMK-PD loss | URM loss | AMK-PD tok% | URM tok% |
|------|-------------|-----------|-------------|----------|-------------|----------|
| 50 | 1.432 | 3.757 | 1.657 | 1.954 | 39.3% | 34.7% |
| 100 | 1.904 | 3.201 | 1.569 | 1.544 | 39.7% | 39.6% |
| 200 | 0.698 | 1.180 | 1.518 | 1.407 | 38.9% | 44.4% |
| 500 | 0.762 | 1.111 | 1.509 | 1.287 | 39.3% | 48.1% |
| 700 | 0.386 | 0.997 | 1.455 | 1.237 | 40.4% | 49.0% |
| 850 | 0.578 | 1.196 | 1.435 | 1.225 | 40.5% | 49.0% |

**관찰된 현상:**
- AMK-PD의 gnorm: step 200 이후 0.4~0.7 범위로 하락하며, 하락 추세 지속
- URM의 gnorm: 0.7~1.3 범위에서 진동하며 안정적으로 유지
- AMK-PD의 loss/tok_acc 개선 속도가 URM 대비 현저히 느림
- AMK-PD의 학습 곡선 정체 시점과 gnorm 하락 시점이 일치

---

## 4. 구조적 차이 분석

### 4.1 어텐션 출력의 수학적 구조

**URM (Softmax Attention):**
```
output = O_proj(softmax(QK^T / √d) · V)
```
V의 가중 평균. 학습이 수렴해도 출력의 크기(norm) ≠ 0.

**AMK-PD (Mean Shift):**
```
C = (W · V) / Norm
m = C - V
output = W_O(m)
```
V의 가중 평균과 자기 자신의 **차이**. 학습이 수렴하면 C → V이므로 m → 0.

### 4.2 m → 0 이 발생하는 경로

Mean Shift 벡터 `m = C - V`가 0에 수렴하는 시나리오:

1. **어텐션 맵의 대각 지배**: W의 대각 성분이 커지면 각 토큰의 가중 평균 C가 자기 자신 V에 수렴
2. **동류 토큰의 V 수렴**: 같은 클래스 토큰끼리 V 표현이 유사해지면 가중 평균 ≈ 자기 자신
3. **다항식 커널의 샤프닝**: `relu(·)^2`가 W를 sparse하게 만들어 소수 이웃만 참조 → C ≈ V 가속

### 4.3 Gradient 체인 비교

**AMK-PD에서 W_Q까지의 gradient 경로:**
```
∂L/∂W_Q = ... × softplus(dt) × W_O × ∂m/∂W × 2·relu(·) × elu'(·) × H_context
```
축소 병목 3개: `softplus(dt)`, `W_O` (zero-init), `2·relu(·)` (sparse 시 작아짐)

**URM에서 W_Q까지의 gradient 경로:**
```
∂L/∂W_Q = ... × O_proj × ∂(softmax·V)/∂Q × hidden_states
```
softmax gradient는 확률 분포의 성질로 보존됨. 축소 병목 없음.

### 4.4 W_O의 gradient

```
∂L/∂W_O = (∂L/∂m_proj)^T · m_concat
```
m_concat → 0이면 ∂L/∂W_O → 0. W_O가 성장하기 전에 m이 줄어들면 W_O는 작게 머무른다.

### 4.5 softplus(dt) 곱셈

```python
Q_interact = Q_in + softplus(dt) × W_O(m)
```
어텐션 기여에 스칼라 게이트가 곱해짐. `softplus(0.1) ≈ 0.744`. dt가 학습으로 줄어들면 게이트도 줄어든다. URM에는 이 스케일링이 없다 (잔차를 직접 더함).

---

## 5. 구조 대비 표

| 설계 요소 | AMK-PD | URM |
|-----------|--------|-----|
| 어텐션 출력 | `C - V` (차이) | `softmax·V` (가중합) |
| 출력 사영 init | zero-init | truncated normal |
| 잔차 스케일링 | `softplus(dt)×` 곱셈 | 직접 더하기 |
| 커널 함수 | `relu(·)^2` (sparse) | softmax (확률 분포) |
| 정규화 위치 | Pre-Norm | Post-Norm |
| 정규화 파라미터 | LayerNorm (학습 가능) | RMSNorm (파라미터 없음) |

---

## 6. 요약

gradient norm 하락의 구조적 원인은 Mean Shift 공식 `m = C - V`에 있다. 학습이 진행되어 어텐션이 수렴하면 가중 평균 C가 자기 자신 V에 가까워지면서 m → 0이 되고, 이 m에 의존하는 gradient 경로 전체가 축소된다. W_O zero-init, softplus(dt) 게이트, relu^2 커널은 이 축소를 가속시키는 요인이다.
