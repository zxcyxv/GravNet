# QK-RMSNorm 도입 근거

## 1. 배경

AMK-PD의 커널 어텐션은 다음 순서로 계산된다.

```
score_ij = (Q_i · K_j) / √d_h
W_ij     = ReLU(score_ij)^p
C_i      = (Σ_j W_ij · V_j) / (Σ_j W_ij + ε)   ← 분모 정규화
m_i      = C_i − V_i                               ← Mean Shift 벡터
```

여기서 `p = kernel_power`, `d_h = head_dim`이다.

---

## 2. 순전파 분석: C는 bounded

`C_i = WV / Norm`의 구조를 분석한다.

ReLU 이후 `W_ij ≥ 0`이고, 정규화에 의해 각 행의 가중치 합이 1이 된다:

$$\alpha_{ij} = \frac{W_{ij}}{\sum_k W_{ik}} \geq 0, \quad \sum_j \alpha_{ij} = 1$$

따라서 `C_i = Σ_j α_ij · V_j`는 V 벡터들의 볼록 결합(Convex Combination)이다. Jensen 부등식에 의해:

$$\|C_i\| \leq \max_j \|V_j\|$$

**결론: Q, K의 노름이 아무리 커도 순전파에서 C의 분산은 폭발하지 않는다.**

---

## 3. 역전파 분석: gradient는 bounded되지 않는다

Q/K 노름의 드리프트가 역전파에 미치는 영향을 분석한다.

### 3.1 기호 정의

학습 중 Q, K 노름이 초기 대비 α배 커졌다고 가정한다:

$$Q_i = \alpha \tilde{q}_i, \quad K_j = \alpha \tilde{k}_j$$

여기서 `q̃`, `k̃`는 α 드리프트 이전의 값이다.

### 3.2 순전파에서 α의 소거

$$\text{score}_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_h}} = \alpha^2 \cdot \tilde{s}_{ij}$$

$$W_{ij} = \text{ReLU}(\alpha^2 \tilde{s}_{ij})^p = \alpha^{2p} \cdot \text{ReLU}(\tilde{s}_{ij})^p \quad (\tilde{s}_{ij} > 0 \text{ 인 경우})$$

$$\text{Norm}_i = \sum_j W_{ij} = \alpha^{2p} \cdot \widetilde{\text{Norm}}_i$$

$$C_i = \frac{\alpha^{2p} \sum_j \text{ReLU}(\tilde{s}_{ij})^p \cdot V_j}{\alpha^{2p} \cdot \widetilde{\text{Norm}}_i} = \tilde{C}_i$$

α가 완전히 소거되어 C는 α에 독립이다. 이것이 순전파 안정성의 원인이다.

### 3.3 역전파에서 α가 소거되지 않는다

상류 gradient `∂L/∂C_i = δ_i` (α에 독립)로부터 역전파를 추적한다.

**Step 1: ∂L/∂W_ij**

몫의 미분으로부터:

$$\frac{\partial C_i}{\partial W_{ij}} = \frac{V_j \cdot \text{Norm}_i - \left(\sum_k W_{ik} V_k\right) \cdot 1}{\text{Norm}_i^2} = \frac{V_j - C_i}{\text{Norm}_i}$$

따라서:

$$\frac{\partial L}{\partial W_{ij}} = \delta_i \cdot \frac{V_j - C_i}{\text{Norm}_i} \propto \frac{1}{\alpha^{2p}}$$

`(V_j - C_i)`는 α에 독립, `Norm_i ∝ α^{2p}`이므로 gradient는 Q/K 노름 성장에 따라 감쇠한다.

**Step 2: ∂L/∂score_ij**

power kernel의 미분:

$$\frac{\partial W_{ij}}{\partial \text{score}_{ij}} = p \cdot \text{ReLU}(\text{score}_{ij})^{p-1} \cdot \mathbf{1}_{[\text{score}_{ij} > 0]}
= p \cdot (\alpha^2 \tilde{s}_{ij})^{p-1} \propto \alpha^{2(p-1)}$$

연쇄 법칙:

$$\frac{\partial L}{\partial \text{score}_{ij}} = \frac{\partial L}{\partial W_{ij}} \cdot \frac{\partial W_{ij}}{\partial \text{score}_{ij}} \propto \alpha^{-2p} \cdot \alpha^{2(p-1)} = \alpha^{-2}$$

**p에 무관하게** score gradient는 α^{-2}로 감쇠한다.

**Step 3: ∂L/∂Q_i**

$$\frac{\partial L}{\partial Q_i} = \sum_j \frac{\partial L}{\partial \text{score}_{ij}} \cdot \frac{\partial \text{score}_{ij}}{\partial Q_i}
= \sum_j \frac{\partial L}{\partial \text{score}_{ij}} \cdot \frac{K_j}{\sqrt{d_h}}$$

$$\propto \sum_j \alpha^{-2} \cdot \alpha \tilde{k}_j = \alpha^{-1} \cdot \sum_j \frac{\partial L}{\partial \tilde{s}_{ij}} \cdot \frac{\tilde{k}_j}{\sqrt{d_h}}$$

$$\boxed{\frac{\partial L}{\partial Q_i} \propto \alpha^{-1}}$$

마찬가지로 `∂L/∂K_j ∝ α^{-1}`. W_QKV의 gradient도 동일한 α^{-1} 의존성을 갖는다.

### 3.4 결과 요약

| Q/K 노름 변화 | ∂L/∂Q, ∂L/∂K | 결과 |
|---|---|---|
| α → ∞ (노름 증가) | → 0 | **Vanishing gradient**: 어텐션 패턴 학습 중단 |
| α → 0 (노름 감소) | → ∞ | **Exploding gradient**: 학습 불안정 |

**핵심**: 순전파에서 α^{2p}가 Norm에 의해 소거되지만, 역전파에서는 이 소거 과정이 gradient에 α^{-2}의 잔여 의존성을 남긴다. 이 의존성은 p에 무관하므로 kernel_power를 바꾸어도 해소되지 않는다.

---

## 4. QK-RMSNorm이 문제를 해소하는 이유

내적 직전에 Q, K에 RMSNorm을 적용하면:

$$\tilde{Q}_i = \text{rms\_norm}(Q_i), \quad \tilde{K}_j = \text{rms\_norm}(K_j)$$

RMSNorm의 정의에 의해:

$$\|\tilde{Q}_i\|_{\text{RMS}} = 1 \quad \Rightarrow \quad \|\tilde{Q}_i\| = \sqrt{d_h}$$

즉, 학습 중 Q나 K의 노름이 아무리 변하더라도 정규화 이후의 `Q̃`, `K̃`의 노름은 `√d_h`로 고정된다. 위 분석의 α는 항상 1이 되어 gradient 스케일이 학습 전 구간에 걸쳐 일정하게 유지된다.

또한 RoPE 이후에 정규화를 적용하므로, RoPE가 인코딩한 상대 위치 정보는 각도(방향)에 담겨 있고 RMSNorm은 크기만 조정하므로 위치 정보가 손실되지 않는다.

---

## 5. 선행 연구

QK-Norm (또는 QK-LayerNorm)은 대규모 모델에서 어텐션 불안정을 해소하기 위해 독립적으로 재발견된 기법이다.

- **ViT-22B** (Dehghani et al., 2023): QK-Norm을 도입하여 22B 파라미터 ViT의 학습 안정성 확보
- **Stable Diffusion 3** (Esser et al., 2024): MM-DiT 어텐션에 QK-Norm 적용

---

## 6. 구현

```python
# RoPE 적용 이후, 내적 직전
Q_proj = rms_norm(Q_proj)   # [B, H, N, d_h]
K_proj = rms_norm(K_proj)   # [B, H, N, d_h]

scale = self.head_dim ** -0.5
W = torch.matmul(Q_proj, K_proj.transpose(-1, -2)) * scale + self.attn_bias
W = F.relu(W)
```

`rms_norm`은 파라미터 없는 함수이므로 추가 학습 파라미터가 없다.
