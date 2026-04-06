"""
AMK-PD: Asymmetric Mapped Kernel Particle Dynamics
PyTorch 구현

아키텍처 개요 (URM 3중 루프 + Carry 구조):
- 외부 루프(loops): carry를 유지하며 반복 호출. 샘플별 adaptive halting.
- 중간 루프(H_cycles): H-1회 no_grad burn-in + 마지막 1회 gradient 추적.
- 내부 루프(L_cycles): K개 블록 순차 통과. X는 각 블록 내부 정보 공간(H_context)에 주입.
- Markovian gradient isolation: forward 1회의 gradient depth = L_cycles × K.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext
from dataclasses import dataclass, replace


# ============================================================
# 유틸리티 (URM에서 이식)
# ============================================================

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """Truncated LeCun Normal 초기화 (JAX 방식, URM models/common.py 원본)."""
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


def rms_norm(hidden_states: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """파라미터 없는 RMSNorm (URM models/layers.py 원본)."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)


# ============================================================
# RoPE (Rotary Positional Embedding) — URM layers.py 이식
# ============================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


CosSin = tuple[torch.Tensor, torch.Tensor]


@dataclass
class AMKPDCarry:
    """URM의 URMCarry에 대응하는 carry 구조체 (ref/urm.py:14-18)"""
    current_hidden: torch.Tensor   # [B, N, d] — detached 은닉 상태
    steps: torch.Tensor            # [B] int32 — 외부 루프 카운터
    halted: torch.Tensor           # [B] bool — 샘플별 정지 플래그
    current_inputs: torch.Tensor   # [B, N] long — 저장된 입력 토큰
    current_labels: torch.Tensor   # [B, N] long — 저장된 정답 라벨


def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """URM과 동일한 RoPE 적용. q, k: [B, N, H, d_h], cos, sin: [N, d_h]"""
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


# ============================================================
# AMK_Block: Micro-Layer
# ============================================================

class AMK_Block(nn.Module):
    """
    AMK_Block (Micro-Layer)

    은닉 상태 Q_in 과 초기 상태 X 를 입력받아
    4단계 연산(스펙트럼 생성 → 보흐너 사영 → 중력 응집 → ConvSwiGLU)을 거쳐
    업데이트된 Q_out 을 반환합니다.

    Args:
        d_model        (int)  : 은닉 차원 d
        num_heads      (int)  : 멀티헤드 어텐션 헤드 수
        dt             (float): Explicit Euler 스텝 사이즈 초기값 (학습 가능)
        expansion_ratio(int)  : ConvSwiGLU 내부 팽창 비율 m
        conv_kernel_size(int) : Depthwise Conv1D 커널 크기
        kernel_power   (int)  : 인력 행렬 다항식 거듭제곱 차수 (1=선형, 2=이차, 3=삼차)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dt: float = 0.5,
        expansion_ratio: int = 4,
        conv_kernel_size: int = 3,
        kernel_power: int = 2,
    ):
        super().__init__()
        self.d_model = d_model          # d
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.kernel_power = kernel_power
        # URM과 동일한 intermediate 크기: round(expansion * d * 2/3), 256 정렬
        inner_dim = (-(round(expansion_ratio * d_model * 2 / 3) // -256)) * 256

        # ── 학습 가능한 스텝 파라미터 ──────────────────────────────────
        # self.dt  = nn.Parameter(torch.tensor(float(dt)))   # Δt: scalar (고정값 1.0으로 대체)

        # ── 멀티헤드 선형 사영 (QKV fused) ────────────────────────────────
        self.W_QKV = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_O_aux = nn.Linear(d_model, d_model, bias=False)

        # ── ConvSwiGLU ───────────────────────────────────────────────
        # 선형 팽창: d → 2*inner_dim  (G 와 U 로 분할)
        self.W_up = nn.Linear(d_model, 2 * inner_dim, bias=False)

        # Depthwise Conv1D (URM 방식: padding=k//2, 출력 슬라이싱)
        self.dw_conv = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=inner_dim,
            bias=True,
        )

        # 최종 투영: inner_dim → d
        self.W_down = nn.Linear(inner_dim, d_model, bias=False)

        # ── 정규화: 파라미터 없는 rms_norm (URM과 동일) ──────────────────

        # ── 시각화용 텔레메트리 (Visualization Telemetry) ────────────────
        self.log_viz = False
        self.viz_W = []
        self.viz_m = []
        self.viz_H = []  # H_context = LayerNorm(Q) + X — 실제 중력 작용 공간

        # ── bypass 모니터링용 activation norm ────────────────────────────
        self._last_m_norm: torch.Tensor | float = 0.0
        self._last_C_norm: torch.Tensor | float = 0.0

    @property
    def last_m_norm(self) -> float:
        v = self._last_m_norm
        return v.item() if isinstance(v, torch.Tensor) else v

    @property
    def last_C_norm(self) -> float:
        v = self._last_C_norm
        return v.item() if isinstance(v, torch.Tensor) else v

    # ----------------------------------------------------------
    def forward(self, Q_in: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        """
        Args:
            Q_in    : [B, N, d]  현재 은닉 상태 (입력 주입은 L_cycle 레벨에서 수행)
            cos_sin : (cos, sin) RoPE 캐시 — 각각 [N, d_h]
        Returns:
            Q_out: [B, N, d]  업데이트된 은닉 상태
        """
        B, N, d = Q_in.shape        # Batch, Seq_len, Dim
        H = self.num_heads
        d_h = self.head_dim

        # ══════════════════════════════════════════════════════
        # Step 1: 멀티헤드 선형 사영 + 헤드 분할 (fused QKV)
        # ══════════════════════════════════════════════════════

        qkv = self.W_QKV(Q_in).view(B, N, 3, H, d_h)
        Q_pre = qkv[:, :, 0]   # [B, N, H, d_h]
        K_pre = qkv[:, :, 1]   # [B, N, H, d_h]

        # ══════════════════════════════════════════════════════
        # Step 2: RoPE 적용 (내적 이전, Q/K 벡터에 회전)
        # ══════════════════════════════════════════════════════

        cos, sin = cos_sin
        Q_pre, K_pre = apply_rotary_pos_emb(Q_pre, K_pre, cos, sin)

        Q_proj = Q_pre.transpose(1, 2)  # [B, H, N, d_h]
        K_proj = K_pre.transpose(1, 2)  # [B, H, N, d_h]
        V_proj = qkv[:, :, 2].transpose(1, 2)  # [B, H, N, d_h]

        # RoPE 적용 이후, 내적 직전에 RMSNorm 추가
        Q_proj = rms_norm(Q_proj)
        K_proj = rms_norm(K_proj)

        # ══════════════════════════════════════════════════════
        # Step 3: 헤드별 다항식 인력 행렬 (RoPE 내적 스칼라 → ELU+1 → power)
        #
        # ReLU 대신 ELU+1을 내적 스칼라에 적용:
        #   S_ij = Q_i · K_j / √d_h  (음수 가능, RoPE 상대위치 포함)
        #   W_ij = (ELU(S_ij) + 1)^p  (항상 > 0, 단조 변환으로 RoPE 정보 보존)
        #
        # backup2의 ELU+1은 벡터에 적용 후 내적 → RoPE와 양립 불가
        # 스칼라에 적용하면 RoPE 상대위치 순서 구조를 보존하면서 dense W 보장
        # ══════════════════════════════════════════════════════

        scale = self.head_dim ** -0.5
        S = torch.matmul(Q_proj, K_proj.transpose(-1, -2)) * scale  # [B, H, N, N]
        W = F.elu(S) + 1.0  # 항상 > 0, dense W 보장
        if self.kernel_power == 2:
            W = W * W
        elif self.kernel_power == 4:
            W = W * W; W = W * W
        elif self.kernel_power != 1:
            W = W ** self.kernel_power

        # ══════════════════════════════════════════════════════
        # Step 5: 헤드별 Mean Shift
        # ══════════════════════════════════════════════════════

        Attraction = torch.matmul(W, V_proj)               # [B, H, N, d_h]
        Norm = W.sum(dim=-1, keepdim=True) + 1e-6            # [B, H, N, 1]
        C = Attraction / Norm                               # [B, H, N, d_h]
        m = C - V_proj                                      # [B, H, N, d_h]

        # ══════════════════════════════════════════════════════
        # Step 6: 헤드 병합 + 출력 사영 (W_O zero-init)
        # ══════════════════════════════════════════════════════

        m_concat = m.transpose(1, 2).contiguous().view(B, N, d)           # [B, N, d]
        m_proj = self.W_O_aux(m_concat)                                    # [B, N, d]

        # graph break 방지: .item() 없이 텐서로 저장
        self._last_m_norm = m_concat.detach().norm(dim=-1).mean()
        self._last_C_norm = C.detach().transpose(1, 2).contiguous().view(B, N, d).norm(dim=-1).mean()

        # ══════════════════════════════════════════════════════
        # Step 7: 잔차 + Post-Addition Norm (분산 팽창 억제)
        # ══════════════════════════════════════════════════════

        Q_interact = rms_norm(Q_in + m_proj)  # [B, N, d]

        # ── 시각화 캐싱 (헤드별 데이터 저장) ────────────────────────────────
        if self.log_viz:
            # 헤드별 정규화된 인력 행렬: [B, H, N, N]
            W_norm = W / (W.sum(dim=-1, keepdim=True) + 1e-6)
            self.viz_W.append(W_norm.detach())             # [B, H, N, N]
            self.viz_m.append(m.detach())                  # [B, H, N, d_h] (헤드별)
            self.viz_H.append(Q_in.detach())               # [B, N, d]
            
        # ══════════════════════════════════════════════════════
        # Step 4: ConvSwiGLU with Micro-Residual
        # ══════════════════════════════════════════════════════

        # 선형 팽창 후 G / U 분할 (Q_interact는 이미 norm1으로 정규화됨)
        GU = self.W_up(Q_interact)          # [B, N, 2*m*d]
        G, U = GU.chunk(2, dim=-1)          # G: [B, N, m*d],  U: [B, N, m*d]

        # SwiGLU 게이팅
        H_ffn = F.silu(G) * U               # [B, N, m*d]

        # Depthwise Conv1D: (B, channels, length) 형식 필요
        N = Q_interact.shape[1]
        H_ffn_t  = H_ffn.transpose(1, 2)           # [B, inner, N]
        H_conv_t = self.dw_conv(H_ffn_t)           # [B, inner, N'] (패딩으로 길어질 수 있음)
        H_conv_t = F.silu(H_conv_t[..., :N])       # [B, inner, N] 슬라이싱 + 활성화
        H_conv   = H_conv_t.transpose(1, 2).contiguous()  # [B, N, inner]

        # 최종 투영
        H_out = self.W_down(H_conv)          # [B, N, d]

        # Micro-Residual + Post-Addition Norm (분산 팽창 억제)
        Q_out = rms_norm(Q_interact + H_out)  # [B, N, d]

        return Q_out  # [B, N, d]


# ============================================================
# AMKPDModel: 전체 아키텍처
# ============================================================

class AMKPDModel(nn.Module):
    """
    AMK-PD Model (URM 3중 루프 + Carry 구조)

    외부 루프(loops): carry 유지, 샘플별 adaptive halting.
    중간 루프(H_cycles): H-1회 no_grad burn-in + 마지막 1회 gradient.
    내부 루프(L_cycles): K개 블록 통과. X는 각 블록 내부 H_context에 주입.

    Args:
        vocab_size     (int)  : 어휘 크기
        d_model        (int)  : 은닉 차원 d
        num_heads      (int)  : 멀티헤드 어텐션 헤드 수
        num_layers     (int)  : L_cycle당 AMK_Block 수 K
        loops          (int)  : 외부 루프 최대 횟수 (carry 재호출 횟수)
        H_cycles       (int)  : 중간 루프 횟수 (H-1회 no_grad + 1회 grad)
        L_cycles       (int)  : 내부 루프 횟수 (X 주입 + 블록 통과)
        dt             (float): Euler 스텝 초기값 (각 블록이 독립적으로 학습)
        kernel_power   (int)  : 인력 행렬 다항식 거듭제곱 차수
        expansion_ratio(int)  : ConvSwiGLU 팽창 비율
        conv_kernel_size(int) : Depthwise Conv 커널 크기
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        loops: int = 6,
        H_cycles: int = 2,
        L_cycles: int = 1,
        dt: float = 0.5,
        kernel_power: int = 2,
        expansion_ratio: int = 4,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.d_model    = d_model
        self.num_layers = num_layers   # K
        self.loops      = loops        # 외부 루프 최대 횟수
        self.H_cycles   = H_cycles     # 중간 루프 (H-1 no_grad + 1 grad)
        self.L_cycles   = L_cycles     # 내부 루프
        self.burn_in_no_grad = True    # H_cycles-1 burn-in 시 no_grad 사용 여부

        # ── 토큰 임베딩 (URM: √d 스케일링) ────────────────────────────
        self.embed_scale = math.sqrt(d_model)
        self.embedding   = nn.Embedding(vocab_size, d_model)

        # ── RoPE (URM과 동일, additive pos_emb 대체) ─────────────────
        self.rotary_emb = RotaryEmbedding(
            dim=d_model // num_heads,
            max_position_embeddings=8192,
        )

        # ── Carry 리셋용 초기 벡터 (URM urm.py:101-104) ──────────────
        self.register_buffer(
            'init_hidden',
            trunc_normal_init_(torch.empty(d_model), std=1.0),
        )

        # ── K 개의 AMK_Block (루프 간 가중치 공유) ────────────────────
        self.blocks = nn.ModuleList([
            AMK_Block(
                d_model=d_model,
                num_heads=num_heads,
                dt=dt,
                expansion_ratio=expansion_ratio,
                conv_kernel_size=conv_kernel_size,
                kernel_power=kernel_power,
            )
            for _ in range(num_layers)
        ])

        # ── Q Head (URM urm.py:81,107-108 패턴) ──────────────────────
        # 출력 2개: [q_halt_logit, q_continue_logit]
        self.halt_head = nn.Linear(d_model, 2)

        # ── Language Modeling Head ────────────────────────────────────
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ── 시각화용 텔레메트리 ───────────────────────────────────────────
        self.log_viz = False
        self.viz_Q = []
        self.viz_H_global = []

        # ── 가중치 초기화 ─────────────────────────────────────────────
        self._init_weights()

    # ----------------------------------------------------------
    def _init_weights(self):
        """URM과 동일한 Truncated LeCun Normal 초기화."""
        d = self.d_model
        embed_std = 1.0 / math.sqrt(d)  # 1/√d (URM embed_init_std)

        # 임베딩 / LM head
        trunc_normal_init_(self.embedding.weight, std=embed_std)
        trunc_normal_init_(self.lm_head.weight, std=1.0 / math.sqrt(d))

        # Q head: weight=0, bias=-5 (URM urm.py:107-108)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -5.0)

        # 블록 내부 가중치
        for block in self.blocks:
            hidden = block.d_model
            std_h = 1.0 / math.sqrt(hidden)
            inner_dim = block.W_down.in_features
            std_inter = 1.0 / math.sqrt(inner_dim)

            # QKV, O_aux
            trunc_normal_init_(block.W_QKV.weight, std=std_h)
            trunc_normal_init_(block.W_O_aux.weight, std=std_h)

            # ConvSwiGLU: gate_up → std_h, down → std_inter
            trunc_normal_init_(block.W_up.weight, std=std_h)
            trunc_normal_init_(block.W_down.weight, std=std_inter)

    # ----------------------------------------------------------
    def initial_carry(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> AMKPDCarry:
        """에폭 시작 시 carry 초기화 (URM urm.py:180-188)"""
        return AMKPDCarry(
            current_hidden=torch.empty(
                batch_size, seq_len, self.d_model,
                dtype=self.init_hidden.dtype, device=device,
            ),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            current_inputs=torch.empty(batch_size, seq_len, dtype=torch.long, device=device),
            current_labels=torch.empty(batch_size, seq_len, dtype=torch.long, device=device),
        )

    # ----------------------------------------------------------
    def reset_carry(
        self, reset_flag: torch.Tensor, carry: AMKPDCarry
    ) -> AMKPDCarry:
        """halted 샘플만 init_hidden으로 리셋 (URM urm.py:134-140)"""
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            self.init_hidden,
            carry.current_hidden,
        )
        return replace(carry, current_hidden=new_hidden)

    # ----------------------------------------------------------
    def _run_blocks(self, Q: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        """K개 블록 순차 통과 = 1 L_cycle의 블록 체인"""
        for block in self.blocks:
            Q = block(Q, cos_sin)
        return Q

    # ----------------------------------------------------------
    def forward(
        self,
        carry: AMKPDCarry,
        batch: tuple,  # (inputs: [B, N], labels: [B, N])
    ):
        """
        URM 3중 루프 구조의 단일 외부 루프 호출.

        Args:
            carry: 이전 상태 (initial_carry 또는 이전 forward의 반환값)
            batch: (inputs, labels) 텐서 튜플

        Returns:
            (new_carry, logits, halt_logits)
                new_carry   : 갱신된 AMKPDCarry (current_hidden detached)
                logits      : [B, N, vocab_size]
                halt_logits : [B, 1] (raw logits, sigmoid 미적용)
        """
        inputs, labels = batch

        # ── Step A: 데이터 수락 (halted 샘플만 새 데이터 수락) ─────────
        new_carry = self.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        # halted 샘플: 새 입력/라벨 수락, non-halted: 기존 유지
        halted_mask_inp = carry.halted.view(-1, 1).expand_as(inputs)
        new_inputs = torch.where(halted_mask_inp, inputs, carry.current_inputs)
        new_labels = torch.where(halted_mask_inp, labels, carry.current_labels)

        # ── Step B: 입력 임베딩 (current_inputs 기준) ─────────────────
        seq_len = new_inputs.shape[1]
        X = self.embed_scale * self.embedding(new_inputs)  # [B, N, d]
        cos, sin = self.rotary_emb()           # 각각 [max_seq, d_h]
        cos_sin = (cos[:seq_len], sin[:seq_len])  # [N, d_h]로 슬라이싱

        Q = new_carry.current_hidden  # [B, N, d]

        # ── Step C: 3중 루프 본체 ─────────────────────────────────────
        # H_cycles-1회: burn-in (no_grad 선택 가능)
        if self.H_cycles > 1:
            ctx = torch.no_grad() if self.burn_in_no_grad else nullcontext()
            with ctx:
                for _h in range(self.H_cycles - 1):
                    for _l in range(self.L_cycles):
                        Q = Q + X                     # L_cycle 시작 시 입력 주입
                        Q = self._run_blocks(Q, cos_sin)

        # 마지막 1 H_cycle: gradient 추적 (URM urm.py:159-162)
        if self.log_viz:
            self.viz_Q = []
            self.viz_H_global = []
            for b in self.blocks:
                b.viz_H = []
            self.viz_Q.append(Q.detach())

        for _l in range(self.L_cycles):
            Q = Q + X                             # L_cycle 시작 시 입력 주입
            Q = self._run_blocks(Q, cos_sin)

            if self.log_viz:
                self.viz_Q.append(Q.detach())
                if self.blocks[-1].viz_H:
                    self.viz_H_global.append(self.blocks[-1].viz_H[-1])

        # ── Step D: 출력 헤드 ─────────────────────────────────────────
        logits = self.lm_head(Q)                           # [B, N, vocab_size]

        # URM urm.py:166 — 첫 번째 토큰 기준 Q값 (AMK-PD는 mean pooling 사용)
        q_logits = self.halt_head(Q.mean(dim=1))           # [B, 2]
        q_halt_logits    = q_logits[..., 0]                # [B]
        q_continue_logits = q_logits[..., 1]               # [B]

        # ── Step E: carry 갱신 + halting 판정 ─────────────────────────
        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.loops)

            if self.training and self.loops > 1:
                # URM urm.py:221 — q_halt_logits > 0 이면 조기 정지
                halted = halted | (q_halt_logits > 0)

                # Halt exploration (URM urm.py:224-226)
                halt_exploration_prob = 0.1
                explore_mask = torch.rand_like(q_halt_logits) < halt_exploration_prob
                min_halt_steps = explore_mask * torch.randint_like(
                    new_steps, low=2, high=self.loops + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

        new_carry = AMKPDCarry(
            current_hidden=Q.detach(),   # Markovian isolation (URM urm.py:164)
            steps=new_steps,
            halted=halted,
            current_inputs=new_inputs,
            current_labels=new_labels,
        )

        return new_carry, logits, (q_halt_logits, q_continue_logits)


# ============================================================
# 빠른 동작 확인용 스크립트
# ============================================================

if __name__ == "__main__":
    # ── 하이퍼파라미터 ────────────────────────────────────────────────
    VOCAB_SIZE      = 32_000
    D_MODEL         = 128
    NUM_HEADS       = 8
    NUM_LAYERS      = 3      # K
    LOOPS           = 6      # 외부 루프
    H_CYCLES        = 2      # 중간 루프 (1회 no_grad + 1회 grad)
    L_CYCLES        = 1      # 내부 루프
    DT              = 0.1
    KERNEL_POWER    = 2
    EXPANSION_RATIO = 4
    CONV_KERNEL     = 3

    BATCH_SIZE      = 2
    SEQ_LEN         = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 모델 초기화 ───────────────────────────────────────────────────
    model = AMKPDModel(
        vocab_size       = VOCAB_SIZE,
        d_model          = D_MODEL,
        num_heads        = NUM_HEADS,
        num_layers       = NUM_LAYERS,
        loops            = LOOPS,
        H_cycles         = H_CYCLES,
        L_cycles         = L_CYCLES,
        dt               = DT,
        kernel_power     = KERNEL_POWER,
        expansion_ratio  = EXPANSION_RATIO,
        conv_kernel_size = CONV_KERNEL,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── 더미 입력 ─────────────────────────────────────────────────────
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    target    = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    print(f"Input shape: {input_ids.shape}")  # [B, N]

    # ── Carry 기반 다중 Forward ───────────────────────────────────────
    model.train()
    carry = model.initial_carry(BATCH_SIZE, SEQ_LEN, device)
    batch = (input_ids, target)

    # ── URM 패턴: 배치당 1 forward, carry가 루프 간 상태를 전달 ─────────
    # 데이터로더 반복 자체가 외부 루프. 같은 배치를 여러 번 쓰지 않음.
    print(f"\nSimulating {LOOPS} batches (H_cycles={H_CYCLES}, L_cycles={L_CYCLES}):")
    print("1 batch = 1 forward (URM pretrain.py pattern)")

    for t in range(LOOPS):
        # 각 t는 별개의 배치 호출로 가정 (carry가 상태를 이어받음)
        carry, logits, (q_halt, q_cont) = model(carry, batch)
        print(
            f"  Batch {t+1:2d} | "
            f"logits: {tuple(logits.shape)} | "
            f"q_halt: {q_halt.tolist()} | q_cont: {q_cont.tolist()} | "
            f"halted: {carry.halted.tolist()} | steps: {carry.steps.tolist()}"
        )

        # 배치마다 1회 loss + backward (carry.detach()로 gradient 격리됨)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            target.reshape(-1),
        )
        loss.backward()
        print(f"    → loss={loss.item():.4f}, backward OK")

    print(f"\n{LOOPS} batches processed. Ready for optimizer.step()")
