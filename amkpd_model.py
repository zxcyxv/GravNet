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
# RoPE (Rotary Positional Embedding)
# ============================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=81, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, :])

    def forward(self, seq_len=81):
        return (
            self.cos_cached[:, :seq_len, ...],
            self.sin_cached[:, :seq_len, ...]
        )

@dataclass
class AMKPDCarry:
    """URM의 URMCarry에 대응하는 carry 구조체 (ref/urm.py:14-18)"""
    current_hidden: torch.Tensor   # [B, N, d] — detached 은닉 상태
    steps: torch.Tensor            # [B] int32 — 외부 루프 카운터
    halted: torch.Tensor           # [B] bool — 샘플별 정지 플래그
    current_inputs: torch.Tensor   # [B, N] long — 저장된 입력 토큰
    current_labels: torch.Tensor   # [B, N] long — 저장된 정답 라벨


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, cos, sin):
    return (q * cos) + (rotate_half(q) * sin)


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
        self.dt  = nn.Parameter(torch.tensor(float(dt)))   # Δt: scalar

        # ── 멀티헤드 선형 사영 ─────────────────────────────────────────
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_O.weight)

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

        # ── 정규화 (Pre-Norm) ──────────────────────────────────────────
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ── 시각화용 텔레메트리 (Visualization Telemetry) ────────────────
        self.log_viz = False
        self.viz_W = []
        self.viz_m = []
        self.viz_H = []  # H_context = LayerNorm(Q) + X — 실제 중력 작용 공간

    # ----------------------------------------------------------
    def forward(self, Q_in: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q_in : [B, N, d]  현재 은닉 상태 (Q 누적기)
            X    : [B, N, d]  입력 앵커 (디리클레 경계 조건)
        Returns:
            Q_out: [B, N, d]  업데이트된 은닉 상태
        """
        B, N, d = Q_in.shape        # Batch, Seq_len, Dim
        H = self.num_heads
        d_h = self.head_dim

        # ══════════════════════════════════════════════════════
        # Step 1: Pre-Norm + Information Injection
        # ══════════════════════════════════════════════════════

        Q_norm1 = self.norm1(Q_in)                  # [B, N, d]
        H_context = Q_norm1 + X                     # [B, N, d]

        # ══════════════════════════════════════════════════════
        # Step 2: 멀티헤드 선형 사영 + 헤드 분할
        # ══════════════════════════════════════════════════════

        Q_proj = self.W_Q(H_context).view(B, N, H, d_h).transpose(1, 2)  # [B, H, N, d_h]
        K_proj = self.W_K(H_context).view(B, N, H, d_h).transpose(1, 2)  # [B, H, N, d_h]
        V_proj = self.W_V(H_context).view(B, N, H, d_h).transpose(1, 2)  # [B, H, N, d_h]

        # ══════════════════════════════════════════════════════
        # Step 3: 헤드별 비대칭 Bochner 매핑 (ELU+1 양수 보장)
        # ══════════════════════════════════════════════════════

        Phi_Q = F.elu(Q_proj) + 1.0  # [B, H, N, d_h]
        Phi_K = F.elu(K_proj) + 1.0  # [B, H, N, d_h]

        # ══════════════════════════════════════════════════════
        # Step 4: 헤드별 다항식 인력 행렬
        # ══════════════════════════════════════════════════════

        W = torch.matmul(Phi_Q, Phi_K.transpose(-1, -2))  # [B, H, N, N]
        W = F.relu(W) ** self.kernel_power                  # [B, H, N, N]

        # ══════════════════════════════════════════════════════
        # Step 5: 헤드별 Mean Shift
        # ══════════════════════════════════════════════════════

        Attraction = torch.matmul(W, V_proj)               # [B, H, N, d_h]
        Norm = W.sum(dim=-1, keepdim=True) + 1.0            # [B, H, N, 1]
        C = Attraction / Norm                               # [B, H, N, d_h]
        m = C - V_proj                                      # [B, H, N, d_h]

        # ══════════════════════════════════════════════════════
        # Step 6: 헤드 병합 + 출력 사영 (W_O zero-init)
        # ══════════════════════════════════════════════════════

        m_concat = m.transpose(1, 2).contiguous().view(B, N, d)  # [B, N, d]
        m_proj = self.W_O(m_concat)                               # [B, N, d]

        # ══════════════════════════════════════════════════════
        # Step 7: PDE 상태 업데이트 (순수 잔차, 야코비안 I 보장)
        # ══════════════════════════════════════════════════════

        dt_safe = F.softplus(self.dt)
        Q_interact = Q_in + dt_safe * m_proj  # [B, N, d]

        # ── 시각화 캐싱 ──────────────────────────────────────────────────
        if self.log_viz:
            W_avg = W.mean(dim=1)                          # [B, N, N] (헤드 평균)
            Norm_avg = W_avg.sum(dim=-1, keepdim=True) + 1.0
            self.viz_W.append(W_avg / Norm_avg)            # 정규화된 [B, N, N]
            self.viz_m.append(m_concat)                    # concat 후, W_O 전
            self.viz_H.append(H_context)                   # 중력 작용 공간
            
        # ══════════════════════════════════════════════════════
        # Step 4: ConvSwiGLU with Micro-Residual
        # ══════════════════════════════════════════════════════

        Q_norm2 = self.norm2(Q_interact)    # Pre-Norm for SwiGLU

        # 선형 팽창 후 G / U 분할
        GU = self.W_up(Q_norm2)             # [B, N, 2*m*d]
        G, U = GU.chunk(2, dim=-1)          # G: [B, N, m*d],  U: [B, N, m*d]

        # SwiGLU 게이팅
        H_ffn = F.silu(G) * U               # [B, N, m*d]

        # Depthwise Conv1D: (B, channels, length) 형식 필요
        N = Q_norm2.shape[1]
        H_ffn_t  = H_ffn.transpose(1, 2)           # [B, inner, N]
        H_conv_t = self.dw_conv(H_ffn_t)           # [B, inner, N'] (패딩으로 길어질 수 있음)
        H_conv_t = F.silu(H_conv_t[..., :N])       # [B, inner, N] 슬라이싱 + 활성화
        H_conv   = H_conv_t.transpose(1, 2).contiguous()  # [B, N, inner]

        # 최종 투영
        H_out = self.W_down(H_conv)          # [B, N, d]

        # Micro-Residual (필수)
        Q_out = Q_interact + H_out           # [B, N, d]

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

        # ── 토큰 임베딩 ───────────────────────────────────────────────
        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.pos_emb     = nn.Embedding(8192, d_model)
        self.input_norm  = nn.LayerNorm(d_model)

        # ── Carry 리셋용 초기 벡터 (URM urm.py:101-104) ──────────────
        self.register_buffer(
            'init_hidden',
            torch.randn(d_model) * 0.02,
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

        # ── 최종 출력 정규화 ──────────────────────────────────────────
        self.final_norm = nn.LayerNorm(d_model)

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
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        # q_head: weight=0, bias=-5 두 출력 모두 (URM urm.py:107-108)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -5.0)

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
    def _run_blocks(self, Q: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """K개 블록 순차 통과 = 1 L_cycle의 블록 체인 (X는 각 블록 내부 정보 공간에 주입)"""
        for block in self.blocks:
            Q = block(Q, X)
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
        pos = torch.arange(seq_len, device=new_inputs.device).unsqueeze(0)
        X = self.embedding(new_inputs) + self.pos_emb(pos)
        X = self.input_norm(X)  # [B, N, d]

        Q = new_carry.current_hidden  # [B, N, d]

        # ── Step C: 3중 루프 본체 ─────────────────────────────────────
        # H_cycles-1회: burn-in (no_grad 선택 가능)
        if self.H_cycles > 1:
            ctx = torch.no_grad() if self.burn_in_no_grad else nullcontext()
            with ctx:
                for _h in range(self.H_cycles - 1):
                    for _l in range(self.L_cycles):
                        Q = self._run_blocks(Q, X)    # X는 블록 내부 정보 공간에 주입

        # 마지막 1 H_cycle: gradient 추적 (URM urm.py:159-162)
        if self.log_viz:
            self.viz_Q = []
            self.viz_H_global = []
            for b in self.blocks:
                b.viz_H = []
            self.viz_Q.append(Q.detach())

        for _l in range(self.L_cycles):
            Q = self._run_blocks(Q, X)            # X는 블록 내부 정보 공간에 주입

            if self.log_viz:
                self.viz_Q.append(Q.detach())
                if self.blocks[-1].viz_H:
                    self.viz_H_global.append(self.blocks[-1].viz_H[-1])

        # ── Step D: 출력 헤드 ─────────────────────────────────────────
        Q_norm = self.final_norm(Q)                        # [B, N, d]
        logits = self.lm_head(Q_norm)                      # [B, N, vocab_size]

        # URM urm.py:166 — 첫 번째 토큰 기준 Q값 (AMK-PD는 mean pooling 사용)
        q_logits = self.halt_head(Q_norm.mean(dim=1))      # [B, 2]
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
