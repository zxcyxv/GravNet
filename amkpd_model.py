"""
AMK-PD: Asymmetric Mapped Kernel Particle Dynamics
PyTorch 구현

아키텍처 개요:
- 입력 시퀀스 X ∈ R^(B×N×d)를 초기 상태로 하여 거시적 루프(Macro-Loop)를 M번 반복
- 단일 거시적 루프 내부에 K개의 AMK_Block이 직렬 배치 (Micro-Depth)
- TBPTL (Truncated Backpropagation Through Loops) 최적화 적용
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


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
        d_spectral     (int)  : 스펙트럼 특징 차원 D
        dt             (float): Explicit Euler 스텝 사이즈 초기값 (학습 가능)
        expansion_ratio(int)  : ConvSwiGLU 내부 팽창 비율 m
        conv_kernel_size(int) : Depthwise Conv1D 커널 크기
    """

    def __init__(
        self,
        d_model: int,
        d_spectral: int,
        dt: float = 0.1,
        expansion_ratio: int = 4,
        conv_kernel_size: int = 3,
        use_w_v: bool = False,
    ):
        super().__init__()
        self.d_model = d_model          # d
        self.d_spectral = d_spectral    # D
        self.use_w_v = use_w_v
        inner_dim = expansion_ratio * d_model  # m * d

        # ── 학습 가능한 스텝 파라미터 ──────────────────────────────────
        self.dt  = nn.Parameter(torch.tensor(float(dt)))   # Δt: scalar
        # lam은 블록 내부에서 제거 → AMKPDModel 수준으로 격상됨

        # Zero-Initialized Projection (Residual Scale Mismatch 해결)
        self.m_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.m_proj.weight)
        nn.init.zeros_(self.m_proj.bias)


        # ── Step 1: 상태 의존적 스펙트럼 생성 ─────────────────────────
        # 하이퍼네트워크: q_pool(d) → Ω(d×D) 를 위해 d → d*D 사영
        # SpectralNorm 으로 립시츠 연속성 보장
        self.hyper_q = spectral_norm(
            nn.Linear(d_model, d_model * d_spectral, bias=False)
        )  # W_hq: d → d*D
        self.hyper_k = spectral_norm(
            nn.Linear(d_model, d_model * d_spectral, bias=False)
        )  # W_hk: d → d*D

        # ── Step 2: 전역 편향 벡터 (학습 가능) ────────────────────────
        self.B_Q = nn.Parameter(torch.randn(d_spectral) * 0.02)  # [D]
        self.B_K = nn.Parameter(torch.randn(d_spectral) * 0.02)  # [D]

        # ── Step 3: Value 사영 (Option A vs Option B) ──────────────────
        if self.use_w_v:
            self.W_V = nn.Linear(d_model, d_model, bias=False)       # [d, d]
        else:
            self.W_V = None

        # ── Step 4: ConvSwiGLU ────────────────────────────────────────
        # 선형 팽창: d → 2*m*d  (G 와 U 로 분할)
        self.W_up = nn.Linear(d_model, 2 * inner_dim, bias=False)

        # Depthwise Conv1D: same-padding 으로 시퀀스 길이 보존
        padding = (conv_kernel_size - 1) // 2
        self.dw_conv = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=conv_kernel_size,
            padding=padding,
            groups=inner_dim,   # depthwise: 채널마다 독립적 컨볼루션
            bias=False,
        )

        # 최종 투영: m*d → d
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
            Q_in : [B, N, d]  현재 은닉 상태 (순수 잔차 누적기)
            X    : [B, N, d]  입력 정보 안커 (input_norm 통과 후 값)
        Returns:
            Q_out: [B, N, d]  업데이트된 은닉 상태
        """
        B, N, d = Q_in.shape        # Batch, Seq_len, Dim
        D = self.d_spectral

        # ══════════════════════════════════════════════════════
        # Step 1: State-Dependent Spectral Generation
        # ══════════════════════════════════════════════════════

        # Pre-Norm 적용
        Q_norm1 = self.norm1(Q_in)                  # [B, N, d]

        # 핵심: 정보 공간(Information Space)에서만 X 주입
        # 누적기 Q에는 직접 더하지 않음 — 노름 폭발 원천 차단
        H_context = Q_norm1 + X                     # [B, N, d]

        # 시퀀스 축(N)에 대해 Mean Pooling → 거시 상태 벡터 (H_context 기준)
        q_pool = H_context.mean(dim=1)              # [B, d]

        # 하이퍼네트워크 통과 → d×D 행렬로 복원
        Omega_Q = self.hyper_q(q_pool).view(B, d, D)  # [B, d, D]
        Omega_K = self.hyper_k(q_pool).view(B, d, D)  # [B, d, D]

        # ══════════════════════════════════════════════════════
        # Step 2: Neural Bochner Spectral Mapping (H_context 기준)
        # ══════════════════════════════════════════════════════

        # X의 앵커 정보를 포함한 커널 사영 (RoPE 제거)
        Phi_Q = F.elu(torch.bmm(H_context, Omega_Q) + self.B_Q) + 1.0  # [B, N, D]
        Phi_K = F.elu(torch.bmm(H_context, Omega_K) + self.B_K) + 1.0  # [B, N, D]

        # ══════════════════════════════════════════════════════
        # Step 3: Gravitational Mean Shift (H_context 안커 기준)
        # ══════════════════════════════════════════════════════

        # V 및 anchor: H_context 공간에서 정의 (X의 안커 역할 내재화)
        if getattr(self, 'use_w_v', False):
            V      = self.W_V(H_context)            # [B, N, d]
            anchor = V
        else:
            V      = H_context                      # [B, N, d]
            anchor = H_context

        # 전역 컨텍스트: C = Phi_K^T @ V
        C = torch.bmm(Phi_K.transpose(1, 2), V)     # [B, D, d]

        # 인력: Attraction = Phi_Q @ C
        Attraction = torch.bmm(Phi_Q, C)             # [B, N, d]

        # 정규화 항
        ones_N = torch.ones(B, N, 1, device=Q_in.device, dtype=Q_in.dtype)
        phi_k_sum = torch.bmm(Phi_K.transpose(1, 2), ones_N).squeeze(-1)  # [B, D]
        denom = torch.bmm(Phi_Q, phi_k_sum.unsqueeze(-1)).squeeze(-1)
        Norm = denom.abs() + 1.0

        # Mean Shift 벡터 (H_context 안커 기준)
        m = Attraction / Norm.unsqueeze(-1) - anchor   # [B, N, d]

        # Zero-initialized Projection
        m_proj = self.m_proj(m)

        # Softplus 제약
        dt_safe = F.softplus(self.dt)

        # 순수 잔차 업데이트 — Q 누적기에는 m_proj만 더함 (X 직접 덧셈 없음)
        Q_interact = Q_in + dt_safe * m_proj  # [B, N, d]


        # ── 시각화 캐싱 ──────────────────────────────────────────────────
        if self.log_viz:
            W_raw = torch.bmm(Phi_Q, Phi_K.transpose(1, 2))
            W_norm = W_raw / Norm.unsqueeze(-1)
            self.viz_W.append(W_norm)
            self.viz_m.append(m)
            self.viz_H.append(H_context)  # 실제 동역학 개입 공간 캐싱
            
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
        H_ffn_t  = H_ffn.transpose(1, 2)    # [B, m*d, N]
        H_conv_t = self.dw_conv(H_ffn_t)    # [B, m*d, N]  ← same-padding 으로 N 유지
        H_conv   = H_conv_t.transpose(1, 2) # [B, N, m*d]

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
    AMK-PD Model

    거시적 루프(Macro-Loop)를 M회 반복하며,
    각 루프 안에서 K개의 AMK_Block 을 직렬 통과시킵니다.
    TBPTL 로 초기 N_trunc 루프는 no_grad 로 실행하고,
    이후 루프에서만 그래디언트를 추적합니다.

    Args:
        vocab_size     (int)  : 어휘 크기
        d_model        (int)  : 은닉 차원 d
        d_spectral     (int)  : 스펙트럼 차원 D
        num_layers     (int)  : 거시 루프 내 AMK_Block 수 K (Micro-Depth)
        max_loops      (int)  : 최대 거시 루프 횟수 M
        trunc_loops    (int)  : TBPTL forward-only 구간 N_trunc
        dt             (float): Euler 스텝 초기값 (각 블록이 독립적으로 학습)
        lam            (float): 탄성 주입 강도 초기값 (모델 수준 단일 파라미터)
        expansion_ratio(int)  : ConvSwiGLU 팽창 비율 m
        conv_kernel_size(int) : Depthwise Conv 커널 크기
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        d_spectral: int = 128,
        num_layers: int = 4,
        max_loops: int = 8,
        trunc_loops: int = 4,
        dt: float = 0.1,
        lam: float = 0.1,
        expansion_ratio: int = 4,
        conv_kernel_size: int = 3,
        use_w_v: bool = False,
    ):
        super().__init__()
        assert trunc_loops < max_loops, (
            "trunc_loops 는 max_loops 보다 작아야 합니다. "
            f"(trunc_loops={trunc_loops}, max_loops={max_loops})"
        )

        self.d_model     = d_model
        self.num_layers  = num_layers   # K
        self.max_loops   = max_loops    # M
        self.trunc_loops = trunc_loops  # N_trunc

        # lam 파라미터 제거: X는 이제 블록 내부 정보 공간에서만 작동

        # ── 토큰 임베딩 ───────────────────────────────────────────────
        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.pos_emb     = nn.Embedding(8192, d_model)  # 1D Absolute Positional Embedding
        self.input_norm  = nn.LayerNorm(d_model)

        # ── RoPE 제거: 1D 회전이 ELU 코널과 교환법칙 위배 — 절대 위치 Emb만 사용

        # ── K 개의 AMK_Block (거시 루프 간 가중치 공유) ───────────────
        self.blocks = nn.ModuleList([
            AMK_Block(
                d_model=d_model,
                d_spectral=d_spectral,
                dt=dt,
                expansion_ratio=expansion_ratio,
                conv_kernel_size=conv_kernel_size,
                use_w_v=getattr(self, 'use_w_v', False),
            )
            for _ in range(num_layers)
        ])

        # ── 최종 출력 정규화 ──────────────────────────────────────────
        self.final_norm = nn.LayerNorm(d_model)

        # ── ACT Halting Head ──────────────────────────────────────────
        # q_pool(d) → 정지 확률 스칼라
        self.halt_head = nn.Linear(d_model, 1)

        # ── Language Modeling Head ────────────────────────────────────
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ── 시각화용 텔레메트리 ───────────────────────────────────────────
        self.log_viz = False
        self.viz_Q = []
        self.viz_H_global = []  # H_context 매크로 루프별 스냅샷 (= 실제 KDE 활동 공간)

        # ── 가중치 초기화 ─────────────────────────────────────────────
        self._init_weights()

    # ----------------------------------------------------------
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.halt_head.weight)
        nn.init.zeros_(self.halt_head.bias)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    # ----------------------------------------------------------
    def _run_one_macro_loop(
        self, Q: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """
        단일 거시적 루프: K 개의 AMK_Block에 X 내장 후 차례로 통과.
        Q 누적기에는 X를 직접 더하지 않음. X는 블록 내부 정보 공간에서만 작동.
        """
        for block in self.blocks:
            Q = block(Q, X)   # [B, N, d] -> [B, N, d]
        return Q  # [B, N, d]

    # ----------------------------------------------------------
    def forward(self, input_ids: torch.Tensor):
        """
        TBPTL 을 적용한 전방 패스.

        Args:
            input_ids: [B, N]  토큰 인덱스

        Returns:
            (logits_list, halt_probs_list)
                logits_list     : gradient-tracking 구간 각 루프의
                                  [B, N, vocab_size] 로짓 텐서 리스트
                halt_probs_list : gradient-tracking 구간 각 루프의
                                  [B, 1] 정지 확률 스칼라 리스트
        """
        # 임베딩 및 정규화
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        X = self.embedding(input_ids) + self.pos_emb(pos)   # [B, N, d]
        X = self.input_norm(X)          # [B, N, d]

        # Q^(0) = X 로 초기화
        Q = X                           # [B, N, d]

        logits_list:     list[torch.Tensor] = []
        halt_probs_list: list[torch.Tensor] = []

        if self.log_viz:
            self.viz_Q.append(Q)
            self.viz_H_global = []  # 매 forward 시 리셋
            for b in self.blocks:
                b.viz_H = []

        # ── 거시적 루프 (l = 1 … M) ────────────────────────────────
        for l in range(1, self.max_loops + 1):

            if l <= self.trunc_loops:
                with torch.no_grad():
                    Q = self._run_one_macro_loop(Q, X)  # [B, N, d]
                Q = Q.detach()

            else:
                Q = self._run_one_macro_loop(Q, X)      # [B, N, d]

                # 출력 전 최종 정규화 (모델의 메인 State Q는 건드리지 않고, 출력용으로만 사용)
                Q_out_norm = self.final_norm(Q)         # [B, N, d]

                # ACT Halting Head: 정지 확률 계산
                q_pool_l = Q_out_norm.mean(dim=1)       # [B, d]
                p_halt   = torch.sigmoid(
                    self.halt_head(q_pool_l)            # [B, 1]
                )                                       # [B, 1]  ∈ (0, 1)

                # Language Modeling Head
                logits = self.lm_head(Q_out_norm)       # [B, N, vocab_size]

                logits_list.append(logits)
                halt_probs_list.append(p_halt)

            if self.log_viz:
                self.viz_Q.append(Q)
                # 마지막 블록의 H_context를 매크로 루프 단위로 쓰기
                if self.blocks[-1].viz_H:
                    self.viz_H_global.append(self.blocks[-1].viz_H[-1])

        return logits_list, halt_probs_list


# ============================================================
# 빠른 동작 확인용 스크립트
# ============================================================

if __name__ == "__main__":
    # ── 하이퍼파라미터 ────────────────────────────────────────────────
    VOCAB_SIZE      = 32_000
    D_MODEL         = 128
    D_SPECTRAL      = 64
    NUM_LAYERS      = 3      # K
    MAX_LOOPS       = 6      # M
    TRUNC_LOOPS     = 3      # N_trunc
    DT              = 0.1
    LAM             = 0.1
    EXPANSION_RATIO = 4
    CONV_KERNEL     = 3

    BATCH_SIZE      = 2
    SEQ_LEN         = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 모델 초기화 ───────────────────────────────────────────────────
    model = AMKPDModel(
        vocab_size      = VOCAB_SIZE,
        d_model         = D_MODEL,
        d_spectral      = D_SPECTRAL,
        num_layers      = NUM_LAYERS,
        max_loops       = MAX_LOOPS,
        trunc_loops     = TRUNC_LOOPS,
        dt              = DT,
        lam             = LAM,
        expansion_ratio = EXPANSION_RATIO,
        conv_kernel_size= CONV_KERNEL,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── 더미 입력 ─────────────────────────────────────────────────────
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    print(f"Input shape: {input_ids.shape}")  # [B, N]

    # ── 순전파 ────────────────────────────────────────────────────────
    model.train()
    logits_list, halt_probs_list = model(input_ids)

    print(f"\nGradient-tracking loops: {len(logits_list)}")
    for i, (logits, p_halt) in enumerate(zip(logits_list, halt_probs_list)):
        loop_idx = TRUNC_LOOPS + 1 + i
        print(
            f"  Loop {loop_idx:2d} | "
            f"logits: {tuple(logits.shape)} | "
            f"p_halt: {p_halt.squeeze().tolist()}"
        )

    # ── 간단한 역전파 테스트 ──────────────────────────────────────────
    # 마지막 루프 로짓으로 cross-entropy loss 계산
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    final_logits = logits_list[-1]   # [B, N, vocab_size]

    loss = F.cross_entropy(
        final_logits.reshape(-1, VOCAB_SIZE),   # [B*N, vocab_size]
        target.reshape(-1),                      # [B*N]
    )
    loss.backward()
    print(f"\nCross-Entropy Loss: {loss.item():.4f}")
    print("Backward pass: OK")
