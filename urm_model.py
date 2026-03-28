"""
URM (Universal Reasoning Model) — Standalone PyTorch 재구현
AMK-PD와 동일한 인터페이스를 제공하여 공정한 비교를 가능하게 합니다.

원본: ref/URM/models/urm/urm.py
변경 사항:
  - flash_attn → F.scaled_dot_product_attention
  - pydantic/hydra 의존성 제거
  - 퍼즐 임베딩 제거 (Sudoku 데이터셋과 직접 호환)
  - AMKPDModel과 동일한 carry 구조 및 forward 시그니처
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from dataclasses import dataclass, replace


# ============================================================
# Utilities
# ============================================================

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """JAX 호환 truncated normal init (ref/URM/models/common.py)."""
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


def _find_multiple(a, b):
    """a를 b의 배수로 올림."""
    return (-(a // -b)) * b


def rms_norm(hidden_states: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """파라미터 없는 RMSNorm (URM 원본과 동일)."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)


# ============================================================
# Carry 구조체 (AMKPDCarry와 동일 인터페이스)
# ============================================================

@dataclass
class URMCarry:
    current_hidden: torch.Tensor   # [B, N, d]
    steps: torch.Tensor            # [B] int32
    halted: torch.Tensor           # [B] bool
    current_inputs: torch.Tensor   # [B, N] long
    current_labels: torch.Tensor   # [B, N] long


# ============================================================
# RoPE (Rotary Positional Embedding)
# ============================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=81, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())  # [seq_len, head_dim]
        self.register_buffer("sin_cached", emb.sin())  # [seq_len, head_dim]

    def forward(self, seq_len=None):
        if seq_len is not None:
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, N, H, d_h], cos/sin: [N, d_h]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


# ============================================================
# Attention (SDPA, non-causal)
# ============================================================

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # QKV fused projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Truncated LeCun normal init
        trunc_normal_init_(self.qkv_proj.weight, std=1.0 / (hidden_size ** 0.5))
        trunc_normal_init_(self.o_proj.weight, std=1.0 / (hidden_size ** 0.5))

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        H = self.num_heads
        d_h = self.head_dim

        qkv = self.qkv_proj(hidden_states).view(B, N, 3, H, d_h)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, N, H, d_h]

        # RoPE
        cos, sin = cos_sin
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # [B, N, H, d_h] -> [B, H, N, d_h] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)
        return self.o_proj(attn_output)


# ============================================================
# ConvSwiGLU (URM 원본과 동일한 intermediate 크기 계산)
# ============================================================

class ConvSwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion=4, conv_kernel=2):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.inter = inter

        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.dwconv = nn.Conv1d(
            in_channels=inter, out_channels=inter,
            kernel_size=conv_kernel, padding=conv_kernel // 2,
            groups=inter, bias=True,
        )
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

        # Init
        trunc_normal_init_(self.gate_up_proj.weight, std=1.0 / (hidden_size ** 0.5))
        trunc_normal_init_(self.down_proj.weight, std=1.0 / (inter ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = F.silu(gate) * up
        x_conv = self.dwconv(x_ffn.transpose(1, 2))
        x_conv = x_conv[..., :up.size(1)]
        x_conv = F.silu(x_conv)
        x_conv = x_conv.transpose(1, 2).contiguous()
        return self.down_proj(x_conv)


# ============================================================
# URMBlock (Attention + ConvSwiGLU with post-norm)
# ============================================================

class URMBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, expansion=4, rms_norm_eps=1e-5):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads)
        self.mlp = ConvSwiGLU(hidden_size, expansion)
        self.norm_eps = rms_norm_eps

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(cos_sin, hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, self.norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, self.norm_eps)
        return hidden_states


# ============================================================
# URMModel (AMKPDModel과 동일한 인터페이스)
# ============================================================

class URMModel(nn.Module):
    """
    URM Model — AMKPDModel과 동일한 인터페이스.

    3중 루프: 외부(loops) × 중간(H_cycles) × 내부(L_cycles × num_layers)
    Input injection: hidden = hidden + input_embeddings (각 L_cycle 시작)
    Post-norm RMSNorm (파라미터 없음)

    Args:
        vocab_size      : 어휘 크기
        d_model         : 은닉 차원
        num_heads       : 어텐션 헤드 수
        num_layers      : 블록 수
        loops           : 외부 루프 최대 횟수
        H_cycles        : 중간 루프 횟수 (H-1회 no_grad + 1회 grad)
        L_cycles        : 내부 루프 횟수
        expansion       : MLP 팽창 비율
        rope_theta      : RoPE base frequency
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        loops: int = 16,
        H_cycles: int = 2,
        L_cycles: int = 6,
        expansion: float = 4,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model    = d_model
        self.num_layers = num_layers
        self.loops      = loops
        self.H_cycles   = H_cycles
        self.L_cycles   = L_cycles
        self.burn_in_no_grad = True  # AMK-PD와 동일한 옵션

        embed_scale = math.sqrt(d_model)
        embed_init_std = 1.0 / embed_scale

        # ── 임베딩 ────────────────────────────────────────────────
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        trunc_normal_init_(self.embed_tokens.weight, std=embed_init_std)
        self.embed_scale = embed_scale

        # ── RoPE ──────────────────────────────────────────────────
        self.rotary_emb = RotaryEmbedding(
            dim=d_model // num_heads,
            max_position_embeddings=8192,
            base=rope_theta,
        )

        # ── 블록들 ───────────────────────────────────────────────
        self.layers = nn.ModuleList([
            URMBlock(d_model, num_heads, expansion)
            for _ in range(num_layers)
        ])

        # ── Init hidden ─────────────────────────────────────────
        self.register_buffer(
            'init_hidden',
            trunc_normal_init_(torch.empty(d_model), std=1),
        )

        # ── 출력 헤드 ────────────────────────────────────────────
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        trunc_normal_init_(self.lm_head.weight, std=1.0 / (d_model ** 0.5))

        self.q_head = nn.Linear(d_model, 2, bias=True)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

    # ----------------------------------------------------------
    def initial_carry(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> URMCarry:
        return URMCarry(
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
    def _reset_carry(self, reset_flag: torch.Tensor, carry: URMCarry) -> URMCarry:
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            self.init_hidden,
            carry.current_hidden,
        )
        return replace(carry, current_hidden=new_hidden)

    # ----------------------------------------------------------
    def _run_layers(self, hidden_states: torch.Tensor, cos_sin) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(cos_sin, hidden_states)
        return hidden_states

    # ----------------------------------------------------------
    def forward(
        self,
        carry: URMCarry,
        batch: tuple,
    ):
        """
        AMKPDModel.forward와 동일한 시그니처.

        Args:
            carry: 이전 상태
            batch: (inputs, labels) 텐서 튜플

        Returns:
            (new_carry, logits, (q_halt_logits, q_continue_logits))
        """
        inputs, labels = batch

        # ── Step A: 데이터 수락 (halted 샘플만) ──────────────────
        new_carry = self._reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        halted_mask_inp = carry.halted.view(-1, 1).expand_as(inputs)
        new_inputs = torch.where(halted_mask_inp, inputs, carry.current_inputs)
        new_labels = torch.where(halted_mask_inp, labels, carry.current_labels)

        # ── Step B: 입력 임베딩 ──────────────────────────────────
        input_embeddings = self.embed_scale * self.embed_tokens(new_inputs)  # [B, N, d]
        seq_len = new_inputs.shape[1]
        cos_sin = self.rotary_emb(seq_len)

        hidden_states = new_carry.current_hidden  # [B, N, d]

        # ── Step C: 3중 루프 본체 ────────────────────────────────
        # H_cycles-1회: burn-in
        if self.H_cycles > 1:
            ctx = torch.no_grad() if self.burn_in_no_grad else nullcontext()
            with ctx:
                for _h in range(self.H_cycles - 1):
                    for _l in range(self.L_cycles):
                        hidden_states = hidden_states + input_embeddings
                        hidden_states = self._run_layers(hidden_states, cos_sin)

        # 마지막 1 H_cycle: gradient 추적
        for _l in range(self.L_cycles):
            hidden_states = hidden_states + input_embeddings
            hidden_states = self._run_layers(hidden_states, cos_sin)

        # ── Step D: 출력 헤드 ────────────────────────────────────
        logits = self.lm_head(hidden_states)  # [B, N, vocab_size]

        # Q-head: 첫 번째 토큰 기준 (URM 원본 패턴)
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)  # [B, 2]
        q_halt_logits     = q_logits[..., 0]   # [B]
        q_continue_logits = q_logits[..., 1]   # [B]

        # ── Step E: carry 갱신 + halting 판정 ────────────────────
        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.loops)

            if self.training and self.loops > 1:
                halted = halted | (q_halt_logits > 0)

                halt_exploration_prob = 0.1
                explore_mask = torch.rand_like(q_halt_logits) < halt_exploration_prob
                min_halt_steps = explore_mask * torch.randint_like(
                    new_steps, low=2, high=self.loops + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

        new_carry = URMCarry(
            current_hidden=hidden_states.detach(),
            steps=new_steps,
            halted=halted,
            current_inputs=new_inputs,
            current_labels=new_labels,
        )

        return new_carry, logits, (q_halt_logits, q_continue_logits)


# ============================================================
# 빠른 동작 확인
# ============================================================

if __name__ == "__main__":
    VOCAB_SIZE  = 12
    D_MODEL     = 384
    NUM_HEADS   = 8
    NUM_LAYERS  = 4
    LOOPS       = 16
    H_CYCLES    = 2
    L_CYCLES    = 6
    BATCH_SIZE  = 2
    SEQ_LEN     = 81

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = URMModel(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, loops=LOOPS, H_cycles=H_CYCLES, L_cycles=L_CYCLES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    target    = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    model.train()
    carry = model.initial_carry(BATCH_SIZE, SEQ_LEN, device)
    batch = (input_ids, target)

    for t in range(3):
        carry, logits, (q_halt, q_cont) = model(carry, batch)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1))
        loss.backward()
        print(f"  Batch {t+1} | logits: {tuple(logits.shape)} | loss={loss.item():.4f} | steps: {carry.steps.tolist()}")

    print("OK")
