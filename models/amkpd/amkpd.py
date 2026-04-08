"""
AMK-PD: Asymmetric Mean-shift Kernel Particle Dynamics
URM pretrain.py compatible interface.

Core difference from URM:
    URM:    attn_output = softmax(QK^T/sqrt(d))V → residual
    AMK-PD: m = C - V where C = softmax(QK^T/sqrt(d))V → residual
            (Graph Laplacian diffusion: m = -L_rw V)

Additional: QK-RMSNorm applied to Q/K after RoPE to prevent attention collapse.
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, ConvSwiGLU, RotaryEmbedding, CosSin,
    CastedEmbedding, CastedLinear, apply_rotary_pos_emb,
)
from models.sparse_embedding import CastedSparseEmbedding


# ============================================================
# Config & Carry
# ============================================================

@dataclass
class AMKPDCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class AMKPDConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str = "rope"
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    loops: int = 16
    L_cycles: int = 3
    H_cycles: int = 4
    forward_dtype: str = "bfloat16"


# ============================================================
# AMK_Block: Mean-Shift Micro-Layer
# ============================================================

class AMKBlock(nn.Module):
    """
    Mean Shift attention block.

    Attention:
        P = softmax(QK^T / sqrt(d_h))  — Markov transition matrix
        C = PV                          — Weighted mean centroid
        m = C - V = -L_rw V            — Graph Laplacian diffusion vector

    Followed by ConvSwiGLU MLP (identical to URM).
    """

    def __init__(self, config: AMKPDConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.norm_eps = config.rms_norm_eps

        # QKV fused projection
        self.qkv_proj = CastedLinear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)

        # ConvSwiGLU (identical to URM)
        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, d = hidden_states.shape
        H = self.num_heads
        d_h = self.head_dim

        # QKV projection
        qkv = self.qkv_proj(hidden_states).view(B, N, 3, H, d_h)
        Q_pre = qkv[:, :, 0]  # [B, N, H, d_h]
        K_pre = qkv[:, :, 1]  # [B, N, H, d_h]

        # RoPE
        cos, sin = cos_sin
        Q_pre, K_pre = apply_rotary_pos_emb(Q_pre, K_pre, cos, sin)

        Q_proj = Q_pre.transpose(1, 2)  # [B, H, N, d_h]
        K_proj = K_pre.transpose(1, 2)  # [B, H, N, d_h]
        V_proj = qkv[:, :, 2].transpose(1, 2)  # [B, H, N, d_h]

        # QK-RMSNorm: stabilize attention (V not normalized — preserves m = C - V scale)
        Q_proj = rms_norm(Q_proj, variance_epsilon=self.norm_eps)
        K_proj = rms_norm(K_proj, variance_epsilon=self.norm_eps)

        # Softmax Mean Shift: C = PV, m = C - V
        C = F.scaled_dot_product_attention(Q_proj, K_proj, V_proj, is_causal=False)
        m = C - V_proj  # [B, H, N, d_h] — graph Laplacian diffusion

        # Head merge + output projection
        m_out = m.transpose(1, 2).contiguous().view(B, N, d)
        attn_output = self.o_proj(m_out)

        # Tangent projection: project m onto the tangent plane of the unit sphere at Q
        # m_perp = m - <m, Q> * Q  (Q is already on the sphere via post-norm)
        dot = (attn_output * hidden_states).sum(dim=-1, keepdim=True)
        attn_output = attn_output - dot * hidden_states

        # Post-norm residual = retraction onto the sphere
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)

        # ConvSwiGLU MLP (identical to URM)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)

        return hidden_states


# ============================================================
# AMKPD_Inner: Core Model
# ============================================================

class AMKPD_Inner(nn.Module):
    def __init__(self, config: AMKPDConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embedding
        self.embed_tokens = CastedEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )

        # LM head
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Q head (halting)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Puzzle embedding
        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)

        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len + self.puzzle_emb_len,
            base=config.rope_theta,
        )

        # Transformer layers (AMK blocks)
        self.layers = nn.ModuleList([AMKBlock(config) for _ in range(config.num_layers)])

        # Initial hidden state
        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head init: weight=0, bias=-5 (URM pattern)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> AMKPDCarry:
        return AMKPDCarry(
            current_hidden=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: AMKPDCarry) -> AMKPDCarry:
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            self.init_hidden,
            carry.current_hidden,
        )
        return replace(carry, current_hidden=new_hidden)

    def forward(
        self,
        carry: AMKPDCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[AMKPDCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        hidden_states = carry.current_hidden

        # H_cycles-1: burn-in (no_grad)
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        hidden_states = hidden_states + input_embeddings
                        for layer in self.layers:
                            hidden_states = layer(hidden_states=hidden_states, **seq_info)

        # Last H_cycle: with gradients
        for _ in range(self.config.L_cycles):
            hidden_states = hidden_states + input_embeddings
            for layer in self.layers:
                hidden_states = layer(hidden_states=hidden_states, **seq_info)

        new_carry = replace(carry, current_hidden=hidden_states.detach())

        # Output heads
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


# ============================================================
# AMKPD: Outer Wrapper (matches URM interface)
# ============================================================

class AMKPD(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = AMKPDConfig(**config_dict)
        self.inner = AMKPD_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> AMKPDCarry:
        batch_size = batch["inputs"].shape[0]
        base = self.inner.empty_carry(batch_size)
        return AMKPDCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: AMKPDCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False,
    ) -> Tuple[AMKPDCarry, Dict[str, torch.Tensor]]:

        # Reset halted samples
        new_carry = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)

        # Update batch data for halted samples
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        # Forward pass
        new_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        # Halting logic
        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            if self.training and (self.config.loops > 1):
                halted = halted | (q_halt_logits > 0)

                # Exploration
                halt_exploration_prob = 0.1
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.loops + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

        return (
            AMKPDCarry(
                current_hidden=new_carry.current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            ),
            outputs,
        )
