"""
visualize_head_evolution_old.py — old model (W_Q/W_K/W_V/W_O/W_aux) 체크포인트용 시각화

사용법:
  uv run python visualize_head_evolution_old.py checkpoints/step020000_pacc0.0820.pt 0 20
"""

import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from contextlib import nullcontext
from dataclasses import dataclass, replace
from dataset import SudokuDataset


# ================================================================
# Old AMK_Block (W_Q/W_K/W_V/W_O/W_aux 분리)
# ================================================================

class AMK_Block_Old(nn.Module):
    def __init__(self, d_model, num_heads=8, dt=0.5, expansion_ratio=4,
                 conv_kernel_size=3, kernel_power=2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kernel_power = kernel_power
        inner_dim = (-(round(expansion_ratio * d_model * 2 / 3) // -256)) * 256

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.W_aux = nn.Linear(d_model, d_model, bias=False)

        self.W_up = nn.Linear(d_model, 2 * inner_dim, bias=False)
        self.dw_conv = nn.Conv1d(inner_dim, inner_dim, kernel_size=conv_kernel_size,
                                 padding=conv_kernel_size // 2, groups=inner_dim, bias=True)
        self.W_down = nn.Linear(inner_dim, d_model, bias=False)

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.log_viz = False
        self.viz_W = []
        self.viz_m = []
        self.viz_H = []
        self.last_m_norm = 0.0
        self.last_C_norm = 0.0

    def forward(self, Q_in, X):
        B, N, d = Q_in.shape
        H = self.num_heads
        d_h = self.head_dim

        H_context = Q_in + X

        Q_proj = self.W_Q(H_context).view(B, N, H, d_h).transpose(1, 2)
        K_proj = self.W_K(H_context).view(B, N, H, d_h).transpose(1, 2)
        V_proj = self.W_V(H_context).view(B, N, H, d_h).transpose(1, 2)

        Phi_Q = F.elu(Q_proj) + 1.0
        Phi_K = F.elu(K_proj) + 1.0

        scale = self.head_dim ** -0.5
        W = torch.matmul(Phi_Q, Phi_K.transpose(-1, -2)) * scale
        W = F.relu(W) ** self.kernel_power

        Attraction = torch.matmul(W, V_proj)
        Norm = W.sum(dim=-1, keepdim=True) + 1e-6
        C = Attraction / Norm
        m = C - V_proj

        m_concat = m.transpose(1, 2).contiguous().view(B, N, d)
        C_concat = C.transpose(1, 2).contiguous().view(B, N, d)
        m_proj = self.W_O(m_concat) + self.W_aux(C_concat)

        self.last_m_norm = m_concat.detach().norm(dim=-1).mean().item()
        self.last_C_norm = C_concat.detach().norm(dim=-1).mean().item()

        Q_interact = self.norm1(Q_in + 1.0 * m_proj)

        if self.log_viz:
            W_norm = W / (W.sum(dim=-1, keepdim=True) + 1e-6)
            self.viz_W.append(W_norm.detach())
            self.viz_m.append(m.detach())
            self.viz_H.append(H_context)

        GU = self.W_up(Q_interact)
        G, U = GU.chunk(2, dim=-1)
        H_ffn = F.silu(G) * U
        N_ = Q_interact.shape[1]
        H_ffn_t = H_ffn.transpose(1, 2)
        H_conv_t = self.dw_conv(H_ffn_t)
        H_conv_t = F.silu(H_conv_t[..., :N_])
        H_conv = H_conv_t.transpose(1, 2).contiguous()
        H_out = self.W_down(H_conv)
        Q_out = self.norm2(Q_interact + H_out)

        return Q_out


# ================================================================
# Old AMKPDModel
# ================================================================

@dataclass
class AMKPDCarry:
    current_hidden: torch.Tensor
    steps: torch.Tensor
    halted: torch.Tensor
    current_inputs: torch.Tensor
    current_labels: torch.Tensor


class AMKPDModel_Old(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4,
                 loops=6, H_cycles=2, L_cycles=1, dt=0.5, kernel_power=2,
                 expansion_ratio=4, conv_kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.loops = loops
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.burn_in_no_grad = True

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(8192, d_model)
        self.input_norm = nn.RMSNorm(d_model)
        self.register_buffer('init_hidden', torch.randn(d_model) * 0.02)

        self.blocks = nn.ModuleList([
            AMK_Block_Old(d_model, num_heads, dt, expansion_ratio,
                          conv_kernel_size, kernel_power)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(d_model)
        self.halt_head = nn.Linear(d_model, 2)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.log_viz = False
        self.viz_Q = []
        self.viz_H_global = []

    def initial_carry(self, batch_size, seq_len, device):
        return AMKPDCarry(
            current_hidden=torch.empty(batch_size, seq_len, self.d_model, device=device),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device),
            current_inputs=torch.empty(batch_size, seq_len, dtype=torch.long, device=device),
            current_labels=torch.empty(batch_size, seq_len, dtype=torch.long, device=device),
        )

    def reset_carry(self, reset_flag, carry):
        new_hidden = torch.where(reset_flag.view(-1, 1, 1), self.init_hidden, carry.current_hidden)
        return replace(carry, current_hidden=new_hidden)

    def _run_blocks(self, Q, X):
        for block in self.blocks:
            Q = block(Q, X)
        return Q

    def forward(self, carry, batch):
        inputs, labels = batch

        new_carry = self.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        halted_mask_inp = carry.halted.view(-1, 1).expand_as(inputs)
        new_inputs = torch.where(halted_mask_inp, inputs, carry.current_inputs)
        new_labels = torch.where(halted_mask_inp, labels, carry.current_labels)

        seq_len = new_inputs.shape[1]
        pos = torch.arange(seq_len, device=new_inputs.device).unsqueeze(0)
        X = self.embedding(new_inputs) + self.pos_emb(pos)
        X = self.input_norm(X)

        Q = new_carry.current_hidden

        if self.H_cycles > 1:
            ctx = torch.no_grad() if self.burn_in_no_grad else nullcontext()
            with ctx:
                for _h in range(self.H_cycles - 1):
                    for _l in range(self.L_cycles):
                        Q = self._run_blocks(Q, X)

        if self.log_viz:
            self.viz_Q = []
            self.viz_H_global = []
            for b in self.blocks:
                b.viz_H = []
            self.viz_Q.append(Q.detach())

        for _l in range(self.L_cycles):
            Q = self._run_blocks(Q, X)
            if self.log_viz:
                self.viz_Q.append(Q.detach())
                if self.blocks[-1].viz_H:
                    self.viz_H_global.append(self.blocks[-1].viz_H[-1])

        Q_norm = self.final_norm(Q)
        logits = self.lm_head(Q_norm)

        q_logits = self.halt_head(Q_norm.mean(dim=1))
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.loops)
            if self.training and self.loops > 1:
                halted = halted | (q_halt_logits > 0)
                halt_exploration_prob = 0.1
                explore_mask = torch.rand_like(q_halt_logits) < halt_exploration_prob
                min_halt_steps = explore_mask * torch.randint_like(new_steps, low=2, high=self.loops + 1)
                halted = halted & (new_steps >= min_halt_steps)

        new_carry = AMKPDCarry(
            current_hidden=Q.detach(),
            steps=new_steps, halted=halted,
            current_inputs=new_inputs, current_labels=new_labels,
        )
        return new_carry, logits, (q_halt_logits, q_continue_logits)


# ================================================================
# 시각화 (원본과 동일)
# ================================================================

def draw_sudoku(ax, given, preds, labels):
    ax.set_xlim(0, 9); ax.set_ylim(0, 9)
    ax.set_aspect('equal'); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for i in range(10):
        lw = 2.5 if i % 3 == 0 else 0.5
        ax.axhline(y=i, color='black', linewidth=lw)
        ax.axvline(x=i, color='black', linewidth=lw)
    for idx in range(81):
        r, c = idx // 9, idx % 9
        pd = preds[idx] - 1 if preds[idx] >= 2 else 0
        ld = labels[idx] - 1 if labels[idx] >= 2 else 0
        if given[idx]:
            ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='#e0e0e0'))
            if ld > 0:
                ax.text(c+.5, r+.5, str(ld), ha='center', va='center',
                        fontsize=9, fontweight='bold', color='black')
        elif pd > 0:
            correct = (pd == ld)
            bg = '#e8f4fd' if correct else '#fde8e8'
            tc = '#2196F3' if correct else '#e74c3c'
            ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor=bg))
            ax.text(c+.5, r+.5, str(pd), ha='center', va='center',
                    fontsize=9, fontweight='bold', color=tc)
        else:
            ax.text(c+.5, r+.5, '?', ha='center', va='center',
                    fontsize=8, color='#999')


def draw_head_heatmap(ax, att_9x9, given, preds, labels, query_r, query_c, head_name):
    sns.heatmap(att_9x9, ax=ax, cmap='magma', cbar=False, square=True,
                linewidths=0.4, linecolor='gray', vmin=0)
    for b in [3, 6]:
        ax.axhline(b, color='cyan', linewidth=1.5)
        ax.axvline(b, color='cyan', linewidth=1.5)
    ax.add_patch(plt.Rectangle((query_c, query_r), 1, 1,
                 fill=False, edgecolor='lime', linewidth=2.5))
    for idx in range(81):
        r, c = idx // 9, idx % 9
        pd = preds[idx] - 1 if preds[idx] >= 2 else 0
        ld = labels[idx] - 1 if labels[idx] >= 2 else 0
        if given[idx]:
            ax.text(c+.5, r+.5, str(ld), ha='center', va='center',
                    fontsize=6, color='white', alpha=0.6)
        elif pd > 0:
            marker = '✓' if pd == ld else '✗'
            color = '#00ff00' if pd == ld else '#ff4444'
            ax.text(c+.5, r+.5, marker, ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')
    ax.set_title(head_name, fontsize=9)


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/step020000_pacc0.0820.pt"
    sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    max_loops = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]

    ds = SudokuDataset(a["data_dir"], "test")
    inp, lbl = ds[sample_idx]
    inp = inp.unsqueeze(0).to(device)
    lbl = lbl.unsqueeze(0).to(device)

    model = AMKPDModel_Old(
        vocab_size=11, d_model=a["d_model"], num_heads=a["num_heads"],
        num_layers=a["num_layers"], loops=max_loops, H_cycles=a["H_cycles"],
        L_cycles=a["L_cycles"], kernel_power=a["kernel_power"],
        expansion_ratio=a["expansion_ratio"], conv_kernel_size=a.get("conv_kernel", 2),
    ).to(device)

    # _orig_mod. prefix 제거
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()

    model.log_viz = True
    for b in model.blocks:
        b.log_viz = True

    H = a["num_heads"]
    heads_to_show = [0, 5, 6]
    inp_np = inp[0].cpu().numpy()
    lbl_np = lbl[0].cpu().numpy()
    given = inp_np > 1

    blank_indices = np.where(~given)[0]
    query_idx = blank_indices[len(blank_indices) // 2]
    qr, qc = query_idx // 9, query_idx % 9
    query_digit = lbl_np[query_idx] - 1 if lbl_np[query_idx] >= 2 else '?'

    carry = model.initial_carry(1, 81, device)
    batch = (inp, lbl)

    step_checkpoints = [1, 2, 3, 5, 8, 12, 16, 20]
    step_checkpoints = [s for s in step_checkpoints if s <= max_loops]

    collected = []

    print(f"Query cell: ({qr},{qc}), answer={query_digit}")
    print(f"Running {max_loops} loops ...")

    with torch.no_grad():
        for i in range(1, max_loops + 1):
            for b in model.blocks:
                b.viz_W = []
                b.viz_m = []
                b.viz_H = []

            carry, logits, _ = model(carry, batch)

            if i in step_checkpoints:
                preds = logits[0].argmax(dim=-1).cpu().numpy()
                W_blocks = []
                for b in model.blocks:
                    if b.viz_W:
                        W_blocks.append(b.viz_W[-1][0].cpu().numpy())
                collected.append((i, preds, W_blocks))

                mask = lbl_np != 0
                n_correct = ((preds == lbl_np) & mask).sum()
                n_blanks = mask.sum()
                print(f"  loop {i:>3}: {n_correct}/{n_blanks} cells correct")

    model.log_viz = False
    for b in model.blocks:
        b.log_viz = False

    n_steps = len(collected)
    n_heads_show = len(heads_to_show)
    n_cols = 1 + n_heads_show
    n_rows = n_steps

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, (step, preds, W_blocks) in enumerate(collected):
        ax = axes[ri, 0]
        draw_sudoku(ax, given, preds, lbl_np)
        mask = lbl_np != 0
        n_correct = ((preds == lbl_np) & mask).sum()
        ax.set_title(f"Loop {step}  ({n_correct}/{mask.sum()} correct)", fontsize=10)

        W_last = W_blocks[-1] if W_blocks else None
        for ci, h in enumerate(heads_to_show):
            ax = axes[ri, ci + 1]
            if W_last is not None:
                att = W_last[h, query_idx].reshape(9, 9)
                draw_head_heatmap(ax, att, given, preds, lbl_np, qr, qc, f"H{h}")
            else:
                ax.set_title(f"H{h} (no data)")

    fig.suptitle(
        f"Query=({qr},{qc}) answer={query_digit} — "
        f"Puzzle #{sample_idx}\n"
        f"Gray=Given, ✓=correct pred, ✗=wrong pred. "
        f"Green box=query cell",
        fontsize=13, y=1.0
    )
    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/08_head_evolution_old.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: visualizations/08_head_evolution_old.png")

    print(f"\n{'step':>4} | {'H':>2} | {'att->correct':>12} | {'att->wrong':>10} | {'att->given':>10} | {'att->blank':>10}")
    print("-" * 70)

    for step, preds, W_blocks in collected:
        W_last = W_blocks[-1] if W_blocks else None
        if W_last is None:
            continue
        mask = lbl_np != 0
        correct_blank = (~given) & (preds == lbl_np) & mask
        wrong_blank = (~given) & (preds != lbl_np) & mask

        for h in heads_to_show:
            att = W_last[h, query_idx]
            total = att.sum() + 1e-12
            att_correct = att[correct_blank].sum() / total
            att_wrong = att[wrong_blank].sum() / total
            att_given = att[given].sum() / total
            att_blank = att[~given].sum() / total
            print(f"{step:>4} | H{h} | {att_correct:>11.4f} | {att_wrong:>9.4f} | {att_given:>9.4f} | {att_blank:>9.4f}")
        print()


if __name__ == "__main__":
    main()
