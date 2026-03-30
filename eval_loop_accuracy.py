"""
eval_loop_accuracy.py — 루프 수별 정답률 변화 측정 (old model)

각 루프 스텝마다 전체 테스트 샘플의 token_acc, puzzle_acc를 측정하여
루프를 거칠수록 정답률이 올라가는지/내려가는지 확인.

사용법:
  uv run python eval_loop_accuracy.py checkpoints/step030000_pacc0.0938.pt
  uv run python eval_loop_accuracy.py checkpoints/step030000_pacc0.0938.pt --max_samples 500
"""

import sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from dataclasses import dataclass, replace
from dataset import SudokuDataset
from torch.utils.data import DataLoader


# ================================================================
# Old AMK_Block / AMKPDModel (visualize_head_evolution_old.py와 동일)
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

        Q_interact = self.norm1(Q_in + m_proj)

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

    def forward_single_loop(self, Q, X):
        """H_cycles 전체를 1회 실행하고 Q 반환 (halting 없이)"""
        if self.H_cycles > 1:
            with torch.no_grad():
                for _h in range(self.H_cycles - 1):
                    for _l in range(self.L_cycles):
                        Q = self._run_blocks(Q, X)
        for _l in range(self.L_cycles):
            Q = self._run_blocks(Q, X)
        return Q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_loops", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]

    # 모델 로드
    model = AMKPDModel_Old(
        vocab_size=11, d_model=a["d_model"], num_heads=a["num_heads"],
        num_layers=a["num_layers"], loops=args.max_loops, H_cycles=a["H_cycles"],
        L_cycles=a["L_cycles"], kernel_power=a["kernel_power"],
        expansion_ratio=a["expansion_ratio"], conv_kernel_size=a.get("conv_kernel", 2),
    ).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()

    # 데이터 로드
    ds = SudokuDataset(a["data_dir"], "test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_inputs, all_labels = [], []
    count = 0
    for inp, lbl in loader:
        all_inputs.append(inp)
        all_labels.append(lbl)
        count += inp.shape[0]
        if count >= args.max_samples:
            break
    all_inputs = torch.cat(all_inputs)[:args.max_samples].to(device)
    all_labels = torch.cat(all_labels)[:args.max_samples].to(device)
    N_samples = all_inputs.shape[0]

    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Samples: {N_samples}, Max loops: {args.max_loops}")
    print(f"H_cycles={a['H_cycles']}, L_cycles={a['L_cycles']}, kernel_power={a['kernel_power']}")
    print()

    # 루프별 평가
    print(f"{'loop':>4} | {'token_acc':>10} | {'puzzle_acc':>10} | {'correct/total':>14}")
    print("-" * 50)

    with torch.no_grad():
        # 배치 분할
        bs = args.batch_size
        for loop_i in range(1, args.max_loops + 1):
            total_correct = 0
            total_valid = 0
            total_puzzle_correct = 0

            for start in range(0, N_samples, bs):
                end = min(start + bs, N_samples)
                inp = all_inputs[start:end]
                lbl = all_labels[start:end]
                B = inp.shape[0]
                seq_len = inp.shape[1]

                # 매 루프 수마다 처음부터 다시 실행
                if loop_i == 1 or start == 0:
                    pass  # 아래에서 처리

                # carry 초기화 후 loop_i번 반복
                seq_len = inp.shape[1]
                pos = torch.arange(seq_len, device=device).unsqueeze(0)
                X = model.embedding(inp) + model.pos_emb(pos)
                X = model.input_norm(X)
                Q = model.init_hidden.unsqueeze(0).unsqueeze(0).expand(B, seq_len, -1).clone()

                for _loop in range(loop_i):
                    Q = model.forward_single_loop(Q, X)

                Q_norm = model.final_norm(Q)
                logits = model.lm_head(Q_norm)
                preds = logits.argmax(dim=-1)

                mask = lbl != 0
                correct = (preds == lbl) & mask
                total_correct += correct.sum().item()
                total_valid += mask.sum().item()
                total_puzzle_correct += (correct | ~mask).all(dim=1).sum().item()

            token_acc = total_correct / max(1, total_valid)
            puzzle_acc = total_puzzle_correct / N_samples

            print(f"{loop_i:>4} | {token_acc*100:>9.2f}% | {puzzle_acc*100:>9.2f}% | {total_correct}/{total_valid}")

    print("\nDone.")


if __name__ == "__main__":
    main()
