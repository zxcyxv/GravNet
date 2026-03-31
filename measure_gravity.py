"""
prompt.txt의 3가지 중력 증거 측정:
  1. KDE 단조 증가성 — f(Q) = Σ_j W_ij 가 루프마다 증가하는가
  2. 복원력 야코비안 — ∂m/∂V 의 대각 성분이 음수인가
  3. 벡터장 정렬도 — cos(m_concat, H_out) 가 양수인가

맞춘 문제 vs 틀린 문제 분리 출력.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import create_dataloaders
from amkpd_model import AMKPDModel, AMK_Block, AMKPDCarry


# ── 텔레메트리용 forward hook (m_concat, H_out, density 캡처) ──

class BlockProbe:
    """AMK_Block의 중간값을 캡처하는 프로브."""

    def __init__(self, block: AMK_Block):
        self.block = block
        self.m_proj_list = []     # [B, N, d] — 사영된 중력 벡터
        self.delta_Q_list = []    # [B, N, d] — Q_out - Q_in (최종 상태 변화)
        self.density_list = []    # [B, N] — effective neighbors
        self.constraint_mass_list = []  # [B, N] — 제약 이웃 질량 비율
        self.non_constraint_mass_list = []  # [B, N] — 비제약 질량 비율
        self.V_proj_list = []     # [B, H, N, d_h]
        self.m_list = []          # [B, H, N, d_h]
        self._hook = None
        self._constraint_mask = None  # [N, N] bool
        self._patch_forward()

    def _patch_forward(self):
        """forward를 래핑하여 중간값을 캡처."""
        original_forward = self.block.forward

        probe = self  # closure 참조

        def patched_forward(Q_in, X):
            B, N, d = Q_in.shape
            H = probe.block.num_heads
            d_h = probe.block.head_dim

            H_context = Q_in + X

            qkv = probe.block.W_QKV(H_context).view(B, N, 3, H, d_h)
            Q_proj = qkv[:, :, 0].transpose(1, 2)
            K_proj = qkv[:, :, 1].transpose(1, 2)
            V_proj = qkv[:, :, 2].transpose(1, 2)

            Phi_Q = F.elu(Q_proj) + 1.0
            Phi_K = F.elu(K_proj) + 1.0

            scale = d_h ** -0.5
            W = torch.matmul(Phi_Q, Phi_K.transpose(-1, -2)) * scale
            W = F.relu(W)
            kp = probe.block.kernel_power
            if kp == 2:
                W = W * W
            elif kp == 4:
                W = W * W; W = W * W
            elif kp != 1:
                W = W ** kp

            Attraction = torch.matmul(W, V_proj)
            Norm = W.sum(dim=-1, keepdim=True) + 1e-6
            C = Attraction / Norm
            m = C - V_proj

            # ── 캡처: 정규화 엔트로피 + 제약 질량 분리 (측정1) ──
            W_norm_dist = W / (W.sum(dim=-1, keepdim=True) + 1e-12)  # [B,H,N,N]
            log_p = torch.log(W_norm_dist + 1e-12)
            entropy = -(W_norm_dist * log_p).sum(dim=-1)  # [B, H, N]
            eff_neighbors = entropy.exp()  # [B, H, N]
            eff_neighbors_mean = eff_neighbors.mean(dim=1)  # [B, N]
            probe.density_list.append(eff_neighbors_mean.detach().cpu())

            # 제약 이웃 질량 비율 (행+열+박스 이웃 vs 나머지)
            if probe._constraint_mask is None:
                # [N, N] 마스크 생성 (최초 1회)
                N_cells = N
                cmask = torch.zeros(N_cells, N_cells, dtype=torch.bool, device=W.device)
                for i in range(N_cells):
                    row_i = i // 9
                    col_i = i % 9
                    box_r, box_c = (row_i // 3) * 3, (col_i // 3) * 3
                    for j in range(N_cells):
                        if i == j:
                            continue
                        row_j = j // 9
                        col_j = j % 9
                        box_rj, box_cj = (row_j // 3) * 3, (col_j // 3) * 3
                        if row_i == row_j or col_i == col_j or (box_r == box_rj and box_c == box_cj):
                            cmask[i, j] = True
                probe._constraint_mask = cmask  # [N, N]

            cmask = probe._constraint_mask  # [N, N]
            # W_norm_dist: [B, H, N, N] → 헤드 평균 [B, N, N]
            W_avg = W_norm_dist.mean(dim=1)  # [B, N, N]
            constraint_mass = (W_avg * cmask.unsqueeze(0).float()).sum(dim=-1)  # [B, N]
            non_constraint_mass = (W_avg * (~cmask).unsqueeze(0).float()).sum(dim=-1)  # [B, N]
            probe.constraint_mass_list.append(constraint_mass.detach().cpu())
            probe.non_constraint_mass_list.append(non_constraint_mass.detach().cpu())

            m_concat = m.transpose(1, 2).contiguous().view(B, N, d)
            C_concat = C.transpose(1, 2).contiguous().view(B, N, d)
            m_proj = probe.block.W_O_aux(torch.cat([m_concat, C_concat], dim=-1))

            # ── 캡처: m_proj (측정3) ──
            probe.m_proj_list.append(m_proj.detach().cpu())
            probe.V_proj_list.append(V_proj.detach().cpu())
            probe.m_list.append(m.detach().cpu())

            # graph break 방지
            probe.block._last_m_norm = m_concat.detach().norm(dim=-1).mean()
            probe.block._last_C_norm = C_concat.detach().norm(dim=-1).mean()

            Q_interact = probe.block.norm1(Q_in + 1.0 * m_proj)

            GU = probe.block.W_up(Q_interact)
            G, U = GU.chunk(2, dim=-1)
            H_ffn = F.silu(G) * U

            N_seq = Q_interact.shape[1]
            H_ffn_t = H_ffn.transpose(1, 2)
            H_conv_t = probe.block.dw_conv(H_ffn_t)
            H_conv_t = F.silu(H_conv_t[..., :N_seq])
            H_conv = H_conv_t.transpose(1, 2).contiguous()

            H_out = probe.block.W_down(H_conv)

            Q_out = probe.block.norm2(Q_interact + H_out)

            # ── 캡처: ΔQ = Q_out - Q_in (측정3) ──
            probe.delta_Q_list.append((Q_out - Q_in).detach().cpu())

            return Q_out

        self.block.forward = patched_forward

    def clear(self):
        self.m_proj_list.clear()
        self.delta_Q_list.clear()
        self.density_list.clear()
        self.constraint_mass_list.clear()
        self.non_constraint_mass_list.clear()
        self.V_proj_list.clear()
        self.m_list.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_loops", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    a = ckpt["args"]

    _, test_loader, meta = create_dataloaders(
        a["data_dir"], args.batch_size, 0, device.type == "cuda"
    )

    model = AMKPDModel(
        vocab_size=meta["vocab_size"],
        d_model=a["d_model"],
        num_heads=a["num_heads"],
        num_layers=a["num_layers"],
        loops=args.n_loops,
        H_cycles=a["H_cycles"],
        L_cycles=a["L_cycles"],
        kernel_power=a["kernel_power"],
        expansion_ratio=a["expansion_ratio"],
        conv_kernel_size=a.get("conv_kernel", 2),
    ).to(device)

    # state dict 변환
    state = {}
    for k, v in ckpt["model"].items():
        k = k.replace("_orig_mod.", "")
        state[k] = v
    for bi in range(a["num_layers"]):
        pfx = f"blocks.{bi}"
        if f"{pfx}.W_Q.weight" in state:
            state[f"{pfx}.W_QKV.weight"] = torch.cat([
                state.pop(f"{pfx}.W_Q.weight"),
                state.pop(f"{pfx}.W_K.weight"),
                state.pop(f"{pfx}.W_V.weight"),
            ], dim=0)
            state[f"{pfx}.W_O_aux.weight"] = torch.cat([
                state.pop(f"{pfx}.W_O.weight"),
                state.pop(f"{pfx}.W_aux.weight"),
            ], dim=1)
    model.load_state_dict(state)
    model.eval()

    # ── 마지막 블록에 프로브 설치 ──
    last_block_idx = a["num_layers"] - 1
    probe = BlockProbe(model.blocks[last_block_idx])

    # ── 데이터 수집 ──
    print("Collecting telemetry...")

    # 결과 저장 (샘플별)
    all_results = []  # list of dict per sample

    total_collected = 0
    for inputs, labels in test_loader:
        if total_collected >= args.max_samples:
            break

        B_batch = min(inputs.shape[0], args.max_samples - total_collected)
        inputs = inputs[:B_batch].to(device)
        labels = labels[:B_batch].to(device)
        seq_len = inputs.shape[1]

        carry = model.initial_carry(B_batch, seq_len, device)
        batch = (inputs, labels)

        # 루프별 캡처
        sample_density = []      # [n_loops][B, N]
        sample_constraint = []   # [n_loops][B, N]
        sample_non_constraint = []  # [n_loops][B, N]
        sample_m_proj = []       # [n_loops][B, N, d]
        sample_delta_Q = []      # [n_loops][B, N, d]

        with torch.no_grad():
            for outer in range(args.n_loops):
                probe.clear()
                carry, logits, _ = model(carry, batch)

                # 마지막 블록의 마지막 micro-step 데이터 사용
                if probe.density_list:
                    sample_density.append(probe.density_list[-1][:B_batch])
                    sample_constraint.append(probe.constraint_mass_list[-1][:B_batch])
                    sample_non_constraint.append(probe.non_constraint_mass_list[-1][:B_batch])
                    sample_m_proj.append(probe.m_proj_list[-1][:B_batch])
                    sample_delta_Q.append(probe.delta_Q_list[-1][:B_batch])

        # ── 정확도 판정 ──
        pred = logits.argmax(dim=-1)  # [B, N]
        blank_mask = (inputs < 2)  # blank tokens
        correct_per_sample = []  # bool per sample

        for b in range(B_batch):
            blank_b = blank_mask[b]
            if blank_b.any():
                correct = (pred[b][blank_b] == labels[b][blank_b]).all().item()
            else:
                correct = True
            correct_per_sample.append(correct)

        # ── 샘플별 측정값 계산 ──
        n_collected_loops = len(sample_density)
        for b in range(B_batch):
            blank_b = blank_mask[b].cpu().numpy()
            given_b = ~blank_b

            result = {
                "correct": correct_per_sample[b],
                # 측정1: effective neighbors + 제약 질량
                "eff_neighbors_blank": [],
                "eff_neighbors_given": [],
                "constraint_mass_blank": [],
                "constraint_mass_given": [],
                "non_constraint_mass_blank": [],
                "non_constraint_mass_given": [],
                # 측정3: cos(m_proj, ΔQ) per loop
                "cos_mDQ_blank": [],
                "cos_mDQ_given": [],
                "cos_mDQ_all": [],
                # 부가: ||m_proj|| / ||ΔQ|| 기여도
                "m_proj_ratio_blank": [],
                "m_proj_ratio_given": [],
            }

            for l in range(n_collected_loops):
                density = sample_density[l][b].numpy()  # [N]
                c_mass = sample_constraint[l][b].numpy()  # [N]
                nc_mass = sample_non_constraint[l][b].numpy()  # [N]
                mp = sample_m_proj[l][b].numpy()        # [N, d]
                dq = sample_delta_Q[l][b].numpy()       # [N, d]

                # 측정1: 유효 이웃 수 + 제약/비제약 질량
                result["eff_neighbors_blank"].append(float(density[blank_b].mean()))
                result["eff_neighbors_given"].append(float(density[given_b].mean()))
                result["constraint_mass_blank"].append(float(c_mass[blank_b].mean()))
                result["constraint_mass_given"].append(float(c_mass[given_b].mean()))
                result["non_constraint_mass_blank"].append(float(nc_mass[blank_b].mean()))
                result["non_constraint_mass_given"].append(float(nc_mass[given_b].mean()))

                # 측정3: cos(m_proj, ΔQ) — 중력 사영이 실제 이동에 반영되는가
                mp_norm = np.linalg.norm(mp, axis=-1, keepdims=True) + 1e-12
                dq_norm = np.linalg.norm(dq, axis=-1, keepdims=True) + 1e-12
                cos_sim = np.sum((mp / mp_norm) * (dq / dq_norm), axis=-1)  # [N]

                result["cos_mDQ_blank"].append(float(cos_sim[blank_b].mean()))
                result["cos_mDQ_given"].append(float(cos_sim[given_b].mean()))
                result["cos_mDQ_all"].append(float(cos_sim.mean()))

                # ||m_proj|| / ||ΔQ|| — 중력이 전체 이동에서 차지하는 비율
                mp_norms = np.linalg.norm(mp, axis=-1)  # [N]
                dq_norms = np.linalg.norm(dq, axis=-1) + 1e-12  # [N]
                ratio = mp_norms / dq_norms
                result["m_proj_ratio_blank"].append(float(ratio[blank_b].mean()))
                result["m_proj_ratio_given"].append(float(ratio[given_b].mean()))

            all_results.append(result)

        total_collected += B_batch
        print(f"  Collected {total_collected}/{args.max_samples} samples")

    # ── 측정2: 야코비안 (소수 샘플만 — 계산 비용 큼) ──
    print("Computing Jacobian (small sample)...")
    jacobian_results = {"diag_mean": [], "correct": []}

    # 테스트 데이터 재로드
    test_iter = iter(test_loader)
    inputs_j, labels_j = next(test_iter)
    B_j = min(4, inputs_j.shape[0])
    inputs_j = inputs_j[:B_j].to(device)
    labels_j = labels_j[:B_j].to(device)
    seq_len_j = inputs_j.shape[1]

    carry_j = model.initial_carry(B_j, seq_len_j, device)
    batch_j = (inputs_j, labels_j)

    # 마지막 루프까지 진행 (no_grad)
    with torch.no_grad():
        for _ in range(args.n_loops - 1):
            probe.clear()
            carry_j, _, _ = model(carry_j, batch_j)

    # 마지막 루프: grad 활성화하여 야코비안 계산
    probe.clear()
    # 수동으로 마지막 블록의 V_proj에서 grad 추적
    block = model.blocks[last_block_idx]

    # carry의 hidden을 사용하여 수동 forward
    Q_in = carry_j.current_hidden.detach().requires_grad_(False)
    seq_len_j = inputs_j.shape[1]
    pos = torch.arange(seq_len_j, device=device).unsqueeze(0)
    X = model.embedding(carry_j.current_inputs) + model.pos_emb(pos)
    X = model.input_norm(X)

    # 마지막 블록 직전까지 진행
    Q = Q_in
    for _h in range(model.H_cycles - 1):
        for _l in range(model.L_cycles):
            Q = model._run_blocks(Q, X)

    for _l in range(model.L_cycles - 1):
        Q = model._run_blocks(Q, X)

    # 마지막 블록 전까지의 블록들 통과
    for bi in range(last_block_idx):
        Q = model.blocks[bi](Q, X)

    # 마지막 블록: V_proj에서 grad 추적
    H_context = Q + X
    d_model = a["d_model"]
    num_heads = a["num_heads"]
    d_h = d_model // num_heads

    qkv = block.W_QKV(H_context).view(B_j, seq_len_j, 3, num_heads, d_h)
    V_proj = qkv[:, :, 2].transpose(1, 2)  # [B, H, N, d_h]
    V_proj = V_proj.detach().requires_grad_(True)

    Q_proj = qkv[:, :, 0].transpose(1, 2).detach()
    K_proj = qkv[:, :, 1].transpose(1, 2).detach()

    Phi_Q = F.elu(Q_proj) + 1.0
    Phi_K = F.elu(K_proj) + 1.0

    scale = d_h ** -0.5
    W = torch.matmul(Phi_Q, Phi_K.transpose(-1, -2)) * scale
    W = F.relu(W)
    kp = a["kernel_power"]
    if kp == 2:
        W = W * W
    elif kp == 4:
        W = W * W; W = W * W
    elif kp != 1:
        W = W ** kp

    Attraction = torch.matmul(W, V_proj)
    Norm = W.sum(dim=-1, keepdim=True) + 1e-6
    C = Attraction / Norm
    m = C - V_proj  # [B, H, N, d_h]

    # ∂m/∂V 의 대각 성분: 각 (b, h, n, k)에 대해 ∂m[b,h,n,k]/∂V[b,h,n,k]
    # 효율적 계산: m의 각 성분에 대해 backward
    # 간단하게 몇 개 셀만 샘플링
    diag_values = []
    sample_cells = list(range(0, seq_len_j, 9))[:9]  # 9개 셀 샘플

    for b_idx in range(min(2, B_j)):
        for cell_idx in sample_cells:
            for h_idx in range(min(4, num_heads)):
                # m[b, h, n, :] 의 합에 대해 V[b, h, n, :] 로 편미분
                if V_proj.grad is not None:
                    V_proj.grad.zero_()
                m_scalar = m[b_idx, h_idx, cell_idx].sum()
                m_scalar.backward(retain_graph=True)
                if V_proj.grad is not None:
                    # 대각 성분: ∂m[b,h,n,k]/∂V[b,h,n,k]의 평균
                    grad_diag = V_proj.grad[b_idx, h_idx, cell_idx]  # [d_h]
                    diag_values.append(float(grad_diag.mean().item()))

    jacobian_diag_mean = float(np.mean(diag_values)) if diag_values else 0.0
    jacobian_diag_std = float(np.std(diag_values)) if diag_values else 0.0
    print(f"  Jacobian diag(∂m/∂V) mean: {jacobian_diag_mean:.4f} ± {jacobian_diag_std:.4f}")
    print(f"  (Expected: ≈ -1.0 for restoring force)")

    # ── 맞춘 문제 vs 틀린 문제 분리 ──
    correct_samples = [r for r in all_results if r["correct"]]
    wrong_samples = [r for r in all_results if not r["correct"]]

    print(f"\nTotal samples: {len(all_results)}")
    print(f"  Correct: {len(correct_samples)}")
    print(f"  Wrong:   {len(wrong_samples)}")

    def aggregate(samples, key):
        """샘플들의 루프별 값을 평균."""
        if not samples:
            return []
        n_loops = len(samples[0][key])
        return [float(np.mean([s[key][l] for s in samples])) for l in range(n_loops)]

    # ── 시각화 ──
    os.makedirs("visualizations", exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(26, 12))

    n_loops_plot = len(all_results[0]["eff_neighbors_blank"]) if all_results else 0
    loop_x = range(1, n_loops_plot + 1)

    # 제약 이웃의 baseline: 각 셀은 행8+열8+박스8-중복 = 20개 제약 이웃, 60개 비제약
    # 제약 질량 uniform baseline = 20/80 = 0.25 (자기 자신 제외)
    CONSTRAINT_BASELINE = 20.0 / 80.0

    for group, label, row in [
        (correct_samples, "Correct", 0),
        (wrong_samples, "Wrong", 1),
    ]:
        if not group:
            continue

        # ── 측정1a: 유효 이웃 수 ──
        eff_blank = aggregate(group, "eff_neighbors_blank")
        eff_given = aggregate(group, "eff_neighbors_given")
        ax = axes[row, 0]
        ax.plot(loop_x, eff_blank, 'ro-', markersize=3, label='Blank')
        ax.plot(loop_x, eff_given, 'bs-', markersize=3, label='Given')
        ax.set_title(f"Effective Neighbors ({label})\nexp(entropy)", fontsize=10)
        ax.set_xlabel("Outer Loop")
        ax.set_ylabel("Effective # Neighbors")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 측정1b: 제약 질량 비율 ──
        c_blank = aggregate(group, "constraint_mass_blank")
        c_given = aggregate(group, "constraint_mass_given")
        nc_blank = aggregate(group, "non_constraint_mass_blank")
        nc_given = aggregate(group, "non_constraint_mass_given")

        ax = axes[row, 1]
        ax.plot(loop_x, c_blank, 'ro-', markersize=3, linewidth=2, label='Constraint (blank)')
        ax.plot(loop_x, nc_blank, 'r--', markersize=2, alpha=0.6, label='Non-constraint (blank)')
        ax.plot(loop_x, c_given, 'bs-', markersize=3, linewidth=2, label='Constraint (given)')
        ax.plot(loop_x, nc_given, 'b--', markersize=2, alpha=0.6, label='Non-constraint (given)')
        ax.axhline(y=CONSTRAINT_BASELINE, color='gray', linestyle=':', alpha=0.6,
                   label=f'Uniform baseline ({CONSTRAINT_BASELINE:.2f})')
        ax.set_title(f"Constraint vs Non-constraint Mass ({label})\n"
                     f"(행+열+박스 이웃 질량 vs 나머지)", fontsize=10)
        ax.set_xlabel("Outer Loop")
        ax.set_ylabel("Attention Mass Ratio")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 단조 증가 검증 (제약 질량이 증가하는가)
        if c_blank:
            mono = sum(1 for i in range(1, len(c_blank)) if c_blank[i] >= c_blank[i-1])
            ax.text(0.02, 0.98, f"Constraint↑: {mono}/{len(c_blank)-1} steps",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ── 측정2: 야코비안 ──
        ax = axes[row, 2]
        ax.bar(['diag(∂m/∂V)'], [jacobian_diag_mean],
               yerr=[jacobian_diag_std], color='purple', alpha=0.7, capsize=10)
        ax.axhline(y=-1.0, color='red', linestyle='--', linewidth=2, label='Ideal: -I')
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f"Restoring Force Jacobian\n"
                     f"diag(∂m/∂V) = {jacobian_diag_mean:.4f} ± {jacobian_diag_std:.4f}\n"
                     f"(⚠ architecturally trivial: m=C-V → ∂m/∂V≈-I)", fontsize=9)
        ax.set_ylabel("Jacobian diagonal mean")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 측정3: cos(m_proj, ΔQ) + ||m_proj||/||ΔQ|| ──
        cos_blank = aggregate(group, "cos_mDQ_blank")
        cos_given = aggregate(group, "cos_mDQ_given")
        cos_all = aggregate(group, "cos_mDQ_all")
        ratio_blank = aggregate(group, "m_proj_ratio_blank")
        ratio_given = aggregate(group, "m_proj_ratio_given")

        ax = axes[row, 3]
        ax.plot(loop_x, cos_blank, 'ro-', markersize=3, label='cos Blank')
        ax.plot(loop_x, cos_given, 'bs-', markersize=3, label='cos Given')
        ax.plot(loop_x, cos_all, 'k^-', markersize=3, label='cos All', linewidth=2)
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)

        # 오른쪽 축: 기여 비율
        ax2 = ax.twinx()
        ax2.plot(loop_x, ratio_blank, 'r:', markersize=2, alpha=0.6, label='||m_proj||/||ΔQ|| blank')
        ax2.plot(loop_x, ratio_given, 'b:', markersize=2, alpha=0.6, label='||m_proj||/||ΔQ|| given')
        ax2.set_ylabel("||m_proj|| / ||ΔQ||", fontsize=8)
        ax2.legend(loc='lower right', fontsize=6)

        ax.set_title(f"Gravity → Movement Alignment ({label})\ncos(m_proj, ΔQ) + magnitude ratio", fontsize=10)
        ax.set_xlabel("Outer Loop")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-1, 1)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

        mean_cos = float(np.mean(cos_all)) if cos_all else 0
        mean_ratio = float(np.mean(ratio_blank)) if ratio_blank else 0
        ax.text(0.02, 0.02, f"cos={mean_cos:.3f}, ratio={mean_ratio:.2f}",
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round',
                          facecolor='lightgreen' if mean_cos > 0.3 else 'lightyellow',
                          alpha=0.7))

    plt.suptitle("Gravity Evidence: Measurements from prompt.txt\n"
                 "(Top = Correct puzzles, Bottom = Wrong puzzles)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("visualizations/11_gravity_evidence.png", dpi=200)
    plt.close()

    # ── JSON 저장 ──
    metrics = {
        "n_correct": len(correct_samples),
        "n_wrong": len(wrong_samples),
        "jacobian_diag_mean": jacobian_diag_mean,
        "jacobian_diag_std": jacobian_diag_std,
        "correct": {
            "eff_neighbors_blank": aggregate(correct_samples, "eff_neighbors_blank"),
            "eff_neighbors_given": aggregate(correct_samples, "eff_neighbors_given"),
            "constraint_mass_blank": aggregate(correct_samples, "constraint_mass_blank"),
            "constraint_mass_given": aggregate(correct_samples, "constraint_mass_given"),
            "non_constraint_mass_blank": aggregate(correct_samples, "non_constraint_mass_blank"),
            "non_constraint_mass_given": aggregate(correct_samples, "non_constraint_mass_given"),
            "cos_mDQ_blank": aggregate(correct_samples, "cos_mDQ_blank"),
            "cos_mDQ_given": aggregate(correct_samples, "cos_mDQ_given"),
            "cos_mDQ_all": aggregate(correct_samples, "cos_mDQ_all"),
            "m_proj_ratio_blank": aggregate(correct_samples, "m_proj_ratio_blank"),
            "m_proj_ratio_given": aggregate(correct_samples, "m_proj_ratio_given"),
        },
        "wrong": {
            "eff_neighbors_blank": aggregate(wrong_samples, "eff_neighbors_blank"),
            "eff_neighbors_given": aggregate(wrong_samples, "eff_neighbors_given"),
            "constraint_mass_blank": aggregate(wrong_samples, "constraint_mass_blank"),
            "constraint_mass_given": aggregate(wrong_samples, "constraint_mass_given"),
            "non_constraint_mass_blank": aggregate(wrong_samples, "non_constraint_mass_blank"),
            "non_constraint_mass_given": aggregate(wrong_samples, "non_constraint_mass_given"),
            "cos_mDQ_blank": aggregate(wrong_samples, "cos_mDQ_blank"),
            "cos_mDQ_given": aggregate(wrong_samples, "cos_mDQ_given"),
            "cos_mDQ_all": aggregate(wrong_samples, "cos_mDQ_all"),
            "m_proj_ratio_blank": aggregate(wrong_samples, "m_proj_ratio_blank"),
            "m_proj_ratio_given": aggregate(wrong_samples, "m_proj_ratio_given"),
        },
    }
    with open("visualizations/gravity_evidence.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("GRAVITY EVIDENCE SUMMARY")
    print("=" * 60)

    print(f"\n[측정1a] 유효 이웃 수:")
    for group_name, group in [("Correct", correct_samples), ("Wrong", wrong_samples)]:
        if not group:
            print(f"  {group_name}: (no samples)")
            continue
        d = aggregate(group, "eff_neighbors_blank")
        print(f"  {group_name} blanks: {d[0]:.1f} → {d[-1]:.1f} effective neighbors")

    print(f"\n[측정1b] 제약 이웃 질량 비율 (uniform baseline = {CONSTRAINT_BASELINE:.3f}):")
    for group_name, group in [("Correct", correct_samples), ("Wrong", wrong_samples)]:
        if not group:
            print(f"  {group_name}: (no samples)")
            continue
        c = aggregate(group, "constraint_mass_blank")
        nc = aggregate(group, "non_constraint_mass_blank")
        mono = sum(1 for i in range(1, len(c)) if c[i] >= c[i-1])
        print(f"  {group_name} blanks: constraint {c[0]:.3f} → {c[-1]:.3f} "
              f"(↑ {mono}/{len(c)-1} steps), "
              f"non-constraint {nc[0]:.3f} → {nc[-1]:.3f}")
        if c[-1] > CONSTRAINT_BASELINE:
            print(f"    ✓ 제약 이웃 질량 > uniform baseline → 제약 구조로 수렴")
        else:
            print(f"    ✗ 제약 이웃 질량 ≤ uniform baseline")

    print(f"\n[측정2] 복원력 야코비안:")
    print(f"  diag(∂m/∂V) = {jacobian_diag_mean:.4f} ± {jacobian_diag_std:.4f}")
    print(f"  ⚠ 아키텍처 자명: m=C-V → ∂m/∂V = ∂C/∂V - I ≈ -I (W_ii/ΣW ≈ 0)")

    print(f"\n[측정3] 중력→이동 정렬도 cos(m_proj, ΔQ) + 기여 비율:")
    for group_name, group in [("Correct", correct_samples), ("Wrong", wrong_samples)]:
        if not group:
            print(f"  {group_name}: (no samples)")
            continue
        c = aggregate(group, "cos_mDQ_all")
        r = aggregate(group, "m_proj_ratio_blank")
        mean_c = np.mean(c) if c else 0
        mean_r = np.mean(r) if r else 0
        print(f"  {group_name}: cos(m_proj,ΔQ)={mean_c:.3f}, ||m_proj||/||ΔQ||={mean_r:.3f}")
        if mean_c > 0.5:
            print(f"    ✓ 중력이 실제 이동 방향을 지배")
        elif mean_c > 0.2:
            print(f"    △ 중력이 이동에 부분적 기여")
        else:
            print(f"    ✗ 중력과 이동 방향 무관")

    print(f"\nVisualization: visualizations/11_gravity_evidence.png")
    print(f"Metrics: visualizations/gravity_evidence.json")


if __name__ == "__main__":
    main()
