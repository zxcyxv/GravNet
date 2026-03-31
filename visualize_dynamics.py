"""
prompt.txt 기반 4가지 핵심 시각화:
1. Head-wise Topological Specialization Map (헤드별 위상 제약 특화도)
2. Hypersphere Angular Trajectory (초구면 각거리 궤적)
3. Kernel Sparsity via Gini/Tsallis (다항식 커널 희소성)
4. Head Orthogonality (헤드 간 이동벡터 직교성)
"""

import os
import argparse
import json
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F

from dataset import create_dataloaders
from amkpd_model import AMKPDModel, AMK_Block, AMKPDCarry


# ── Sudoku topology helpers ──────────────────────────────────────────────

def get_row_indices(idx):
    """idx (0-80)와 같은 행(row)에 있는 셀 인덱스들 (자기 자신 제외)"""
    row = idx // 9
    return [row * 9 + c for c in range(9) if row * 9 + c != idx]

def get_col_indices(idx):
    """idx와 같은 열(col)에 있는 셀 인덱스들 (자기 자신 제외)"""
    col = idx % 9
    return [r * 9 + col for r in range(9) if r * 9 + col != idx]

def get_box_indices(idx):
    """idx와 같은 3x3 박스에 있는 셀 인덱스들 (자기 자신 제외)"""
    row, col = idx // 9, idx % 9
    box_r, box_c = (row // 3) * 3, (col // 3) * 3
    return [
        (box_r + dr) * 9 + (box_c + dc)
        for dr in range(3) for dc in range(3)
        if (box_r + dr) * 9 + (box_c + dc) != idx
    ]


def gini(array):
    """Gini coefficient: 0=perfect equality, 1=perfect inequality"""
    array = np.abs(array.flatten())
    array = np.sort(array) + 1e-12
    n = len(array)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def tsallis_entropy(p, q=2.0):
    """Tsallis entropy: lower = more sparse (q=2 is default)"""
    p = np.abs(p) + 1e-12
    p = p / p.sum()
    return float((1.0 - np.sum(p ** q)) / (q - 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="체크포인트 경로")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_loops_viz", type=int, default=20,
                        help="Number of outer loops for visualization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Checkpoint loading ──
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    a = ckpt["args"]

    # ── 2. Data ──
    train_loader, test_loader, meta = create_dataloaders(
        a["data_dir"], args.batch_size, 0, device.type == "cuda"
    )

    model = AMKPDModel(
        vocab_size=meta["vocab_size"],
        d_model=a["d_model"],
        num_heads=a["num_heads"],
        num_layers=a["num_layers"],
        loops=args.n_loops_viz,
        H_cycles=a["H_cycles"],
        L_cycles=a["L_cycles"],
        kernel_power=a["kernel_power"],
        expansion_ratio=a["expansion_ratio"],
        conv_kernel_size=a.get("conv_kernel", 2),
    ).to(device)

    # _orig_mod. prefix 제거 + old W_Q/W_K/W_V/W_O/W_aux → fused 변환
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

    # ── 3. Feature extraction via telemetry ──
    print("Extracting features ...")
    inputs, labels = next(iter(test_loader))
    B_viz = min(16, inputs.shape[0])
    inputs = inputs[:B_viz].to(device)
    labels = labels[:B_viz].to(device)
    seq_len = inputs.shape[1]
    H = a["num_heads"]

    # Enable telemetry
    model.log_viz = True
    for b in model.blocks:
        b.log_viz = True

    carry = model.initial_carry(B_viz, seq_len, device)
    batch = (inputs, labels)

    # Per-loop collection: W_per_head[loop][block] = [B, H, N, N]
    # m_per_head[loop][block] = [B, H, N, d_h]
    global_Q = [carry.current_hidden.detach().cpu()]
    W_per_head = []  # list of list of tensors
    m_per_head = []
    K_blocks = len(model.blocks)

    with torch.no_grad():
        for outer in range(args.n_loops_viz):
            for b in model.blocks:
                b.viz_W = []
                b.viz_m = []
                b.viz_H = []
            model.viz_Q = []

            carry, logits, _ = model(carry, batch)

            for q in model.viz_Q:
                if isinstance(q, torch.Tensor):
                    global_Q.append(q.cpu())

            # Collect per-block W and m for this loop
            n_micro = len(model.blocks[0].viz_W)
            for mi in range(n_micro):
                loop_W = []
                loop_m = []
                for k in range(K_blocks):
                    loop_W.append(model.blocks[k].viz_W[mi].cpu())  # [B, H, N, N]
                    loop_m.append(model.blocks[k].viz_m[mi].cpu())  # [B, H, N, d_h]
                W_per_head.append(loop_W)
                m_per_head.append(loop_m)

    # Disable telemetry
    model.log_viz = False
    for b in model.blocks:
        b.log_viz = False

    os.makedirs("visualizations", exist_ok=True)
    metrics = {}

    # Use sample 0 for all per-sample analyses
    inp_0 = inputs[0].cpu().numpy()
    lbl_0 = labels[0].cpu().numpy()
    given_mask = (inp_0 > 1)  # 1=blank, 2-10 = digits 1-9
    blank_indices = np.where(~given_mask)[0]

    N_seq = seq_len  # 81
    n_loops_collected = len(W_per_head)
    d_h = a["d_model"] // H

    print(f"Collected {n_loops_collected} micro-steps, {len(global_Q)} Q snapshots")
    print(f"Blanks: {len(blank_indices)}, Given: {given_mask.sum()}")

    # ════════════════════════════════════════════════════════════════════
    # 1. Head-wise Topological Specialization Map
    # ════════════════════════════════════════════════════════════════════
    print("\n[1/4] Head-wise Topological Specialization Map ...")

    # Pick a blank cell near the center
    query_idx = blank_indices[len(blank_indices) // 2] if len(blank_indices) > 0 else 40

    row_neighbors = set(get_row_indices(query_idx))
    col_neighbors = set(get_col_indices(query_idx))
    box_neighbors = set(get_box_indices(query_idx))

    # Use the last micro-step, last block
    W_last = W_per_head[-1][-1][0].numpy()  # [H, N, N], sample 0

    fig, axes = plt.subplots(2, (H + 1) // 2, figsize=(4 * ((H + 1) // 2), 8))
    axes = axes.flatten()

    head_topology_mass = {"row": [], "col": [], "box": []}

    for h in range(H):
        ax = axes[h]
        att_row = W_last[h, query_idx]  # [81]
        grid_9x9 = att_row.reshape(9, 9)
        sns.heatmap(grid_9x9, ax=ax, cmap='YlOrRd', cbar=True, cbar_kws={'shrink': 0.6},
                    square=True, linewidths=0.5, linecolor='gray')
        ax.set_title(f"Head {h}", fontsize=11)

        # In-Topology Mass Ratio
        total_mass = att_row.sum() + 1e-12
        row_mass = att_row[list(row_neighbors)].sum() / total_mass
        col_mass = att_row[list(col_neighbors)].sum() / total_mass
        box_mass = att_row[list(box_neighbors)].sum() / total_mass
        head_topology_mass["row"].append(float(row_mass))
        head_topology_mass["col"].append(float(col_mass))
        head_topology_mass["box"].append(float(box_mass))

        ax.set_xlabel(f"R:{row_mass:.2f} C:{col_mass:.2f} B:{box_mass:.2f}", fontsize=8)

    for i in range(H, len(axes)):
        axes[i].axis('off')

    query_r, query_c = query_idx // 9, query_idx % 9
    fig.suptitle(
        f"Head-wise Attraction for Blank Cell ({query_r},{query_c}) [idx={query_idx}]\n"
        f"R=Row mass, C=Col mass, B=Box mass",
        fontsize=13
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("visualizations/01_head_topology_specialization.png", dpi=200)
    plt.close()

    # Bar chart: per-head topology mass
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(H)
    width = 0.25
    ax.bar(x - width, head_topology_mass["row"], width, label='Row', color='#e74c3c')
    ax.bar(x, head_topology_mass["col"], width, label='Column', color='#3498db')
    ax.bar(x + width, head_topology_mass["box"], width, label='Box', color='#2ecc71')
    ax.set_xlabel("Head")
    ax.set_ylabel("In-Topology Mass Ratio")
    ax.set_title("Head Topological Specialization (Row / Col / Box)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{i}" for i in range(H)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    # Uniform baseline: each constraint has 8 neighbors out of 80 other cells
    ax.axhline(y=8.0 / 80.0, color='gray', linestyle='--', alpha=0.5, label='Uniform (8/80)')
    plt.tight_layout()
    plt.savefig("visualizations/01_head_topology_bar.png", dpi=200)
    plt.close()

    metrics["head_topology_mass"] = head_topology_mass

    # ════════════════════════════════════════════════════════════════════
    # 2. Hypersphere Angular Trajectory
    # ════════════════════════════════════════════════════════════════════
    print("[2/4] Hypersphere Angular Trajectory ...")

    n_q = len(global_Q)
    cos_sims_given = []
    cos_sims_blank = []
    angular_velocities = []

    for l in range(1, n_q):
        Q_prev = global_Q[l - 1][0].numpy()  # [81, d], sample 0
        Q_curr = global_Q[l][0].numpy()

        # Per-token cosine similarity
        dot = np.sum(Q_prev * Q_curr, axis=-1)
        norm_prev = np.linalg.norm(Q_prev, axis=-1) + 1e-12
        norm_curr = np.linalg.norm(Q_curr, axis=-1) + 1e-12
        cos_sim = dot / (norm_prev * norm_curr)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        cos_sims_given.append(float(cos_sim[given_mask].mean()))
        cos_sims_blank.append(float(cos_sim[~given_mask].mean()))

        # Angular velocity = arccos(cos_sim)
        angles = np.arccos(cos_sim)
        angular_velocities.append(float(angles.mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = range(1, n_q)
    ax1.plot(steps, cos_sims_given, 'b-o', markersize=3, label='Given cells', alpha=0.8)
    ax1.plot(steps, cos_sims_blank, 'r-o', markersize=3, label='Blank cells', alpha=0.8)
    ax1.set_xlabel("Step (Q snapshot)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Cosine Similarity S_C(Q^l, Q^{l-1})")
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, angular_velocities, 'g-s', markersize=3, linewidth=2)
    ax2.set_xlabel("Step (Q snapshot)")
    ax2.set_ylabel("Angular Velocity (radians)")
    ax2.set_title("Angular Velocity on Hypersphere")
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Projected Dynamics: Hypersphere Angular Trajectory", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("visualizations/02_angular_trajectory.png", dpi=200)
    plt.close()

    metrics["cosine_sim_given"] = cos_sims_given
    metrics["cosine_sim_blank"] = cos_sims_blank
    metrics["angular_velocity"] = angular_velocities

    # ════════════════════════════════════════════════════════════════════
    # 3. Kernel Sparsity via Gini / Tsallis Entropy
    # ════════════════════════════════════════════════════════════════════
    print("[3/4] Kernel Sparsity (Gini / Tsallis) ...")

    # Per head, per micro-step: Gini and Tsallis of W[h] for all query tokens
    gini_per_step = {h: [] for h in range(H)}  # gini_per_step[head][step]
    tsallis_per_step = {h: [] for h in range(H)}

    for step_idx in range(n_loops_collected):
        # Use last block
        W_step = W_per_head[step_idx][-1][0].numpy()  # [H, N, N], sample 0
        for h in range(H):
            W_h = W_step[h]  # [N, N]
            # Average Gini across all query tokens
            ginis = [gini(W_h[q]) for q in range(N_seq)]
            tsallises = [tsallis_entropy(W_h[q]) for q in range(N_seq)]
            gini_per_step[h].append(float(np.mean(ginis)))
            tsallis_per_step[h].append(float(np.mean(tsallises)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap('tab10')

    for h in range(H):
        color = cmap(h)
        ax1.plot(range(1, n_loops_collected + 1), gini_per_step[h],
                 marker='o', markersize=2, color=color, label=f'Head {h}', alpha=0.8)
        ax2.plot(range(1, n_loops_collected + 1), tsallis_per_step[h],
                 marker='o', markersize=2, color=color, label=f'Head {h}', alpha=0.8)

    ax1.set_xlabel("Micro-step")
    ax1.set_ylabel("Gini Coefficient")
    ax1.set_title("Gini Coefficient (higher = sparser)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Micro-step")
    ax2.set_ylabel("Tsallis Entropy (q=2)")
    ax2.set_title("Tsallis Entropy (lower = sparser)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Polynomial Kernel Sparsity Across Loops", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("visualizations/03_kernel_sparsity.png", dpi=200)
    plt.close()

    metrics["gini_per_head"] = {str(h): gini_per_step[h] for h in range(H)}
    metrics["tsallis_per_head"] = {str(h): tsallis_per_step[h] for h in range(H)}

    # ════════════════════════════════════════════════════════════════════
    # 4. Head Orthogonality (m_h 간 코사인 거리)
    # ════════════════════════════════════════════════════════════════════
    print("[4/4] Head Orthogonality of Net Force Vectors ...")

    # Per micro-step: compute H x H cosine similarity of m_h
    # m_h shape per step/block: [B, H, N, d_h]
    # Flatten m_h to [H, B*N*d_h] then compute pairwise cosine

    ortho_matrices = []  # list of [H, H] arrays
    mean_off_diag = []   # mean abs off-diagonal cosine sim

    checkpoints_to_plot = []
    # Sample a few steps for heatmap plots
    plot_indices = [0, n_loops_collected // 4, n_loops_collected // 2,
                    3 * n_loops_collected // 4, n_loops_collected - 1]
    plot_indices = sorted(set([max(0, min(i, n_loops_collected - 1)) for i in plot_indices]))

    for step_idx in range(n_loops_collected):
        # Use last block, sample 0
        m_step = m_per_head[step_idx][-1][0].numpy()  # [H, N, d_h]

        # Flatten each head: [H, N*d_h]
        m_flat = m_step.reshape(H, -1)
        # Normalize
        norms = np.linalg.norm(m_flat, axis=1, keepdims=True) + 1e-12
        m_normed = m_flat / norms

        # Cosine similarity matrix [H, H]
        cos_mat = m_normed @ m_normed.T
        ortho_matrices.append(cos_mat)

        # Mean absolute off-diagonal
        mask = ~np.eye(H, dtype=bool)
        mean_off_diag.append(float(np.abs(cos_mat[mask]).mean()))

        if step_idx in plot_indices:
            checkpoints_to_plot.append((step_idx, cos_mat.copy()))

    # Plot: H x H heatmaps at selected steps
    n_plots = len(checkpoints_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for i, (step_idx, cos_mat) in enumerate(checkpoints_to_plot):
        ax = axes[i]
        sns.heatmap(cos_mat, ax=ax, vmin=-1, vmax=1, cmap='RdBu_r', center=0,
                    annot=True, fmt='.2f', square=True, cbar=(i == n_plots - 1),
                    xticklabels=[f"H{h}" for h in range(H)],
                    yticklabels=[f"H{h}" for h in range(H)])
        ax.set_title(f"Step {step_idx + 1}", fontsize=10)

    plt.suptitle("Head-wise m Vector Cosine Similarity (0 = Orthogonal)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("visualizations/04_head_orthogonality_heatmaps.png", dpi=200)
    plt.close()

    # Plot: mean off-diagonal over steps
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, n_loops_collected + 1), mean_off_diag, 'b-o', markersize=3)
    ax.set_xlabel("Micro-step")
    ax.set_ylabel("Mean |cos(m_h, m_j)| (off-diagonal)")
    ax.set_title("Head Force Vector Orthogonality Over Loops\n(Lower = more independent heads)")
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Perfect orthogonality')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Mode collapse')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/04_head_orthogonality_trend.png", dpi=200)
    plt.close()

    metrics["head_orthogonality_mean_offdiag"] = mean_off_diag

    # ════════════════════════════════════════════════════════════════════
    # 5. Class-Conditioned Latent Projection (Neural Collapse 시각화)
    # ════════════════════════════════════════════════════════════════════
    print("[5/7] Class-Conditioned Neural Collapse Trajectory ...")

    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False

    n_q = len(global_Q)
    # 선택할 루프 체크포인트 (최대 8개)
    q_checkpoints = [0, 1, 2, 4, 8, 12, 16, n_q - 1]
    q_checkpoints = sorted(set([min(c, n_q - 1) for c in q_checkpoints]))

    # 모든 체크포인트의 Q를 한번에 차원축소
    Q_all = np.concatenate([global_Q[c][0].numpy() for c in q_checkpoints], axis=0)  # [n_ckpt*81, d]
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    Q_2d_all = reducer.fit_transform(Q_all)  # [n_ckpt*81, 2]
    Q_2d_split = Q_2d_all.reshape(len(q_checkpoints), N_seq, 2)

    cmap_9 = plt.get_cmap('tab10')
    n_cols = min(4, len(q_checkpoints))
    n_rows = (len(q_checkpoints) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for ci, ckpt_idx in enumerate(q_checkpoints):
        ax = axes[ci]
        ax.set_title(f"Q snapshot {ckpt_idx}", fontsize=11)
        px = Q_2d_split[ci, :, 0]
        py = Q_2d_split[ci, :, 1]

        for c in range(1, 10):
            mask_c = (lbl_0 == c + 1)  # labels use 2-10 for digits 1-9
            if not mask_c.any():
                continue
            color = cmap_9(c)
            # Blank cells: circle
            blank_c = mask_c & ~given_mask
            if blank_c.any():
                ax.scatter(px[blank_c], py[blank_c], c=[color], marker='o',
                           s=50, alpha=0.7, zorder=3)
            # Given cells: star
            given_c = mask_c & given_mask
            if given_c.any():
                ax.scatter(px[given_c], py[given_c], c=[color], marker='*',
                           s=180, edgecolors='black', linewidth=0.5, zorder=5)
        ax.grid(True, alpha=0.3)

    for i in range(len(q_checkpoints), len(axes)):
        axes[i].axis('off')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for c in range(1, 10):
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=cmap_9(c), markersize=8,
                                       label=str(c)))
    legend_elements.append(Line2D([0], [0], marker='*', color='w',
                                   markerfacecolor='gray', markersize=12,
                                   markeredgecolor='black', label='Given'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=10, fontsize=9)

    plt.suptitle("Class-Conditioned Latent Space (Colors = Ground Truth Digits 1-9)", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig("visualizations/05_class_conditioned_collapse.png", dpi=200)
    plt.close()

    # ── NC1 / Silhouette 정량 메트릭 (루프별) ──
    print("[5b/8] NC1 & Silhouette Score across loops ...")

    def compute_nc1(features, labels):
        """NC1 = Tr(Sigma_W) / Tr(Sigma_B)
        Within-class / between-class variance ratio (scalar form).
        Lower = better clustering. Numerically stable even when N < d.
        """
        classes = np.unique(labels)
        global_mean = features.mean(axis=0)
        N = features.shape[0]

        sw = 0.0  # sum of within-class squared distances
        sb = 0.0  # sum of between-class squared distances

        for c in classes:
            mask_c = (labels == c)
            X_c = features[mask_c]
            n_c = X_c.shape[0]
            if n_c == 0:
                continue
            mu_c = X_c.mean(axis=0)
            sw += np.sum((X_c - mu_c) ** 2)
            sb += n_c * np.sum((mu_c - global_mean) ** 2)

        sw /= N
        sb /= N

        if sb < 1e-12:
            return float('inf')
        return float(sw / sb)

    nc1_history = []
    silhouette_history = []

    for c_idx in range(n_q):
        Q_snap = global_Q[c_idx][0].numpy()  # [81, d], sample 0
        labs = lbl_0  # [81]

        # NC1
        nc1_val = compute_nc1(Q_snap, labs)
        nc1_history.append(nc1_val)

        # Silhouette (requires >= 2 classes and >= 2 samples per class to be meaningful)
        unique_labs = np.unique(labs)
        if len(unique_labs) >= 2:
            sil = silhouette_score(Q_snap, labs)
            silhouette_history.append(float(sil))
        else:
            silhouette_history.append(0.0)

    # Plot NC1 & Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(n_q), nc1_history, 'r-o', markersize=3, linewidth=2)
    ax1.set_xlabel("Q snapshot")
    ax1.set_ylabel("NC1 (Tr(Σ_W) / Tr(Σ_B))")
    ax1.set_title("NC1: Within/Between Class Variance Ratio\n(Lower = better digit clustering)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(n_q), silhouette_history, 'b-o', markersize=3, linewidth=2)
    ax2.set_xlabel("Q snapshot")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score\n(Higher = better digit clustering, max=1)")
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Hidden State Digit Clustering Metrics Across Loops", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("visualizations/05b_nc1_silhouette.png", dpi=200)
    plt.close()

    metrics["nc1"] = nc1_history
    metrics["silhouette"] = silhouette_history

    # ════════════════════════════════════════════════════════════════════
    # 6. Label-Sorted Attraction Matrix + Block Diagonal Ratio (BDR)
    # ════════════════════════════════════════════════════════════════════
    print("[6/8] Label-Sorted Attraction Matrix + BDR ...")

    # ── BDR 계산 함수 ──
    # BDR = (같은 숫자 블록 내부 W 합) / (전체 W 합)
    # Uniform baseline: 9 블록 × 9×9 / 81² = 1/9 ≈ 0.111
    def compute_bdr(W_matrix, labels):
        """Block Diagonal Ratio: 같은 레이블 블록 내부 질량 비율"""
        total = W_matrix.sum() + 1e-12
        in_block = 0.0
        for c in np.unique(labels):
            mask_c = (labels == c)
            in_block += W_matrix[np.ix_(mask_c, mask_c)].sum()
        return float(in_block / total)

    # Column Periodicity Ratio (CPR): 같은 열(위치 차이가 9의 배수) 내부 질량 비율
    # Uniform baseline: 각 셀에 같은 열 셀 8개 + 자기 자신 = 9/81 = 1/9 ≈ 0.111
    def compute_cpr(W_matrix):
        """Column Periodicity Ratio: 같은 열 셀 간 질량 비율"""
        N = W_matrix.shape[0]
        total = W_matrix.sum() + 1e-12
        col_mask = np.zeros((N, N), dtype=bool)
        for i in range(N):
            col_i = i % 9
            for j in range(N):
                if j % 9 == col_i:
                    col_mask[i, j] = True
        return float(W_matrix[col_mask].sum() / total)

    # Row Periodicity Ratio (RPR): 같은 행 셀 간 질량 비율
    def compute_rpr(W_matrix):
        """Row Periodicity Ratio: 같은 행 셀 간 질량 비율"""
        N = W_matrix.shape[0]
        total = W_matrix.sum() + 1e-12
        row_mask = np.zeros((N, N), dtype=bool)
        for i in range(N):
            row_i = i // 9
            for j in range(N):
                if j // 9 == row_i:
                    row_mask[i, j] = True
        return float(W_matrix[row_mask].sum() / total)

    # ── 루프별 BDR/CPR/RPR 추적 ──
    bdr_history = []
    cpr_history = []
    rpr_history = []

    for step_idx in range(n_loops_collected):
        W_step = W_per_head[step_idx][-1][0].numpy()  # [H, N, N]
        W_avg = W_step.mean(axis=0)  # [N, N]
        bdr_history.append(compute_bdr(W_avg, lbl_0))
        cpr_history.append(compute_cpr(W_avg))
        rpr_history.append(compute_rpr(W_avg))

    # ── BDR/CPR/RPR 추이 플롯 ──
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = range(1, n_loops_collected + 1)
    ax.plot(steps, bdr_history, 'r-o', markersize=3, linewidth=2, label='BDR (Same Digit)')
    ax.plot(steps, cpr_history, 'b-s', markersize=3, linewidth=2, label='CPR (Same Column)')
    ax.plot(steps, rpr_history, 'g-^', markersize=3, linewidth=2, label='RPR (Same Row)')
    ax.axhline(y=1/9, color='gray', linestyle='--', alpha=0.6, label=f'Uniform baseline (1/9={1/9:.3f})')
    ax.set_xlabel("Micro-step")
    ax.set_ylabel("Mass Ratio")
    ax.set_title("Block Diagonal Ratio (BDR) vs Column/Row Periodicity\n"
                 "BDR>1/9 = digit identity learned, CPR>1/9 = column structure, RPR>1/9 = row structure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(cpr_history), max(bdr_history), max(rpr_history)) * 1.15)
    plt.tight_layout()
    plt.savefig("visualizations/06_bdr_trend.png", dpi=200)
    plt.close()

    metrics["block_diagonal_ratio"] = bdr_history
    metrics["column_periodicity_ratio"] = cpr_history
    metrics["row_periodicity_ratio"] = rpr_history

    # ── 루프별 Label-Sorted 히트맵 (선택된 체크포인트) ──
    ls_checkpoints = [0, n_loops_collected // 4, n_loops_collected // 2, n_loops_collected - 1]
    ls_checkpoints = sorted(set([max(0, min(c, n_loops_collected - 1)) for c in ls_checkpoints]))

    sorted_indices = np.argsort(lbl_0)
    sorted_labels = lbl_0[sorted_indices]
    boundaries = np.where(sorted_labels[:-1] != sorted_labels[1:])[0] + 1

    n_ls = len(ls_checkpoints)
    fig, axes = plt.subplots(1, n_ls, figsize=(6 * n_ls, 5.5))
    if n_ls == 1:
        axes = [axes]

    for ci, step_idx in enumerate(ls_checkpoints):
        W_step = W_per_head[step_idx][-1][0].numpy().mean(axis=0)  # [N, N]
        sorted_W = W_step[sorted_indices][:, sorted_indices]
        bdr_val = bdr_history[step_idx]

        ax = axes[ci]
        sns.heatmap(sorted_W, ax=ax, cmap='viridis', cbar=(ci == n_ls - 1), square=True)
        for b in boundaries:
            ax.axhline(b, color='white', linewidth=0.7, alpha=0.5)
            ax.axvline(b, color='white', linewidth=0.7, alpha=0.5)
        ax.set_title(f"Step {step_idx+1}\nBDR={bdr_val:.4f}", fontsize=10)

        # 블록 레이블
        prev = 0
        for b in list(boundaries) + [len(sorted_labels)]:
            mid = (prev + b) / 2
            digit = sorted_labels[prev] - 1 if sorted_labels[prev] >= 2 else sorted_labels[prev]
            ax.text(mid, -1.5, str(digit), ha='center', fontsize=7, color='red')
            prev = b

    plt.suptitle("Label-Sorted Attraction Matrix Across Loops\n"
                 "(9×9 bright blocks on diagonal = digit identity learned)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig("visualizations/06_label_sorted_attraction.png", dpi=200)
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # 7. Spatial 2D Receptive Field (헤드별 9x9 그리드 인력)
    # ════════════════════════════════════════════════════════════════════
    print("[7/8] Head-wise Spatial 2D Receptive Field ...")

    # 빈칸 중 하나 선택
    target_q_idx = blank_indices[len(blank_indices) // 2]
    target_q_r, target_q_c = target_q_idx // 9, target_q_idx % 9
    target_q_digit = lbl_0[target_q_idx] - 1 if lbl_0[target_q_idx] >= 2 else '?'

    # 루프 체크포인트 여러 개에서 헤드 평균 W의 변화
    rf_checkpoints = [0, n_loops_collected // 4, n_loops_collected // 2, n_loops_collected - 1]
    rf_checkpoints = sorted(set([max(0, min(c, n_loops_collected - 1)) for c in rf_checkpoints]))

    fig, axes = plt.subplots(len(rf_checkpoints), H + 1, figsize=(3 * (H + 1), 3.5 * len(rf_checkpoints)))
    if len(rf_checkpoints) == 1:
        axes = axes[np.newaxis, :]

    for ri, step_idx in enumerate(rf_checkpoints):
        W_step = W_per_head[step_idx][-1][0].numpy()  # [H, N, N]

        # 헤드 평균 (첫 열)
        W_avg = W_step.mean(axis=0)  # [N, N]
        att_avg = W_avg[target_q_idx].reshape(9, 9)
        ax = axes[ri, 0]
        sns.heatmap(att_avg, ax=ax, cmap='magma', cbar=False, square=True,
                    linewidths=0.5, linecolor='gray')
        # 3x3 경계
        for b in [3, 6]:
            ax.axhline(b, color='cyan', linewidth=1.5)
            ax.axvline(b, color='cyan', linewidth=1.5)
        # 쿼리 셀 표시
        ax.add_patch(plt.Rectangle((target_q_c, target_q_r), 1, 1,
                     fill=False, edgecolor='lime', linewidth=2.5))
        ax.set_title(f"Step {step_idx+1}\nAvg" if ri == 0 else f"Step {step_idx+1}\nAvg", fontsize=9)
        if ri == 0:
            ax.set_title(f"Avg\nStep {step_idx+1}", fontsize=9)

        # 각 헤드
        for h in range(H):
            att_h = W_step[h, target_q_idx].reshape(9, 9)
            ax = axes[ri, h + 1]
            sns.heatmap(att_h, ax=ax, cmap='magma', cbar=False, square=True,
                        linewidths=0.3, linecolor='gray')
            for b in [3, 6]:
                ax.axhline(b, color='cyan', linewidth=1)
                ax.axvline(b, color='cyan', linewidth=1)
            ax.add_patch(plt.Rectangle((target_q_c, target_q_r), 1, 1,
                         fill=False, edgecolor='lime', linewidth=2))
            if ri == 0:
                ax.set_title(f"H{h}\nStep {step_idx+1}", fontsize=9)
            else:
                ax.set_title(f"Step {step_idx+1}", fontsize=9)

            # 힌트 셀 표시 (헤드별 첫 행에서만)
            if ri == len(rf_checkpoints) - 1:
                for idx in range(81):
                    ir, ic = idx // 9, idx % 9
                    if given_mask[idx]:
                        digit = lbl_0[idx] - 1 if lbl_0[idx] >= 2 else '?'
                        ax.text(ic + 0.5, ir + 0.5, str(digit),
                                ha='center', va='center', color='white',
                                fontsize=6, fontweight='bold', alpha=0.7)

    fig.suptitle(
        f"Spatial 2D Receptive Field — Query=({target_q_r},{target_q_c}), "
        f"Answer={target_q_digit}\n"
        f"Rows=Loop steps, Cols=Avg+Heads. Green box=Query cell, Cyan=Box boundaries",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("visualizations/07_spatial_receptive_field.png", dpi=200)
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # 8. Particle Trajectory (입자 궤적 — 중력 이동 직접 시각화)
    # ════════════════════════════════════════════════════════════════════
    print("[8/10] Particle Trajectory in 2D ...")

    # PCA를 마지막 Q에 fit → 모든 snapshot에 동일 projection 적용
    from sklearn.decomposition import PCA as PCA2
    pca_traj = PCA2(n_components=2)
    Q_final = global_Q[-1][0].numpy()  # [81, d]
    pca_traj.fit(Q_final)

    # 모든 snapshot 투영
    Q_projected = []  # list of [81, 2]
    for c_idx in range(n_q):
        Q_projected.append(pca_traj.transform(global_Q[c_idx][0].numpy()))

    # ── 8a: 전체 궤적 (선택된 셀들) ──
    # 빈칸 중 다양한 위치에서 셀 선택 (최대 12개)
    selected_blanks = blank_indices[::max(1, len(blank_indices) // 6)][:6]
    given_indices = np.where(given_mask)[0]
    selected_given = given_indices[::max(1, len(given_indices) // 6)][:6]
    selected_cells = np.concatenate([selected_blanks, selected_given])

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 1: 궤적 + 화살표 (선택된 셀)
    ax = axes[0]
    cmap_digit = plt.get_cmap('tab10')

    for cell_idx in selected_cells:
        traj = np.array([Q_projected[t][cell_idx] for t in range(n_q)])  # [n_q, 2]
        digit = lbl_0[cell_idx] - 1 if lbl_0[cell_idx] >= 2 else 0
        color = cmap_digit(digit)
        is_blank = not given_mask[cell_idx]
        ls = '-' if is_blank else '--'
        lw = 1.5 if is_blank else 0.8

        # 궤적 선
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=lw, linestyle=ls, alpha=0.6)
        # 시작점 (삼각형)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker='^', s=40, zorder=5, alpha=0.8)
        # 끝점 (원 or 별)
        marker = 'o' if is_blank else '*'
        ms = 60 if is_blank else 120
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker=marker, s=ms,
                   edgecolors='black', linewidth=0.5, zorder=6)
        # 이동 화살표 (마지막 3 스텝)
        for t in range(max(0, n_q - 4), n_q - 1):
            dx = traj[t+1, 0] - traj[t, 0]
            dy = traj[t+1, 1] - traj[t, 1]
            ax.annotate('', xy=(traj[t+1, 0], traj[t+1, 1]),
                        xytext=(traj[t, 0], traj[t, 1]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2, alpha=0.7))

    ax.set_title("Selected Cell Trajectories\n(△=start, ●=blank end, ★=given end)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: 전체 81셀의 시작→끝 이동 (화살표)
    ax = axes[1]
    for cell_idx in range(N_seq):
        start = Q_projected[0][cell_idx]
        end = Q_projected[-1][cell_idx]
        digit = lbl_0[cell_idx] - 1 if lbl_0[cell_idx] >= 2 else 0
        color = cmap_digit(digit)
        dx, dy = end[0] - start[0], end[1] - start[1]
        ax.annotate('', xy=(end[0], end[1]), xytext=(start[0], start[1]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8, alpha=0.5))
        # 끝점만 표시
        marker = 'o' if not given_mask[cell_idx] else '*'
        ms = 20 if not given_mask[cell_idx] else 50
        ax.scatter(end[0], end[1], color=color, marker=marker, s=ms, alpha=0.7, zorder=5)

    ax.set_title("All 81 Cells: Start → End Displacement\n(arrows show total movement)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: 스냅샷 오버레이 (루프 진행에 따른 분포 변화)
    ax = axes[2]
    snap_indices = [0, n_q // 4, n_q // 2, n_q - 1]
    snap_indices = sorted(set([min(s, n_q - 1) for s in snap_indices]))
    alphas = np.linspace(0.15, 1.0, len(snap_indices))
    sizes = np.linspace(10, 50, len(snap_indices))

    for si, snap_idx in enumerate(snap_indices):
        pts = Q_projected[snap_idx]
        for cell_idx in range(N_seq):
            digit = lbl_0[cell_idx] - 1 if lbl_0[cell_idx] >= 2 else 0
            color = cmap_digit(digit)
            ax.scatter(pts[cell_idx, 0], pts[cell_idx, 1],
                       color=color, s=sizes[si], alpha=alphas[si], zorder=si + 1)
        # 테두리로 스냅샷 구분
        ax.scatter([], [], color='gray', s=sizes[si], alpha=alphas[si],
                   label=f"Snap {snap_idx}")

    ax.set_title("Snapshot Overlay (faint=early, bold=final)\n(colors=digit 1-9)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 공통 범례
    from matplotlib.lines import Line2D as Line2D2
    legend_els = []
    for c in range(1, 10):
        legend_els.append(Line2D2([0], [0], marker='o', color='w',
                                   markerfacecolor=cmap_digit(c), markersize=8, label=str(c)))
    fig.legend(handles=legend_els, loc='lower center', ncol=9, fontsize=9)

    plt.suptitle("Particle Dynamics: Cell Trajectories in Latent Space Across Loops", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig("visualizations/08_particle_trajectories.png", dpi=200)
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # 9. Force Convergence (||m|| 감소 = 평형 도달 증거)
    # ════════════════════════════════════════════════════════════════════
    print("[9/10] Force Convergence ||m|| ...")

    # m_per_head[step][block] = [B, H, N, d_h]
    # 각 micro-step에서의 평균 ||m|| 계산
    force_norm_history = []       # 전체 평균
    force_norm_blank = []         # 빈칸만
    force_norm_given = []         # 주어진 셀만

    for step_idx in range(n_loops_collected):
        m_step = m_per_head[step_idx][-1][0].numpy()  # [H, N, d_h], sample 0
        # 헤드 평균 후 셀별 norm
        m_avg = m_step.mean(axis=0)  # [N, d_h]
        norms = np.linalg.norm(m_avg, axis=-1)  # [N]

        force_norm_history.append(float(norms.mean()))
        force_norm_blank.append(float(norms[~given_mask].mean()))
        force_norm_given.append(float(norms[given_mask].mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps_x = range(1, n_loops_collected + 1)
    ax1.plot(steps_x, force_norm_history, 'k-o', markersize=3, linewidth=2, label='All cells')
    ax1.plot(steps_x, force_norm_blank, 'r-s', markersize=3, linewidth=1.5, label='Blank cells')
    ax1.plot(steps_x, force_norm_given, 'b-^', markersize=3, linewidth=1.5, label='Given cells')
    ax1.set_xlabel("Micro-step")
    ax1.set_ylabel("Mean ||m|| (force magnitude)")
    ax1.set_title("Mean-Shift Force Magnitude Across Loops\n(Decreasing = approaching equilibrium)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 이동 거리 (Q snapshot 간 L2 거리)
    displacement_history = []
    displacement_blank = []
    displacement_given = []

    for t in range(1, n_q):
        Q_prev = global_Q[t-1][0].numpy()  # [81, d]
        Q_curr = global_Q[t][0].numpy()
        disp = np.linalg.norm(Q_curr - Q_prev, axis=-1)  # [81]
        displacement_history.append(float(disp.mean()))
        displacement_blank.append(float(disp[~given_mask].mean()))
        displacement_given.append(float(disp[given_mask].mean()))

    ax2.plot(range(1, n_q), displacement_history, 'k-o', markersize=3, linewidth=2, label='All cells')
    ax2.plot(range(1, n_q), displacement_blank, 'r-s', markersize=3, linewidth=1.5, label='Blank cells')
    ax2.plot(range(1, n_q), displacement_given, 'b-^', markersize=3, linewidth=1.5, label='Given cells')
    ax2.set_xlabel("Q snapshot")
    ax2.set_ylabel("Mean ||ΔQ|| (displacement per step)")
    ax2.set_title("Cell Displacement Per Loop Step\n(Decreasing = convergence to fixed point)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Dynamical System Convergence: Force & Displacement", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("visualizations/09_force_convergence.png", dpi=200)
    plt.close()

    metrics["force_norm_all"] = force_norm_history
    metrics["force_norm_blank"] = force_norm_blank
    metrics["force_norm_given"] = force_norm_given
    metrics["displacement_all"] = displacement_history
    metrics["displacement_blank"] = displacement_blank
    metrics["displacement_given"] = displacement_given

    # ════════════════════════════════════════════════════════════════════
    # 10. Unsupervised Clustering Emergence (라벨 없이 뭉침 검증)
    # ════════════════════════════════════════════════════════════════════
    print("[10/10] Unsupervised Clustering Emergence ...")

    from sklearn.neighbors import NearestNeighbors

    def hopkins_statistic(X, n_samples=None):
        """Hopkins statistic: 0.5 = uniform random, →1.0 = clustered."""
        n, d = X.shape
        if n_samples is None:
            n_samples = min(n // 2, 30)
        if n_samples < 2:
            return 0.5

        rng = np.random.RandomState(42)

        # Sample n_samples points from X
        sample_idx = rng.choice(n, n_samples, replace=False)
        X_sample = X[sample_idx]

        # Generate n_samples uniform random points in the data bounding box
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_random = rng.uniform(X_min, X_max, (n_samples, d))

        # Nearest neighbor distances
        nn = NearestNeighbors(n_neighbors=2).fit(X)

        # For real samples: distance to nearest OTHER point in X
        u_dist = nn.kneighbors(X_sample, n_neighbors=2)[0][:, 1]  # skip self
        # For random points: distance to nearest point in X
        w_dist = nn.kneighbors(X_random, n_neighbors=1)[0][:, 0]

        u_sum = u_dist.sum()
        w_sum = w_dist.sum()

        return float(w_sum / (u_sum + w_sum + 1e-12))

    def pairwise_distance_cv(X):
        """Coefficient of variation of pairwise distances.
        Higher = more structure (some close + some far)."""
        from scipy.spatial.distance import pdist
        dists = pdist(X)
        if len(dists) == 0 or dists.std() < 1e-12:
            return 0.0
        return float(dists.std() / (dists.mean() + 1e-12))

    def constraint_vs_nonconstraint_distance(X, mask_given):
        """제약 이웃 vs 비이웃 간 평균 거리 비율.
        < 1.0 이면 제약 이웃끼리 더 가까움."""
        from scipy.spatial.distance import cdist
        D = cdist(X, X)  # [81, 81]
        constraint_dists = []
        non_constraint_dists = []
        for i in range(81):
            neighbors = set(get_row_indices(i)) | set(get_col_indices(i)) | set(get_box_indices(i))
            non_neighbors = set(range(81)) - neighbors - {i}
            for j in neighbors:
                constraint_dists.append(D[i, j])
            for j in non_neighbors:
                non_constraint_dists.append(D[i, j])
        c_mean = np.mean(constraint_dists) if constraint_dists else 1.0
        nc_mean = np.mean(non_constraint_dists) if non_constraint_dists else 1.0
        return float(c_mean / (nc_mean + 1e-12))

    hopkins_history = []
    cv_history = []
    constraint_ratio_history = []

    for c_idx in range(n_q):
        Q_snap = global_Q[c_idx][0].numpy()
        hopkins_history.append(hopkins_statistic(Q_snap))
        cv_history.append(pairwise_distance_cv(Q_snap))
        constraint_ratio_history.append(
            constraint_vs_nonconstraint_distance(Q_snap, given_mask))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.plot(range(n_q), hopkins_history, 'g-o', markersize=3, linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Random (no structure)')
    ax1.set_xlabel("Q snapshot")
    ax1.set_ylabel("Hopkins Statistic")
    ax1.set_title("Hopkins Statistic\n(0.5=random, →1.0=clusters exist)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(n_q), cv_history, 'm-o', markersize=3, linewidth=2)
    ax2.set_xlabel("Q snapshot")
    ax2.set_ylabel("Pairwise Distance CV (σ/μ)")
    ax2.set_title("Pairwise Distance Variation\n(Higher = more spatial structure)")
    ax2.grid(True, alpha=0.3)

    ax3.plot(range(n_q), constraint_ratio_history, 'c-o', markersize=3, linewidth=2)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.6, label='No preference')
    ax3.set_xlabel("Q snapshot")
    ax3.set_ylabel("Constraint / Non-constraint Distance Ratio")
    ax3.set_title("Constraint Neighbor Proximity\n(<1.0 = constraint neighbors closer)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Unsupervised Structure Emergence Across Loops", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("visualizations/10_clustering_emergence.png", dpi=200)
    plt.close()

    metrics["hopkins"] = hopkins_history
    metrics["pairwise_distance_cv"] = cv_history
    metrics["constraint_neighbor_ratio"] = constraint_ratio_history

    # ── Save metrics ──
    with open("visualizations/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone! Results saved to visualizations/")
    print(f"  01_head_topology_specialization.png  - Head-wise attraction heatmaps")
    print(f"  01_head_topology_bar.png             - In-Topology Mass Ratio bar chart")
    print(f"  02_angular_trajectory.png            - Cosine sim + angular velocity")
    print(f"  03_kernel_sparsity.png               - Gini & Tsallis per head")
    print(f"  04_head_orthogonality_heatmaps.png   - H x H cosine sim matrices")
    print(f"  04_head_orthogonality_trend.png      - Off-diagonal trend")
    print(f"  05_class_conditioned_collapse.png    - Neural Collapse trajectory")
    print(f"  05b_nc1_silhouette.png               - NC1 & Silhouette digit clustering")
    print(f"  06_label_sorted_attraction.png       - Block-diagonal check")
    print(f"  06_bdr_trend.png                     - BDR/CPR/RPR trend")
    print(f"  07_spatial_receptive_field.png        - 9x9 receptive field per head")
    print(f"  08_particle_trajectories.png         - Cell trajectories in latent 2D")
    print(f"  09_force_convergence.png             - ||m|| and ||ΔQ|| convergence")
    print(f"  10_clustering_emergence.png          - Hopkins, CV, constraint proximity")


if __name__ == "__main__":
    main()
