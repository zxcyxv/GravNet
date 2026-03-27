import os
import argparse
import json
import time
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch.nn.functional as F

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from dataset import create_dataloaders
from amkpd_model import AMKPDModel, AMK_Block, AMKPDCarry
from train import build_optimizer_and_scheduler, compute_loss

def calculate_effective_rank(Q):
    ranks = []
    for b in range(Q.shape[0]):
        # Singular Value Decomposition on (N, d)
        U, S, V = torch.linalg.svd(Q[b].float(), full_matrices=False)
        p = S / (S.sum() + 1e-9)
        entropy = -torch.sum(p * torch.log(p + 1e-9))
        ranks.append(torch.exp(entropy).item())
    return np.mean(ranks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_spectral", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=None, help="저장된 모델 가중치 (예: checkpoints/best.pt)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. DataLoader
    train_loader, test_loader, meta = create_dataloaders(args.data_dir, args.batch_size, 0, device.type == "cuda")
    
    # 2. Checkpoint Pre-loading for Model Architecture Specs
    ckpt_args = {}
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        ckpt_args = ckpt.get("args", {})
        if ckpt_args:
            print(f"Loaded config from checkpoint: d_model={ckpt_args.get('d_model')}, d_spectral={ckpt_args.get('d_spectral')}")
            args.d_model = ckpt_args.get("d_model", args.d_model)
            args.d_spectral = ckpt_args.get("d_spectral", args.d_spectral)

    model = AMKPDModel(
        vocab_size=meta["vocab_size"],
        d_model=args.d_model,
        d_spectral=args.d_spectral,
        num_layers=ckpt_args.get("num_layers", 3),
        loops=ckpt_args.get("loops", 6),
        H_cycles=ckpt_args.get("H_cycles", 2),
        L_cycles=ckpt_args.get("L_cycles", 1),
        kernel_power=ckpt_args.get("kernel_power", 2),
        use_w_v=ckpt_args.get("use_w_v", False),
    ).to(device)

    # Fake train args
    class TrainArgs:
        weight_decay = 0.1
        lr = 3e-4
        warmup_steps = 100
        halt_loss_coef = 0.01
        grad_accum = 1
        max_grad_norm = 1.0
        loops = ckpt_args.get("loops", 6)

    t_args = TrainArgs()
    optimizer, scheduler = build_optimizer_and_scheduler(model, t_args, args.train_steps)
    
    # 3. Checkpoint Restoring or Pre-training
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        model.load_state_dict(ckpt["model"])
        print("Checkpoint weights loaded successfully. Skipping pre-training.")
    else:
        if args.checkpoint is not None:
            print(f"Warning: Checkpoint {args.checkpoint} not found. Proceeding with pre-training.")
            
        model.train()
        step = 0
        seq_len = meta["seq_len"]
        carry = model.initial_carry(args.batch_size, seq_len, device)
        pbar = tqdm(total=args.train_steps, desc="Pre-Training for Viz")
        while step < args.train_steps:
            for inputs, labels in train_loader:
                if step >= args.train_steps: break
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.shape[0] != carry.current_hidden.shape[0]:
                    carry = model.initial_carry(inputs.shape[0], seq_len, device)
                batch = (inputs, labels)
                optimizer.zero_grad()
                # Deep Supervision: 매 외부 루프마다 loss + backward 누적
                for _t in range(t_args.loops):
                    carry, logits, halt_logits = model(carry, batch)
                    loss, _ = compute_loss(logits, halt_logits, carry.current_labels, t_args)
                    (loss / t_args.loops).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step += 1
                pbar.update(1)
        pbar.close()

    # 4. Extracting Features (Native Telemetry via multi-forward carry loop)
    model.eval()
    print("Extracting features (1 batch, up to 64 samples) ...")
    inputs, labels = next(iter(test_loader))
    inputs = inputs[:min(64, inputs.shape[0])].to(device)
    labels = labels[:inputs.shape[0]].to(device)
    B_viz = inputs.shape[0]
    seq_len = inputs.shape[1]

    # 텔레메트리 켜기
    model.log_viz = True
    for b in model.blocks:
        b.log_viz = True

    # carry 기반 다중 forward로 텔레메트리 수집
    carry = model.initial_carry(B_viz, seq_len, device)
    batch = (inputs, labels)

    # 외부 루프별 텔레메트리 수집
    global_Q = []
    global_H = []
    global_W = []
    global_m = []
    K_blocks = len(model.blocks)
    M_loops = model.loops

    with torch.no_grad():
        # 초기 상태 수집
        global_Q.append(carry.current_hidden.detach().cpu())

        for outer in range(M_loops):
            # 각 forward 호출 전 블록 텔레메트리 리셋
            for b in model.blocks:
                b.viz_W = []
                b.viz_m = []
                b.viz_H = []
            model.viz_Q = []
            model.viz_H_global = []

            carry, logits, halt_logits = model(carry, batch)

            # forward 내부에서 수집된 텔레메트리 추출
            # viz_Q: L_cycles + 1개 (gradient H_cycle의 각 L_cycle 경계)
            for q in model.viz_Q:
                if isinstance(q, torch.Tensor):
                    global_Q.append(q.detach().cpu())
            for h in model.viz_H_global:
                if isinstance(h, torch.Tensor):
                    global_H.append(h.detach().cpu())

            # 블록별 W, m 수집 (gradient H_cycle에서만)
            n_micro = len(model.blocks[0].viz_W)
            for mi in range(n_micro):
                loop_m_tensors = []
                for k_idx in range(K_blocks):
                    global_W.append(model.blocks[k_idx].viz_W[mi].detach().cpu())
                    loop_m_tensors.append(model.blocks[k_idx].viz_m[mi].detach().cpu())
                global_m.append(loop_m_tensors)

    # 동역학적인 Jacobian Norm (Hutchinson Estimator)
    print("Calculating Dynamical Jacobian Norms via Hutchinson Estimator...")
    global_J_norm = []

    X_input = model.embedding(inputs[0:1]) + model.pos_emb(torch.arange(seq_len, device=inputs.device).unsqueeze(0))
    X_input = model.input_norm(X_input)

    n_jacobian_steps = len(global_Q) - 1
    for l in range(n_jacobian_steps):
        Q_l = global_Q[l][0:1].to(device).detach().clone().requires_grad_(True)
        # X 주입 후 첫 번째 블록 야코비안 계산 (URM L_cycle 패턴)
        Q_injected = Q_l + X_input
        model.blocks[0].viz_m = []
        with torch.enable_grad():
            _ = model.blocks[0](Q_injected)
            m_vec = model.blocks[0].viz_m[-1]

            norm_sq_sum = 0
            for _ in range(5):
                v = torch.randn_like(m_vec)
                vjp = torch.autograd.grad(m_vec, Q_l, v, retain_graph=True)[0]
                norm_sq_sum += (vjp ** 2).sum().item()
            global_J_norm.append(math.sqrt(norm_sq_sum / 5.0))

    # ── B. Maximal Lyapunov Exponent (MLE) 근사 측정 ──
    print("Calculating Maximal Lyapunov Exponent (MLE) ...")

    epsilon = 1e-5
    perturbation = torch.randn_like(X_input)
    perturbation = perturbation / torch.norm(perturbation, dim=-1, keepdim=True) * epsilon
    X_perturbed = X_input + perturbation

    Q_orig = X_input.clone()
    Q_pert = X_perturbed.clone()

    lyapunov_exponents = []
    log_sum = 0.0

    with torch.no_grad():
        for l in range(n_jacobian_steps):
            # L_cycle: X 주입 + 블록 통과
            Q_orig = Q_orig + X_input
            Q_pert = Q_pert + X_perturbed
            for block in model.blocks:
                Q_orig = block(Q_orig)
                Q_pert = block(Q_pert)

            delta_Q = torch.norm(Q_pert - Q_orig, dim=-1).mean().item()

            log_sum += math.log(delta_Q / epsilon)
            mle = log_sum / (l + 1)
            lyapunov_exponents.append(mle)

            Q_pert = Q_orig + (Q_pert - Q_orig) / (delta_Q + 1e-12) * epsilon

    # 텔레메트리 끄기
    model.log_viz = False
    for b in model.blocks:
        b.log_viz = False

    # M_loops 재정의: 실제 수집된 외부 루프 수 (Jacobian/MLE 스텝)
    M_loops = n_jacobian_steps

    os.makedirs("visualizations", exist_ok=True)
    metrics_log = {}
    
    # ── 1. Shannon Entropy & Heatmap ───────────────────────────────────────────
    print("Plotting 1. Attraction Map Shannon Entropy ...")
    entropies = []
    for W_tensor in global_W:
        P = torch.abs(W_tensor)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-9)
        H = -torch.sum(P * torch.log(P + 1e-9), dim=-1).mean().item()
        entropies.append(H)
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(entropies)+1), entropies, marker='o', color='purple')
    plt.title("Shannon Entropy of Normalized Attraction Map over Micro-Layers")
    plt.xlabel("Micro-Layer (l)")
    plt.ylabel("Mean Entropy H(W)")
    plt.axhline(y=np.log(81), color='r', linestyle='--', label="Log(N)=4.39 (Uniform Flatness)")
    plt.legend()
    plt.grid(True)
    plt.savefig("visualizations/01_Entropy_Line.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(global_W[-1][0].numpy(), cmap='coolwarm', center=0)
    plt.title(f"Attraction Map Heatmap W (Sample 1, Last Micro-Layer)\nEntropy: {entropies[-1]:.3f}")
    plt.savefig("visualizations/01_Entropy_Heatmap.png")
    plt.close()

    # Calculate and store heatmap metrics
    metrics_log["heatmap_entropies"] = entropies
    last_W = global_W[-1][0].numpy()
    max_probs = np.max(np.abs(last_W) / (np.abs(last_W).sum(axis=-1, keepdims=True) + 1e-9), axis=-1)
    metrics_log["heatmap_mean_max_prob"] = float(np.mean(max_probs))
    
    def gini(array):
        array = array.flatten()
        if np.amin(array) < 0: array -= np.amin(array)
        array += 1e-9
        array = np.sort(array)
        index = np.arange(1, array.shape[0]+1)
        n = array.shape[0]
        return float((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
    
    metrics_log["heatmap_gini_coefficient"] = gini(last_W)

    # ── 2. Topological Trajectory (UMAP/PCA) ──────────────────────────────────
    print("Plotting 2. Topological Trajectory ...")
    loops = len(global_Q)
    Q_cat = torch.stack(global_Q, dim=0) # [Loops, B, N, d]
    
    B, N_seq, d = inputs.shape[0], inputs.shape[1], args.d_model
    Q_traj = Q_cat[:, 0, :, :].numpy() # [Loops, 81, d]
    
    inp_0 = inputs[0].cpu().numpy()
    given_mask = (inp_0 > 1)  # 1 is blank, 2-10 are 1-9
    
    Q_flat = Q_traj.reshape(loops * N_seq, d)
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        print(" Using UMAP...")
    else:
        reducer = PCA(n_components=2)
        print(" Using PCA...")
        
    Q_proj_flat = reducer.fit_transform(Q_flat)
    Q_proj = Q_proj_flat.reshape((loops, N_seq, 2))
    
    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(1, loops - 1)) for i in range(loops)]
    
    # Plot Trajectories
    for n in range(N_seq):
        traj_x = Q_proj[:, n, 0]
        traj_y = Q_proj[:, n, 1]
        
        is_given = given_mask[n]
        color_start = 'mediumblue' if is_given else 'tomato'
        marker = 's' if is_given else 'o'
        
        # 완전 겹치는 0번 토큰들을 분리해 눈으로 확인하기 위한 위치 흩뿌림(Jitter) 추가
        jx, jy = (np.random.normal(0, 0.4), np.random.normal(0, 0.4)) if not is_given else (0, 0)
        
        plt.plot(traj_x + jx, traj_y + jy, color='gray', alpha=0.3, linewidth=1, zorder=1)
        # Start
        plt.scatter(traj_x[0] + jx, traj_y[0] + jy, c=color_start, marker=marker, s=50, alpha=0.8, zorder=2)
        # End
        plt.scatter(traj_x[-1] + jx, traj_y[-1] + jy, c='black' if is_given else 'green', marker=marker, s=80, edgecolors='w', zorder=3)
        
    # Legend hacking
    plt.scatter([], [], c='mediumblue', marker='s', label='Given (Start)')
    plt.scatter([], [], c='black', marker='s', label='Given (End)')
    plt.scatter([], [], c='tomato', marker='o', label='Empty (Start)')
    plt.scatter([], [], c='green', marker='o', label='Empty (End)')
    
    plt.title("Topological Trajectory Tracking Across Macro-Loops")
    plt.legend()
    plt.grid(True)
    plt.savefig("visualizations/02_Trajectory.png")
    plt.close()

    # Calculate and store trajectory metrics
    dist_moved_given = np.linalg.norm(Q_traj[-1, given_mask] - Q_traj[0, given_mask], axis=-1).mean().item()
    dist_moved_blank = np.linalg.norm(Q_traj[-1, ~given_mask] - Q_traj[0, ~given_mask], axis=-1).mean().item()
    metrics_log["trajectory_dist_moved_given"] = float(dist_moved_given)
    metrics_log["trajectory_dist_moved_blank"] = float(dist_moved_blank)

    if len(np.unique(given_mask)) > 1:
        sil_score = silhouette_score(Q_traj[-1], given_mask)
    else:
        sil_score = 0.0
    metrics_log["trajectory_silhouette_score"] = float(sil_score)

    # ── 3. Effective Dimensionality (Rank Collapse) ───────────────────────────
    print("Plotting 3. Effective Dimensionality ...")
    eff_ranks = []
    for l in range(loops):
        Q_l = global_Q[l] # [B, N, d]
        er = calculate_effective_rank(Q_l)
        eff_ranks.append(er)
        
    plt.figure(figsize=(10, 4))
    plt.plot(range(loops), eff_ranks, marker='s', color='green', linewidth=2)
    plt.title("Effective Dimensionality (Rank Collapse Detection)")
    plt.xlabel("Macro-Loop (0=Input Layer)")
    plt.ylabel("Effective Rank")
    plt.axhline(y=args.d_model, color='r', linestyle=':', label=f"Max Possible ({args.d_model})")
    plt.axhline(y=args.d_model / 4, color='orange', linestyle='--', label=f"Safe Threshold (d/4 = {args.d_model/4})")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, args.d_model * 1.1)
    plt.savefig("visualizations/03_Effective_Rank.png")
    plt.close()

    metrics_log["effective_ranks"] = [float(er) for er in eff_ranks]
    # ── 3.5. Step-by-Step Dynamics & Internal Parameters (추가된 코드) ──
    print("Calculating Step-by-Step Dynamics & Internal Parameters ...")
    
    # 1) 각 거시적 루프(Macro-Loop) 간의 이동 거리 (Delta Q)
    step_moved_given = []
    step_moved_blank = []
    for l in range(1, loops):
        # Q_cat shape: [Loops, B, N, d]
        delta_Q = np.linalg.norm(Q_cat[l, 0].numpy() - Q_cat[l-1, 0].numpy(), axis=-1) # [81]
        step_moved_given.append(float(delta_Q[given_mask].mean()))
        step_moved_blank.append(float(delta_Q[~given_mask].mean()))
        
    metrics_log["dynamics_step_movement_given"] = step_moved_given
    metrics_log["dynamics_step_movement_blank"] = step_moved_blank

    # 2) 모델이 스스로 학습한 블록별 스텝 사이즈 (dt_safe)
    learned_dts = []
    for k_idx, block in enumerate(model.blocks):
        dt_val = F.softplus(block.dt).item()
        learned_dts.append(float(dt_val))
    metrics_log["dynamics_learned_dt_per_block"] = learned_dts

    # 3) 각 루프/블록별 중력 벡터(m)의 실제 크기 (Net Force Magnitude)
    # m 벡터가 정말로 0에 수렴해서 입자가 멈춘 것인지 확인
    m_vector_norms = []
    for m_idx in range(M_loops):
        loop_m_norms = []
        for k_idx in range(K_blocks):
            # 오염되지 않은 global_m 리스트에서 직접 조회
            m_tensor = global_m[m_idx][k_idx] # [B, N, d]
            m_norm = torch.norm(m_tensor[0], dim=-1).mean().item()
            loop_m_norms.append(float(m_norm))
        m_vector_norms.append(loop_m_norms) # [M_loops, K_blocks]
        
    metrics_log["dynamics_mean_shift_net_force"] = m_vector_norms
    # ── 4. Advanced Manifold Evaluation Metrics ───────────────────────────────
    print("Calculating 4. Advanced Manifold Metrics ...")
    
    # Extract labels for the first sample
    lbl = labels[0].cpu().numpy()  # [81]
    
    mad_gaps = []
    target_silhouettes = []
    explained_vars_top1 = []
    explained_vars_top9 = []
    explained_vars_top27 = []
    
    # global_H가 있으면 H_context 공간 사용 (= 실제 KDE 작용 공간), 없으면 Q로 폴백
    use_H = len(global_H) > 0
    n_manifold_loops = len(global_H) if use_H else loops

    for l in range(n_manifold_loops):
        feat_l = global_H[l][0].numpy() if use_H else global_Q[l][0].numpy()  # [81, d]
        
        # A) Target-Conditioned Silhouette Score
        mask = lbl != 0
        if len(np.unique(lbl[mask])) > 1:
            try:
                sil = silhouette_score(feat_l[mask], lbl[mask])
            except Exception:
                sil = 0.0
        else:
            sil = 0.0
        target_silhouettes.append(float(sil))
        
        # B) Class-Conditioned MAD & MAD-Gap
        Q_v = feat_l[mask]
        l_v = lbl[mask]
        if len(Q_v) > 1:
            Q_norm_v = Q_v / (np.linalg.norm(Q_v, axis=1, keepdims=True) + 1e-9)
            dist_mat = 1.0 - (Q_norm_v @ Q_norm_v.T)
            
            target_mask = (l_v[:, None] == l_v[None, :])
            np.fill_diagonal(target_mask, False)
            remote_mask = (l_v[:, None] != l_v[None, :])
            
            mad_tgt = dist_mat[target_mask].mean() if target_mask.any() else 0.0
            mad_rem = dist_mat[remote_mask].mean() if remote_mask.any() else 0.0
            mad_gap = mad_rem - mad_tgt
        else:
            mad_gap = 0.0
            
        mad_gaps.append(float(mad_gap))

        # ── A. Neural Collapse Indices (NC1, NC2) ──
        if len(np.unique(l_v)) > 1:
            classes = np.unique(l_v)
            num_classes = len(classes)
            global_mean = np.mean(Q_v, axis=0)

            class_means = []
            within_class_scatter = 0.0
            for c in classes:
                Q_c = Q_v[l_v == c]
                mu_c = np.mean(Q_c, axis=0)
                class_means.append(mu_c)
                within_class_scatter += np.sum(np.linalg.norm(Q_c - mu_c, axis=1)**2)

            within_class_scatter /= len(Q_v)

            class_means = np.array(class_means)
            between_class_scatter = np.sum(np.linalg.norm(class_means - global_mean, axis=1)**2) / num_classes

            nc1 = within_class_scatter / (between_class_scatter + 1e-9)

            M_mat = (class_means - global_mean).T  # [d, num_classes]
            M_norm_sq = np.linalg.norm(M_mat, 'fro')**2 + 1e-9
            cos_mat = (M_mat.T @ M_mat) / M_norm_sq

            ideal_etf = (num_classes / (num_classes - 1)) * np.eye(num_classes) - (1 / (num_classes - 1)) * np.ones((num_classes, num_classes))
            nc2 = np.linalg.norm(cos_mat - ideal_etf, 'fro')
        else:
            nc1, nc2 = 0.0, 0.0

        metrics_log.setdefault("manifold_nc1_collapse", []).append(float(nc1))
        metrics_log.setdefault("manifold_nc2_etf", []).append(float(nc2))

        # C) Singular Spectrum Explained Variance
        U, S, V = np.linalg.svd(feat_l, full_matrices=False)
        S_sq = S**2
        tot_var = S_sq.sum() + 1e-9
        explained_vars_top1.append(float(S_sq[:1].sum() / tot_var))
        explained_vars_top9.append(float(S_sq[:9].sum() / tot_var))
        explained_vars_top27.append(float((S_sq[:27].sum() if len(S_sq)>=27 else S_sq.sum()) / tot_var))

    metrics_log["manifold_target_silhouette_scores"] = target_silhouettes
    metrics_log["manifold_mad_gaps"] = mad_gaps
    metrics_log["manifold_explained_variance_top1"] = explained_vars_top1
    metrics_log["manifold_explained_variance_top9"] = explained_vars_top9
    metrics_log["manifold_explained_variance_top27"] = explained_vars_top27
    metrics_log["dynamics_jacobian_norms"] = global_J_norm
    metrics_log["dynamics_maximal_lyapunov_exponent"] = lyapunov_exponents
    
    # ── 5. Anchor-Target Ratio (ATR) & Bipartite Trajectory ─────────────────
    print("Calculating and Plotting 5. Target-Conditioned Bipartite Trajectory & ATR ...")
    
    ATR_history = []
    cohered_blanks_history = []
    
    # ATR/응집 계산 대상: H_context 공간 (= 실제 중력 작용 공간)
    # 이분 구조 투영(2D):은 Q 공간 (= 눈에 보이는 입자 움직임) 사용
    H_arr = global_H if use_H else global_Q
    h_loops = len(H_arr)
    H_flat = np.stack([H_arr[l][0].numpy() for l in range(h_loops)], axis=0).reshape(h_loops * N_seq, -1)
    if HAS_UMAP:
        reducer_bipartite = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer_bipartite = PCA(n_components=2)
        
    H_proj_flat_bip = reducer_bipartite.fit_transform(H_flat)
    H_proj_bip = H_proj_flat_bip.reshape((h_loops, N_seq, 2))
    
    # Q는 이분구조 시각화(Bipartite plot)에서만 사용
    Q_proj_flat_bip = reducer_bipartite.fit_transform(Q_flat) if loops > 0 else H_proj_flat_bip
    Q_proj_bip = Q_proj_flat_bip.reshape((loops, N_seq, 2))
    
    # 클래스 1~9를 위한 색상 팔레트 (tab10의 1~9 인덱스 사용)
    palette = plt.get_cmap('tab10')
    
    for l in range(h_loops):
        H_l = H_arr[l][0].numpy() # [81, d] — 실제 중력 작용 공간
        
        # 1. 클래스별 힌트 셀(Given Cells)의 질량 중심(Centroid) 계산 (H_context 기준)
        centroids = {}
        for c in range(1, 10):
            c_given_mask = given_mask & (lbl == c)
            if c_given_mask.any():
                centroids[c] = H_l[c_given_mask].mean(axis=0)
                
        # 2D 시각화 공간(H_proj 기준)의 질량 중심
        centroids_2d = {}
        for c in range(1, 10):
            c_given_mask = given_mask & (lbl == c)
            if c_given_mask.any():
                centroids_2d[c] = H_proj_bip[l][c_given_mask].mean(axis=0)
                
        # 2D 시각화 공간 상에서의 앵커(힌트 셀) 군집 간 평균 거리
        if len(centroids_2d) > 1:
            cent_coords_2d = np.array(list(centroids_2d.values()))
            from scipy.spatial.distance import pdist
            inter_cent_dist_2d = np.mean(pdist(cent_coords_2d))
            # 육안으로 보이는 응집(Cohesion) 기준: 앵커 간 평균 거리의 15% 이내 진입
            cohesion_threshold_2d = inter_cent_dist_2d * 0.15 
        else:
            cohesion_threshold_2d = 1e-9
        
        # 2. 빈칸 셀의 고차원 ATR 계산 및 2D 시각적 응집 포인트 개수 카운팅
        d_targets = []
        d_remotes = []
        cohered_blanks = 0
        
        for idx in range(N_seq):
            if not given_mask[idx]: # 빈칸 셀인 경우
                true_c = lbl[idx]
                if true_c in centroids:
                    # 1) H_context 고차원 거리 (ATR 지표)
                    d_target = np.linalg.norm(H_l[idx] - centroids[true_c])
                    d_targets.append(d_target)
                    
                    # 2) H_proj 2D 화면 거리 (응집 카운팅)
                    d_target_2d = np.linalg.norm(H_proj_bip[l, idx] - centroids_2d[true_c])
                    if d_target_2d < cohesion_threshold_2d:
                        cohered_blanks += 1
                        
                    # 오답 앵커들과의 평균 거리 (H_context 기준)
                    remote_dists = [np.linalg.norm(H_l[idx] - centroids[k]) for k in centroids if k != true_c]
                    if remote_dists:
                        d_remotes.append(np.mean(remote_dists))
        
        if d_targets and d_remotes:
            mean_d_target = np.mean(d_targets)
            mean_d_remote = np.mean(d_remotes)
            # ATR: Target 거리 / Remote 거리. 1.0 미만으로 수렴해야 정상 작동을 의미함.
            ATR = mean_d_target / (mean_d_remote + 1e-9)
            ATR_history.append(float(ATR))
        else:
            ATR_history.append(0.0)
            
        cohered_blanks_history.append(int(cohered_blanks))

    # 3. 이분 궤적(Bipartite Trajectory) 시각화 플로팅 (레이어별 차례대로 배치)
    cols = math.ceil(loops / 2)
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 10))
    axes = axes.flatten()
    
    for l in range(loops):
        ax = axes[l]
        ax.set_title(f"Macro-Loop {l}", fontsize=12)
        
        for idx in range(N_seq):
            true_c = lbl[idx]
            color_c = palette(true_c)
            px = Q_proj_bip[l, idx, 0]
            py = Q_proj_bip[l, idx, 1]
            
            if given_mask[idx]:
                # 힌트 셀 (경계 조건 앵커): 별 모양 마커
                ax.scatter(px, py, c=[color_c], marker='*', s=200, edgecolors='black', linewidth=1.0, zorder=5)
            else:
                # 빈칸 셀 (자유 입자): 원형 마커
                jx, jy = np.random.normal(0, 0.1), np.random.normal(0, 0.1)
                ax.scatter(px + jx, py + jy, c=[color_c], marker='o', s=50, edgecolors='white', linewidth=0.5, alpha=0.7, zorder=4)
                
        ax.grid(True, alpha=0.3)
        
    # 남는 subplot 숨기기
    for l in range(loops, len(axes)):
        axes[l].axis('off')

    plt.suptitle("Target-Conditioned Bipartite Separation across Macro-Loops", fontsize=16, y=0.95)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=15, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, markeredgecolor='white')
    ]
    fig.legend(handles=custom_lines, labels=['Given Cells (Anchors)', 'Blank Cells (Moving Particles)'], loc='lower center', ncol=2, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig("visualizations/04_Bipartite_Trajectory.png", dpi=300)
    plt.close()

    metrics_log["manifold_anchor_target_ratio"] = ATR_history
    metrics_log["manifold_cohered_blank_cells"] = cohered_blanks_history
    # ── 6. Multimodal Attraction Distribution (다봉 분포 검증) ─────────────────
    print("Calculating and Plotting 6. Multimodal Attraction Profile ...")
    
    last_W = global_W[-1][0].numpy()  # 마지막 루프/마지막 레이어의 Attraction Map [81, 81]
    
    # 아무 빈칸(Blank Cell) 하나를 쿼리로 선택
    blank_indices = np.where(~given_mask)[0]
    if len(blank_indices) > 0:
        query_idx = blank_indices[0]
        att_vector = last_W[query_idx]  # [81]
        
        plt.figure(figsize=(14, 4))
        plt.bar(range(N_seq), att_vector, color='lightgray', label='Blank Cells (Ignored)')
        
        # 정답 힌트가 있는 앵커(Given Cells)들을 붉은색으로 하이라이트
        given_indices = np.where(given_mask)[0]
        if len(given_indices) > 0:
            plt.bar(given_indices, att_vector[given_indices], color='tomato', label='Given Cells (Anchors)')
        
        # 다봉 분포 피크 측정을 위한 균등 분포(Uniform) 기준선
        uniform_baseline = 1.0 / N_seq
        plt.axhline(y=uniform_baseline, color='blue', linestyle='--', alpha=0.5, label='Uniform Baseline')
        
        plt.title(f"Attraction Distribution for Blank Cell {query_idx} (Checking for Multi-peaked Mass)")
        plt.xlabel("Key Token Index (0~80)")
        plt.ylabel("Attraction Probability Mass")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("visualizations/05_Multimodal_Attraction.png", dpi=300)
        plt.close()
        
        # 다봉 분포 수치 검증 지표 저장
        mass_on_given = att_vector[given_indices].sum()
        metrics_log["attraction_mass_on_given_cells"] = float(mass_on_given)
        
        # 균등 분포 대비 1.5배 이상의 확률 질량을 가진 "유의미한 피크(Peak)" 개수 카운팅
        peak_threshold = uniform_baseline * 1.5
        significant_peaks = np.sum(att_vector > peak_threshold)
        peaks_on_given = np.sum(att_vector[given_indices] > peak_threshold)
        
        metrics_log["attraction_significant_peaks_total"] = int(significant_peaks)
        metrics_log["attraction_significant_peaks_on_given"] = int(peaks_on_given)

    with open("visualizations/metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=4)

    print("Visualizations and advanced metrics complete! Results saved to 'visualizations' folder.")
if __name__ == "__main__":
    main()
