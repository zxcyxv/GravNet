"""
정답을 맞춘 퍼즐에 대해서만:
1. 루프별 예측 변화를 실제 스도쿠 그리드로 시각화
2. 헤드별 위상 특화도 분석
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import torch.nn.functional as F
from amkpd_model import AMKPDModel
from dataset import SudokuDataset

# ── Sudoku topology ──
def get_row_indices(idx):
    row = idx // 9
    return [row * 9 + c for c in range(9) if row * 9 + c != idx]

def get_col_indices(idx):
    col = idx % 9
    return [r * 9 + col for r in range(9) if r * 9 + col != idx]

def get_box_indices(idx):
    row, col = idx // 9, idx % 9
    box_r, box_c = (row // 3) * 3, (col // 3) * 3
    return [
        (box_r + dr) * 9 + (box_c + dc)
        for dr in range(3) for dc in range(3)
        if (box_r + dr) * 9 + (box_c + dc) != idx
    ]


def draw_sudoku_grid(ax, given, preds, labels, title="", highlight_wrong=True):
    """
    9x9 스도쿠 그리드를 그린다.
    given: [81] bool - 주어진 셀
    preds: [81] int - 모델 예측 (0=blank, 2-10 → 1-9)
    labels: [81] int - 정답 (0=blank in label means given, 2-10 → 1-9)
    """
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=4)

    # 3x3 박스 굵은 선
    for i in range(10):
        lw = 2.5 if i % 3 == 0 else 0.5
        ax.axhline(y=i, color='black', linewidth=lw)
        ax.axvline(x=i, color='black', linewidth=lw)

    for idx in range(81):
        r, c = idx // 9, idx % 9
        pred_val = preds[idx]
        label_val = labels[idx]

        # 토큰 값 → 숫자 (2→1, 3→2, ..., 10→9), 0 or 1 → blank
        pred_digit = pred_val - 1 if pred_val >= 2 else 0
        label_digit = label_val - 1 if label_val >= 2 else 0

        if given[idx]:
            # 주어진 셀: 회색 배경, 검정 글씨
            ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='#e0e0e0', edgecolor='none'))
            if label_digit > 0:
                ax.text(c + 0.5, r + 0.5, str(label_digit),
                        ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        else:
            # 빈칸: 모델 예측 표시
            if pred_digit > 0:
                if pred_digit == label_digit:
                    # 정답: 파란 글씨
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='#e8f4fd', edgecolor='none'))
                    ax.text(c + 0.5, r + 0.5, str(pred_digit),
                            ha='center', va='center', fontsize=11, fontweight='bold', color='#2196F3')
                else:
                    # 오답: 빨간 글씨 + 연한 빨간 배경
                    if highlight_wrong:
                        ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='#fde8e8', edgecolor='none'))
                    ax.text(c + 0.5, r + 0.5, str(pred_digit),
                            ha='center', va='center', fontsize=11, fontweight='bold', color='#e74c3c')
            else:
                ax.text(c + 0.5, r + 0.5, '?',
                        ha='center', va='center', fontsize=9, color='#999999')


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/step002500_pacc0.0299.pt"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    max_loops = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    print(f"Checkpoint: {ckpt_path} (step={ckpt['global_step']})")

    ds = SudokuDataset(a["data_dir"], "test")
    n = min(n_samples, len(ds))
    inputs = torch.stack([ds[i][0] for i in range(n)]).to(device)
    labels = torch.stack([ds[i][1] for i in range(n)]).to(device)

    model = AMKPDModel(
        vocab_size=11, d_model=a["d_model"], num_heads=a["num_heads"],
        num_layers=a["num_layers"], loops=max_loops, H_cycles=a["H_cycles"],
        L_cycles=a["L_cycles"], kernel_power=a["kernel_power"],
        expansion_ratio=a["expansion_ratio"], conv_kernel_size=a.get("conv_kernel", 2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    B, seq_len = inputs.shape
    carry = model.initial_carry(B, seq_len, device)
    batch = (inputs, labels)

    # ── 루프별 예측 수집 ──
    checkpoints = [1, 2, 3, 5, 10, 15, 20, 25, 30]
    checkpoints = [c for c in checkpoints if c <= max_loops]

    loop_preds = {}   # {loop_num: [B, 81] predictions}
    loop_correct = {} # {loop_num: [B] bool puzzle correct}

    # 헤드별 W 수집 (마지막 체크포인트에서)
    model.log_viz = True
    for b in model.blocks:
        b.log_viz = True

    print(f"Running {max_loops} loops on {n} samples ...")
    with torch.no_grad():
        for i in range(1, max_loops + 1):
            # 텔레메트리 리셋
            for b in model.blocks:
                b.viz_W = []
                b.viz_m = []
                b.viz_H = []

            carry, logits, _ = model(carry, batch)

            if i in checkpoints:
                preds = logits.argmax(dim=-1)  # [B, 81]
                mask = labels != 0
                puzzle_correct = ((preds == labels) | ~mask).all(dim=1)  # [B]
                loop_preds[i] = preds.cpu().numpy()
                loop_correct[i] = puzzle_correct.cpu().numpy()

                n_correct = puzzle_correct.sum().item()
                tok_acc = ((preds == labels) & mask).float().sum() / mask.float().sum()
                print(f"  loop {i:>3}: puzzle_acc={n_correct}/{n} ({n_correct/n*100:.1f}%), tok_acc={tok_acc.item()*100:.1f}%")

    # 마지막 루프의 헤드별 W 수집
    last_W_per_block = []
    for b in model.blocks:
        if b.viz_W:
            last_W_per_block.append(b.viz_W[-1].cpu().numpy())  # [B, H, N, N]

    model.log_viz = False
    for b in model.blocks:
        b.log_viz = False

    # ── 정답 퍼즐 찾기 (마지막 체크포인트 기준) ──
    last_ckpt = checkpoints[-1]
    correct_mask = loop_correct[last_ckpt]
    correct_indices = np.where(correct_mask)[0]

    print(f"\nCorrect puzzles at loop {last_ckpt}: {len(correct_indices)}/{n}")

    if len(correct_indices) == 0:
        print("No correct puzzles found. Showing best partial matches instead.")
        # 가장 많은 셀을 맞춘 퍼즐 상위 3개
        mask_np = (labels.cpu().numpy() != 0)
        preds_last = loop_preds[last_ckpt]
        labels_np = labels.cpu().numpy()
        cell_correct = (preds_last == labels_np) & mask_np
        n_correct_cells = cell_correct.sum(axis=1)
        correct_indices = np.argsort(n_correct_cells)[-3:][::-1]
        print(f"  Showing top-3 by cell accuracy: indices={correct_indices.tolist()}")
        print(f"  Cell correct counts: {n_correct_cells[correct_indices].tolist()}")

    os.makedirs("visualizations", exist_ok=True)

    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # ════════════════════════════════════════════════════════════════════
    # 1. 루프별 스도쿠 그리드 변화 (정답/최고 퍼즐)
    # ════════════════════════════════════════════════════════════════════
    n_show = min(3, len(correct_indices))
    for si in range(n_show):
        pidx = correct_indices[si]
        given = inputs_np[pidx] > 1
        lbl = labels_np[pidx]

        n_ckpts = len(checkpoints)
        fig, axes = plt.subplots(1, n_ckpts + 1, figsize=(3.2 * (n_ckpts + 1), 3.5))

        # 정답 그리드
        draw_sudoku_grid(axes[0], given, lbl, lbl, title="Ground Truth", highlight_wrong=False)

        # 루프별 예측
        for ci, ck in enumerate(checkpoints):
            preds_ck = loop_preds[ck][pidx]
            mask = lbl != 0
            n_cell_correct = ((preds_ck == lbl) & mask).sum()
            n_blanks = mask.sum()
            draw_sudoku_grid(axes[ci + 1], given, preds_ck, lbl,
                             title=f"Loop {ck}\n({n_cell_correct}/{n_blanks} cells)")

        fig.suptitle(f"Puzzle #{pidx} — Prediction Evolution Across Loops", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(f"visualizations/correct_puzzle_{si}_loops.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: correct_puzzle_{si}_loops.png (puzzle #{pidx})")

    # ════════════════════════════════════════════════════════════════════
    # 2. 정답 퍼즐의 헤드별 위상 특화도
    # ════════════════════════════════════════════════════════════════════
    if last_W_per_block and len(correct_indices) > 0:
        pidx = correct_indices[0]
        given = inputs_np[pidx] > 1
        lbl = labels_np[pidx]
        blank_indices = np.where(~given)[0]

        H = a["num_heads"]
        # 마지막 블록의 W: [B, H, N, N]
        W_last = last_W_per_block[-1][pidx]  # [H, 81, 81]

        # 여러 빈칸에 대해 평균 topology mass 계산
        head_row_mass = np.zeros(H)
        head_col_mass = np.zeros(H)
        head_box_mass = np.zeros(H)

        for bi in blank_indices:
            row_nb = set(get_row_indices(bi))
            col_nb = set(get_col_indices(bi))
            box_nb = set(get_box_indices(bi))
            for h in range(H):
                att = W_last[h, bi]
                total = att.sum() + 1e-12
                head_row_mass[h] += att[list(row_nb)].sum() / total
                head_col_mass[h] += att[list(col_nb)].sum() / total
                head_box_mass[h] += att[list(box_nb)].sum() / total

        n_blanks = len(blank_indices)
        head_row_mass /= n_blanks
        head_col_mass /= n_blanks
        head_box_mass /= n_blanks

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(H)
        width = 0.25
        ax.bar(x - width, head_row_mass, width, label='Row', color='#e74c3c')
        ax.bar(x, head_col_mass, width, label='Column', color='#3498db')
        ax.bar(x + width, head_box_mass, width, label='Box', color='#2ecc71')
        ax.set_xlabel("Head")
        ax.set_ylabel("In-Topology Mass Ratio (avg over all blanks)")
        ax.set_title(f"Head Topological Specialization — Correct Puzzle #{pidx}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{i}" for i in range(H)])
        ax.axhline(y=8.0 / 80.0, color='gray', linestyle='--', alpha=0.5, label='Uniform (8/80)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("visualizations/correct_head_topology.png", dpi=200)
        plt.close()
        print(f"  Saved: correct_head_topology.png")

        # 개별 빈칸 히트맵 (정답을 맞춘 셀 하나)
        preds_last = loop_preds[last_ckpt][pidx]
        mask = lbl != 0
        correct_blanks = [bi for bi in blank_indices if preds_last[bi] == lbl[bi]]

        if correct_blanks:
            query_idx = correct_blanks[len(correct_blanks) // 2]
            qr, qc = query_idx // 9, query_idx % 9

            n_cols = min(H, 4)
            n_rows = (H + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = np.array(axes).flatten()

            for h in range(H):
                ax = axes[h]
                att = W_last[h, query_idx].reshape(9, 9)
                import seaborn as sns
                sns.heatmap(att, ax=ax, cmap='YlOrRd', cbar=True,
                            cbar_kws={'shrink': 0.6}, square=True,
                            linewidths=0.5, linecolor='gray')
                # 쿼리 셀 표시
                ax.add_patch(patches.Rectangle((qc, qr), 1, 1,
                             fill=False, edgecolor='blue', linewidth=3))
                total = att.sum() + 1e-12
                r_m = att.flatten()[list(set(get_row_indices(query_idx)))].sum() / total
                c_m = att.flatten()[list(set(get_col_indices(query_idx)))].sum() / total
                b_m = att.flatten()[list(set(get_box_indices(query_idx)))].sum() / total
                ax.set_title(f"Head {h}", fontsize=10)
                ax.set_xlabel(f"R:{r_m:.2f} C:{c_m:.2f} B:{b_m:.2f}", fontsize=8)

            for i in range(H, len(axes)):
                axes[i].axis('off')

            pred_digit = preds_last[query_idx] - 1 if preds_last[query_idx] >= 2 else '?'
            label_digit = lbl[query_idx] - 1 if lbl[query_idx] >= 2 else '?'
            fig.suptitle(
                f"Correct Blank ({qr},{qc}) — pred={pred_digit}, answer={label_digit}\n"
                f"Puzzle #{pidx}",
                fontsize=12
            )
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.savefig("visualizations/correct_cell_head_heatmap.png", dpi=200)
            plt.close()
            print(f"  Saved: correct_cell_head_heatmap.png (cell ({qr},{qc}))")

    print("\nDone!")


if __name__ == "__main__":
    main()
