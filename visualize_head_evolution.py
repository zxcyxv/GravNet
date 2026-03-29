"""
가설 검증: H0/H5/H6의 attention이 루프를 거치며 꺼지는 이유가
"이미 정답을 채워넣은 셀"이기 때문인지 확인.

출력: 루프별로 [스도쿠 예측 그리드 | H0 | H5 | H6 히트맵] 을 나란히 배치.
히트맵 위에 정답/오답 여부를 오버레이.
"""

import os, sys, math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from amkpd_model import AMKPDModel
from dataset import SudokuDataset


def draw_sudoku(ax, given, preds, labels):
    """9x9 스도쿠 그리드. 파란=정답, 빨간=오답, 회색배경=given"""
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
    """9x9 히트맵 + 정답/오답 오버레이"""
    sns.heatmap(att_9x9, ax=ax, cmap='magma', cbar=False, square=True,
                linewidths=0.4, linecolor='gray', vmin=0)
    # 3x3 박스 경계
    for b in [3, 6]:
        ax.axhline(b, color='cyan', linewidth=1.5)
        ax.axvline(b, color='cyan', linewidth=1.5)
    # 쿼리 셀
    ax.add_patch(plt.Rectangle((query_c, query_r), 1, 1,
                 fill=False, edgecolor='lime', linewidth=2.5))
    # 각 셀에 정답 여부 표시
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
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/step002500_pacc0.0299.pt"
    sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 퍼즐 인덱스
    max_loops = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]

    ds = SudokuDataset(a["data_dir"], "test")
    inp, lbl = ds[sample_idx]
    inp = inp.unsqueeze(0).to(device)   # [1, 81]
    lbl = lbl.unsqueeze(0).to(device)

    model = AMKPDModel(
        vocab_size=11, d_model=a["d_model"], num_heads=a["num_heads"],
        num_layers=a["num_layers"], loops=max_loops, H_cycles=a["H_cycles"],
        L_cycles=a["L_cycles"], kernel_power=a["kernel_power"],
        expansion_ratio=a["expansion_ratio"], conv_kernel_size=a.get("conv_kernel", 2),
    ).to(device)
    # _orig_mod. prefix 제거 + old W_Q/W_K/W_V/W_O/W_aux → fused 변환
    state = {}
    for k, v in ckpt["model"].items():
        k = k.replace("_orig_mod.", "")
        state[k] = v
    # old checkpoint: W_Q + W_K + W_V → W_QKV
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

    # 텔레메트리 ON
    model.log_viz = True
    for b in model.blocks:
        b.log_viz = True

    H = a["num_heads"]
    heads_to_show = [0, 5, 6]  # 관심 헤드
    inp_np = inp[0].cpu().numpy()
    lbl_np = lbl[0].cpu().numpy()
    given = inp_np > 1

    # 빈칸 중앙 부근을 쿼리로 선택
    blank_indices = np.where(~given)[0]
    query_idx = blank_indices[len(blank_indices) // 2]
    qr, qc = query_idx // 9, query_idx % 9
    query_digit = lbl_np[query_idx] - 1 if lbl_np[query_idx] >= 2 else '?'

    # ── 루프별 수집 ──
    carry = model.initial_carry(1, 81, device)
    batch = (inp, lbl)

    step_checkpoints = [1, 2, 3, 5, 8, 12, 16, 20]
    step_checkpoints = [s for s in step_checkpoints if s <= max_loops]

    collected = []  # list of (step, preds, W_per_block)

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
                preds = logits[0].argmax(dim=-1).cpu().numpy()  # [81]
                # 각 블록의 W 수집 (마지막 micro-step)
                W_blocks = []
                for b in model.blocks:
                    if b.viz_W:
                        W_blocks.append(b.viz_W[-1][0].cpu().numpy())  # [H, 81, 81]
                collected.append((i, preds, W_blocks))

                mask = lbl_np != 0
                n_correct = ((preds == lbl_np) & mask).sum()
                n_blanks = mask.sum()
                print(f"  loop {i:>3}: {n_correct}/{n_blanks} cells correct")

    model.log_viz = False
    for b in model.blocks:
        b.log_viz = False

    # ════════════════════════════════════════════════════════════════
    # 메인 시각화: 루프별 [스도쿠 | H0 | H5 | H6] 가로 배치, 블록별 세로
    # ════════════════════════════════════════════════════════════════
    n_steps = len(collected)
    n_heads_show = len(heads_to_show)
    # 마지막 블록만 사용 (prompt의 07번과 동일)
    n_cols = 1 + n_heads_show  # 스도쿠 + 헤드 3개
    n_rows = n_steps

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, (step, preds, W_blocks) in enumerate(collected):
        # 스도쿠 그리드
        ax = axes[ri, 0]
        draw_sudoku(ax, given, preds, lbl_np)
        mask = lbl_np != 0
        n_correct = ((preds == lbl_np) & mask).sum()
        ax.set_title(f"Loop {step}  ({n_correct}/{mask.sum()} correct)", fontsize=10)

        # 헤드별 히트맵 (마지막 블록)
        W_last = W_blocks[-1] if W_blocks else None  # [H, 81, 81]
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
    plt.savefig("visualizations/08_head_evolution.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: visualizations/08_head_evolution.png")

    # ════════════════════════════════════════════════════════════════
    # 추가: 쿼리 셀의 attention이 "정답 맞춘 셀"에 얼마나 가는지 수치 추적
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'step':>4} | {'H':>2} | {'att→correct':>12} | {'att→wrong':>10} | {'att→given':>10} | {'att→blank':>10}")
    print("-" * 70)

    for step, preds, W_blocks in collected:
        W_last = W_blocks[-1] if W_blocks else None
        if W_last is None:
            continue
        mask = lbl_np != 0
        correct_blank = (~given) & (preds == lbl_np) & mask
        wrong_blank = (~given) & (preds != lbl_np) & mask

        for h in heads_to_show:
            att = W_last[h, query_idx]  # [81]
            total = att.sum() + 1e-12
            att_correct = att[correct_blank].sum() / total
            att_wrong = att[wrong_blank].sum() / total
            att_given = att[given].sum() / total
            att_blank = att[~given].sum() / total
            print(f"{step:>4} | H{h} | {att_correct:>11.4f} | {att_wrong:>9.4f} | {att_given:>9.4f} | {att_blank:>9.4f}")
        print()


if __name__ == "__main__":
    main()
