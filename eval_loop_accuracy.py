"""
eval_loop_accuracy.py — 루프 수별 정답률 변화 측정

각 루프 스텝마다 전체 테스트 샘플의 token_acc, puzzle_acc를 측정하여
루프를 거칠수록 정답률이 올라가는지/내려가는지 확인.

사용법:
  uv run python eval_loop_accuracy.py checkpoints/step030000_pacc0.0938.pt
  uv run python eval_loop_accuracy.py checkpoints/step030000_pacc0.0938.pt --max_samples 500
"""

import argparse
import torch
from dataset import SudokuDataset
from torch.utils.data import DataLoader
from amkpd_model import AMKPDModel


def load_checkpoint(ckpt_path, device, max_loops):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]

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
    return model, a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_loops", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, a = load_checkpoint(args.ckpt_path, device, args.max_loops)

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

                # carry 초기화 후 loop_i번 반복
                pos = torch.arange(seq_len, device=device).unsqueeze(0)
                X = model.embedding(inp) + model.pos_emb(pos)
                X = model.input_norm(X)
                Q = model.init_hidden.unsqueeze(0).unsqueeze(0).expand(B, seq_len, -1).clone()

                for _loop in range(loop_i):
                    # H_cycles burn-in + grad cycle
                    if model.H_cycles > 1:
                        for _h in range(model.H_cycles - 1):
                            for _l in range(model.L_cycles):
                                for block in model.blocks:
                                    Q = block(Q, X)
                    for _l in range(model.L_cycles):
                        for block in model.blocks:
                            Q = block(Q, X)

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
