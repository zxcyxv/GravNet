"""
diagnose_gnorm.py — 루프별 × 레이어별 gradient norm 진단

체크포인트를 로드하고, 고정 배치에 대해 N번의 forward-backward를 수행하며
각 루프에서 레이어별 gradient norm을 기록한다.

사용법:
  python diagnose_gnorm.py --checkpoint checkpoints/step_30000.pt
  python diagnose_gnorm.py --checkpoint checkpoints/best.pt --num_batches 8
  python diagnose_gnorm.py --checkpoint checkpoints/best.pt --no_compile --dtype float32
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from amkpd_model import AMKPDCarry
from dataset import create_dataloaders
from train import build_model, build_optimizer, compute_loss


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="루프별/레이어별 gradient norm 진단",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",    required=True, help="체크포인트 경로 (.pt)")
    p.add_argument("--data_dir",      default=None,  help="데이터 경로 (None이면 체크포인트에서 읽음)")
    p.add_argument("--num_loops",     type=int, default=16, help="분석할 루프 수")
    p.add_argument("--num_batches",   type=int, default=4,  help="평균낼 배치 수 (많을수록 신뢰도↑)")
    p.add_argument("--dtype",         default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--no_compile",    action="store_true", help="torch.compile 비활성화")
    p.add_argument("--update_params", action="store_true",
                   help="루프 간 optimizer.step() 수행 (기본: 파라미터 고정 순수 진단)")
    p.add_argument("--output_csv",    default="gnorm_diagnosis.csv")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Gradient 수집
# ─────────────────────────────────────────────────────────────────────────────

def collect_per_param_gnorm(model: nn.Module) -> Dict[str, float]:
    """backward 직후, clip 전 파라미터별 gradient norm 수집."""
    base = getattr(model, "_orig_mod", model)   # compile wrapper 벗기기
    result = {}
    for name, param in base.named_parameters():
        if param.grad is not None:
            result[name] = param.grad.detach().float().norm().item()
        else:
            result[name] = 0.0
    return result


def group_gnorms(per_param: Dict[str, float], num_layers: int) -> Dict[str, float]:
    """파라미터명 → 컴포넌트 그룹별 L2 gnorm."""
    def rss(*prefixes) -> float:
        sq = sum(v ** 2 for k, v in per_param.items()
                 if any(k.startswith(pf) for pf in prefixes))
        return math.sqrt(sq)

    g: Dict[str, float] = {}

    # 블록별 컴포넌트
    for bi in range(num_layers):
        b = f"blocks.{bi}"
        g[f"b{bi}_QKV"]   = rss(f"{b}.W_QKV")
        g[f"b{bi}_O"]     = rss(f"{b}.W_O_aux")
        g[f"b{bi}_bias"]  = rss(f"{b}.attn_bias")
        g[f"b{bi}_up"]    = rss(f"{b}.W_up")
        g[f"b{bi}_dn"]    = rss(f"{b}.W_down")
        g[f"b{bi}_conv"]  = rss(f"{b}.dw_conv")
        # 블록 합계
        g[f"b{bi}_TOTAL"] = rss(f"{b}.")

    # 글로벌 컴포넌트
    g["embed"]    = rss("embedding")
    g["lm_head"]  = rss("lm_head")
    g["halt_head"]= rss("halt_head")

    # 전체 gnorm (clip 전)
    g["TOTAL"] = math.sqrt(sum(v ** 2 for v in per_param.values()))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_compact_table(records: List[Dict], num_layers: int):
    """루프 × 블록 합계 + 전체 gnorm 요약 테이블."""
    block_cols = [f"b{bi}_TOTAL" for bi in range(num_layers)]
    other_cols  = ["embed", "lm_head", "halt_head", "TOTAL"]
    cols = block_cols + other_cols

    W = 8
    header = f"{'loop':>5} | " + " | ".join(f"{c:>{W}}" for c in cols)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("[블록 합계 gnorm — 루프별 요약]")
    print(header)
    print(sep)

    for rec in records:
        row = f"{rec['loop']:>5} | " + " | ".join(f"{rec.get(c, 0.0):>{W}.4f}" for c in cols)
        print(row)

    print(sep)
    avg_row = "  avg | " + " | ".join(
        f"{sum(r.get(c, 0.0) for r in records) / len(records):>{W}.4f}" for c in cols
    )
    print(avg_row)
    print("=" * len(header))


def print_detail_table(records: List[Dict], num_layers: int):
    """루프 × 컴포넌트 상세 테이블 (블록 0만 상세, 나머지는 합계)."""
    detail_cols = ["b0_QKV", "b0_O", "b0_bias", "b0_up", "b0_dn", "b0_conv"]
    rest_block_cols = [f"b{bi}_TOTAL" for bi in range(1, num_layers)]
    cols = detail_cols + rest_block_cols + ["embed", "lm_head", "TOTAL"]

    W = 8
    header = f"{'loop':>5} | " + " | ".join(f"{c:>{W}}" for c in cols)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("[컴포넌트 상세 gnorm — block 0 세분화]")
    print(header)
    print(sep)

    for rec in records:
        row = f"{rec['loop']:>5} | " + " | ".join(f"{rec.get(c, 0.0):>{W}.4f}" for c in cols)
        print(row)

    print("=" * len(header) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── 체크포인트 로드 ────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] 체크포인트 없음: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_args: dict = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)

    print(f"Checkpoint step={ckpt.get('global_step', '?')}  "
          f"puzzle_acc={ckpt.get('best_puzzle_acc', '?'):.4f}")

    # ── 데이터 ────────────────────────────────────────────────────────────
    data_dir  = args.data_dir or ckpt_args.get("data_dir", "data/sudoku-extreme-1k-aug-1000")
    batch_size = ckpt_args.get("batch_size", 128)

    # train_loader와 동일한 구조 (SudokuGroupDataset, IterableDataset)
    # epochs_per_iter=1 → 1 epoch 분량만 순회
    train_loader, _, meta = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        epochs_per_iter=1,
        seed=42,
        num_workers=0,
        pin_memory=False,
    )

    # 고정 배치 확보 (train 분포에서)
    fixed_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for inputs, labels in train_loader:
        fixed_batches.append((inputs.to(device), labels.to(device)))
        if len(fixed_batches) >= args.num_batches:
            break

    if not fixed_batches:
        print("[ERROR] 배치 로드 실패", file=sys.stderr)
        sys.exit(1)

    B, N = fixed_batches[0][0].shape
    print(f"Batches : {len(fixed_batches)} × [{B}, {N}]")

    # ── 모델 ──────────────────────────────────────────────────────────────
    model = build_model(ckpt_args, meta["vocab_size"]).to(device)

    model.load_state_dict(ckpt["model"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params  : {n_params:,}")

    if not args.no_compile:
        try:
            model = torch.compile(model, dynamic=False)
            print("torch.compile : enabled")
        except Exception as e:
            print(f"torch.compile : skipped ({e})")

    # ── Mixed precision ───────────────────────────────────────────────────
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    use_amp   = (amp_dtype != torch.float32) and (device.type == "cuda")
    autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
    print(f"dtype   : {args.dtype}  amp={use_amp}")

    # ── compute_loss용 args ───────────────────────────────────────────────
    loss_ns = SimpleNamespace(
        halt_loss_coef=ckpt_args.get("halt_loss_coef", 0.01),
        loops=args.num_loops,
    )

    # ── Optimizer (update_params 모드 시) ─────────────────────────────────
    optimizer: Optional[torch.optim.Optimizer] = None
    if args.update_params:
        opt_ns = SimpleNamespace(**{
            "lr": ckpt_args.get("lr", 1e-4),
            "weight_decay": ckpt_args.get("weight_decay", 1.0),
            "max_grad_norm": ckpt_args.get("max_grad_norm", 1.0),
        })
        optimizer = build_optimizer(model, opt_ns)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("optimizer state loaded")
            except Exception as e:
                print(f"optimizer state load 실패 (무시): {e}")
        print("모드    : training (optimizer.step 수행)")
    else:
        print("모드    : 순수 진단 (파라미터 고정)")

    num_layers = ckpt_args.get("num_layers", 6)

    # ── 루프별 gnorm 수집 ─────────────────────────────────────────────────
    # accumulated[loop_idx][group_key] = [gnorm, gnorm, ...]  (배치별)
    accumulated: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    print(f"\n{'='*60}")
    print(f"루프 {args.num_loops}회 × 배치 {len(fixed_batches)}개 진단 시작")
    print(f"{'='*60}")

    for batch_idx, (inputs, labels) in enumerate(fixed_batches):
        model.train()
        model.zero_grad()

        seq_len = inputs.shape[1]
        carry   = model.initial_carry(inputs.shape[0], seq_len, device)
        batch   = (inputs, labels)

        print(f"\n  Batch {batch_idx + 1}/{len(fixed_batches)}")
        print(f"  {'loop':>4}  {'TOTAL':>8}  {'max_block':>10}  {'embed':>8}  {'lm_head':>8}")
        print(f"  {'-'*50}")

        for loop_idx in range(1, args.num_loops + 1):
            # ── Forward ──────────────────────────────────────────────────
            with autocast_ctx:
                carry, logits, q_logits = model(carry, batch)
                loss, _ = compute_loss(logits, q_logits, carry.current_labels, loss_ns)

            # ── Backward ─────────────────────────────────────────────────
            loss.backward()

            # ── Gradient 수집 (clip 전 raw gnorm) ─────────────────────
            per_param = collect_per_param_gnorm(model)
            groups    = group_gnorms(per_param, num_layers)

            for k, v in groups.items():
                accumulated[loop_idx][k].append(v)

            # 진행 상황 한 줄 출력
            max_block_gnorm = max(groups.get(f"b{bi}_TOTAL", 0.0) for bi in range(num_layers))
            print(f"  {loop_idx:>4}  {groups['TOTAL']:>8.4f}  {max_block_gnorm:>10.4f}"
                  f"  {groups['embed']:>8.4f}  {groups['lm_head']:>8.4f}")

            # ── 파라미터 업데이트 (선택) ──────────────────────────────
            if args.update_params and optimizer is not None:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.zero_grad()

    # ── 배치 평균 집계 ────────────────────────────────────────────────────
    avg_records: List[Dict] = []
    all_group_keys: List[str] = []

    for loop_idx in range(1, args.num_loops + 1):
        rec: Dict = {"loop": loop_idx}
        for k, vals in accumulated[loop_idx].items():
            rec[k] = sum(vals) / max(1, len(vals))
        avg_records.append(rec)
        if not all_group_keys:
            all_group_keys = [k for k in rec if k != "loop"]

    # ── 출력 ──────────────────────────────────────────────────────────────
    print_compact_table(avg_records, num_layers)
    print_detail_table(avg_records, num_layers)

    # 가장 gnorm이 컸던 루프
    worst_loop = max(avg_records, key=lambda r: r.get("TOTAL", 0.0))
    print(f"최대 gnorm 루프: loop={worst_loop['loop']}  TOTAL={worst_loop['TOTAL']:.4f}")

    # 각 루프에서 gnorm이 가장 큰 블록
    print("\n루프별 최대 gnorm 블록:")
    for rec in avg_records:
        block_gnorms = {f"b{bi}": rec.get(f"b{bi}_TOTAL", 0.0) for bi in range(num_layers)}
        worst_block = max(block_gnorms, key=block_gnorms.get)
        print(f"  loop {rec['loop']:>2}: {worst_block} = {block_gnorms[worst_block]:.4f}"
              f"  (TOTAL={rec['TOTAL']:.4f})")

    # ── CSV 저장 ──────────────────────────────────────────────────────────
    csv_path = Path(args.output_csv)
    fieldnames = ["loop"] + all_group_keys
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(avg_records)
    print(f"\nCSV 저장: {csv_path}")


if __name__ == "__main__":
    main()
