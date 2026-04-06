"""
train_gnorm_diag.py — 루프별 × 레이어별 gradient norm 진단용 훈련 스크립트

train.py와 완전히 동일한 훈련 로직을 사용하되,
optimizer step 직후에 per-layer gnorm을 JSONL로 기록하는 로직만 추가한다.

추가된 로직:
  - scaler.unscale_() 직후, clip_grad_norm_() 직전에
    파라미터별 gradient norm을 수집
  - carry.steps (현재 루프 번호)와 함께 JSONL에 기록
  - 기본 100 스텝 후 자동 종료 (--diag_steps로 조정)

사용법:
  python train_gnorm_diag.py --resume checkpoints/best.pt --diag_steps 200
  python train_gnorm_diag.py --resume checkpoints/best.pt --diag_steps 200 --no_compile
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# train.py의 모든 함수/클래스를 그대로 가져온다
import train as _train
from train import (
    parse_args,
    build_model,
    build_optimizer,
    cosine_schedule_with_warmup,
    compute_loss,
    compute_metrics,
    evaluate,
    CheckpointManager,
)


# ─────────────────────────────────────────────────────────────────────────────
# Gnorm 수집 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _per_layer_gnorms(model: nn.Module, num_layers: int) -> dict:
    """unscale_ 이후 파라미터별 gradient norm을 수집해 컴포넌트별로 묶는다."""
    base = getattr(model, "_orig_mod", model)

    per_param = {
        name: p.grad.detach().float().norm().item()
        for name, p in base.named_parameters()
        if p.grad is not None
    }

    def rss(*prefixes) -> float:
        sq = sum(v ** 2 for k, v in per_param.items()
                 if any(k.startswith(pf) for pf in prefixes))
        return math.sqrt(sq)

    g = {}
    for bi in range(num_layers):
        b = f"blocks.{bi}"
        g[f"b{bi}_QKV"]   = rss(f"{b}.W_QKV")
        g[f"b{bi}_O"]     = rss(f"{b}.W_O_aux")
        g[f"b{bi}_bias"]  = rss(f"{b}.attn_bias")
        g[f"b{bi}_up"]    = rss(f"{b}.W_up")
        g[f"b{bi}_dn"]    = rss(f"{b}.W_down")
        g[f"b{bi}_conv"]  = rss(f"{b}.dw_conv")
        g[f"b{bi}_TOTAL"] = rss(f"{b}.")
    g["embed"]     = rss("embedding")
    g["lm_head"]   = rss("lm_head")
    g["halt_head"] = rss("halt_head")
    g["TOTAL_pre_clip"] = math.sqrt(sum(v ** 2 for v in per_param.values()))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 진단용 train() — train.py의 train()과 단 한 곳만 다르다
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    """train.py의 train()과 동일. optimizer step 직전에 per-layer gnorm 기록만 추가."""

    import time
    import csv
    from collections import deque
    from tqdm import tqdm

    # ── train.py와 동일한 초기화 블록 ─────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"Device : {device}")

    assert args.epochs % args.eval_interval == 0
    total_iters     = args.epochs // args.eval_interval
    epochs_per_iter = args.eval_interval

    from dataset import create_dataloaders, vectorized_sudoku_augment
    train_loader, test_loader, meta = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs_per_iter=epochs_per_iter,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    batches_per_epoch   = meta["total_groups"] // args.batch_size
    total_steps_approx  = batches_per_epoch * args.epochs // args.grad_accum

    model = build_model(vars(args), meta["vocab_size"]).to(device)
    if args.full_grad:
        model.burn_in_no_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params:,}")

    if not args.no_compile:
        try:
            model = torch.compile(model, dynamic=False)
            print("torch.compile : enabled")
        except Exception as e:
            print(f"torch.compile : skipped ({e})")

    if args.debug_nan:
        torch.autograd.set_detect_anomaly(True)

    _dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype  = _dtype_map[args.dtype]
    use_amp    = amp_dtype != torch.float32
    if use_amp and amp_dtype == torch.float16 and device.type != "cuda":
        use_amp = False

    autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
    use_scaler   = use_amp and (amp_dtype == torch.float16)
    scaler       = torch.amp.GradScaler("cuda", enabled=use_scaler)

    optimizer = build_optimizer(model, args)

    start_iter      = 0
    global_step     = 0
    best_puzzle_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_iter      = ckpt.get("iter_id", 0) + 1
        global_step     = ckpt["global_step"]
        best_puzzle_acc = ckpt.get("best_puzzle_acc", 0.0)
        print(f"Resumed : {args.resume}  iter={start_iter}  step={global_step}")

    ckpt_mgr = CheckpointManager(args.checkpoint_dir, args.save_top_k)

    # ── [추가] JSONL 초기화 ────────────────────────────────────────────────
    diag_path = Path(args.checkpoint_dir) / "gnorm_per_loop.jsonl"
    diag_file = open(diag_path, "w")
    print(f"Gnorm log : {diag_path}")
    print(f"진단 스텝 : {args.diag_steps}  (이후 자동 종료)")
    # ──────────────────────────────────────────────────────────────────────

    # train.py의 grad_log (csv)는 유지
    grad_log_path   = Path(args.checkpoint_dir) / "grad_metrics.csv"
    grad_log_fields = [
        "step", "gnorm",
        "b0_W_QKV", "b0_W_O_aux", "b0_W_up", "b0_W_down", "b0_dw_conv",
        "b0_norm1", "b0_norm2",
        "embedding", "pos_emb", "input_norm",
        "final_norm", "lm_head", "halt_head",
        "b0_m_norm", "b0_C_norm", "other",
    ]
    with open(grad_log_path, "w", newline="") as f:
        csv.writer(f).writerow(grad_log_fields)

    _known_prefixes = []
    for bi in range(args.num_layers):
        for comp in ["W_QKV", "W_O_aux", "W_up", "W_down", "dw_conv", "norm1", "norm2"]:
            _known_prefixes.append(f"blocks.{bi}.{comp}")
    _known_prefixes += ["embedding", "pos_emb", "input_norm", "final_norm", "lm_head", "halt_head"]

    def collect_grad_norms(model_ref, step, gnorm):
        row = {"step": step, "gnorm": f"{gnorm:.6f}"}
        param_gnorms = {}
        for name, p in model_ref.named_parameters():
            param_gnorms[name] = p.grad.norm().item() if p.grad is not None else 0.0

        def sum_gnorm(*prefixes):
            return sum(v**2 for n, v in param_gnorms.items()
                       if any(n.startswith(px) for px in prefixes)) ** 0.5

        row["b0_W_QKV"]   = f"{sum_gnorm('blocks.0.W_QKV'):.6f}"
        row["b0_W_O_aux"] = f"{sum_gnorm('blocks.0.W_O_aux'):.6f}"
        row["b0_W_up"]    = f"{sum_gnorm('blocks.0.W_up'):.6f}"
        row["b0_W_down"]  = f"{sum_gnorm('blocks.0.W_down'):.6f}"
        row["b0_dw_conv"] = f"{sum_gnorm('blocks.0.dw_conv'):.6f}"
        row["b0_norm1"]   = f"{sum_gnorm('blocks.0.norm1'):.6f}"
        row["b0_norm2"]   = f"{sum_gnorm('blocks.0.norm2'):.6f}"
        row["embedding"]  = f"{sum_gnorm('embedding'):.6f}"
        row["pos_emb"]    = f"{sum_gnorm('pos_emb'):.6f}"
        row["input_norm"] = f"{sum_gnorm('input_norm'):.6f}"
        row["final_norm"] = f"{sum_gnorm('final_norm'):.6f}"
        row["lm_head"]    = f"{sum_gnorm('lm_head'):.6f}"
        row["halt_head"]  = f"{sum_gnorm('halt_head'):.6f}"
        b0 = model_ref.blocks[0] if hasattr(model_ref, 'blocks') else model_ref._orig_mod.blocks[0]
        row["b0_m_norm"]  = f"{b0.last_m_norm:.6f}"
        row["b0_C_norm"]  = f"{b0.last_C_norm:.6f}"
        other_sq = sum(v**2 for n, v in param_gnorms.items()
                       if not any(n.startswith(px) for px in _known_prefixes))
        row["other"] = f"{other_sq**0.5:.6f}"
        with open(grad_log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=grad_log_fields).writerow(row)

    total_start  = time.time()
    step_times   = deque(maxlen=100)
    step_samples = deque(maxlen=100)
    last_grad_norm = 0.0
    seq_len      = meta["seq_len"]

    carry = model.initial_carry(args.batch_size, seq_len, device)

    for iter_id in range(start_iter, total_iters):
        model.train()
        optimizer.zero_grad()

        epoch_start = iter_id * epochs_per_iter
        epoch_end   = epoch_start + epochs_per_iter
        iter_loss   = 0.0
        iter_steps  = 0

        pbar = tqdm(
            enumerate(train_loader),
            desc=f"Iter {iter_id+1}/{total_iters} (ep {epoch_start}-{epoch_end})",
            dynamic_ncols=True,
            leave=True,
        )

        for batch_idx, (inputs, labels) in pbar:
            t0 = time.perf_counter()

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if inputs.shape[0] != carry.current_hidden.shape[0]:
                carry = model.initial_carry(inputs.shape[0], seq_len, device)

            if args.augment:
                inputs, labels = vectorized_sudoku_augment(inputs, labels)

            batch = (inputs, labels)

            with autocast_ctx:
                carry, logits, q_logits = model(carry, batch)
                loss, loss_log = compute_loss(logits, q_logits, carry.current_labels, args)
                scaled_loss = loss / args.grad_accum

            if math.isnan(loss_log["loss"]):
                tqdm.write(f"\n[FATAL] NaN loss at step {global_step}!")
                for name, p in model.named_parameters():
                    if p.isnan().any():
                        tqdm.write(f"  NaN in param '{name}'")
                diag_file.close()
                sys.exit(1)

            scaler.scale(scaled_loss).backward()

            is_update_step = (batch_idx + 1) % args.grad_accum == 0
            if is_update_step:
                scaler.unscale_(optimizer)

                # ── [추가] per-layer gnorm 기록 (unscale_ 후, clip 전) ────
                layer_gnorms = _per_layer_gnorms(model, args.num_layers)
                record = {
                    "step":              global_step,
                    "carry_steps_mean":  round(carry.steps.float().mean().item(), 2),
                    "carry_steps_hist":  carry.steps.tolist(),
                    **layer_gnorms,
                }
                diag_file.write(json.dumps(record) + "\n")
                diag_file.flush()
                # ──────────────────────────────────────────────────────────

                last_grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                ).item()

                if global_step % args.log_interval == 0:
                    collect_grad_norms(model, global_step, last_grad_norm)

                lr_this_step = args.lr * cosine_schedule_with_warmup(
                    global_step, total_steps_approx, args.warmup_steps, args.lr_min_ratio
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_this_step

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # ── [추가] 지정 스텝 도달 시 종료 ─────────────────────────
                if global_step >= (ckpt.get("global_step", 0) if args.resume else 0) + args.diag_steps:
                    tqdm.write(f"\n[DIAG] {args.diag_steps} 스텝 완료. 종료.")
                    diag_file.close()
                    return
                # ──────────────────────────────────────────────────────────

            elapsed = time.perf_counter() - t0
            step_times.append(elapsed)
            step_samples.append(inputs.shape[0])
            throughput = sum(step_samples) / max(1e-9, sum(step_times))

            iter_loss  += loss_log["loss"]
            iter_steps += 1

            current_lr = optimizer.param_groups[0]["lr"]
            avg_steps  = carry.steps.float().mean().item()

            should_log = is_update_step and global_step % args.log_interval == 0
            if should_log:
                with torch.no_grad():
                    m = compute_metrics(logits, carry.current_labels)

            pbar.set_postfix(
                loss=f"{loss_log['loss']:.4f}",
                lr=f"{current_lr:.2e}",
                gnorm=f"{last_grad_norm:.3f}",
                samp_s=f"{throughput:.0f}",
                steps=f"{avg_steps:.1f}",
                refresh=False,
            )

            if should_log:
                elapsed_total = time.time() - total_start
                eta_sec = (elapsed_total / max(1, global_step)) * max(0, total_steps_approx - global_step)
                h, rem  = divmod(int(eta_sec), 3600)
                m_, s_  = divmod(rem, 60)
                mem_str = ""
                if device.type == "cuda":
                    gb = torch.cuda.memory_allocated(device) / 1024 ** 3
                    mem_str = f"  mem={gb:.2f}GB"
                b0 = getattr(model, "_orig_mod", model).blocks[0]
                tqdm.write(
                    f"[{global_step:6d}/~{total_steps_approx}]"
                    f"  loss={loss_log['loss']:.4f}"
                    f" (main={loss_log['main_loss']:.4f}"
                    f" halt={loss_log['halt_loss']:.4f})"
                    f"  tok={m['token_acc']*100:.2f}%"
                    f"  puzz={m['puzzle_acc']*100:.2f}%"
                    f"  lr={current_lr:.2e}"
                    f"  gnorm={last_grad_norm:.3f}"
                    f"  m={b0.last_m_norm:.3f} C={b0.last_C_norm:.3f}"
                    f"  {throughput:.0f}samp/s"
                    f"  ETA {h:02d}:{m_:02d}:{s_:02d}"
                    f"{mem_str}"
                )

            if is_update_step and global_step > 0 and global_step % args.eval_steps == 0:
                eval_res = evaluate(model, test_loader, device, args, autocast_ctx)
                model.train()
                tqdm.write(
                    f"\n{'─'*70}\n"
                    f"[EVAL  step={global_step}]\n"
                    f"  loss       = {eval_res['eval_loss']:.4f}\n"
                    f"  token_acc  = {eval_res['eval_token_acc']*100:.2f}%\n"
                    f"  puzzle_acc = {eval_res['eval_puzzle_acc']*100:.2f}%\n"
                    f"  avg_steps  = {eval_res['eval_avg_steps']:.1f}\n"
                    f"{'─'*70}"
                )
                puzzle_acc = eval_res["eval_puzzle_acc"]
                is_best    = puzzle_acc > best_puzzle_acc
                if is_best:
                    best_puzzle_acc = puzzle_acc
                state = {
                    "iter_id": iter_id, "global_step": global_step,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "best_puzzle_acc": best_puzzle_acc,
                    "eval_results": eval_res, "args": vars(args),
                }
                fname = f"step{global_step:06d}_pacc{puzzle_acc:.4f}.pt"
                ckpt_mgr.save(state, puzzle_acc, fname)
                tqdm.write(f"  [ckpt] saved → {args.checkpoint_dir}/{fname}")
                if is_best:
                    torch.save(state, Path(args.checkpoint_dir) / "best.pt")
                    tqdm.write(f"  [ckpt] ★ NEW BEST  puzzle_acc={puzzle_acc:.4f}")

        avg_loss = iter_loss / max(1, iter_steps)
        tqdm.write(
            f"\n{'═'*70}\n"
            f"Iter {iter_id+1}/{total_iters} 완료  avg_loss={avg_loss:.4f}  step={global_step}\n"
            f"{'═'*70}"
        )

    diag_file.close()
    tqdm.write(f"\n훈련 완료.  Best puzzle_acc = {best_puzzle_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap
    import sys as _sys

    # --diag_steps를 먼저 파싱한 뒤 sys.argv에서 제거해
    # train.py의 parse_args()가 모르는 인자로 인한 오류를 막는다
    _diag_parser = _ap.ArgumentParser(add_help=False)
    _diag_parser.add_argument("--diag_steps", type=int, default=100)
    _diag_args, _remaining = _diag_parser.parse_known_args()
    _sys.argv = [_sys.argv[0]] + _remaining   # --diag_steps 제거

    args = parse_args()
    args.diag_steps = _diag_args.diag_steps

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.checkpoint_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train(args)
