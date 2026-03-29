"""
train_urm.py — URM 훈련 스크립트

train.py (AMK-PD)와 동일한 인터페이스, 동일한 로그 포맷.
공정 비교를 위해 동일한 옵티마이저/스케줄러/데이터셋 사용.

사용 예시:
  uv run python train_urm.py --d_model 384 --num_heads 8 --num_layers 4 --batch_size 64 --loops 16 --H_cycles 2 --L_cycles 6
"""

import argparse
import csv
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from urm_model import URMModel, URMCarry
from dataset import create_dataloaders, vectorized_sudoku_augment


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train URM on Sudoku",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 데이터 ────────────────────────────────────────────────────────────
    g = p.add_argument_group("Data")
    g.add_argument("--data_dir",    default="data/sudoku-extreme-1k-aug-1000",
                   help="데이터셋 루트 경로")
    g.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader 워커 수 (Windows 는 0 권장)")
    g.add_argument("--augment",     action="store_true",
                   help="GPU-side 온라인 증강 활성화")

    # ── 모델 아키텍처 ─────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--d_model",         type=int,   default=512,  help="은닉 차원 d")
    g.add_argument("--num_heads",       type=int,   default=8,    help="어텐션 헤드 수")
    g.add_argument("--num_layers",      type=int,   default=4,    help="블록 수")
    g.add_argument("--loops",           type=int,   default=16,   help="외부 루프 최대 횟수")
    g.add_argument("--H_cycles",        type=int,   default=2,    help="중간 루프 (H-1회 no_grad + 1회 grad)")
    g.add_argument("--L_cycles",        type=int,   default=6,    help="내부 루프")
    g.add_argument("--expansion",       type=float, default=4,    help="MLP 팽창 비율")
    g.add_argument("--full_grad",       action="store_true",      help="H_cycles burn-in에서 no_grad 끄기")

    # ── 훈련 ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",        type=int,   default=10)
    g.add_argument("--batch_size",    type=int,   default=256)
    g.add_argument("--lr",            type=float, default=3e-4,  help="최대 학습률")
    g.add_argument("--weight_decay",  type=float, default=0.1)
    g.add_argument("--warmup_steps",  type=int,   default=500,   help="LR 선형 워밍업 스텝 수")
    g.add_argument("--grad_accum",    type=int,   default=1,     help="그래디언트 누적 스텝 수")
    g.add_argument("--max_grad_norm", type=float, default=1.0,   help="Gradient clipping")

    # ── Loss 계수 ─────────────────────────────────────────────────────────
    g = p.add_argument_group("Loss")
    g.add_argument("--halt_loss_coef", type=float, default=0.01,
                   help="Halting BCE 손실 가중치")

    # ── 속도 ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Speed")
    g.add_argument("--compile", action="store_true",
                   help="torch.compile 활성화")
    g.add_argument("--dtype",   choices=["float32", "bfloat16", "float16"], default="float32",
                   help="Mixed precision dtype")

    # ── 로깅 & 체크포인팅 ─────────────────────────────────────────────────
    g = p.add_argument_group("Logging")
    g.add_argument("--checkpoint_dir", default="checkpoints_urm")
    g.add_argument("--log_interval",   type=int, default=50,  help="상세 로그 출력 주기")
    g.add_argument("--eval_interval",  type=int, default=5000, help="테스트셋 평가 주기")
    g.add_argument("--save_top_k",     type=int, default=3,   help="상위 K개 체크포인트 보존")
    g.add_argument("--resume",         type=str, default=None, help="재개할 체크포인트 경로")

    g = p.add_argument_group("Debug")
    g.add_argument("--debug_nan",      action="store_true")

    g.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 옵티마이저 & 스케줄러 (train.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer_and_scheduler(
    model:       nn.Module,
    args:        argparse.Namespace,
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(k in name for k in ("bias",)):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = torch.cuda.is_available()
    optim_kwargs: dict = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        param_groups,
        lr     = args.lr,
        betas  = (0.9, 0.95),
        eps    = 1e-8,
        **optim_kwargs,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Loss (train.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    logits:            torch.Tensor,
    q_logits:          Tuple[torch.Tensor, torch.Tensor],
    labels:            torch.Tensor,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, N = labels.shape
    V    = logits.shape[-1]
    q_halt_logits, q_continue_logits = q_logits

    main_loss = F.cross_entropy(
        logits.reshape(B * N, V),
        labels.reshape(B * N),
        ignore_index=0,
    )
    total = main_loss

    halt_coef = getattr(args, 'halt_loss_coef', 0.01)
    q_loss = torch.zeros(1, device=total.device).squeeze()
    if halt_coef > 0 and torch.isfinite(q_halt_logits).all():
        with torch.no_grad():
            mask = labels != 0
            preds = logits.argmax(dim=-1)
            acc = ((preds == labels) & mask).float().sum(-1) / mask.float().sum(-1).clamp(min=1)
            q_target = 2 * acc - 1

        q_margin = q_halt_logits - q_continue_logits
        q_loss = F.mse_loss(q_margin, q_target)
        total = total + halt_coef * q_loss

    log = {
        "loss":      total.item(),
        "main_loss": main_loss.item(),
        "halt_loss": q_loss.item(),
    }
    return total, log


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (train.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, object]:
    mask    = labels != 0
    n_valid = mask.float().sum()
    preds   = logits.argmax(dim=-1)
    correct = (preds == labels) & mask
    return {
        "token_acc":  (correct.float().sum() / n_valid).item(),
        "puzzle_acc": (correct | ~mask).all(dim=1).float().mean().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (train.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:        nn.Module,
    loader:       torch.utils.data.DataLoader,
    device:       torch.device,
    args:         argparse.Namespace,
    autocast_ctx,
) -> Dict[str, float]:
    model.eval()

    total_loss       = 0.0
    token_correct    = 0.0
    token_total      = 0.0
    puzzle_correct   = 0.0
    puzzle_total     = 0
    total_steps_sum  = 0.0

    for inputs, labels in tqdm(loader, desc="  [eval]", leave=False, dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = inputs.shape[0]
        seq_len = inputs.shape[1]

        carry = model.initial_carry(B, seq_len, device)
        batch = (inputs, labels)

        for _ in range(args.loops):
            with autocast_ctx:
                carry, logits, q_logits = model(carry, batch)

        _, loss_log = compute_loss(logits, q_logits, carry.current_labels, args)
        total_loss += loss_log["loss"]

        m    = compute_metrics(logits, carry.current_labels)
        mask = carry.current_labels != 0

        token_correct  += m["token_acc"]  * mask.float().sum().item()
        token_total    += mask.float().sum().item()
        puzzle_correct += m["puzzle_acc"] * B
        puzzle_total   += B
        total_steps_sum += carry.steps.float().sum().item()

    n_batches = max(1, len(loader))
    result = {
        "eval_loss":       total_loss / n_batches,
        "eval_token_acc":  token_correct / max(1.0, token_total),
        "eval_puzzle_acc": puzzle_correct / max(1, puzzle_total),
        "eval_avg_steps":  total_steps_sum / max(1, puzzle_total),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 관리 (train.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, ckpt_dir: str, save_top_k: int = 3):
        self.ckpt_dir   = Path(ckpt_dir)
        self.save_top_k = save_top_k
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._saved: List[Tuple[float, Path]] = []

    def save(self, state: dict, score: float, filename: str) -> None:
        path = self.ckpt_dir / filename
        torch.save(state, path)
        self._saved.append((score, path))
        self._saved.sort(key=lambda x: -x[0])
        while len(self._saved) > self.save_top_k:
            _, old_path = self._saved.pop()
            if old_path.exists() and old_path.name != "best.pt":
                old_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# 훈련 메인
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:

    # ── 재현성 ────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device : {device}")

    # ── 데이터 ────────────────────────────────────────────────────────────
    train_loader, test_loader, meta = create_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
    )
    print(
        f"Dataset: train={meta['train_size']:,}  test={meta['test_size']:,}"
        f"  vocab={meta['vocab_size']}  seq_len={meta['seq_len']}"
    )

    # ── 모델 ──────────────────────────────────────────────────────────────
    model = URMModel(
        vocab_size  = meta["vocab_size"],
        d_model     = args.d_model,
        num_heads   = args.num_heads,
        num_layers  = args.num_layers,
        loops       = args.loops,
        H_cycles    = args.H_cycles,
        L_cycles    = args.L_cycles,
        expansion   = args.expansion,
    ).to(device)
    if args.full_grad:
        model.burn_in_no_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params:,}")

    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile : enabled")
        except Exception as e:
            print(f"torch.compile : skipped ({e})")

    # ── NaN 디버깅 훅 ─────────────────────────────────────────────────────
    if args.debug_nan:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly Detection Enabled for NaN debugging.")

        def nan_forward_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"\n[DEBUG] NaN/Inf in forward output of {module.__class__.__name__}")
            elif isinstance(output, tuple):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor) and (torch.isnan(out).any() or torch.isinf(out).any()):
                        print(f"\n[DEBUG] NaN/Inf in forward output[{i}] of {module.__class__.__name__}")

        for name, m in model.named_modules():
            m.register_forward_hook(nan_forward_hook)

    # ── Mixed precision ───────────────────────────────────────────────────
    _dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype  = _dtype_map[args.dtype]
    use_amp    = amp_dtype != torch.float32
    if use_amp and amp_dtype == torch.float16 and device.type != "cuda":
        print("float16 is CUDA-only — falling back to float32")
        use_amp = False

    autocast_ctx = torch.autocast(
        device_type = device.type,
        dtype       = amp_dtype,
        enabled     = use_amp,
    )
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if use_amp:
        print(f"Mixed precision : {args.dtype}")

    # ── 옵티마이저 & 스케줄러 ─────────────────────────────────────────────
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps     = steps_per_epoch * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, total_steps)
    print(
        f"Optimizer: AdamW  lr={args.lr}  wd={args.weight_decay}"
        f"  warmup={args.warmup_steps}  total_steps={total_steps:,}"
    )

    # ── 체크포인트 재개 ───────────────────────────────────────────────────
    start_epoch     = 0
    global_step     = 0
    best_puzzle_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch     = ckpt["epoch"] + 1
        global_step     = ckpt["global_step"]
        best_puzzle_acc = ckpt.get("best_puzzle_acc", 0.0)
        print(f"Resumed : {args.resume}  epoch={start_epoch}  step={global_step}")

    ckpt_mgr    = CheckpointManager(args.checkpoint_dir, args.save_top_k)

    # ── Gradient metric 로깅 설정 ──────────────────────────────────────────
    grad_log_path = Path(args.checkpoint_dir) / "grad_metrics.csv"
    grad_log_fields = [
        "step", "gnorm",
        # layer 0
        "l0_qkv_proj", "l0_o_proj",
        "l0_gate_up_proj", "l0_down_proj", "l0_dwconv",
        # layer 1
        "l1_qkv_proj", "l1_o_proj",
        "l1_gate_up_proj", "l1_down_proj", "l1_dwconv",
        # global components
        "embed_tokens", "lm_head", "q_head",
        # 누락 체크
        "other",
    ]
    with open(grad_log_path, "w", newline="") as f:
        csv.writer(f).writerow(grad_log_fields)

    # 각 필드가 커버하는 prefix 목록 (other 계산용)
    _known_prefixes = [
        "layers.0.self_attn.qkv_proj", "layers.0.self_attn.o_proj",
        "layers.0.mlp.gate_up_proj", "layers.0.mlp.down_proj", "layers.0.mlp.dwconv",
        "layers.1.self_attn.qkv_proj", "layers.1.self_attn.o_proj",
        "layers.1.mlp.gate_up_proj", "layers.1.mlp.down_proj", "layers.1.mlp.dwconv",
        "embed_tokens", "lm_head", "q_head",
    ]

    def collect_grad_norms(model, step, gnorm):
        """컴포넌트별 gradient norm 수집 → CSV 1행"""
        row = {"step": step, "gnorm": f"{gnorm:.6f}"}

        param_gnorms = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_gnorms[name] = p.grad.norm().item()
            else:
                param_gnorms[name] = 0.0

        def sum_gnorm(*prefixes):
            total = 0.0
            for name, val in param_gnorms.items():
                if any(name.startswith(px) for px in prefixes):
                    total += val ** 2
            return total ** 0.5

        # layer 0
        row["l0_qkv_proj"]     = f"{sum_gnorm('layers.0.self_attn.qkv_proj'):.6f}"
        row["l0_o_proj"]       = f"{sum_gnorm('layers.0.self_attn.o_proj'):.6f}"
        row["l0_gate_up_proj"] = f"{sum_gnorm('layers.0.mlp.gate_up_proj'):.6f}"
        row["l0_down_proj"]    = f"{sum_gnorm('layers.0.mlp.down_proj'):.6f}"
        row["l0_dwconv"]       = f"{sum_gnorm('layers.0.mlp.dwconv'):.6f}"
        # layer 1
        row["l1_qkv_proj"]     = f"{sum_gnorm('layers.1.self_attn.qkv_proj'):.6f}"
        row["l1_o_proj"]       = f"{sum_gnorm('layers.1.self_attn.o_proj'):.6f}"
        row["l1_gate_up_proj"] = f"{sum_gnorm('layers.1.mlp.gate_up_proj'):.6f}"
        row["l1_down_proj"]    = f"{sum_gnorm('layers.1.mlp.down_proj'):.6f}"
        row["l1_dwconv"]       = f"{sum_gnorm('layers.1.mlp.dwconv'):.6f}"
        # global
        row["embed_tokens"]    = f"{sum_gnorm('embed_tokens'):.6f}"
        row["lm_head"]         = f"{sum_gnorm('lm_head'):.6f}"
        row["q_head"]          = f"{sum_gnorm('q_head'):.6f}"
        # 누락 파라미터 체크
        other_sq = 0.0
        for name, val in param_gnorms.items():
            if not any(name.startswith(px) for px in _known_prefixes):
                other_sq += val ** 2
        row["other"] = f"{other_sq ** 0.5:.6f}"

        with open(grad_log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=grad_log_fields)
            w.writerow(row)

    total_start = time.time()

    step_times:   deque = deque(maxlen=100)
    step_samples: deque = deque(maxlen=100)
    last_grad_norm = 0.0

    seq_len = meta["seq_len"]

    # ── Epoch 루프 ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()

        carry = model.initial_carry(args.batch_size, seq_len, device)

        epoch_loss  = 0.0
        epoch_steps = 0

        pbar = tqdm(
            enumerate(train_loader),
            total        = len(train_loader),
            desc         = f"Epoch {epoch + 1:02d}/{args.epochs}",
            dynamic_ncols = True,
            leave        = True,
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

            # ── Forward ───────────────────────────────────────────────
            with autocast_ctx:
                carry, logits, q_logits = model(carry, batch)
                loss, loss_log = compute_loss(
                    logits, q_logits, carry.current_labels, args
                )
                scaled_loss = loss / args.grad_accum

            # ── NaN 감지 ──────────────────────────────────────────────
            if math.isnan(loss_log["loss"]):
                tqdm.write(f"\n[FATAL] NaN loss detected at global step {global_step}!")
                import sys
                sys.exit(1)

            # ── Backward ──────────────────────────────────────────────
            scaler.scale(scaled_loss).backward()

            # ── 옵티마이저 스텝 ───────────────────────────────────────
            is_update_step = (batch_idx + 1) % args.grad_accum == 0
            if is_update_step:
                scaler.unscale_(optimizer)
                last_grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                ).item()
                # gradient metric 기록 (clip 후, zero_grad 전) — log_interval 주기에만
                if global_step % args.log_interval == 0:
                    collect_grad_norms(model, global_step, last_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # ── 처리량 측정 ───────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            step_times.append(elapsed)
            step_samples.append(inputs.shape[0])
            throughput = sum(step_samples) / max(1e-9, sum(step_times))

            # ── 배치 지표 ─────────────────────────────────────────────
            with torch.no_grad():
                m = compute_metrics(logits, carry.current_labels)

            epoch_loss  += loss_log["loss"]
            epoch_steps += 1

            # ── tqdm postfix ──────────────────────────────────────────
            current_lr = optimizer.param_groups[0]["lr"]
            avg_steps  = carry.steps.float().mean().item()

            pbar.set_postfix(
                loss     = f"{loss_log['loss']:.4f}",
                tok_acc  = f"{m['token_acc'] * 100:.1f}%",
                puzz_acc = f"{m['puzzle_acc'] * 100:.1f}%",
                lr       = f"{current_lr:.2e}",
                gnorm    = f"{last_grad_norm:.3f}",
                samp_s   = f"{throughput:.0f}",
                steps    = f"{avg_steps:.1f}",
                refresh  = False,
            )

            # ── 상세 로그 ─────────────────────────────────────────────
            if is_update_step and global_step % args.log_interval == 0:
                elapsed_total = time.time() - total_start
                eta_sec       = (elapsed_total / max(1, global_step)) * max(0, total_steps - global_step)
                h, rem        = divmod(int(eta_sec), 3600)
                m_, s_        = divmod(rem, 60)

                mem_str = ""
                if device.type == "cuda":
                    gb = torch.cuda.memory_allocated(device) / 1024 ** 3
                    mem_str = f"  mem={gb:.2f}GB"

                tqdm.write(
                    f"[{global_step:6d}/{total_steps}]"
                    f"  loss={loss_log['loss']:.4f}"
                    f" (main={loss_log['main_loss']:.4f}"
                    f" halt={loss_log['halt_loss']:.4f})"
                    f"  tok={m['token_acc'] * 100:.2f}%"
                    f"  puzz={m['puzzle_acc'] * 100:.2f}%"
                    f"  avg_steps={avg_steps:.1f}"
                    f"  lr={current_lr:.2e}"
                    f"  gnorm={last_grad_norm:.3f}"
                    f"  {throughput:.0f}samp/s"
                    f"  ETA {h:02d}:{m_:02d}:{s_:02d}"
                    f"{mem_str}"
                )

            # ── 평가 & 체크포인팅 ─────────────────────────────────────
            if is_update_step and global_step > 0 and global_step % args.eval_interval == 0:
                eval_res = evaluate(model, test_loader, device, args, autocast_ctx)
                model.train()
                carry = model.initial_carry(args.batch_size, seq_len, device)

                tqdm.write(
                    f"\n{'─' * 70}\n"
                    f"[EVAL  step={global_step}]\n"
                    f"  loss       = {eval_res['eval_loss']:.4f}\n"
                    f"  token_acc  = {eval_res['eval_token_acc']  * 100:.2f}%\n"
                    f"  puzzle_acc = {eval_res['eval_puzzle_acc'] * 100:.2f}%\n"
                    f"  avg_steps  = {eval_res['eval_avg_steps']:.1f}\n"
                    f"{'─' * 70}\n"
                )

                puzzle_acc = eval_res["eval_puzzle_acc"]
                is_best    = puzzle_acc > best_puzzle_acc
                if is_best:
                    best_puzzle_acc = puzzle_acc

                state = {
                    "epoch":           epoch,
                    "global_step":     global_step,
                    "model":           model.state_dict(),
                    "optimizer":       optimizer.state_dict(),
                    "scheduler":       scheduler.state_dict(),
                    "scaler":          scaler.state_dict(),
                    "best_puzzle_acc": best_puzzle_acc,
                    "eval_results":    eval_res,
                    "args":            vars(args),
                }

                fname = f"step{global_step:06d}_pacc{puzzle_acc:.4f}.pt"
                ckpt_mgr.save(state, puzzle_acc, fname)
                tqdm.write(f"  [ckpt] saved → {args.checkpoint_dir}/{fname}")

                if is_best:
                    best_path = Path(args.checkpoint_dir) / "best.pt"
                    torch.save(state, best_path)
                    tqdm.write(
                        f"  [ckpt] ★ NEW BEST  puzzle_acc={puzzle_acc:.4f} → {best_path}"
                    )

        # ── 에포크 요약 ───────────────────────────────────────────────────
        avg_loss    = epoch_loss / max(1, epoch_steps)
        elapsed_tot = time.time() - total_start
        h, rem      = divmod(int(elapsed_tot), 3600)
        m_, s_      = divmod(rem, 60)

        tqdm.write(
            f"\nEpoch {epoch + 1:02d}/{args.epochs} 완료"
            f"  avg_loss={avg_loss:.4f}"
            f"  best_puzzle_acc={best_puzzle_acc:.4f}"
            f"  elapsed={h:02d}:{m_:02d}:{s_:02d}\n"
        )

    tqdm.write(f"\n훈련 완료.  Best puzzle_acc = {best_puzzle_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.checkpoint_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved → {args.checkpoint_dir}/config.json")
    print()

    train(args)
