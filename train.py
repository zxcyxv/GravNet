"""
train.py — AMK-PD 훈련 스크립트 (URM-compatible)

URM 원본과 동일한 구조:
  - Group-based IterableDataset (매 epoch 1000 groups 순회, group당 1 sample)
  - epochs_per_iter = eval_interval (2000 epochs를 하나의 iter로 묶음)
  - total_iters = epochs / eval_interval
  - Cosine LR with min_ratio=1.0 (= warmup 후 상수)
  - weight_decay=1.0, lr=1e-4
  - 배치당 1 forward (carry 지속)

사용 예시:
  uv run python train.py
  uv run python train.py --epochs 50000 --eval_interval 2000 --batch_size 128
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

from amkpd_model import AMKPDModel, AMKPDCarry
from dataset import create_dataloaders, vectorized_sudoku_augment, SudokuDataset
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train AMK-PD on Sudoku (URM-compatible)",
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
    g.add_argument("--d_model",         type=int,   default=128,  help="은닉 차원 d")
    g.add_argument("--num_heads",       type=int,   default=8,    help="멀티헤드 어텐션 헤드 수")
    g.add_argument("--num_layers",      type=int,   default=4,    help="K: L_cycle당 블록 수")
    g.add_argument("--loops",           type=int,   default=16,   help="외부 루프 최대 횟수")
    g.add_argument("--H_cycles",        type=int,   default=2,    help="중간 루프 (H-1회 no_grad + 1회 grad)")
    g.add_argument("--L_cycles",        type=int,   default=6,    help="내부 루프 (X 주입 + 블록 통과)")
    g.add_argument("--dt",              type=float, default=1.0,  help="Euler 스텝 초기값")
    g.add_argument("--expansion_ratio", type=int,   default=4,    help="ConvSwiGLU 팽창 비율")
    g.add_argument("--conv_kernel",     type=int,   default=2,    help="Depthwise conv 커널 크기")
    g.add_argument("--kernel_power",    type=int,   default=4,    help="인력 행렬 다항식 거듭제곱 차수")
    g.add_argument("--full_grad",       action="store_true",      help="H_cycles burn-in에서 no_grad 끄기")

    # ── 훈련 (URM defaults) ──────────────────────────────────────────────
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",         type=int,   default=50000,
                   help="URM epochs (1 epoch = total_groups / batch_size batches)")
    g.add_argument("--eval_interval",  type=int,   default=2000,
                   help="평가 주기 (URM epochs 단위). epochs_per_iter와 동일.")
    g.add_argument("--batch_size",     type=int,   default=128,
                   help="Global batch size (URM: 128)")
    g.add_argument("--lr",             type=float, default=1e-4,
                   help="최대 학습률 (URM: 1e-4)")
    g.add_argument("--weight_decay",   type=float, default=1.0,
                   help="Weight decay (URM: 1.0)")
    g.add_argument("--warmup_steps",   type=int,   default=2000,
                   help="LR 선형 워밍업 스텝 수 (URM: 2000)")
    g.add_argument("--lr_min_ratio",   type=float, default=1.0,
                   help="Cosine decay 최소 비율 (1.0 = 상수 LR, URM: 1.0)")
    g.add_argument("--grad_accum",     type=int,   default=1,
                   help="그래디언트 누적 스텝 수")
    g.add_argument("--max_grad_norm",  type=float, default=1.0,
                   help="Gradient clipping")

    # ── Loss 계수 ─────────────────────────────────────────────────────────
    g = p.add_argument_group("Loss")
    g.add_argument("--halt_loss_coef", type=float, default=0.01,
                   help="Halting loss 가중치")

    # ── 속도 ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Speed")
    g.add_argument("--no_compile", action="store_true",
                   help="torch.compile 비활성화")
    g.add_argument("--dtype",   choices=["float32", "bfloat16", "float16"], default="bfloat16",
                   help="Mixed precision dtype")

    # ── 로깅 & 체크포인팅 ─────────────────────────────────────────────────
    g = p.add_argument_group("Logging")
    g.add_argument("--checkpoint_dir", default="checkpoints")
    g.add_argument("--log_interval",   type=int, default=50,
                   help="상세 로그 출력 주기 (옵티마이저 스텝)")
    g.add_argument("--eval_steps",     type=int, default=5000,
                   help="스텝 단위 평가 주기 (iter 중간에도 eval)")
    g.add_argument("--save_top_k",     type=int, default=3,
                   help="상위 K개 체크포인트 보존")
    g.add_argument("--resume",         type=str, default=None,
                   help="재개할 체크포인트 경로")

    g = p.add_argument_group("Debug")
    g.add_argument("--debug_nan",      action="store_true",
                   help="NaN/Inf 감지 훅 활성화")

    g.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule (URM-compatible: cosine with min_ratio)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float = 1.0,
) -> float:
    """
    URM과 동일한 LR 스케줄.
    min_ratio=1.0이면 warmup 후 상수 LR (decay 없음).
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine_decay


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(k in name for k in ("bias", "norm")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = torch.cuda.is_available()
    optim_kwargs = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        **optim_kwargs,
    )
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    logits: torch.Tensor,
    q_logits: Tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, N = labels.shape
    V = logits.shape[-1]
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
        "loss": total.item(),
        "main_loss": main_loss.item(),
        "halt_loss": q_loss.item(),
    }
    return total, log


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, object]:
    mask = labels != 0
    n_valid = mask.float().sum()
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & mask
    return {
        "token_acc": (correct.float().sum() / n_valid).item(),
        "puzzle_acc": (correct | ~mask).all(dim=1).float().mean().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    autocast_ctx,
    max_samples: int = 512,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    token_correct = 0.0
    token_total = 0.0
    puzzle_correct = 0.0
    puzzle_total = 0
    total_steps_sum = 0.0
    n_batches = 0

    # 큰 배치로 한 번에 처리 (test set이 작으므로)
    all_inputs, all_labels = [], []
    count = 0
    for inputs, labels in loader:
        all_inputs.append(inputs)
        all_labels.append(labels)
        count += inputs.shape[0]
        if count >= max_samples:
            break

    all_inputs = torch.cat(all_inputs, dim=0)[:max_samples].to(device)
    all_labels = torch.cat(all_labels, dim=0)[:max_samples].to(device)

    # 배치 분할 (GPU 메모리에 맞게)
    eval_bs = min(256, all_inputs.shape[0])
    for start in range(0, all_inputs.shape[0], eval_bs):
        end = min(start + eval_bs, all_inputs.shape[0])
        inputs = all_inputs[start:end]
        labels = all_labels[start:end]
        B = inputs.shape[0]
        seq_len = inputs.shape[1]

        carry = model.initial_carry(B, seq_len, device)
        batch = (inputs, labels)

        for _ in range(args.loops):
            with autocast_ctx:
                carry, logits, q_logits = model(carry, batch)

        _, loss_log = compute_loss(logits, q_logits, carry.current_labels, args)
        total_loss += loss_log["loss"]

        m = compute_metrics(logits, carry.current_labels)
        mask = carry.current_labels != 0

        token_correct += m["token_acc"] * mask.float().sum().item()
        token_total += mask.float().sum().item()
        puzzle_correct += m["puzzle_acc"] * B
        puzzle_total += B
        total_steps_sum += carry.steps.float().sum().item()
        n_batches += 1

    n_batches = max(1, n_batches)
    return {
        "eval_loss": total_loss / n_batches,
        "eval_token_acc": token_correct / max(1.0, token_total),
        "eval_puzzle_acc": puzzle_correct / max(1, puzzle_total),
        "eval_avg_steps": total_steps_sum / max(1, puzzle_total),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 관리
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, ckpt_dir: str, save_top_k: int = 3):
        self.ckpt_dir = Path(ckpt_dir)
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
    torch.set_float32_matmul_precision("high")
    print(f"Device : {device}")

    # ── URM epoch/iter 구조 ────────────────────────────────────────────────
    # total_iters = epochs / eval_interval
    # 각 iter = eval_interval epochs 분량의 데이터
    # 각 iter 끝에 eval 수행
    assert args.epochs % args.eval_interval == 0, \
        f"epochs({args.epochs}) must be divisible by eval_interval({args.eval_interval})"
    total_iters = args.epochs // args.eval_interval
    epochs_per_iter = args.eval_interval

    print(f"URM structure: {args.epochs} epochs = {total_iters} iters × {epochs_per_iter} epochs/iter")

    # ── 데이터 (URM Group-based) ──────────────────────────────────────────
    train_loader, test_loader, meta = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs_per_iter=epochs_per_iter,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # total_steps 계산: URM 방식
    # 1 epoch = total_groups / batch_size batches (drop_last)
    batches_per_epoch = meta["total_groups"] // args.batch_size
    total_steps_approx = batches_per_epoch * args.epochs // args.grad_accum
    print(
        f"Dataset: total_groups={meta['total_groups']:,}  "
        f"total_examples={meta['train_size']:,}  "
        f"test={meta['test_size']:,}"
    )
    print(
        f"Training: {batches_per_epoch} batches/epoch × {args.epochs} epochs "
        f"= ~{total_steps_approx:,} optimizer steps"
    )

    # ── 모델 ──────────────────────────────────────────────────────────────
    model = AMKPDModel(
        vocab_size=meta["vocab_size"],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        loops=args.loops,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        dt=args.dt,
        kernel_power=args.kernel_power,
        expansion_ratio=args.expansion_ratio,
        conv_kernel_size=args.conv_kernel,
    ).to(device)
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

    # ── NaN 디버깅 ─────────────────────────────────────────────────────────
    if args.debug_nan:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly Detection Enabled.")

    # ── Mixed precision ───────────────────────────────────────────────────
    _dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = _dtype_map[args.dtype]
    use_amp = amp_dtype != torch.float32
    if use_amp and amp_dtype == torch.float16 and device.type != "cuda":
        print("float16 is CUDA-only — falling back to float32")
        use_amp = False

    autocast_ctx = torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    )
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if use_amp:
        print(f"Mixed precision : {args.dtype}")

    # ── 옵티마이저 ────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, args)
    print(
        f"Optimizer: AdamW  lr={args.lr}  wd={args.weight_decay}"
        f"  warmup={args.warmup_steps}  lr_min_ratio={args.lr_min_ratio}"
        f"  ~total_steps={total_steps_approx:,}"
    )

    # ── 체크포인트 재개 ───────────────────────────────────────────────────
    start_iter = 0
    global_step = 0
    best_puzzle_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_iter = ckpt.get("iter_id", 0) + 1
        global_step = ckpt["global_step"]
        best_puzzle_acc = ckpt.get("best_puzzle_acc", 0.0)
        print(f"Resumed : {args.resume}  iter={start_iter}  step={global_step}")

    ckpt_mgr = CheckpointManager(args.checkpoint_dir, args.save_top_k)

    # ── Gradient metric 로깅 ──────────────────────────────────────────────
    grad_log_path = Path(args.checkpoint_dir) / "grad_metrics.csv"
    grad_log_fields = [
        "step", "gnorm",
        "b0_W_QKV", "b0_W_O_aux",
        "b0_W_up", "b0_W_down", "b0_dw_conv",
        "b0_norm1", "b0_norm2",
        "embedding", "pos_emb", "input_norm",
        "final_norm", "lm_head", "halt_head",
        "b0_m_norm", "b0_C_norm",
        "other",
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

        row["b0_W_QKV"] = f"{sum_gnorm('blocks.0.W_QKV'):.6f}"
        row["b0_W_O_aux"] = f"{sum_gnorm('blocks.0.W_O_aux'):.6f}"
        row["b0_W_up"] = f"{sum_gnorm('blocks.0.W_up'):.6f}"
        row["b0_W_down"] = f"{sum_gnorm('blocks.0.W_down'):.6f}"
        row["b0_dw_conv"] = f"{sum_gnorm('blocks.0.dw_conv'):.6f}"
        row["b0_norm1"] = f"{sum_gnorm('blocks.0.norm1'):.6f}"
        row["b0_norm2"] = f"{sum_gnorm('blocks.0.norm2'):.6f}"
        row["embedding"] = f"{sum_gnorm('embedding'):.6f}"
        row["pos_emb"] = f"{sum_gnorm('pos_emb'):.6f}"
        row["input_norm"] = f"{sum_gnorm('input_norm'):.6f}"
        row["final_norm"] = f"{sum_gnorm('final_norm'):.6f}"
        row["lm_head"] = f"{sum_gnorm('lm_head'):.6f}"
        row["halt_head"] = f"{sum_gnorm('halt_head'):.6f}"
        b0 = model_ref.blocks[0]
        row["b0_m_norm"] = f"{b0.last_m_norm:.6f}"
        row["b0_C_norm"] = f"{b0.last_C_norm:.6f}"
        other_sq = 0.0
        for name, val in param_gnorms.items():
            if not any(name.startswith(px) for px in _known_prefixes):
                other_sq += val ** 2
        row["other"] = f"{other_sq ** 0.5:.6f}"

        with open(grad_log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=grad_log_fields)
            w.writerow(row)

    total_start = time.time()
    step_times: deque = deque(maxlen=100)
    step_samples: deque = deque(maxlen=100)
    last_grad_norm = 0.0
    seq_len = meta["seq_len"]

    # ── 메인 훈련 루프 (URM 구조) ──────────────────────────────────────────
    # outer loop: total_iters 회 반복
    # inner loop: train_loader가 epochs_per_iter epochs 분량의 배치를 yield
    # 각 outer iter 끝에 eval

    # carry는 iter 시작 시 초기화 (URM: carry는 train_state에 1번 초기화 후 계속 사용)
    carry = model.initial_carry(args.batch_size, seq_len, device)

    for iter_id in range(start_iter, total_iters):
        model.train()
        optimizer.zero_grad()

        epoch_start = iter_id * epochs_per_iter
        epoch_end = epoch_start + epochs_per_iter

        iter_loss = 0.0
        iter_steps = 0

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

            # 배치 크기 변동 시 carry 재초기화
            if inputs.shape[0] != carry.current_hidden.shape[0]:
                carry = model.initial_carry(inputs.shape[0], seq_len, device)

            if args.augment:
                inputs, labels = vectorized_sudoku_augment(inputs, labels)

            batch = (inputs, labels)

            # ── Forward: 배치당 1회 (carry 지속) ─────────────────────────
            with autocast_ctx:
                carry, logits, q_logits = model(carry, batch)
                loss, loss_log = compute_loss(logits, q_logits, carry.current_labels, args)
                scaled_loss = loss / args.grad_accum

            if math.isnan(loss_log["loss"]):
                tqdm.write(f"\n[FATAL] NaN loss at step {global_step}!")
                for name, p in model.named_parameters():
                    if p.isnan().any():
                        tqdm.write(f"  NaN in param '{name}'")
                import sys
                sys.exit(1)

            # ── Backward ─────────────────────────────────────────────────
            scaler.scale(scaled_loss).backward()

            # ── Optimizer step ───────────────────────────────────────────
            is_update_step = (batch_idx + 1) % args.grad_accum == 0
            if is_update_step:
                scaler.unscale_(optimizer)
                last_grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                ).item()

                if global_step % args.log_interval == 0:
                    collect_grad_norms(model, global_step, last_grad_norm)

                # URM-style LR: cosine with min_ratio (applied manually)
                lr_this_step = args.lr * cosine_schedule_with_warmup(
                    global_step, total_steps_approx, args.warmup_steps, args.lr_min_ratio
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_this_step

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            # ── 처리량 측정 ──────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            step_times.append(elapsed)
            step_samples.append(inputs.shape[0])
            throughput = sum(step_samples) / max(1e-9, sum(step_times))

            iter_loss += loss_log["loss"]
            iter_steps += 1

            current_lr = optimizer.param_groups[0]["lr"]
            avg_steps = carry.steps.float().mean().item()

            # 매 배치 metrics 계산 생략 — 로그 주기에만 계산
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

            # ── 상세 로그 ────────────────────────────────────────────────
            if should_log:
                elapsed_total = time.time() - total_start
                eta_sec = (elapsed_total / max(1, global_step)) * max(0, total_steps_approx - global_step)
                h, rem = divmod(int(eta_sec), 3600)
                m_, s_ = divmod(rem, 60)

                mem_str = ""
                if device.type == "cuda":
                    gb = torch.cuda.memory_allocated(device) / 1024 ** 3
                    mem_str = f"  mem={gb:.2f}GB"

                b0 = model.blocks[0]
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

            # ── 스텝 단위 eval & 체크포인팅 ─────────────────────────────────
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
                is_best = puzzle_acc > best_puzzle_acc
                if is_best:
                    best_puzzle_acc = puzzle_acc

                state = {
                    "iter_id": iter_id,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_puzzle_acc": best_puzzle_acc,
                    "eval_results": eval_res,
                    "args": vars(args),
                }

                fname = f"step{global_step:06d}_pacc{puzzle_acc:.4f}.pt"
                ckpt_mgr.save(state, puzzle_acc, fname)
                tqdm.write(f"  [ckpt] saved → {args.checkpoint_dir}/{fname}")

                if is_best:
                    best_path = Path(args.checkpoint_dir) / "best.pt"
                    torch.save(state, best_path)
                    tqdm.write(f"  [ckpt] ★ NEW BEST  puzzle_acc={puzzle_acc:.4f}")

        # ── Iter 끝: 요약 ────────────────────────────────────────────────────
        avg_loss = iter_loss / max(1, iter_steps)
        current_epoch = (iter_id + 1) * epochs_per_iter
        tqdm.write(
            f"\n{'═'*70}\n"
            f"Iter {iter_id+1}/{total_iters} 완료 (epoch {current_epoch}/{args.epochs})"
            f"  avg_loss={avg_loss:.4f}  step={global_step}\n"
            f"{'═'*70}"
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
