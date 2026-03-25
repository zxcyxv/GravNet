"""
train.py — AMK-PD 훈련 스크립트

속도 최적화:
  - torch.compile (reduce-overhead 모드, 선택적)
  - Mixed precision: bfloat16 / float16 + GradScaler
  - Fused AdamW (CUDA 한정)
  - Gradient accumulation
  - cudnn.benchmark
  - GPU-side 벡터화 증강 (for-loop 없음)
  - pin_memory + persistent_workers DataLoader

모니터링:
  - 스텝별: loss(main/aux/act), token_acc, puzzle_acc, grad_norm, lr, 처리량, ETA
  - 평가 시: 전체 test 셋 + 루프별 puzzle_acc 분해

사용 예시:
  uv run python train.py --d_model 128 --batch_size 256 --epochs 10
  uv run python train.py --compile --dtype bfloat16 --augment --grad_accum 2
"""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from amkpd_model import AMKPDModel
from dataset import create_dataloaders, vectorized_sudoku_augment


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train AMK-PD on Sudoku",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 데이터 ────────────────────────────────────────────────────────────
    g = p.add_argument_group("Data")
    g.add_argument("--data_dir",    default="data/sudoku-extreme-1k-aug-1000",
                   help="데이터셋 루트 경로")
    g.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader 워커 수 (Windows 는 0 권장)")
    g.add_argument("--augment",     action="store_true",
                   help="GPU-side 온라인 증강 활성화 (이미 증강된 train 셋에 추가 다양성)")

    # ── 모델 아키텍처 ─────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--d_model",         type=int,   default=128,  help="은닉 차원 d")
    g.add_argument("--d_spectral",      type=int,   default=64,   help="스펙트럼 차원 D")
    g.add_argument("--num_layers",      type=int,   default=3,    help="K: 루프당 블록 수")
    g.add_argument("--max_loops",       type=int,   default=6,    help="M: 최대 거시 루프")
    g.add_argument("--trunc_loops",     type=int,   default=3,    help="N_trunc: no_grad 루프 수")
    g.add_argument("--dt",              type=float, default=0.1,  help="Euler 스텝 초기값 (학습)")
    g.add_argument("--lam",             type=float, default=0.1,  help="탄성 주입 강도 초기값 (학습)")
    g.add_argument("--expansion_ratio", type=int,   default=4,    help="ConvSwiGLU 팽창 비율 m")
    g.add_argument("--conv_kernel",     type=int,   default=3,    help="Depthwise conv 커널 크기")
    g.add_argument("--use_w_v",         action="store_true",      help="Mean Shift 계산 시 W_V 선형 변환 공간 사용 (Option B)")

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
    g.add_argument("--aux_loss_coef", type=float, default=0.3,
                   help="중간 루프 CE 손실 가중치 (0 = 비활성)")
    g.add_argument("--act_loss_coef", type=float, default=0.01,
                   help="ACT halting BCE 손실 가중치 (0 = 비활성)")

    # ── 속도 ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Speed")
    g.add_argument("--compile", action="store_true",
                   help="torch.compile 활성화 (C++ 컴파일러 필요)")
    g.add_argument("--dtype",   choices=["float32", "bfloat16", "float16"], default="float32",
                   help="Mixed precision dtype")

    # ── 로깅 & 체크포인팅 ─────────────────────────────────────────────────
    g = p.add_argument_group("Logging")
    g.add_argument("--checkpoint_dir", default="checkpoints")
    g.add_argument("--log_interval",   type=int, default=50,  help="상세 로그 출력 주기 (옵티마이저 스텝)")
    g.add_argument("--eval_interval",  type=int, default=5000, help="테스트셋 평가 주기 (옵티마이저 스텝)")
    g.add_argument("--save_top_k",     type=int, default=3,   help="상위 K개 체크포인트 보존")
    g.add_argument("--resume",         type=str, default=None, help="재개할 체크포인트 경로")

    g = p.add_argument_group("Debug")
    g.add_argument("--debug_nan",      action="store_true",   help="NaN/Inf 감지 훅 및 안티-익셉션 활성화")

    g.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 옵티마이저 & 스케줄러
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer_and_scheduler(
    model:       nn.Module,
    args:        argparse.Namespace,
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    - Bias / LayerNorm / 1D 파라미터 → weight_decay=0
    - 나머지 → weight_decay=args.weight_decay
    - CUDA 에서는 fused AdamW 사용 (CUDA 커널 병합으로 속도 향상)
    - Cosine decay + linear warmup 스케줄러
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D 파라미터(bias, norm weight/bias, spectral bias 등) 는 decay 제외
        if param.ndim <= 1 or any(k in name for k in ("bias", "norm", "B_Q", "B_K")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Fused AdamW: CUDA 에서만 사용 (단일 커널로 파라미터 업데이트)
    use_fused = torch.cuda.is_available()
    optim_kwargs: dict = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        param_groups,
        lr     = args.lr,
        betas  = (0.9, 0.95),
        eps    = 1e-8,
        **optim_kwargs,
    )

    # Cosine decay with linear warmup
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    logits_list:     List[torch.Tensor],  # [(B, 81, V), ...]
    halt_probs_list: List[torch.Tensor],  # [(B, 1), ...]
    labels:          torch.Tensor,         # (B, 81)
    args:            argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    전체 손실 = main_CE  +  aux_coef × aux_CE  +  act_coef × act_BCE

    - main_CE : 마지막 루프 logits 에 대한 Cross-Entropy
    - aux_CE  : gradient-tracking 구간 중간 루프들의 CE 평균
    - act_BCE : ACT 정지 확률 정규화 (마지막 루프→1, 나머지→0)
    """
    B, N    = labels.shape
    V       = logits_list[0].shape[-1]
    flat_lbl = labels.reshape(B * N)

    def ce(logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(B * N, V),
            flat_lbl,
            ignore_index=0,   # PAD (레이블에 없으므로 사실상 no-op)
        )

    # 메인 손실
    main_loss = ce(logits_list[-1])
    total     = main_loss

    # 보조 손실: 중간 루프 (마지막 루프 제외)
    aux_loss = torch.zeros(1, device=total.device).squeeze()
    if args.aux_loss_coef > 0 and len(logits_list) > 1:
        aux_loss = torch.stack([ce(l) for l in logits_list[:-1]]).mean()
        total    = total + args.aux_loss_coef * aux_loss

    # ACT 정규화 손실
    act_loss = torch.zeros(1, device=total.device).squeeze()
    if args.act_loss_coef > 0 and len(halt_probs_list) > 1:
        # halt_stack: (B, L) — 루프별 정지 확률
        halt_stack  = torch.cat(halt_probs_list, dim=1)  # (B, L)
        halt_target = torch.zeros_like(halt_stack)
        halt_target[:, -1] = 1.0                          # 마지막 루프만 halting=1
        # NaN/Inf 방어: Q 발산 시 halt_stack이 NaN이 될 수 있으므로 스킵
        if torch.isfinite(halt_stack).all():
            act_loss = F.binary_cross_entropy(halt_stack.clamp(1e-6, 1 - 1e-6), halt_target)
        total    = total + args.act_loss_coef * act_loss

    log = {
        "loss":      total.item(),
        "main_loss": main_loss.item(),
        "aux_loss":  aux_loss.item(),
        "act_loss":  act_loss.item(),
    }
    return total, log


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    logits_list: List[torch.Tensor],  # [(B, 81, V), ...]
    labels:      torch.Tensor,         # (B, 81)
) -> Dict[str, object]:
    """
    - token_acc  : 81개 셀 중 맞춘 비율 (최종 루프 기준)
    - puzzle_acc : 81개 전부 맞춰야 1 (최종 루프 기준)
    - per_loop_puzzle_acc : 루프별 puzzle_acc 리스트
    """
    mask    = labels != 0       # (B, 81) — 항상 True (레이블에 PAD 없음)
    n_valid = mask.float().sum()

    loop_puzzle_accs = []
    for logits in logits_list:
        preds          = logits.argmax(dim=-1)              # (B, 81)
        correct        = (preds == labels) & mask           # (B, 81)
        all_correct    = (correct | ~mask).all(dim=1)       # (B,)
        loop_puzzle_accs.append(all_correct.float().mean().item())

    # 최종 루프 기준 대표 지표
    final_preds   = logits_list[-1].argmax(dim=-1)          # (B, 81)
    final_correct = (final_preds == labels) & mask          # (B, 81)

    return {
        "token_acc":           (final_correct.float().sum() / n_valid).item(),
        "puzzle_acc":          (final_correct | ~mask).all(dim=1).float().mean().item(),
        "per_loop_puzzle_acc": loop_puzzle_accs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:        nn.Module,
    loader:       torch.utils.data.DataLoader,
    device:       torch.device,
    args:         argparse.Namespace,
    autocast_ctx,
) -> Dict[str, float]:
    """테스트셋 전체를 순회하며 평균 지표를 반환합니다."""
    model.eval()

    total_loss       = 0.0
    token_correct    = 0.0
    token_total      = 0.0
    puzzle_correct   = 0.0
    puzzle_total     = 0
    loop_puzz_corr: Optional[List[float]] = None

    for inputs, labels in tqdm(loader, desc="  [eval]", leave=False, dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)  # (B, 81)
        labels = labels.to(device, non_blocking=True)  # (B, 81)

        with autocast_ctx:
            logits_list, halt_probs_list = model(inputs)

        _, loss_log = compute_loss(logits_list, halt_probs_list, labels, args)
        total_loss += loss_log["loss"]

        m    = compute_metrics(logits_list, labels)
        B    = inputs.shape[0]
        mask = labels != 0

        token_correct  += m["token_acc"]  * mask.float().sum().item()
        token_total    += mask.float().sum().item()
        puzzle_correct += m["puzzle_acc"] * B
        puzzle_total   += B

        lpa = m["per_loop_puzzle_acc"]
        if loop_puzz_corr is None:
            loop_puzz_corr = [0.0] * len(lpa)
        for i, v in enumerate(lpa):
            loop_puzz_corr[i] += v * B

    n_batches = max(1, len(loader))
    result = {
        "eval_loss":       total_loss / n_batches,
        "eval_token_acc":  token_correct / max(1.0, token_total),
        "eval_puzzle_acc": puzzle_correct / max(1, puzzle_total),
    }
    if loop_puzz_corr:
        for i, v in enumerate(loop_puzz_corr):
            result[f"eval_loop{i + 1}_puzzle_acc"] = v / max(1, puzzle_total)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 관리
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """상위 K개 체크포인트를 puzzle_acc 기준으로 보존합니다."""

    def __init__(self, ckpt_dir: str, save_top_k: int = 3):
        self.ckpt_dir   = Path(ckpt_dir)
        self.save_top_k = save_top_k
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._saved: List[Tuple[float, Path]] = []  # (score, path)

    def save(self, state: dict, score: float, filename: str) -> None:
        path = self.ckpt_dir / filename
        torch.save(state, path)

        self._saved.append((score, path))
        self._saved.sort(key=lambda x: -x[0])  # 내림차순

        # Top-K 초과 시 최하위 삭제 (best.pt 는 제외)
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
    torch.backends.cudnn.benchmark = True  # 고정된 입력 크기(81)에서 최적 CUDA 커널 자동 선택
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
    model = AMKPDModel(
        vocab_size       = meta["vocab_size"],
        d_model          = args.d_model,
        d_spectral       = args.d_spectral,
        num_layers       = args.num_layers,
        max_loops        = args.max_loops,
        trunc_loops      = args.trunc_loops,
        dt               = args.dt,
        lam              = args.lam,
        expansion_ratio  = args.expansion_ratio,
        conv_kernel_size = args.conv_kernel,
        use_w_v          = getattr(args, 'use_w_v', False),
    ).to(device)

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

        def nan_backward_hook(module, grad_input, grad_output):
            for i, g in enumerate(grad_input):
                if isinstance(g, torch.Tensor) and (torch.isnan(g).any() or torch.isinf(g).any()):
                    print(f"\n[DEBUG] NaN/Inf in backward grad_input[{i}] of {module.__class__.__name__}")
            for i, g in enumerate(grad_output):
                if isinstance(g, torch.Tensor) and (torch.isnan(g).any() or torch.isinf(g).any()):
                    print(f"\n[DEBUG] NaN/Inf in backward grad_output[{i}] of {module.__class__.__name__}")

        for name, m in model.named_modules():
            m.register_forward_hook(nan_forward_hook)
            m.register_full_backward_hook(nan_backward_hook)

    # ── Mixed precision ───────────────────────────────────────────────────
    _dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype  = _dtype_map[args.dtype]
    # bfloat16 은 CPU 에서도 지원; float16 은 CUDA 전용
    use_amp    = amp_dtype != torch.float32
    if use_amp and amp_dtype == torch.float16 and device.type != "cuda":
        print("float16 is CUDA-only — falling back to float32")
        use_amp = False

    autocast_ctx = torch.autocast(
        device_type = device.type,
        dtype       = amp_dtype,
        enabled     = use_amp,
    )
    # GradScaler: float16 에서만 필요 (bfloat16 은 동적 범위가 충분)
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
    total_start = time.time()

    # 처리량 추적: 최근 100 마이크로-스텝의 이동 평균
    step_times:   deque = deque(maxlen=100)
    step_samples: deque = deque(maxlen=100)
    last_grad_norm = 0.0

    # ── Epoch 루프 ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()

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

            inputs = inputs.to(device, non_blocking=True)  # (B, 81)
            labels = labels.to(device, non_blocking=True)  # (B, 81)

            # GPU-side 벡터화 온라인 증강 (선택적)
            if args.augment:
                inputs, labels = vectorized_sudoku_augment(inputs, labels)

            # ── Forward ───────────────────────────────────────────────────
            with autocast_ctx:
                logits_list, halt_probs_list = model(inputs)
                loss, loss_log = compute_loss(logits_list, halt_probs_list, labels, args)
                loss = loss / args.grad_accum   # 그래디언트 누적 스케일링

            # ── NaN 발생 감지 및 종료 ────────────────────────────────────────
            if math.isnan(loss_log["loss"]):
                tqdm.write(f"\n[FATAL] NaN loss detected at global step {global_step}!")
                tqdm.write("--- Model Parameter Status ---")
                has_nan_param = False
                for name, p in model.named_parameters():
                    if p.isnan().any():
                        tqdm.write(f"Parameter '{name}' contains NaN.")
                        has_nan_param = True
                    if p.grad is not None and p.grad.isnan().any():
                        tqdm.write(f"Gradient of '{name}' contains NaN.")
                if not has_nan_param:
                    tqdm.write("All parameters are finite. Forward pass exploded! Please check activation scaling.")
                tqdm.write("Exiting training to prevent further NaN propagation.")
                import sys
                sys.exit(1)

            # ── Backward ──────────────────────────────────────────────────
            scaler.scale(loss).backward()

            # ── 옵티마이저 스텝 (grad_accum 마이크로-스텝마다) ─────────────
            is_update_step = (batch_idx + 1) % args.grad_accum == 0
            if is_update_step:
                scaler.unscale_(optimizer)
                last_grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                ).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # ── 처리량 측정 ───────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            step_times.append(elapsed)
            step_samples.append(inputs.shape[0])
            throughput = sum(step_samples) / max(1e-9, sum(step_times))

            # ── 배치 지표 (cheap, no_grad) ────────────────────────────────
            with torch.no_grad():
                m = compute_metrics(logits_list, labels)

            epoch_loss  += loss_log["loss"]
            epoch_steps += 1

            # ── tqdm postfix (매 스텝 갱신) ───────────────────────────────
            current_lr = optimizer.param_groups[0]["lr"]
            halt_means = [f"{p.mean().item():.2f}" for p in halt_probs_list]

            pbar.set_postfix(
                loss     = f"{loss_log['loss']:.4f}",
                tok_acc  = f"{m['token_acc'] * 100:.1f}%",
                puzz_acc = f"{m['puzzle_acc'] * 100:.1f}%",
                lr       = f"{current_lr:.2e}",
                gnorm    = f"{last_grad_norm:.3f}",
                samp_s   = f"{throughput:.0f}",
                halt     = "[" + ",".join(halt_means) + "]",
                refresh  = False,
            )

            # ── 상세 로그 (log_interval 옵티마이저 스텝마다) ──────────────
            if is_update_step and global_step % args.log_interval == 0:
                # ETA 계산
                elapsed_total = time.time() - total_start
                eta_sec       = (elapsed_total / max(1, global_step)) * max(0, total_steps - global_step)
                h, rem        = divmod(int(eta_sec), 3600)
                m_, s_        = divmod(rem, 60)

                lpa_str = "  ".join(
                    f"L{i+1}={v * 100:.1f}%"
                    for i, v in enumerate(m["per_loop_puzzle_acc"])
                )

                mem_str = ""
                if device.type == "cuda":
                    gb = torch.cuda.memory_allocated(device) / 1024 ** 3
                    mem_str = f"  mem={gb:.2f}GB"

                tqdm.write(
                    f"[{global_step:6d}/{total_steps}]"
                    f"  loss={loss_log['loss']:.4f}"
                    f" (main={loss_log['main_loss']:.4f}"
                    f" aux={loss_log['aux_loss']:.4f}"
                    f" act={loss_log['act_loss']:.4f})"
                    f"  tok={m['token_acc'] * 100:.2f}%"
                    f"  puzz={m['puzzle_acc'] * 100:.2f}%"
                    f"  [{lpa_str}]"
                    f"  lr={current_lr:.2e}"
                    f"  gnorm={last_grad_norm:.3f}"
                    f"  {throughput:.0f}samp/s"
                    f"  ETA {h:02d}:{m_:02d}:{s_:02d}"
                    f"{mem_str}"
                )

            # ── 평가 & 체크포인팅 (eval_interval 옵티마이저 스텝마다) ──────
            if is_update_step and global_step > 0 and global_step % args.eval_interval == 0:
                eval_res = evaluate(model, test_loader, device, args, autocast_ctx)
                model.train()

                # 루프별 puzzle_acc 정리
                loop_str = "  ".join(
                    f"L{k.replace('eval_loop', '').replace('_puzzle_acc', '')}="
                    f"{v * 100:.2f}%"
                    for k, v in eval_res.items()
                    if k.startswith("eval_loop")
                )

                tqdm.write(
                    f"\n{'─' * 70}\n"
                    f"[EVAL  step={global_step}]\n"
                    f"  loss       = {eval_res['eval_loss']:.4f}\n"
                    f"  token_acc  = {eval_res['eval_token_acc']  * 100:.2f}%\n"
                    f"  puzzle_acc = {eval_res['eval_puzzle_acc'] * 100:.2f}%\n"
                    f"  per-loop   : {loop_str}\n"
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
                tqdm.write(f"  [ckpt] saved → checkpoints/{fname}")

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
    # Windows multiprocessing 보호 (num_workers > 0 시 필수)
    args = parse_args()

    # 설정 저장
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.checkpoint_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved → {args.checkpoint_dir}/config.json")
    print()

    train(args)
