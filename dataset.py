"""
dataset.py — AMK-PD Sudoku DataLoader

핵심 설계:
  - 메모리 매핑(mmap_mode='r'): 1.3GB 파일을 RAM에 올리지 않고 디스크에서 슬라이싱
  - 벡터화 온라인 증강: GPU-side, Python for-loop 없이 전체 배치에 한 번에 적용
  - DataLoader: pin_memory / persistent_workers / prefetch_factor 로 CPU↔GPU 병목 제거

토큰 인코딩:
  0  = PAD (데이터에 없음)
  1  = blank (입력에서 빈 칸)
  2-10 = Sudoku 숫자 1-9
  vocab_size = 11
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SudokuDataset(Dataset):
    """
    메모리-매핑 Sudoku 데이터셋.

    __getitem__ 이 호출될 때마다 디스크에서 해당 슬라이스만 읽어오므로
    전체 1.3 GB 를 RAM 에 올릴 필요가 없습니다.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        split_dir = Path(data_dir) / split

        # mmap_mode='r': 배열 전체를 메모리에 올리지 않음 — 필요한 행만 lazy load
        self.inputs = np.load(split_dir / "all__inputs.npy", mmap_mode="r")  # (N, 81)
        self.labels = np.load(split_dir / "all__labels.npy", mmap_mode="r")  # (N, 81)

        with open(split_dir / "dataset.json") as f:
            meta = json.load(f)

        self.vocab_size = meta["vocab_size"]              # 11
        self.seq_len    = meta["seq_len"]                  # 81
        self.pad_id     = meta["pad_id"]                   # 0
        self.ignore_id  = meta.get("ignore_label_id", 0)  # 0

        assert self.inputs.shape[1] == self.seq_len, "seq_len mismatch"
        assert len(self.inputs) == len(self.labels), "inputs/labels length mismatch"

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # .copy() : mmap 슬라이스 → 쓰기 가능한 C-연속 배열 → tensor
        # .long() : uint8(test) 및 int64(train) 모두 통일
        inp = torch.from_numpy(self.inputs[idx].copy()).long()  # (81,)
        lbl = torch.from_numpy(self.labels[idx].copy()).long()  # (81,)
        return inp, lbl


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    data_dir:    str,
    batch_size:  int  = 256,
    num_workers: int  = 0,
    pin_memory:  bool = True,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    훈련/테스트 DataLoader 와 메타데이터를 반환합니다.

    성능 옵션:
      - persistent_workers : worker 프로세스 재시작 비용 제거
      - prefetch_factor     : 다음 배치를 미리 준비
      - pin_memory          : CPU→GPU 전송을 page-locked 메모리로 가속
    """
    train_ds = SudokuDataset(data_dir, "train")
    test_ds  = SudokuDataset(data_dir, "test")

    meta = {
        "vocab_size": train_ds.vocab_size,
        "seq_len":    train_ds.seq_len,
        "pad_id":     train_ds.pad_id,
        "ignore_id":  train_ds.ignore_id,
        "train_size": len(train_ds),
        "test_size":  len(test_ds),
    }

    shared_kwargs = dict(
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        persistent_workers = num_workers > 0,
        prefetch_factor    = 4 if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle    = True,
        drop_last  = True,   # 그래디언트 노이즈 방지
        **shared_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = batch_size * 2,  # backward 없으므로 2배 배치 허용
        shuffle    = False,
        drop_last  = False,
        **shared_kwargs,
    )

    return train_loader, test_loader, meta


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized GPU-side augmentation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def vectorized_sudoku_augment(
    inputs: torch.Tensor,  # (B, 81) long, on device
    labels: torch.Tensor,  # (B, 81) long, on device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-side 벡터화 shuffle_sudoku 증강. Python for-loop 없음.

    build_sudoku_dataset.py 의 shuffle_sudoku 와 동일한 변환을 배치 전체에 동시 적용:
      1. 숫자 치환 : ID 2-10 (= Sudoku 1-9) 을 무작위 순열, blank(1)은 고정
      2. 행 치환   : 3개 band 치환 후 band 내 row 치환
      3. 열 치환   : 3개 stack 치환 후 stack 내 col 치환
      4. 전치     : 50% 확률

    모든 연산은 torch.gather / torch.where 기반 벡터 연산.
    """
    B      = inputs.shape[0]
    device = inputs.device

    # ── 1. 숫자 치환 맵: (B, 11) ─────────────────────────────────────────
    # argsort(noise) 로 for-loop 없이 B개의 독립 순열 생성
    digit_map        = torch.empty(B, 11, dtype=torch.long, device=device)
    digit_map[:, 0]  = 0   # PAD → PAD
    digit_map[:, 1]  = 1   # blank → blank (변화 없음)
    digit_map[:, 2:] = torch.rand(B, 9, device=device).argsort(dim=1) + 2  # (B, 9) → [2,10]

    # ── 2. 행 순열: band(3) × within_row(3×3) ────────────────────────────
    band_perm  = torch.rand(B, 3,    device=device).argsort(dim=1)     # (B, 3)
    within_row = torch.rand(B, 3, 3, device=device).argsort(dim=2)     # (B, 3, 3)
    row_perm   = (band_perm.unsqueeze(2) * 3 + within_row).view(B, 9)  # (B, 9)

    # ── 3. 열 순열: stack(3) × within_col(3×3) ───────────────────────────
    stack_perm = torch.rand(B, 3,    device=device).argsort(dim=1)     # (B, 3)
    within_col = torch.rand(B, 3, 3, device=device).argsort(dim=2)     # (B, 3, 3)
    col_perm   = (stack_perm.unsqueeze(2) * 3 + within_col).view(B, 9) # (B, 9)

    # ── 4. 행+열 순열을 2D gather 로 적용 ────────────────────────────────
    def apply_spatial_perm(boards: torch.Tensor) -> torch.Tensor:
        # boards: (B, 81) → (B, 9, 9) → gather rows → gather cols → (B, 9, 9)
        g = boards.view(B, 9, 9)                                         # (B, 9, 9)
        g = g.gather(1, row_perm.unsqueeze(2).expand(B, 9, 9))          # 행 순열
        g = g.gather(2, col_perm.unsqueeze(1).expand(B, 9, 9))          # 열 순열
        return g                                                          # (B, 9, 9)

    inp_2d = apply_spatial_perm(inputs)  # (B, 9, 9)
    lbl_2d = apply_spatial_perm(labels)  # (B, 9, 9)

    # ── 5. 전치 (50% 확률, torch.where 로 분기 제거) ─────────────────────
    t_flag = (torch.rand(B, device=device) < 0.5).view(B, 1, 1)         # (B, 1, 1)
    inp_2d = torch.where(t_flag.expand(B, 9, 9), inp_2d.transpose(1, 2), inp_2d)
    lbl_2d = torch.where(t_flag.expand(B, 9, 9), lbl_2d.transpose(1, 2), lbl_2d)

    # ── 6. 숫자 치환: gather(digit_map, boards) ──────────────────────────
    inp_out = digit_map.gather(1, inp_2d.reshape(B, 81))                 # (B, 81)
    lbl_out = digit_map.gather(1, lbl_2d.reshape(B, 81))                 # (B, 81)

    return inp_out, lbl_out
