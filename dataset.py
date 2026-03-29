"""
dataset.py — AMK-PD Sudoku DataLoader (URM-compatible)

URM과 동일한 Group-based IterableDataset:
  - 매 epoch마다 group(=원본 퍼즐) 순서를 셔플
  - 각 group에서 1개의 augmentation을 랜덤 샘플링 (i.i.d.)
  - 배치 내 100% 퍼즐 다양성 보장
  - 작은 epoch (total_groups / batch_size batches) × 많은 반복

토큰 인코딩:
  0  = PAD / ignore_label
  1  = blank (입력에서 빈 칸)
  2-10 = Sudoku 숫자 1-9
  vocab_size = 11
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


# ─────────────────────────────────────────────────────────────────────────────
# URM-style group-based batch sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    batch_size: int,
):
    """
    URM의 _sample_batch와 동일한 로직.
    group_order를 순회하며, 각 group에서 1개의 puzzle을 랜덤 선택,
    그 puzzle에서 1개의 example을 랜덤 선택하여 배치를 채운다.
    """
    batch = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < batch_size):
        group_id = group_order[start_index]
        # group 내에서 랜덤 puzzle 선택
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # puzzle 내에서 랜덤 example 선택
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, batch_size - current_size)
        batch.append(puzzle_start + rng.choice(puzzle_size, append_size, replace=False))
        current_size += append_size

    return start_index, np.concatenate(batch)


class SudokuGroupDataset(IterableDataset):
    """
    URM과 동일한 Group-based IterableDataset.

    1 epoch = total_groups개의 group을 셔플하여 순회.
    epochs_per_iter개의 epoch을 하나의 iteration으로 묶어서 yield.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        batch_size: int = 128,
        epochs_per_iter: int = 1,
        seed: int = 42,
        test_mode: bool = False,
    ):
        super().__init__()
        split_dir = Path(data_dir) / split

        # mmap for large arrays
        self.inputs = np.load(split_dir / "all__inputs.npy", mmap_mode="r")
        self.labels = np.load(split_dir / "all__labels.npy", mmap_mode="r")

        # index arrays in memory
        self.group_indices = np.load(split_dir / "all__group_indices.npy")
        self.puzzle_indices = np.load(split_dir / "all__puzzle_indices.npy")

        with open(split_dir / "dataset.json") as f:
            meta = json.load(f)

        self.vocab_size = meta["vocab_size"]
        self.seq_len = meta["seq_len"]
        self.total_groups = meta["total_groups"]

        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        self.seed = seed
        self.test_mode = test_mode
        self._iters = 0

    def _iter_train(self):
        self._iters += 1
        rng = np.random.Generator(np.random.Philox(seed=self.seed + self._iters))

        # epochs_per_iter개의 epoch에 해당하는 group 순서 생성
        group_order = np.concatenate([
            rng.permutation(self.group_indices.size - 1)
            for _ in range(self.epochs_per_iter)
        ])
        start_index = 0

        while start_index < group_order.size:
            start_index, batch_indices = _sample_batch(
                rng,
                group_order=group_order,
                puzzle_indices=self.puzzle_indices,
                group_indices=self.group_indices,
                start_index=start_index,
                batch_size=self.batch_size,
            )

            # drop last incomplete batch
            if batch_indices.size < self.batch_size:
                break

            inp = torch.from_numpy(self.inputs[batch_indices].copy()).long()
            lbl = torch.from_numpy(self.labels[batch_indices].copy()).long()

            # ignore_label: labels에서 0인 셀은 loss 계산에서 제외
            yield inp, lbl

    def _iter_test(self):
        total = len(self.inputs)
        start = 0
        while start < total:
            end = min(total, start + self.batch_size)
            idx = np.arange(start, end)
            inp = torch.from_numpy(self.inputs[idx].copy()).long()
            lbl = torch.from_numpy(self.labels[idx].copy()).long()
            yield inp, lbl
            start = end

    def __iter__(self):
        if self.test_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


# ─────────────────────────────────────────────────────────────────────────────
# Legacy MapDataset (for visualization scripts)
# ─────────────────────────────────────────────────────────────────────────────

class SudokuDataset(Dataset):
    """메모리-매핑 Sudoku 데이터셋. 시각화 스크립트에서 사용."""

    def __init__(self, data_dir: str, split: str = "train"):
        split_dir = Path(data_dir) / split
        self.inputs = np.load(split_dir / "all__inputs.npy", mmap_mode="r")
        self.labels = np.load(split_dir / "all__labels.npy", mmap_mode="r")

        with open(split_dir / "dataset.json") as f:
            meta = json.load(f)

        self.vocab_size = meta["vocab_size"]
        self.seq_len = meta["seq_len"]
        self.pad_id = meta["pad_id"]
        self.ignore_id = meta.get("ignore_label_id", 0)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.from_numpy(self.inputs[idx].copy()).long()
        lbl = torch.from_numpy(self.labels[idx].copy()).long()
        return inp, lbl


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    epochs_per_iter: int = 2000,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    URM 방식의 Group-based IterableDataset DataLoader.

    train: IterableDataset (batch_size=None, dataset이 배치를 직접 생성)
    test: MapDataset (기존과 동일)
    """
    train_ds = SudokuGroupDataset(
        data_dir, "train",
        batch_size=batch_size,
        epochs_per_iter=epochs_per_iter,
        seed=seed,
        test_mode=False,
    )

    test_ds = SudokuDataset(data_dir, "test")

    meta = {
        "vocab_size": train_ds.vocab_size,
        "seq_len": train_ds.seq_len,
        "total_groups": train_ds.total_groups,
        "train_size": len(train_ds.inputs),
        "test_size": len(test_ds),
    }

    # IterableDataset: batch_size=None (dataset이 배치를 직접 yield)
    # num_workers=0 on Windows
    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
    """
    B      = inputs.shape[0]
    device = inputs.device

    digit_map        = torch.empty(B, 11, dtype=torch.long, device=device)
    digit_map[:, 0]  = 0
    digit_map[:, 1]  = 1
    digit_map[:, 2:] = torch.rand(B, 9, device=device).argsort(dim=1) + 2

    band_perm  = torch.rand(B, 3,    device=device).argsort(dim=1)
    within_row = torch.rand(B, 3, 3, device=device).argsort(dim=2)
    row_perm   = (band_perm.unsqueeze(2) * 3 + within_row).view(B, 9)

    stack_perm = torch.rand(B, 3,    device=device).argsort(dim=1)
    within_col = torch.rand(B, 3, 3, device=device).argsort(dim=2)
    col_perm   = (stack_perm.unsqueeze(2) * 3 + within_col).view(B, 9)

    def apply_spatial_perm(boards: torch.Tensor) -> torch.Tensor:
        g = boards.view(B, 9, 9)
        g = g.gather(1, row_perm.unsqueeze(2).expand(B, 9, 9))
        g = g.gather(2, col_perm.unsqueeze(1).expand(B, 9, 9))
        return g

    inp_2d = apply_spatial_perm(inputs)
    lbl_2d = apply_spatial_perm(labels)

    t_flag = (torch.rand(B, device=device) < 0.5).view(B, 1, 1)
    inp_2d = torch.where(t_flag.expand(B, 9, 9), inp_2d.transpose(1, 2), inp_2d)
    lbl_2d = torch.where(t_flag.expand(B, 9, 9), lbl_2d.transpose(1, 2), lbl_2d)

    inp_out = digit_map.gather(1, inp_2d.reshape(B, 81))
    lbl_out = digit_map.gather(1, lbl_2d.reshape(B, 81))

    return inp_out, lbl_out
