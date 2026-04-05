"""
체크포인트 로드 후 테스트 데이터셋 전체에 대해 평가하는 스크립트.

사용 예시:
  uv run python eval_checkpoint.py --checkpoint checkpoints/best.pt
  uv run python eval_checkpoint.py --checkpoint checkpoints/best.pt --max_samples 0  # 전체 평가
"""
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import create_dataloaders
from train import build_model


def compute_metrics(logits, labels):
    mask = labels != 0
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & mask
    n_valid = mask.float().sum()
    return {
        "token_acc": (correct.float().sum() / n_valid).item(),
        "puzzle_acc": (correct | ~mask).all(dim=1).float().mean().item(),
    }


@torch.no_grad()
def evaluate_dataset(model, loader, loops, device, max_samples=0, desc="eval"):
    model.eval()
    total_loss = 0.0
    token_correct = 0.0
    token_total = 0.0
    puzzle_correct = 0.0
    puzzle_total = 0

    for inputs, labels in tqdm(loader, desc=desc, leave=False):
        if max_samples > 0 and puzzle_total >= max_samples:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        B = inputs.shape[0]
        seq_len = inputs.shape[1]

        carry = model.initial_carry(B, seq_len, device)
        batch = (inputs, labels)

        for _ in range(loops):
            carry, logits, q_logits = model(carry, batch)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            carry.current_labels.view(-1),
            ignore_index=0,
        )
        total_loss += loss.item()

        m = compute_metrics(logits, carry.current_labels)
        mask = carry.current_labels != 0
        token_correct += m["token_acc"] * mask.float().sum().item()
        token_total += mask.float().sum().item()
        puzzle_correct += m["puzzle_acc"] * B
        puzzle_total += B

    n_batches = max(1, puzzle_total // max(1, B))
    return {
        "loss": total_loss / n_batches,
        "token_acc": token_correct / max(1.0, token_total),
        "puzzle_acc": puzzle_correct / max(1, puzzle_total),
        "n_samples": puzzle_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="데이터셋 경로 (미지정 시 체크포인트의 data_dir 사용)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="최대 평가 샘플 수 (0 = 전체)")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 체크포인트 로드
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  step={ckpt['global_step']}, best_puzzle_acc={ckpt['best_puzzle_acc']:.4f}")
    print(f"  d_model={ckpt_args['d_model']}, num_heads={ckpt_args['num_heads']}, "
          f"num_layers={ckpt_args['num_layers']}, loops={ckpt_args['loops']}")

    # 데이터 로드
    data_dir = args.data_dir or ckpt_args.get("data_dir", "data/sudoku")
    train_loader, test_loader, meta = create_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # 모델 생성 (train.py의 build_model 재사용) + 가중치 로드
    model = build_model(ckpt_args, meta["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    loops = ckpt_args["loops"]

    # 훈련 데이터로 평가
    print(f"\n{'='*60}")
    print(f"[TRAIN set] loops={loops}")
    print(f"{'='*60}")
    train_res = evaluate_dataset(model, train_loader, loops, device,
                                  max_samples=args.max_samples, desc="train eval")
    print(f"  loss       = {train_res['loss']:.4f}")
    print(f"  token_acc  = {train_res['token_acc'] * 100:.2f}%")
    print(f"  puzzle_acc = {train_res['puzzle_acc'] * 100:.2f}%")
    print(f"  n_samples  = {train_res['n_samples']}")

    # 테스트 데이터로 평가
    print(f"\n{'='*60}")
    print(f"[TEST set] loops={loops}")
    print(f"{'='*60}")
    test_res = evaluate_dataset(model, test_loader, loops, device,
                                 max_samples=args.max_samples, desc="test eval")
    print(f"  loss       = {test_res['loss']:.4f}")
    print(f"  token_acc  = {test_res['token_acc'] * 100:.2f}%")
    print(f"  puzzle_acc = {test_res['puzzle_acc'] * 100:.2f}%")
    print(f"  n_samples  = {test_res['n_samples']}")

    # 비교
    print(f"\n{'='*60}")
    print(f"[COMPARISON]")
    print(f"{'='*60}")
    print(f"  Train pacc: {train_res['puzzle_acc']*100:.2f}%  vs  Test pacc: {test_res['puzzle_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
