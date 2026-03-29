import sys
import torch
import torch.nn.functional as F
from amkpd_model import AMKPDModel
from dataset import SudokuDataset

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/step002500_pacc0.0299.pt"
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 256
split = sys.argv[3] if len(sys.argv) > 3 else "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
a = ckpt["args"]
print(f"Checkpoint: {ckpt_path} (step={ckpt['global_step']})")
print(f"  d_model={a['d_model']}, layers={a['num_layers']}, loops={a['loops']}, batch_size={a['batch_size']}")

ds = SudokuDataset(a["data_dir"], split)
n = min(n_samples, len(ds))
inputs = torch.stack([ds[i][0] for i in range(n)]).to(device)
labels = torch.stack([ds[i][1] for i in range(n)]).to(device)
print(f"  {split} samples: {n}")

model = AMKPDModel(
    vocab_size=11, d_model=a["d_model"], num_heads=a["num_heads"],
    num_layers=a["num_layers"], loops=100, H_cycles=a["H_cycles"],
    L_cycles=a["L_cycles"], kernel_power=a["kernel_power"],
    expansion_ratio=a["expansion_ratio"], conv_kernel_size=a.get("conv_kernel", 2),
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

B, seq_len = inputs.shape
carry = model.initial_carry(B, seq_len, device)
batch = (inputs, labels)

checkpoints = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

print("\nloops      loss   tok_acc  puzz_acc   Q_diff")
print("-" * 55)

prev_Q = None
with torch.no_grad():
    for i in range(1, 101):
        carry, logits, _ = model(carry, batch)

        if i in checkpoints:
            Q = carry.current_hidden
            q_diff = (Q - prev_Q).abs().mean().item() if prev_Q is not None else 0.0
            prev_Q = Q.clone()

            mask = labels != 0
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0
            ).item()
            preds = logits.argmax(dim=-1)
            correct = ((preds == labels) & mask).float().sum() / mask.float().sum()
            puzz = (((preds == labels) | ~mask).all(dim=1)).float().mean()
            print(f"{i:>5}  {loss:>8.4f}  {correct.item()*100:>7.2f}%  {puzz.item()*100:>7.2f}%  {q_diff:>.8f}")
