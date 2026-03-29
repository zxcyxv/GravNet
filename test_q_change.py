import torch
from amkpd_model import AMKPDModel
from dataset import SudokuDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("checkpoints/step002500_pacc0.0000.pt", map_location=device, weights_only=False)
a = ckpt["args"]

ds = SudokuDataset("data/sudoku-extreme-1k-aug-1000", "train")
inputs = torch.stack([ds[i][0] for i in range(8)]).to(device)
labels = torch.stack([ds[i][1] for i in range(8)]).to(device)

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

print("loop  Q_mean       Q_std        Q_diff       halted  steps  logits_mean")
print("-" * 85)

prev_Q = None
with torch.no_grad():
    for i in range(1, 21):
        carry, logits, _ = model(carry, batch)
        Q = carry.current_hidden
        q_diff = (Q - prev_Q).abs().mean().item() if prev_Q is not None else 0.0
        print(f"{i:>4}  {Q.mean().item():>+12.6f}  {Q.std().item():>10.6f}  {q_diff:>12.8f}  {carry.halted.tolist()}  {carry.steps.tolist()}  {logits.mean().item():>+10.4f}")
        prev_Q = Q.clone()
