run_name="URM-sudoku"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

python pretrain.py \
data_path=data/sudoku-extreme-1k-aug-1000 \
arch=urm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
epochs=50000 \
eval_interval=2000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
