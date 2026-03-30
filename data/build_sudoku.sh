#!/bin/bash
uv run python -m data.build_sudoku_dataset --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
