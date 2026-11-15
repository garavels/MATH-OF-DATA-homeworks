# AGENTS.md

## Environment
- Python with all dependencies (numpy, scipy, torch 2.11 with MPS, torchvision 0.26, jupyter/nbconvert): `/opt/anaconda3/envs/data_analysis/bin/python` (Jupyter kernel `python3`).
- Hardware: Apple M4 Pro GPU via MPS. bf16 (autocast or matmuls) is not faster than fp32 on MPS here; Low Power Mode on battery makes training ~1.5x slower.

## Notebooks
- `Part 1`-`Part 5` notebooks live at the repo root; `lib/` holds the helpers and data for Parts 2-4.
- Run a notebook headlessly: `caffeinate -i jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "<notebook>.ipynb"`.
  Do not edit a notebook while nbconvert is executing it: the file is overwritten when execution finishes.

## Part 5 (CIFAR-10 optimizers)
- CIFAR-10 is stored in `./data` (gitignored). The Toronto server is slow (~50 KB/s per connection); to re-download, fetch
  `https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz` with at most 8 parallel HTTP range requests and check MD5 `c58f30108f718f92721af3b95e74349a`.
- Finished training runs are cached in `part5_results_<EPOCHS>ep.json`; delete it (or single entries) to retrain. All 8 runs take
  about 50 minutes on the M4 Pro on AC power (~1.5x longer in Low Power Mode); keep the machine plugged in so timings stay comparable.
