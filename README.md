# CST PatchTST on Weather

English | [日本語](README.ja.md)

Train PatchTST from scratch to predict 96 consecutive Weather records from
the preceding 336 records across 21 measurements at a weather station. At the
nominal 10-minute sampling interval, this is roughly a 16-hour forecast from
56 hours of history.

The [PatchTST Weather setting](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/scripts/PatchTST/weather.sh)
uses three encoder blocks, width 128 and FFN width 256. The CST variant
replaces the 256-to-128 output projection in each FFN, retains its bias and
uses 512 Atoms with fixed two-dimensional index grids. The grids describe
latent feature indices. The attention, shared prediction head, normalization
and remaining FFN operations are from the reference architecture.

This comparison uses one seed, batch size 64, 20 epochs and an RTX 4090. The
ordinary parameters use a base learning rate of 0.0001 and the Atom group
uses 0.00048. Both groups follow the reference type-3 decay factor. We fit
per-channel scaling on the training segment and use the Weather
chronological split. MSE and MAE are reported in the standardized space.
The test set is evaluated using the checkpoint with lowest validation MSE.

## Run

Use a CUDA-enabled PyTorch installation. Obtain `weather.csv` from the
[benchmark data linked by the PatchTST authors](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/README.md).
The experiment used the 52,696-row [Weather CSV mirror](https://www.uni-hildesheim.de/gitlab/domuro/PatchTST/-/blob/f720f50a210751dfbfb84c6008c19dd2cf6d8d44/PatchTST_self_supervised/dataset/weather.csv),
SHA-256 `9b2b19b13e342d9c9b9c51620d42b4be232df7fba30e7f5c4968b7c79ab138ad`.
The records include one repeated timestamp and one 100-minute interval.
The preprocessing keeps the benchmark's original rows.

```bash
python -m pip install -r requirements.txt
for variant in baseline cst; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py --variant "$variant" \
    --data data/weather.csv --output "output/weather/$variant" \
    --batch-size 64 --seed 2021 --epochs 20 --atoms 512 --atom-lr 0.00048
done
```

Set `--pilot 25` with a fresh output directory to record 25 timed updates after
eight warmup updates. See the [upstream license](upstream/LICENSE) for the
vendored architecture.
