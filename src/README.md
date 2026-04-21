# `src/` — PiCo core package

This folder holds the Python package itself. Each subdirectory has a single
role:

| Subdirectory | What it contains |
| --- | --- |
| [`models/`](models/) | Model classes (`iCoVAE`, `VanillaVAE`, `TabularEncoder`, `BaselineNN`, `PiCoSK`) and the variational objectives used during training. |
| [`utils/`](utils/) | Data-loading helpers (`process_depmap_gdsc`, `process_depmap_gdsc_transneo`, `Manual`) and analysis helpers (`PerfComp`, `calculate_feat_imps`) used by both training scripts and the paper's figure notebooks. |
| [`scripts/`](scripts/) | Command-line entry points that drive training. See the script-level table below. |
| [`data/`](data/) | Downloaded raw data lives here at runtime (DepMap, GDSC, TransNEO). The directory is ignored by git — it's created on first run of a training script. |
| [`fonts/`](fonts/) | Source Sans 3 TTF files used by the paper's figure code. Also ignored by git and downloaded on demand by `demo/plot_helpers.py:apply_paper_style()`. |

## Training scripts at a glance

| Script | Runs |
| --- | --- |
| `scripts/icovae_hopt.py` | Stage-1 iCoVAE hyperparameter optimisation (Optuna) + 10-seed refit. |
| `scripts/vae_hopt.py` | Stage-1 VAE baseline (same shape as iCoVAE without the CRISPR constraints). |
| `scripts/nn_hopt.py` | MLP-from-raw-features baseline (Fig. 3 "MLP" column). |
| `scripts/pico_sk_hopt.py` | Stage-2 prediction head (ElasticNet / SVR / Random forest) on top of a pretrained iCoVAE or VAE encoder. |
| `scripts/sk_hopt.py` | Same Stage-2 regressors directly on raw features / PCA (no representation learning). |
| `scripts/schedule_jobs*.py` | Submit batches of the above jobs to a SLURM HPC cluster (see [`docs/examples/slurm/`](../docs/examples/slurm/)). |

Run any training script with `--help` to see its CLI options; the overall
pipeline (Stage 1 then Stage 2) is documented in the top-level
[README.md](../README.md) under §4.

## Reusing the code on your own data

The entry point for custom data is [`utils/data_utils.py`](utils/data_utils.py):
write a function that returns `x` (input features), `s` (auxiliary
constraints), `y` (target) and a list of held-out sample identifiers. The
existing `process_depmap_gdsc` and `process_depmap_gdsc_transneo` functions
are worked examples. Wire your function into `process_data` in the same
file, then pass your dataset name via the `-dataset` flag on each training
script.

## Where outputs go

Every script writes to `src/data/outputs/{dataset}/{target}/{experiment}/{model}/…`.
This path is assumed by the paper's figure notebooks
([`results_analysis/*.ipynb`](../results_analysis/)) and by the demo
notebooks ([`demo/`](../demo/)).
