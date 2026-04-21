# SLURM submission templates

These three scripts are illustrative templates showing how we ran the full
training pipeline on the Cambridge CSD3 Wilkes3 GPU cluster. They are not
expected to run unmodified on any other site — they target a specific SLURM
account, a specific conda environment name, and specific output paths.

SLURM is a widely-used job scheduler for High-Performance Computing (HPC)
clusters. `sbatch icovae_hopt.sh` submits one training job; the directives at
the top (`#SBATCH ...`) tell the scheduler how much wall time, how many GPUs
and so on to allocate.

| Script | Runs |
| --- | --- |
| `icovae_hopt.sh` | Stage-1 iCoVAE hyperparameter optimisation (`src/scripts/icovae_hopt.py`) on one drug / target. |
| `vae_hopt.sh` | Same as above, but for the VAE baseline. |
| `nn_hopt.sh` | Stage-2 MLP-from-raw-features baseline (`src/scripts/nn_hopt.py`). |

## What you need to edit to run these at your site

All three scripts currently hard-code settings specific to Cambridge CSD3.
Before submitting elsewhere, edit:

1. **Account and partition** — `#SBATCH -A ` and `#SBATCH -p
   ampere` (or `icelake` for CPU). Replace with your own SLURM account /
   partition names.
2. **Mail notification address** — `#SBATCH --mail-user=...`. Replace or
   delete the three `#SBATCH --mail-type=...` lines if you don't want mail.
3. **Output log path** — `#SBATCH --output=slurm_out/...`. Point this at a
   directory that exists on your cluster.
4. **Environment setup** — `module load rhel8/default-amp`, `module load cuda/...`,
   and `conda activate pico`. Adapt to your site's module system and conda
   environment name.
5. **CLI options** (`options="-dataset ... -target ..."`) — at the bottom of each
   script, uncomment the example options line that matches the experiment you
   want to run, or pass your own.

Once edited, submit via `sbatch <script>.sh` from a login node.

## Estimated wall time per job

On a single NVIDIA A100 (80 GB):

- Stage-1 hyperparameter optimisation (150 Optuna trials) for one drug:
  approximately 8 hours.
- Stage-1 refit (10 seeds) using pre-chosen hyperparameters from another
  drug: approximately 1 hour.

See the main [README.md](../../../README.md) for context on when you'd want
to rerun these vs just using the bundled demo.
