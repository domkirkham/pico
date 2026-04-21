---
license: cc-by-4.0
language:
  - en
tags:
  - cancer-biology
  - drug-response-prediction
  - variational-autoencoder
  - representation-learning
  - CRISPR
  - gene-expression
  - transfer-learning
  - interpretable-ml
  - bioinformatics
library_name: pytorch
pipeline_tag: tabular-regression
datasets:
  - DepMap 23Q2
  - GDSC2
  - TransNEO
---

# PiCo: Prediction with interpretable Constraints

PiCo is a two-stage framework that learns robust, interpretable low-dimensional representations of gene expression data by leveraging causal auxiliary information from CRISPR-Cas9 screens, then uses these representations for downstream prediction of drug and treatment response in cancer.

## Table of Contents

- [Model Details](#model-details)
- [Uses](#uses)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Model Examination](#model-examination)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications](#technical-specifications)
- [Citation](#citation)
- [Model Card Authors](#model-card-authors)
- [Model Card Contact](#model-card-contact)

## Model Details

### Model Description

PiCo consists of two stages. In Stage 1, an **interpretable Constrained VAE (iCoVAE)** — a semi-supervised variational autoencoder — is pretrained to learn low-dimensional representations of mRNA gene expression data with specific latent dimensions constrained to capture information from CRISPR-Cas9 genetic loss-of-function screens. In Stage 2, the pretrained iCoVAE encoder is frozen and its representations are used with simple linear prediction models to predict downstream targets such as drug response in cancer cell lines or treatment outcome in patients.

The model builds on the semi-supervised VAE framework (Kingma et al., 2014) and the Characteristic Capturing VAE (CCVAE; Joy et al., 2021), extending them to continuous-valued auxiliary constraints and adding a bounded importance-weighting term (tanh-stabilised log-ratio) for training stability when labels are continuous.

- **Developed by:** Dom Kirkham, Riccardo Masina, Stephen-John Sammut, Sach Mukherjee, Oscar M. Rueda
- **Funded by:** UKRI (grants MC_UU_00002/16 and MC_UU_00040/5), NIHR Cambridge Biomedical Research Centre, Cancer Research UK (Career Establishment Award RCCCEA-May22/100002; Pre-doctoral Research Bursary RCCPDB-May23/100003), Breast Cancer Now, Lister Institute
- **Model type:** Semi-supervised variational autoencoder (generative model) with linear prediction heads
- **License:** CC-BY 4.0

### Model Inputs and Outputs

| Stage | Input | Output |
| --- | --- | --- |
| Stage 1 (iCoVAE training) | `x`: bulk mRNA gene expression in log₂(TPM + 1) format (d = 1500 or d = 918 after filtering); `s`: CRISPR gene effect (Chronos) for a chosen set of constraint genes | Pretrained encoder `q(z \| x)` with latent dimension L ∈ {32, 64}; decoder `p(x \| z)`; per-constraint regressors `q(s^(i) \| z_s^(i))` |
| Stage 1 (inference) | `x` | Posterior mean `z = μ(x)` where `z_s` is directly interpretable as a linear predictor of each constraint gene effect |
| Stage 2 (downstream prediction) | `z` (optionally concatenated with clinical or hand-engineered features `c`) | Scalar drug response (log IC50), residual cancer burden score, or probability of pathological complete response |

### Model Sources

- **Repository:** [github.com/domkirkham/pico](https://github.com/domkirkham/pico)
- **Paper:** [Interventionally-guided representation learning for robust and interpretable AI models in cancer medicine](https://www.biorxiv.org/content/10.1101/2025.07.21.662350) (Kirkham et al., 2025)
- **Demo notebooks:** [demo/demo_ccl.ipynb](demo/demo_ccl.ipynb) (cell-line drug response), [demo/demo_transneo.ipynb](demo/demo_transneo.ipynb) (TransNEO transfer)

## Uses

### Direct Use

- Generating interpretable low-dimensional representations of cancer cell line or patient gene expression data informed by CRISPR screening data.
- Drug response prediction (log IC50) in cancer cell lines from gene expression, including zero-shot prediction for cancer types not seen during training.

### Downstream Use

- Treatment response prediction in clinical settings: transferring models trained on cell line data to predict patient outcomes (residual cancer burden score, pathological complete response) in breast cancer patients receiving neoadjuvant chemotherapy.
- The PiCo framework can be applied to any pair of related datasets where transfer learning is appropriate, beyond cancer biology.

### Out-of-Scope Use

- **Clinical decision-making.** This model is a research tool and has not been validated for direct clinical deployment. It should not be used to make treatment decisions without further validation and regulatory approval.
- **Non-expression data.** The model has been trained and validated only on mRNA gene expression data. Use with other molecular data types (proteomics, imaging, methylation) has not been evaluated.
- **Non-cancer settings.** Generalisation to diseases outside of cancer has not been tested.
- **Individual patient prognosis.** The model predicts population-level associations and should not be interpreted as providing individual-level clinical predictions.

## Bias, Risks, and Limitations

- **Cell line representation bias.** Cancer cell line panels (DepMap, GDSC) may not represent the full diversity of human cancers. Certain cancer types and lineages are overrepresented while others have very few samples.
- **Constraint selection dependency.** Model performance varies across drugs and cancer types depending on which gene effects are selected as constraints. Optimal constraint selection cannot be determined *a priori* for a given prediction task.
- **No general-purpose representation.** iCoVAE representations are currently drug-specific due to tailored constraint selection. A single general-utility representation across all drugs is not generated.
- **Cell line–patient gap.** While PiCo demonstrates bench-to-bedside transfer, cell line models do not capture the tumour microenvironment, immune response, or pharmacokinetic factors that influence clinical outcomes.
- **Limited latent capacity.** The number of constrained dimensions is bounded by the latent space size, limiting how many biological mechanisms can be explicitly captured.
- **Demographic bias.** The model does not use or predict any demographic variables (sex, race, ethnicity). However, cell line and patient cohort composition may reflect historical biases in cancer research.

### Recommendations

- Users should validate PiCo predictions against independent clinical data before drawing clinical conclusions.
- Constraint selection should be performed carefully for each new drug or prediction task, guided by the likelihood-ratio test procedure described in Section 4.2 of the paper.
- Performance should be assessed separately for each cancer type of interest, as improvements vary across cancer types and broadly reflect the relevance of drug targets to each type.

## How to Get Started with the Model

Fastest path — run the included demos in Google Colab (no install required):

- [demo/demo_ccl.ipynb](demo/demo_ccl.ipynb) — zero-shot drug-response prediction on held-out cancer types for AZD6738, trametinib, oxaliplatin, 5-fluorouracil; reproduces Figs. 3 + 4 (aggregated boxplot, per-drug pointplot, permutation feature importance) using the paper's exact plotting code
- [demo/demo_transneo.ipynb](demo/demo_transneo.ipynb) — cell-line to patient transfer learning on the TransNEO breast cancer cohort; reproduces Figs. 5c (RCB regression, two-panel CV + ARTemis+PBCP pointplot across all six paper feature sets), 5d (pCR classification), and the Clinical+z+RNA permutation feature importance panels

Local install:

```bash
git clone https://github.com/domkirkham/pico.git
cd pico
conda create --name pico python=3.9.12 -y
conda activate pico
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

See the [repository README](README.md) for detailed usage instructions, reproduction guidance (`results_analysis/ccl_drug_resp.ipynb` and `results_analysis/transneo_treatment_resp.ipynb`), and instructions for running PiCo with custom data.

## Training Details

### Training Data

**Stage 1 (iCoVAE pretraining):**

- **Input (X):** RNA-Seq gene expression data in log₂(TPM + 1) format from DepMap 23Q2 (1019 cell lines). Top 1500 genes by variance (cell line experiments) or LINCS1000 landmark gene set (clinical experiments).
- **Auxiliary (S):** CRISPR-Cas9 gene effect estimates (Chronos model) from DepMap 23Q2, covering approximately 17,000 genes across 1019 cell lines.

**Stage 2 (drug response prediction):**

- **Target (y):** log-transformed IC50 values from GDSC2 (October 2023 release), covering 686 cell lines and 65 drugs. Approximately 499 cell lines overlap between DepMap and GDSC.

**Clinical transfer learning:**

- TransNEO breast cancer cohort (training + 5-fold cross-validation) and ARTemis+PBCP (external validation), both from Sammut et al. (2022).

All datasets are publicly available: [DepMap](https://depmap.org), [GDSC](https://cancerrxgene.org), [TransNEO](https://github.com/cclab-brca/neoadjuvant-therapy-response-predictor).

### Training Procedure

#### Preprocessing

- Gene expression values converted to log₂(TPM + 1) format.
- For cell line experiments: top 1500 genes by variance across the training set retained.
- For clinical experiments: LINCS1000 landmark gene set filtered by availability in clinical data (d = 918).
- CRISPR gene effect data used as-is from Chronos model estimates.

#### Training Hyperparameters

Hyperparameters for iCoVAE and VAE selected via 5-fold cross-validation using the Tree-structured Parzen Estimator (TPE) in Optuna (150 trials). Models refitted with 10 random seeds. Adam optimiser used throughout. Search space:

| Hyperparameter | Values |
| --- | --- |
| Batch size | {16, 32} |
| Learning rate | {1e-4, 3e-4, 1e-3} |
| Weight decay | {0, 1e-4} |
| Number of layers | {1, 2} |
| Layer width | {128, 256} |
| Latent dimension `L` | {32, 64} |
| Dropout | 0.1 |

#### Speeds, Sizes, Times

**Stage 1 (iCoVAE / VAE training).** The Stage 1 hyperparameter search (150 Optuna trials) is run **once per experiment** using AFATINIB as the reference drug. The best hyperparameters are then reused for every other drug via the `--test` flag in [`src/scripts/icovae_hopt.py`](src/scripts/icovae_hopt.py) (and likewise [`vae_hopt.py`](src/scripts/vae_hopt.py) for the VAE baseline). A per-drug "refit" therefore trains only 1 enqueued trial + 1 best-hyperparameter refit + 10 random-seed refits = 12 training runs of 300 epochs each, rather than a full 150-trial search.

**Stage 2 (PiCo prediction heads).** Hyperparameter selection is a brute-force grid search over the full search space for each regressor, with 5-fold cross-validation at every grid point, followed by a best-model refit and 10 random-seed refits. Grid sizes: ElasticNet 330 combinations, linear SVR 450, RandomForest 720. Per-fit cost is dominated by the RandomForest trials (200–600 trees per forest).

| Step | Hardware | Wall time |
| --- | --- | --- |
| Stage 1 iCoVAE hyperparameter search on AFATINIB (150 Optuna trials × 300 epochs, ≈ 3 min/trial) | 1× A100 80 GB | ~8 hours (one-off, per experiment/dataset) |
| Stage 1 iCoVAE per-drug refit at AFATINIB's hyperparameters (12 × 300 epochs) | 1× A100 80 GB | ~35 min per drug |
| Stage 1 VAE baseline (same structure) | 1× A100 80 GB | ~8 hours hopt + ~35 min per drug |
| Stage 2 ElasticNet probe (330-point grid × 5-fold CV + 12 refits ≈ 1.7k fits) | CPU | ~5–15 min per (drug × encoder) |
| Stage 2 linear SVR probe (450-point grid × 5-fold CV + 12 refits ≈ 2.3k fits) | CPU | ~15–45 min per (drug × encoder) |
| Stage 2 RandomForest probe (720-point grid × 5-fold CV + 12 refits ≈ 3.6k fits) | CPU | ~2–6 hours per (drug × encoder) |
| TransNEO iCoVAE pretraining (smaller feature set, d = 918; one shared encoder, not per-drug) | 1× A100 80 GB | ~1–8 hours (refit vs fresh hopt) |
| TransNEO Stage 2 probes | CPU | seconds to minutes |
| **Full 65-drug cell-line experiment, Stage 1 only (iCoVAE + VAE)** | A100 | **~90 GPU-hours** |
| **Full 65-drug cell-line experiment, Stage 2 only (3 regressors × 2 encoders × 65 drugs)** | CPU | **~500–900 CPU-hours**, dominated by the RandomForest probes |

Final model checkpoint size (L = 64, 2 hidden layers, width 256): ~4 MB per seed.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- **Cell line OOD:** 16 cancer types with fewer than 40 samples in DepMap, entirely unseen during training.
- **Clinical external validation:** ARTemis+PBCP cohort (independent from TransNEO training set).

#### Factors

- Cancer type (16 held-out types for OOD evaluation)
- Drug target and mechanism of action (65 drugs spanning targeted and non-targeted therapies)
- TP53 mutation status (for interpretability analyses)
- Breast cancer molecular subtype (for clinical analyses)

#### Metrics

- Spearman correlation, Pearson correlation, RMSE for regression tasks
- AUROC, AUPR, F1 score for classification (pCR prediction)
- Generalisation gap: difference between in-distribution (cross-validation) and out-of-distribution performance
- Permutation-based feature importance

### Results

#### Cell-line drug response prediction (out-of-distribution)

Aggregated across all 65 drugs and 10 random seeds, PiCo significantly outperforms all baselines in OOD Spearman correlation (PiCo-ElasticNet vs. VAE-ElasticNet: p = 6.126 × 10⁻²⁶, Wilcoxon signed-rank test) and exhibits a significantly smaller generalisation gap (p = 9.191 × 10⁻²⁹).

Selected drug-level results (OOD, best model per feature extractor):

| Drug | PiCo | VAE | PCA | Raw |
| --- | --- | --- | --- | --- |
| AZD6738 (ceralasertib) | **0.409** | 0.304 | 0.263 | 0.237 |
| Trametinib | **0.361** | 0.278 | 0.320 | 0.278 |
| Docetaxel | **0.494** | 0.440 | 0.346 | 0.385 |
| Oxaliplatin | **0.532** | 0.450 | 0.420 | 0.401 |

#### Clinical treatment response prediction (TransNEO → ARTemis+PBCP)

| Task | Metric | PiCo | VAE | Clinical only |
| --- | --- | --- | --- | --- |
| RCB score | Spearman (external), Clinical+z | **0.758** | 0.724 | 0.591 |
| pCR | AUROC (external), Clinical+z+RNA | **0.910** | 0.893 | — |

### Summary

PiCo consistently improves out-of-distribution generalisation over unconstrained baselines across the majority of drugs tested. The improvements are most pronounced for drugs with well-characterised mechanisms of action and for cancer types where the drug target is particularly relevant. In the clinical setting, iCoVAE representations trained entirely on cell line data provide meaningful predictive signal for patient treatment response without any fine-tuning.

## Model Examination

Interpretability is a central contribution of PiCo. Constrained latent dimensions capture biologically meaningful information that is directly linked to named genes and their CRISPR knockout effects:

- **CRISPR gene effect prediction.** iCoVAE accurately predicts gene effect in unseen cancer types, with higher accuracy for oncogenes (e.g., *TP53*, *MDM2*).
- **Mechanistic interpretability.** Permutation feature importance analyses demonstrate that PiCo relies on biologically relevant features for prediction (e.g., *SHOC2*/*RAF1*/*BRAF* for MEK inhibitor trametinib; *RAD17*/*RNASEH2C* for ATR inhibitor AZD6738; *MDM4*/*TTF2* for oxaliplatin).
- **Mutational context.** Constrained dimensions capture information richer than mutation status alone — for example, *p53* activity rather than solely *TP53* mutation status.
- **Clinical feature correlation.** iCoVAE features recover expected associations with clinical features (e.g., z_TP53 with ESR1 expression, z_ERBB2 with HER2 status).

## Environmental Impact

Estimates follow Lacoste et al. (2019) and the [ML CO₂ Impact Calculator](https://mlco2.github.io/impact):

- **Hardware type:** Stage 1 — NVIDIA A100 80 GB; Stage 2 — Intel Ice Lake CPU (icelake partition)
- **Hours used:** ~90 GPU-hours (Stage 1, full 65-drug cell-line experiment, iCoVAE + VAE) + ~1–16 GPU-hours (TransNEO Stage 1); ~500–900 CPU-hours (Stage 2 probes across all regressors and encoders)
- **Cloud provider / infrastructure:** On-premise HPC (Cambridge CSD3 Wilkes3 GPU cluster + icelake CPU partition)
- **Compute region:** United Kingdom
- **Carbon emitted (estimate):** ~10 kg CO₂e total (Stage 1: ~100 GPU-h × 0.4 kW × ≈ 0.2 kg CO₂/kWh UK grid average ≈ 8 kg; Stage 2: ~700 CPU-h × 0.02 kW × 0.2 kg CO₂/kWh ≈ 3 kg)

Inference on new cohorts using a pretrained encoder + fitted probe is performed in seconds on CPU; the computational cost above relates only to training/reproduction.

## Technical Specifications

### Model Architecture and Objective

The iCoVAE is a semi-supervised variational autoencoder with a split latent space: constrained dimensions `z_s` are individually linked to CRISPR gene effects via univariate linear regression, while unconstrained dimensions `z_\s` capture remaining variation under a standard Gaussian prior. The model optimises a lower bound on the joint marginal likelihood `p(x, s)` augmented with a bounded importance-weighting term (tanh-saturated log-ratio, Eq. 1 of the paper) that stabilises training for continuous-valued auxiliary data. A calibrated decoder with a single shared learned variance is used. See Section 4.1 of the paper for full mathematical specification.

### Compute Infrastructure

#### Hardware

- **Training:** NVIDIA A100 80 GB GPUs on AMD EPYC 7763 nodes (Cambridge CSD3 Wilkes3 cluster, ConnectX-6 interconnect), 1 TB RAM per node.
- **Inference / demo:** any modern CPU with ≥ 8 GB RAM is sufficient; the included Colab demo notebooks run end-to-end in minutes.

#### Software

- Python 3.9.12
- PyTorch 2.0.1 (with `torch.compile`)
- CUDA 11.8, cuDNN 8.9
- scikit-learn (ElasticNet, SVR, random forest, logistic regression)
- Optuna (hyperparameter optimisation)
- statsmodels (constraint selection via likelihood-ratio testing)

See [requirements.txt](requirements.txt) for the full pinned dependency list.

## Citation

**BibTeX:**

```bibtex
@misc{kirkham_interventionally-guided_2025,
    title = {Interventionally-guided representation learning for robust and
             interpretable AI models in cancer medicine},
    url = {https://www.biorxiv.org/content/10.1101/2025.07.21.662350},
    doi = {10.1101/2025.07.21.662350},
    publisher = {bioRxiv},
    author = {Kirkham, Dom and Masina, Riccardo and Sammut, Stephen-John
              and Mukherjee, Sach and Rueda, Oscar M.},
    month = jul,
    year = {2025},
}
```

**APA:**

Kirkham, D., Masina, R., Sammut, S.-J., Mukherjee, S., & Rueda, O. M. (2025). *Interventionally-guided representation learning for robust and interpretable AI models in cancer medicine.* bioRxiv. <https://doi.org/10.1101/2025.07.21.662350>

## Model Card Authors

Dom Kirkham

## Model Card Contact

<dom.kirkham@mrc-bsu.cam.ac.uk>
