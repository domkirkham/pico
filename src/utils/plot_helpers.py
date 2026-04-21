"""Shared plotting helpers for the PiCo demo notebooks.

These functions are lifted near-verbatim from the paper's
``results_analysis/ccl_drug_resp.ipynb`` (cells 39 and 54), so the demo plots look
identical to the published figures — only the data passed in is a subset of what
the paper used.

* ``plot_perf_comp(perf_df, plot_metric, save_to=None, ax=None)``
    Reproduces Fig. 3 / Fig. B1 boxplot showing performance vs. regression
    model, grouped by feature extractor.
* ``plot_feature_importance(pi_df, target, fe, model, experiment, metric, ...)``
    Reproduces Fig. 4 permutation-FI bar plot (vertical, ranked by metric).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib import rc
from matplotlib.colors import rgb_to_hsv
from matplotlib.lines import Line2D
import pandas as pd


_SOURCE_SANS_URLS = {
    "SourceSans3-Regular.ttf":
        "https://github.com/adobe-fonts/source-sans/raw/release/TTF/SourceSans3-Regular.ttf",
    "SourceSans3-Bold.ttf":
        "https://github.com/adobe-fonts/source-sans/raw/release/TTF/SourceSans3-Bold.ttf",
    "SourceSans3-It.ttf":
        "https://github.com/adobe-fonts/source-sans/raw/release/TTF/SourceSans3-It.ttf",
}


def apply_paper_style(font_dir: str | Path | None = None) -> None:
    """Match the published figures' Source Sans 3 typography and seaborn theme.

    Lifted near-verbatim from cell 2 of ``results_analysis/ccl_drug_resp.ipynb``.
    Looks for the three Source Sans 3 TTFs in ``font_dir`` first (repo's
    ``src/fonts/`` by default), then in a per-user cache at
    ``~/.cache/pico_demo_fonts/``, and if still missing downloads them from
    the public Adobe Source Sans repository. Downloads are attempted once and
    cached; if they fail (e.g. offline Colab instance), the function silently
    falls back to the default font.
    """
    import os
    import urllib.request

    search_dirs: list[Path] = []
    if font_dir is not None:
        search_dirs.append(Path(font_dir))
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME",
                                    Path.home() / ".cache")) / "pico_demo_fonts"
    search_dirs.append(cache_dir)

    for fname, url in _SOURCE_SANS_URLS.items():
        resolved = next((d / fname for d in search_dirs if (d / fname).exists()), None)
        if resolved is None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            target = cache_dir / fname
            try:
                urllib.request.urlretrieve(url, target)
                resolved = target
            except Exception:
                resolved = None
        if resolved is not None:
            fm.fontManager.addfont(str(resolved))

    plt.style.use("default")
    sns.set_theme(
        context="paper", style="ticks", palette="colorblind",
        rc={
            "axes.linewidth": 1,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "axes.edgecolor": "grey",
            "xtick.labelcolor": "black",
            "xtick.color": "grey",
            "ytick.labelcolor": "black",
            "ytick.color": "grey",
        },
    )
    rc("font", **{"family": "sans-serif", "sans-serif": ["Source Sans 3"]})
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.it"] = "Source Sans 3:italic"


# ===== Fig. 3 / B1 — Performance comparison boxplot ======================

FE_DICT = {"none": "Raw", "vae": "VAE", "icovae": "iCoVAE", "pca": "PCA"}
MODEL_DICT = {
    "SVR": "SVR",
    "ElasticNet": "ElasticNet",
    "nn": "MLP",
    "RandomForestRegressor": "RF",
}


def plot_perf_comp(
    perf_df: pd.DataFrame,
    plot_metric: str,
    *,
    target_dict: dict | None = None,
    save_to: str | Path | None = None,
    figsize: tuple = (3.5, 3),
):
    """Single-panel performance boxplot (Fig. 3 of the paper).

    Lifted near-verbatim from ``ccl_drug_resp.ipynb`` cell 39. Aggregates each
    (fe, model, target) cell to its mean across seeds before plotting.

    Args:
        perf_df: Long-form ``pd.DataFrame`` with columns
            ``[fe, model, target, seed, <plot_metric>]`` as produced by
            ``PerfComp.calculate_perf``.
        plot_metric: e.g. ``"spearman_r"`` (held-out test) or
            ``"spearman_r_val"`` (cross-validation).
        target_dict: optional ``{drug: drug_class}`` mapping — only used for
            colouring/legend headers; can be ``None``.
        save_to: optional path; ``.png`` and ``.svg`` saved alongside.
        figsize: matplotlib figure size.
    """
    pal = sns.color_palette("colorblind", n_colors=5)
    palette = {"iCoVAE": pal[0], "VAE": pal[1], "PCA": pal[2], "Raw": "grey"}

    perf_df_plot = perf_df.reset_index().copy()
    perf_df_plot["fe"] = perf_df_plot["fe"].map(FE_DICT)
    perf_df_plot["model"] = perf_df_plot["model"].map(MODEL_DICT)
    if target_dict is not None:
        perf_df_plot["drug_target"] = perf_df_plot["target"].map(target_dict)
    if plot_metric.endswith("_val"):
        perf_df_plot = perf_df_plot.dropna(axis=0)

    hue_order_full = ["iCoVAE", "VAE", "PCA", "Raw"]
    hue_order = [h for h in hue_order_full if h in perf_df_plot["fe"].unique()]
    order_full = ["ElasticNet", "SVR", "RF", "MLP"]
    order = [m for m in order_full if m in perf_df_plot["model"].unique()]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(
        data=perf_df_plot.groupby(["fe", "model", "target"]).mean(numeric_only=True),
        x="model", y=plot_metric, ax=ax,
        hue="fe", palette=palette,
        width=0.8, order=order, hue_order=hue_order,
        errorbar=("sd", 1), err_kws={"linewidth": 1}, estimator="mean",
    )
    sns.stripplot(
        data=perf_df_plot.groupby(["fe", "model", "target"]).mean(numeric_only=True),
        x="model", y=plot_metric, ax=ax,
        hue="fe", dodge=True, palette=palette,
        alpha=0.3, jitter=True, order=order, linewidth=0.2,
        hue_order=hue_order, size=4,
    )
    if plot_metric in ("spearman_r", "spearman_r_val"):
        ax.set_ylabel("Spearman correlation")
    elif plot_metric in ("pearson_r", "pearson_r_val"):
        ax.set_ylabel("Pearson correlation")
    elif plot_metric in ("rmse", "rmse_val"):
        ax.set_ylabel("RMSE")
    ax.set_xlabel("Regression model")
    sns.despine(ax=ax)

    fe_handles = [
        Line2D([], [], marker="o", linestyle="none", color=palette[fe], label=fe)
        for fe in hue_order
    ]
    leg = ax.legend(
        handles=fe_handles, bbox_to_anchor=(0.4, 1.25), loc="upper center",
        frameon=False, ncol=4, title="Feature extractor",
    )
    ax.set_ylim(0)

    fig.canvas.draw()
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to.with_suffix(".png"), dpi=600,
                    bbox_inches="tight", bbox_extra_artists=(leg,))
        fig.savefig(save_to.with_suffix(".svg"),
                    bbox_inches="tight", bbox_extra_artists=(leg,))
    return fig, ax


# ===== Fig. 3 right panel — Per-drug pointplot ===========================

MODEL_MARKERS = {"SVR": "o", "ElasticNet": "s",
                 "Random forest": "D", "MLP": "^"}


def plot_perf_pointplot(
    perf_df: pd.DataFrame,
    plot_metric: str,
    *,
    save_to: str | Path | None = None,
    figsize: tuple = (11, 4),
):
    """Per-drug pointplot of performance, every (FE × model) shown separately.

    Lifted from cell 42 of ``ccl_drug_resp.ipynb`` (the ``single_drug=None``
    branch), simplified for the demo's 4-drug subset.
    """
    pal = sns.color_palette("colorblind", n_colors=5)
    palette = {"iCoVAE": pal[0], "VAE": pal[1], "PCA": pal[2], "Raw": "grey"}

    perf_df_plot = perf_df.reset_index().copy()
    perf_df_plot["fe"] = perf_df_plot["fe"].map(FE_DICT)
    # Map model names; keep "Random forest" for the legend marker dict.
    perf_df_plot["model"] = perf_df_plot["model"].replace(
        {"RandomForestRegressor": "Random forest",
         "nn": "MLP", "ElasticNet": "ElasticNet", "SVR": "SVR"}
    )
    if plot_metric.endswith("_val"):
        perf_df_plot = perf_df_plot.dropna(axis=0)

    perf_df_plot["fe_model"] = perf_df_plot["fe"] + "__" + perf_df_plot["model"]

    hue_order_full = ["iCoVAE", "VAE", "PCA", "Raw"]
    hue_order = [h for h in hue_order_full if h in perf_df_plot["fe"].unique()]
    fe_model_order = [
        f"{fe}__{m}" for fe in hue_order for m in MODEL_MARKERS
        if f"{fe}__{m}" in perf_df_plot["fe_model"].values
    ]
    palette_fe_model = {k: palette[k.split("__")[0]] for k in fe_model_order}
    markers_fe_model = [MODEL_MARKERS[k.split("__")[1]] for k in fe_model_order]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Order drugs by mean iCoVAE performance (descending for ρ, ascending for RMSE).
    is_rmse = plot_metric.startswith("rmse")
    target_means = (
        perf_df_plot[perf_df_plot["fe"] == "iCoVAE"]
        .groupby("target")[plot_metric].mean()
        .sort_values(ascending=is_rmse)
    )
    order = target_means.index.tolist()

    sns.pointplot(
        data=perf_df_plot,
        y=plot_metric, x="target",
        hue="fe_model", order=order, hue_order=fe_model_order,
        palette=palette_fe_model, markers=markers_fe_model,
        dodge=0.4, markersize=4, linestyle="none",
        errorbar=("sd", 1), err_kws={"linewidth": 1.5, "alpha": 0.5},
        capsize=0.25, ax=ax,
    )

    if plot_metric in ("spearman_r", "spearman_r_val"):
        ax.set_ylabel("Spearman correlation")
    elif plot_metric in ("pearson_r", "pearson_r_val"):
        ax.set_ylabel("Pearson correlation")
    elif plot_metric in ("rmse", "rmse_val"):
        ax.set_ylabel("RMSE")
    ax.set_xlabel("")
    ax.grid(visible=True, axis="y")

    fe_handles = [
        Line2D([], [], marker="o", linestyle="none", color=palette[fe], label=fe)
        for fe in hue_order
    ]
    model_handles = [
        Line2D([], [], marker=mk, linestyle="none", color="darkgrey", label=md)
        for md, mk in MODEL_MARKERS.items()
        if md in perf_df_plot["model"].unique()
    ]
    leg1 = ax.legend(handles=fe_handles, bbox_to_anchor=(0.1, 1.2),
                     loc="upper left", frameon=False, ncol=4,
                     title="Feature extractor")
    leg2 = ax.legend(handles=model_handles, bbox_to_anchor=(0.5, 1.2),
                     loc="upper left", frameon=False, ncol=4,
                     title="Regression model")
    ax.add_artist(leg1); ax.add_artist(leg2)
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.despine(ax=ax)
    plt.xticks(rotation=90)
    plt.tight_layout()

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to.with_suffix(".png"), dpi=600,
                    bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
        fig.savefig(save_to.with_suffix(".svg"),
                    bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    return fig, ax


# ===== Fig. 4 — Permutation feature importance ============================

CBAR_LABELS = {
    "r": r"$\Delta \% \rho_{{p}}$",
    "s": r"$\Delta \% \rho_{{s}}$",
    "rmse": r"$\Delta\%$ RMSE",
}
METRIC_LABELS = {
    "s": f"Feature importance\n({CBAR_LABELS['s']})",
    "r": f"Feature importance\n({CBAR_LABELS['r']})",
    "rmse": f"Feature importance\n({CBAR_LABELS['rmse']})",
}


def plot_feature_importance(
    pi_df: pd.DataFrame,
    target: str,
    fe: str,
    model: str,
    *,
    metric: str = "s",
    n_feats: int = 32,
    save_to: str | Path | None = None,
):
    """Permutation feature-importance bar plot (Fig. 4 of the paper).

    Lifted near-verbatim from ``ccl_drug_resp.ipynb`` cell 54.

    Args:
        pi_df: ``pd.DataFrame`` produced by ``calculate_feat_imps`` —
            columns include ``dim, seed, r, r_perm, s, s_perm, rmse, rmse_perm``.
        target, fe, model: only used for the saved-file name.
        metric: ``"s"`` (Spearman), ``"r"`` (Pearson), or ``"rmse"``.
        n_feats: max number of latent dims to show (default 32 — paper).
        save_to: optional path stem; ``.png`` and ``.svg`` saved.
    """
    pal = sns.color_palette("colorblind")
    fig, ax = plt.subplots(1, 1, figsize=(3, n_feats / 6))

    pi_df_plot = pi_df.copy()
    pi_df_plot["fi_r"] = 100 * (pi_df_plot["r_perm"] - pi_df_plot["r"]) / pi_df_plot["r"]
    pi_df_plot["fi_s"] = 100 * (pi_df_plot["s_perm"] - pi_df_plot["s"]) / pi_df_plot["s"]
    pi_df_plot["fi_rmse"] = (
        100 * (pi_df_plot["rmse_perm"] - pi_df_plot["rmse"]) / pi_df_plot["rmse"]
    )
    pi_df_plot["dim"] = pi_df_plot["dim"].apply(lambda x: x.split("_")[1])
    pi_df_plot["dim"] = pi_df_plot["dim"].apply(lambda x: rf"$z_{{{x}}}$")

    pi_df_plot = pi_df_plot.groupby(["dim", "seed"]).mean(numeric_only=True).reset_index()
    plot_order = (
        pi_df_plot.groupby("dim").mean(numeric_only=True)
        .sort_values(by=f"{metric}_perm").index.tolist()
    )
    pi_df_plot = pi_df_plot[pi_df_plot["dim"].isin(plot_order[:n_feats])]
    plot_order = plot_order[:n_feats]
    pi_df_plot = pi_df_plot.set_index("dim").loc[plot_order].reset_index()

    if metric == "rmse":
        pal_cm = sns.diverging_palette(
            rgb_to_hsv(pal[1])[0] * 360, rgb_to_hsv(pal[0])[0] * 360,
            s=100, center="light", as_cmap=True,
        )
    else:
        pal_cm = sns.diverging_palette(
            rgb_to_hsv(pal[0])[0] * 360, rgb_to_hsv(pal[1])[0] * 360,
            s=100, center="light", as_cmap=True,
        )

    sns.barplot(
        data=pi_df_plot, y="dim", x=f"fi_{metric}",
        estimator="mean", errorbar=("sd", 1), ax=ax,
        capsize=0.25, dodge=False,
        err_kws={"linewidth": 1, "alpha": 0.5},
    )
    sns.stripplot(
        data=pi_df_plot, y="dim", x=f"fi_{metric}",
        color="black", size=2, alpha=0.5, ax=ax, jitter=True,
    )
    ax.axvline(0, linestyle="--", color="grey", alpha=0.5)
    ax.set_ylim(len(plot_order) - 0.5, -0.5)
    y_tick_pos = list(range(len(plot_order)))
    ax.set_yticks([], [])
    ax.set_ylabel("")
    ax.set_xlabel(METRIC_LABELS[metric])

    for i, bar in enumerate(ax.patches):
        if bar.get_height() > 0:
            line = ax.lines[i]
            xmin, xmax = line.get_xdata()[0], line.get_xdata()[-1]
            if xmin > 0 and xmax > 0:
                bar.set_color(pal_cm(0.75)); colour = pal_cm(1.0)
            elif xmin < 0 and xmax < 0:
                bar.set_color(pal_cm(0.25)); colour = pal_cm(0.0)
            else:
                bar.set_color("lightgrey"); colour = "grey"
        ax.text(
            ax.get_xlim()[1] * 2.0, y_tick_pos[i], plot_order[i],
            ha="left", va="center", color=colour, fontsize=10,
        )
    if metric != "rmse":
        ax.invert_xaxis()
    sns.despine(ax=ax, bottom=True, left=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.05)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(save_to.with_suffix(".svg"), bbox_inches="tight")
    return fig, ax


# ===== Live permutation feature importance (model-centric demo) ==========

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score, auc as sk_auc,
)


def live_permutation_fi(
    z_test_scaled,
    y_test,
    feature_names,
    coeffs,
    intercept,
    seed,
    *,
    n_iter: int = 100,
    classification: bool = False,
    rng=None,
):
    """Shuffle each column of a feature matrix ``n_iter`` times and record the
    impact on regression metrics. Matches ``utils.comp_utils.calculate_feat_imps``
    so the resulting dataframe drops into ``plot_feature_importance``.

    Args:
        z_test_scaled: ``np.ndarray`` of shape ``(n_samples, n_features)``,
            already scaled with the StandardScaler fit at training time.
        y_test: ``np.ndarray`` of shape ``(n_samples,)`` — ground-truth labels.
        feature_names: list of column labels, one per column of
            ``z_test_scaled``. Pass e.g. ``['z_0', 'z_1', ...]``.
        coeffs, intercept: fitted linear-model weights. For ElasticNet,
            ``coeffs = model.coef_``; for LogisticRegression,
            ``coeffs = model.coef_[0]``.
        seed: per-seed seed ID (goes into the output dataframe).
        n_iter: number of permutation repeats per feature.
        classification: if True, computes AUROC (for LogisticRegression);
            otherwise returns Pearson / Spearman / RMSE columns.
        rng: optional ``np.random.Generator``. If None, a fresh one seeded on
            ``seed`` is used.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    coeffs = np.asarray(coeffs).reshape(-1)
    intercept = float(np.asarray(intercept).reshape(-1)[0])

    base_pred = z_test_scaled @ coeffs + intercept
    if classification:
        base_pred = 1.0 / (1.0 + np.exp(-base_pred))
        base_auroc = max(
            roc_auc_score(y_test, base_pred),
            roc_auc_score(y_test, -base_pred),
        )
    else:
        nas = np.isnan(y_test) | np.isnan(base_pred)
        base_r = pearsonr(base_pred[~nas], y_test[~nas])[0]
        base_s = spearmanr(base_pred[~nas], y_test[~nas])[0]
        base_rmse = float(np.sqrt(np.mean((base_pred[~nas] - y_test[~nas]) ** 2)))

    records = []
    n = z_test_scaled.shape[0]
    for j, col_name in enumerate(feature_names):
        for it in range(n_iter):
            perm = rng.permutation(n)
            z_perm = z_test_scaled.copy()
            z_perm[:, j] = z_test_scaled[perm, j]
            pred = z_perm @ coeffs + intercept
            if classification:
                pred = 1.0 / (1.0 + np.exp(-pred))
                auroc = max(
                    roc_auc_score(y_test, pred),
                    roc_auc_score(y_test, -pred),
                )
                records.append({"dim": col_name, "auroc_perm": auroc,
                                "auroc": base_auroc, "seed": seed, "iter": it})
            else:
                nas = np.isnan(y_test) | np.isnan(pred)
                records.append({
                    "dim": col_name,
                    "r_perm": pearsonr(pred[~nas], y_test[~nas])[0],
                    "s_perm": spearmanr(pred[~nas], y_test[~nas])[0],
                    "rmse_perm": float(np.sqrt(np.mean((pred[~nas] - y_test[~nas]) ** 2))),
                    "r": base_r, "s": base_s, "rmse": base_rmse,
                    "seed": seed, "iter": it,
                })
    return pd.DataFrame(records)


# ===== TransNEO Fig. 5c/d — paper feature-set facet pointplot ============
# Mirrors transneo_treatment_resp.ipynb cells 40 + 43.

import json

# Same six suffixes used in transneo_treatment_resp.ipynb cell 38.
TRANSNEO_FEAT_SETS = {
    "RNA":              "_PGR.log2.tpm_11_norep",
    "Rep":              "",
    "Clinical+Rep":     "_Size.at.diagnosis_7",
    "Clinical+Rep+RNA": "_Size.at.diagnosis_18",
    "Clinical+RNA":     "_Size.at.diagnosis_18_norep",
    "Clinical":         "_Size.at.diagnosis_7_norep",
}
TRANSNEO_REP_TYPES = {"vae": "VAE", "icovae_MCL1_16": "PiCo"}


def load_transneo_perf(res_root: str | Path, target: str, model_type: str,
                       *, seeds=tuple(range(10, 110, 10)), hopt_seed: int = 4563):
    """Replicate cell 40 of ``transneo_treatment_resp.ipynb`` to populate
    ``test_metrics_df`` (one row per seed × variant) and ``val_metrics_df``
    (one row per variant, computed from the 5 CV folds of seed 4563)."""
    res_root = Path(res_root)
    val_rows, test_frames = [], []
    for feat_set, ext in TRANSNEO_FEAT_SETS.items():
        for rep_type, rep_label in TRANSNEO_REP_TYPES.items():
            sub = res_root / f"{model_type}_{rep_type + ext}"
            try:
                preds_val = pd.concat(
                    [pd.read_csv(sub / f"z_pred_val_{f}_best_s{hopt_seed}.csv")
                     for f in range(5)], axis=0,
                )
                pv = preds_val[["pred_0", "y"]].dropna()
                if target == "RCB.score":
                    val_rows.append({"rep_type": rep_label, "model_type": model_type,
                                     "feat_sets": feat_set, "n": len(pv),
                                     "val_spearmanr": spearmanr(pv["pred_0"], pv["y"])[0],
                                     "val_pearsonr": pearsonr(pv["pred_0"], pv["y"])[0],
                                     "val_rmse": float(np.sqrt(((pv["pred_0"] - pv["y"]) ** 2).mean()))})
                else:
                    pa = pv.copy(); pa["y"] = (pa["y"] == 0.0).astype(float)
                    auroc = max(roc_auc_score(pa["y"], pa["pred_0"]),
                                roc_auc_score(pa["y"], -pa["pred_0"]))
                    pr, rc, _ = precision_recall_curve(pa["y"], pa["pred_0"], pos_label=1)
                    val_rows.append({"rep_type": rep_label, "model_type": model_type,
                                     "feat_sets": feat_set, "n": len(pv),
                                     "val_auroc": auroc,
                                     "val_aupr": sk_auc(rc, pr),
                                     "val_f1": f1_score(pa["y"], (pa["pred_0"] > 0.5).astype(float))})

                tm = pd.read_csv(sub / "test_metrics.csv").assign(
                    rep_type=rep_label, model_type=model_type,
                    feat_sets=feat_set, seed=list(seeds),
                )
                test_frames.append(tm)
            except FileNotFoundError:
                continue

    test_df = pd.concat(test_frames, axis=0).assign(dataset="test")
    val_df = pd.DataFrame(val_rows).assign(dataset="cv")
    return val_df, test_df


def plot_transneo_perf(val_df: pd.DataFrame, test_df: pd.DataFrame,
                       *, target: str, model: str, metric: str,
                       save_to: str | Path | None = None):
    """Two-panel CV vs external-validation pointplot — paper cell 43.

    ``metric`` is e.g. ``"spearmanr"``, ``"auroc"``, ``"f1"``.
    Plotting code is lifted near-verbatim from cell 43.
    """
    palette = sns.color_palette("colorblind")
    pal_cv = pal_ev = {"VAE": palette[1], "PiCo": palette[0], "NA": "grey"}
    plot_order = ["Clinical", "RNA", "Clinical+RNA", "Rep",
                  "Clinical+Rep", "Clinical+Rep+RNA"]
    facet_order = ["TransNEO\nCross-validation", "ARTemis+PBCP\nExternal validation"]
    metric_hopt, metric_test = f"val_{metric}", f"test_{metric}"

    hopt_df_plot = val_df[val_df["model_type"] == model].copy()
    hopt_df_plot["rep_type_hue"] = hopt_df_plot["rep_type"].copy()
    hopt_df_plot.loc[
        hopt_df_plot["feat_sets"].isin(["Clinical+RNA", "Clinical", "RNA"]),
        "rep_type_hue",
    ] = "NA"

    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.5))
    sns.pointplot(
        data=hopt_df_plot, y="feat_sets", x=metric_hopt, hue="rep_type_hue",
        order=plot_order, palette=pal_cv,
        linestyle="--", linewidth=1, markersize=4, marker="o",
        errorbar=("sd", 1), capsize=0.25,
        err_kws={"linewidth": 1.5, "alpha": 0.25}, ax=axes[0],
    )

    metrics_df_plot = test_df[test_df["model_type"] == model].copy()
    metrics_df_plot["rep_type_hue"] = metrics_df_plot["rep_type"].copy()
    metrics_df_plot.loc[
        metrics_df_plot["feat_sets"].isin(["Clinical+RNA", "Clinical", "RNA"]),
        "rep_type_hue",
    ] = "NA"
    sns.pointplot(
        data=metrics_df_plot, y="feat_sets", x=metric_test, hue="rep_type_hue",
        order=plot_order, palette=pal_ev,
        linestyle="-", linewidth=1, markersize=4, marker="o",
        errorbar=("sd", 1), capsize=0.25,
        err_kws={"linewidth": 1.2, "alpha": 0.5}, ax=axes[1], legend=False,
    )

    line_vae = Line2D([0], [0], label="VAE", color=palette[1], linestyle="-", linewidth=1)
    line_pico = Line2D([0], [0], label="PiCo", color=palette[0], linestyle="-", linewidth=1)
    handles, _ = axes[0].get_legend_handles_labels()
    leg1 = axes[0].legend(handles=handles[3:], bbox_to_anchor=(1.0, 1.6),
                          loc="upper center", frameon=False, ncol=3, fontsize=10)
    leg2 = axes[0].legend(handles=[line_vae, line_pico],
                          bbox_to_anchor=(-0.35, 1.45), loc="upper center",
                          frameon=False, ncol=1, fontsize=10)
    axes[0].add_artist(leg1); axes[0].add_artist(leg2)

    names_dict = {"Clinical": "Clinical", "RNA": "RNA",
                  "Clinical+RNA": "Clinical+RNA", "Rep": r"$\mathbf{z}$",
                  "Clinical+Rep": r"Clinical+$\mathbf{z}$",
                  "Clinical+Rep+RNA": r"Clinical+$\mathbf{z}$+RNA"}
    axes[0].set_yticks(range(len(plot_order)))
    axes[0].set_yticklabels([names_dict[lbl] for lbl in plot_order])

    sns.despine()
    metric_label = {
        "spearmanr": "Spearman correlation", "pearsonr": "Pearson correlation",
        "rmse": "RMSE", "auroc": "AUROC", "aupr": "AUPR", "f1": "F1 score",
    }[metric]
    for i, ax in enumerate(axes):
        ax.set_ylabel("")
        ax.set_xlabel(metric_label, fontsize=10)
        ax.set_title("")
        ax.text(0.5, 1.1, facet_order[i], transform=ax.transAxes,
                ha="center", fontsize=10)
        ax.grid(visible=True, axis="x")
        ax.tick_params(labelsize=10)
        if i > 0:
            sns.despine(left=True, ax=ax)
            ax.tick_params(top=False, bottom=True, left=False, right=False,
                           labelleft=False, labelbottom=True, labelsize=10)
    fig.tight_layout()
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(save_to.with_suffix(".svg"), bbox_inches="tight")
    return fig, axes
