# Setup
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

# from lifelines.utils import k_fold_cross_validation
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, fisher_exact


from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
)

# from scipy._lib._bunch import _make_tuple_bunch

# SignificanceResult = _make_tuple_bunch('SignificanceResult',
#                                        ['statistic', 'pvalue'], [])

from functools import partial

from utils.data_utils import gene_column_renamer, Manual, process_data

# from models.baselines import SingleGeneLasso, SingleGeneLinear, SingleGeneSVR
import torch
import json

from typing import Any, List

import matplotlib.font_manager as fm

# Download the font
# wd_path = "/home/dk538/rds/hpc-work/graphdep"
# font_url = "https://github.com/adobe-fonts/source-sans/blob/release/TTF/SourceSans3-Regular.ttf?raw=True"
# font_path = f"{wd_path}/results_analysis/figures/SourceSans3-Regular.ttf"  # Specify where to save the font
# font_bold_url = "https://github.com/adobe-fonts/source-sans/blob/release/TTF/SourceSans3-Bold.ttf?raw=True"
# font_bold_path = f"{wd_path}/results_analysis/figures/SourceSans3-Bold.ttf"  # Specify where to save the font
# font_it_url = "https://github.com/adobe-fonts/source-sans/blob/release/TTF/SourceSans3-It.ttf?raw=True"
# font_it_path = f"{wd_path}/results_analysis/figures/SourceSans3-It.ttf"  # Specify where to save the font
# urllib.request.urlretrieve(font_url, font_path)
# urllib.request.urlretrieve(font_bold_url, font_bold_path)
# urllib.request.urlretrieve(font_it_url, font_it_path)

# in a terminal, run
# cp ~/rds/hpc-work/graphdep/results_analysis/figures/*ttf ~/.local/share/fonts
# fc-cache -f -v
# rm -fr ~/.cache/matplotlib

# Then restart Jupyter kernel

fm.findfont("Source Sans 3", rebuild_if_missing=True)
# fm.findfont("Source Sans 3:style=italic", rebuild_if_missing=True)

# Set font globally for Matplotlib


plt.style.use("default")
sns.set_theme(
    context="paper",
    style="ticks",
    palette="colorblind",
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


def ensg_column_renamer(name):
    return name.split(".")[0]


def scanb_row_renamer(name):
    return name.split(".")[0]


def tte_cutoff(x, event, t_sfx, e_sfx, k, timepoint):
    return x[
        ~((x[f"{event}_{t_sfx}"] < (k * timepoint)) & (x[f"{event}_{e_sfx}"] == 0))
    ]


def _to_list(x: Any) -> List[Any]:
    if not isinstance(x, list):
        return [x]
    return x


def rep_renamer(x, constraints, prefix="z"):
    if constraints is None:
        return x
    else:
        dim = int(x.split("_")[1])
        if dim < len(constraints):
            return f"{prefix}_{constraints[dim]}"
        else:
            return x


class SignificanceResult:
    def __init__(self, statistic, pvalue):
        self.statistic = statistic
        self.pvalue = pvalue


def cont_tab(x_rd, x_pcr, feat):
    return pd.merge(
        x_rd[feat].value_counts(),
        x_pcr[feat].value_counts(),
        left_index=True,
        right_index=True,
        how="outer",
    ).fillna(0)


def fisher_auc(cont_tab, x, y, feat):
    try:
        res = fisher_exact(cont_tab)
        auc = roc_auc_score(y, x[feat])
        auc_neg = roc_auc_score(y, -1 * x[feat])
        auc = np.max([auc, auc_neg])
    except UserWarning("AUC calculcation failed."):
        res = SignificanceResult(0.0, 1.0)
        auc = 0.0
    return res, auc


def mwu_auc(x, y, x_rd, x_pcr, feat):
    res = mannwhitneyu(x_rd[feat], x_pcr[feat])
    auc = roc_auc_score(y, x[feat])
    auc_neg = roc_auc_score(y, -1 * x[feat])
    auc = np.max([auc, auc_neg])
    return res, auc


class DropCollinear(BaseEstimator, TransformerMixin):
    def __init__(self, thresh):
        self.uncorr_columns = None
        self.thresh = thresh

    def fit(self, X, y):
        cols_to_drop = []

        # Find variables to remove
        X_corr = X.corr()
        large_corrs = X_corr > self.thresh
        indices = np.argwhere(large_corrs.values)
        indices_nodiag = np.array([[m, n] for [m, n] in indices if m != n])

        if indices_nodiag.size > 0:
            indices_nodiag_lowfirst = np.sort(indices_nodiag, axis=1)
            correlated_pairs = np.unique(indices_nodiag_lowfirst, axis=0)
            resp_corrs = np.array(
                [
                    [
                        np.abs(spearmanr(X.iloc[:, m], y).correlation),
                        np.abs(spearmanr(X.iloc[:, n], y).correlation),
                    ]
                    for [m, n] in correlated_pairs
                ]
            )
            element_to_drop = np.argmin(resp_corrs, axis=1)
            list_to_drop = np.unique(
                correlated_pairs[range(element_to_drop.shape[0]), element_to_drop]
            )
            cols_to_drop = X.columns.values[list_to_drop]

        cols_to_keep = [c for c in X.columns.values if c not in cols_to_drop]
        self.uncorr_columns = cols_to_keep

        return self

    def transform(self, X):
        return X[self.uncorr_columns]

    def get_params(self, deep=False):
        return {"thresh": self.thresh}


def k_fold_cross_validation(
    fitters,
    df,
    duration_col,
    event_col=None,
    k=5,
    scoring_method="log_likelihood",
    fitter_kwargs={},
    seed=None,
):  # pylint: disable=dangerous-default-value,too-many-arguments,too-many-locals
    """
    Perform cross validation on a dataset. If multiple models are provided,
    all models will train on each of the k subsets. Altered version of lifelines function, changed to return coeffs

    Parameters
    ----------
    fitters: model
      one or several objects which possess a method: ``fit(self, data, duration_col, event_col)``
      Note that the last two arguments will be given as keyword arguments,
      and that event_col is optional. The objects must also have
      the "predictor" method defined below.
    df: DataFrame
      a Pandas DataFrame with necessary columns `duration_col` and (optional) `event_col`, plus
      other covariates. `duration_col` refers to the lifetimes of the subjects. `event_col`
      refers to whether the 'death' events was observed: 1 if observed, 0 else (censored).
    duration_col: string
        the name of the column in DataFrame that contains the subjects'
        lifetimes.
    event_col: string, optional
        the  name of the column in DataFrame that contains the subjects' death
        observation. If left as None, assume all individuals are uncensored.
    k: int
      the number of folds to perform. n/k data will be withheld for testing on.
    scoring_method: str
        one of {'log_likelihood', 'concordance_index'}
        log_likelihood: returns the average unpenalized partial log-likelihood.
        concordance_index: returns the concordance-index
    fitter_kwargs:
      keyword args to pass into fitter.fit method.
    seed: fix a seed in np.random.seed

    Returns
    -------
    results: list
      (k,1) list of scores for each fold. The scores can be anything.

    See Also
    ---------
    lifelines.utils.sklearn_adapter.sklearn_adapter

    """
    # Make sure fitters is a list
    fitters = _to_list(fitters)

    # Each fitter has its own scores
    fitter_scores_ll = [[] for _ in fitters]

    # Each fitter has its own scores
    fitter_scores_c = [[] for _ in fitters]

    # Each fitter has its own summaries
    fitter_summs = [[] for _ in fitters]

    # Each fitter has its own assumptions check
    # fitter_assumps = [[] for _ in fitters]

    n, _ = df.shape
    df = df.copy()

    if seed is not None:
        np.random.seed(seed)

    if event_col is None:
        event_col = "E"
        df[event_col] = 1.0

    df = df.reindex(np.random.permutation(df.index)).sort_values(event_col)

    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]

    # testing_columns = df.columns.drop([duration_col, event_col])

    for i in range(1, k + 1):
        ix = assignments == i
        training_data = df.loc[~ix]
        testing_data = df.loc[ix]

        z_columns = training_data.columns[
            training_data.columns.str.startswith("z_")
        ].tolist()
        # Also want to standardise other continous variables
        cont_columns = training_data.columns[
            training_data.columns.isin(["Age", "Size.mm"])
        ].tolist()
        z_columns.extend(cont_columns)

        if len(z_columns) > 0:
            training_data_z = training_data.loc[:, z_columns]
            testing_data_z = testing_data.loc[:, z_columns]

            # standardise train and test data, only z columns
            scaler = StandardScaler().fit(training_data_z)
            training_data_z = scaler.transform(training_data_z)
            testing_data_z = scaler.transform(testing_data_z)

            training_data.loc[:, z_columns] = training_data_z
            testing_data.loc[:, z_columns] = testing_data_z

        for fitter, scores_ll, scores_c, summs in zip(
            fitters, fitter_scores_ll, fitter_scores_c, fitter_summs
        ):
            # fit the fitter to the training data
            fitter.fit(
                training_data,
                duration_col=duration_col,
                event_col=event_col,
                **fitter_kwargs,
            )
            scores_ll.append(
                fitter.score(testing_data, scoring_method="log_likelihood")
            )
            scores_c.append(
                fitter.score(testing_data, scoring_method="concordance_index")
            )
            summs.append(fitter.summary)
            # fitter.check_assumptions(training_data, p_value_threshold=0.05)

    # If a single fitter was given as argument, return a single result
    if len(fitters) == 1:
        return fitter_scores_ll[0], fitter_scores_c[0], fitter_summs[0]
    return fitter_scores_ll, fitter_scores_c, fitter_summs


class PerfComp:
    """Class to compare performance of different representations in predicting drug response"""

    # Use cache to avoid loading all results every time
    def __init__(self, targets, experiment, wd_path, dataset="depmap_gdsc"):
        self.targets = targets
        self.wd_path = wd_path
        self.experiment = experiment
        self.dataset = dataset
        self.seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        if dataset == "depmap_gdsc":
            self.sample_info = pd.read_csv(
                f"{wd_path}/data/depmap23q2/Model.csv"
            ).set_index("ModelID")
            s = pd.read_csv(f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv")
            mut_hs = pd.read_csv(
                f"{wd_path}/data/depmap23q2/OmicsSomaticMutationsMatrixHotspot.csv"
            ).set_index("Unnamed: 0")
            mut_dam = pd.read_csv(
                f"{wd_path}/data/depmap23q2/OmicsSomaticMutationsMatrixDamaging.csv"
            ).set_index("Unnamed: 0")
            self.s = s.rename(mapper=gene_column_renamer, axis=1)
            self.mut_hs = mut_hs.rename(mapper=gene_column_renamer, axis=1)
            self.mut_dam = mut_dam.rename(mapper=gene_column_renamer, axis=1)
        self.perf_df = None
        self.perf_df_type = None

    def _load_preds(
        self,
        fe,
        model,
        target,
        dataset,
        experiment,
        seeds,
        # plot_iqr=False,
        plot_scatter=False,
        return_z=False,
        return_val_z=False,
        # merge_muts=None,
        perm_imp=False,
        ld=False,
        cv_num=5,
    ):
        # Loads a single prediction file and calculates metrics, returning a dictionary
        pred_dict_list = []
        # pred_dict_list_val = []
        # weights_df_list = []
        perm_imp_list = []
        for i, seed in enumerate(seeds):
            if fe == "nn":
                curr_folder = f"{self.wd_path}/data/outputs/{dataset}/{target.upper()}/{experiment}/nn"
            else:
                curr_folder = f"{self.wd_path}/data/outputs/{dataset}/{target.upper()}/{experiment}/pico/{model}_{fe}"

            if ld:
                curr_folder = f"{curr_folder}_ld"

            # Load resp and targets
            # Only need to load resp once -- same for each seed
            # if i == 0:
            #     resp = pd.read_csv(
            #         f"{curr_folder}/response_{dataset.lower()}_{drug.upper()}_{target}.csv",
            #         header=0,
            #     )

            # LOAD ARGUMENTS
            if fe == "nn":
                with open(f"{curr_folder}/args_best.txt", "r") as f:
                    args = json.load(f)
            else:
                with open(f"{curr_folder}/args_best_s{seed}.txt", "r") as f:
                    args = json.load(f)

            # Load constraints and feature extractor args if applicable
            if fe != "nn":
                constraints = args["constraints"]
                n_genes = len(constraints)

            # Load predictions - test
            if fe == "nn":
                test_z = pd.read_csv(f"{curr_folder}/pred_test_s{seed}.csv")
            else:
                test_z = pd.read_csv(f"{curr_folder}/z_pred_test_s{seed}.csv")

            test_pred = test_z["pred_0"].to_numpy()
            test_gt = test_z["y"].to_numpy()

            # Load predictions - val for anything not NN
            if fe != "nn":
                val_zs = []
                for i in range(cv_num):
                    # if fe == "nn":
                    #     val_z = pd.read_csv(f"{curr_folder}/pred_val_{i}_best_s{seed}.csv").set_index("ind")
                    # else:
                    val_z = pd.read_csv(
                        f"{curr_folder}/z_pred_val_{i}_best_s{seed}.csv"
                    ).set_index("ind")
                    val_zs.append(val_z)

                val_z = pd.concat(val_zs, axis=0).reset_index()

                val_pred = val_z["pred_0"].to_numpy()
                val_gt = val_z["y"].to_numpy()

            # Add drug response and typing information for plots
            # test_z = pd.merge(
            #     test_z,
            #     self.sample_info[
            #         ["OncotreeLineage", "ModelID", "LegacySubSubtype"]
            #     ],
            #     left_on="COSMIC_ID",
            #     right_on="COSMICID",
            #     how="left",
            # )
            # if merge_muts is not None:
            #     merged_df_full = pd.merge(
            #         test_z[["ModelID"]],
            #         self.mut_hs,
            #         left_on="ModelID",
            #         right_index=True,
            #         how="left",
            #     )
            #     merged_df_full = pd.merge(
            #         merged_df_full,
            #         self.mut_dam,
            #         left_on="ModelID",
            #         right_index=True,
            #         how="left",
            #         suffixes=["_hs", "_dam"],
            #     ).drop(["ModelID"], axis=1)
            #     mut_count = print(
            #         merged_df_full.sum(axis=0).T.sort_values(ascending=False).head(20)
            #     )
            #     test_z = pd.merge(
            #         test_z, self.mut_hs, left_on="ModelID", right_index=True, how="left"
            #     )
            #     test_z = pd.merge(
            #         test_z,
            #         self.mut_dam,
            #         left_on="ModelID",
            #         right_index=True,
            #         how="left",
            #         suffixes=["_hs", "_dam"],
            #     ).drop(["ModelID"], axis=1)

            # METRICS FOR TEST DATA
            nas = np.logical_or(np.isnan(test_pred), np.isnan(test_gt))
            rmse = np.sqrt(np.mean(np.square(test_pred[~nas] - test_gt[~nas])))
            pearson_r = pearsonr(test_pred[~nas], test_gt[~nas])[0]
            spearman_r = spearmanr(test_pred[~nas], test_gt[~nas])[0]

            # METRICS FOR VAL DATA
            if fe != "nn":
                nas_val = np.logical_or(np.isnan(val_pred), np.isnan(val_gt))
                rmse_val = np.sqrt(
                    np.mean(np.square(val_pred[~nas_val] - val_gt[~nas_val]))
                )
                pearson_r_val = pearsonr(val_pred[~nas_val], val_gt[~nas_val])[0]
                spearman_r_val = spearmanr(val_pred[~nas_val], val_gt[~nas_val])[0]

            # test_metrics = pd.read_csv(f"{curr_folder}/test_metrics.csv", header=0)

            if perm_imp and fe.split("_")[0] == "icovae":
                if model == "transfer":
                    with torch.no_grad():
                        reg_model = torch.load(
                            f"{curr_folder}/regressor_s{seed}.pt",
                            map_location=torch.device("cpu"),
                        )
                        reg_weights = reg_model.loc.weight.numpy()[0]
                        reg_bias = reg_model.loc.bias.numpy()
                elif model == "ElasticNet":
                    with open(f"{curr_folder}/regressor_s{seed}.txt") as reg_model:
                        data = json.load(reg_model)
                        reg_weights = np.array(data["coeffs"])
                        reg_bias = np.array(data["intercept"])
                elif model == "LogisticRegression":
                    with open(f"{curr_folder}/regressor_s{seed}.txt") as reg_model:
                        data = json.load(reg_model)
                        reg_weights = np.array(data["coeffs"])
                        reg_bias = np.array(data["intercept"])
                elif model == "SVR":
                    with open(f"{curr_folder}/regressor_s{seed}.txt") as reg_model:
                        data = json.load(reg_model)
                        reg_weights = np.array(data["coeffs"][0])
                        reg_bias = np.array(data["intercept"])

                test_z_rep = test_z.iloc[:, test_z.columns.str.startswith("z")]
                test_z_rep = test_z_rep.rename(
                    mapper=partial(rep_renamer, constraints), axis=1
                )

                for col in test_z_rep.columns:
                    # Do perturbation 100 times for each column
                    for j in range(100):
                        test_z_rep_perm = test_z_rep.copy()
                        test_z_rep_perm[col] = test_z_rep_perm.sample(
                            frac=1
                        ).reset_index(drop=True)[col]
                        test_z_pred_perm = (test_z_rep_perm * reg_weights).sum(
                            axis=1
                        ) + reg_bias

                        nas = np.logical_or(
                            np.isnan(test_z_pred_perm), np.isnan(test_gt)
                        )
                        r_perm = pearsonr(test_z_pred_perm[~nas], test_gt[~nas])[0]
                        s_perm = spearmanr(test_z_pred_perm[~nas], test_gt[~nas])[0]
                        rmse_perm = np.sqrt(
                            np.mean(np.square(test_z_pred_perm[~nas] - test_gt[~nas]))
                        )
                        perm_imp_list.append(
                            {
                                "dim": col,
                                "r_perm": r_perm,
                                "s_perm": s_perm,
                                "rmse_perm": rmse_perm,
                                "r": pearson_r,
                                "s": spearman_r,
                                "rmse": rmse,
                                "seed": seed,
                                "iter": j,
                            }
                        )

                if i == 0:
                    pi_df = pd.DataFrame(perm_imp_list)
                else:
                    pi_df = pd.concat([pi_df, pd.DataFrame(perm_imp_list)], axis=0)

            if plot_scatter:
                pal = sns.color_palette("Set2")
                if fe == "icovae":
                    c = pal[1]
                elif fe == "vae":
                    c = pal[0]
                elif fe == "nn":
                    c = pal[2]
                f, ax = plt.subplots(1, 1, figsize=(3, 3))
                sns.regplot(x=test_pred, y=test_gt, color=c)
                ax.set_xlabel(f"{target.capitalize()} pred. ln(IC50)")
                ax.set_ylabel(f"{target.capitalize()} ln(IC50)")
                sns.despine(ax=ax)
                ax.text(
                    s=rf"$r_p={pearson_r:.3f}$", x=0.1, y=0.8, transform=ax.transAxes
                )
            if fe == "nn":
                pred_dict_list.append(
                    {
                        "target": target,
                        "model": model,
                        "fe": fe,
                        "zdim": np.nan,
                        "n_genes": np.nan,
                        "rmse": rmse,
                        "pearson_r": pearson_r,
                        "spearman_r": spearman_r,
                        "rmse_val": np.nan,
                        "pearson_r_val": np.nan,
                        "spearman_r_val": np.nan,
                        "seed": seed,
                    }
                )
            else:
                pred_dict_list.append(
                    {
                        "target": target,
                        "model": model,
                        "fe": fe,
                        "n_genes": n_genes,
                        "rmse": rmse,
                        "pearson_r": pearson_r,
                        "spearman_r": spearman_r,
                        "rmse_val": rmse_val,
                        "pearson_r_val": pearson_r_val,
                        "spearman_r_val": spearman_r_val,
                        "seed": seed,
                    }
                )

        if return_z:
            if perm_imp:
                return pred_dict_list, test_z, constraints, pi_df
            else:
                return pred_dict_list, test_z, constraints
        elif return_val_z:
            if perm_imp:
                return pred_dict_list, val_z, constraints, pi_df
            else:
                return pred_dict_list, val_z, constraints
        else:
            return pred_dict_list

    def _load_preds_type(
        self,
        fe,
        model,
        target,
        dataset_name,
        experiment,
        seeds,
        plot=False,
        dataset_params={"var_filt_s": None, "var_filt_x": 1},
        ld=False,
        cv_num=5,
    ):
        # Loads a single prediction file and calculates metrics, returning a dictionary

        palette = sns.color_palette("colorblind")

        pred_dict_list = []
        # weights_df_list = []

        for i, seed in enumerate(seeds):
            if fe == "nn":
                curr_folder = f"{self.wd_path}/data/outputs/{dataset_name}/{target.upper()}/{experiment}/nn"
            else:
                curr_folder = f"{self.wd_path}/data/outputs/{dataset_name}/{target.upper()}/{experiment}/pico/{model}_{fe}"

            if ld:
                curr_folder = f"{curr_folder}_ld"

            # Load constraints and feature extractor args if applicable, on first seed
            if i == 0:
                # LOAD ARGUMENTS
                with open(f"{curr_folder}/args_best.txt", "r") as f:
                    args = json.load(f)

                x, s, c, y, test_samples = process_data(
                    dataset=dataset_name, wd_path=self.wd_path, experiment=experiment
                )
                if fe != "nn":
                    constraints = args["constraints"]
                else:
                    constraints = [s.columns[0]]
                genes = [constraint.strip() for constraint in constraints]
                n_genes = len(constraints)

                dataset = Manual(
                    x=x,
                    s=s,
                    y=y,
                    constraints=genes,
                    target=target,
                    params=dataset_params,
                )

            # Load predictions
            if fe == "nn":
                test_z = pd.read_csv(f"{curr_folder}/pred_test_s{seed}.csv")
            else:
                test_z = pd.read_csv(f"{curr_folder}/z_pred_test_s{seed}.csv")

            # test_pred = test_z["pred_0"].to_numpy()
            # test_gt = test_z["y"]

            # Add typing information for plots
            test_z["ModelID"] = test_z["ind"].apply(lambda x: dataset.idx_to_sample[x])
            test_z = test_z.set_index("ModelID")

            # Add type info
            test_z = pd.merge(
                test_z,
                self.sample_info["OncotreeLineage"],
                left_index=True,
                right_index=True,
                how="left",
            )

            # Load predictions - val for anything not NN
            if fe != "nn":
                val_zs = []
                for i in range(cv_num):
                    # if fe == "nn":
                    #     val_z = pd.read_csv(f"{curr_folder}/pred_val_{i}_best_s{seed}.csv").set_index("ind")
                    # else:
                    val_z = pd.read_csv(
                        f"{curr_folder}/z_pred_val_{i}_best_s{seed}.csv"
                    ).set_index("ind")
                    val_zs.append(val_z)

                val_z = pd.concat(val_zs, axis=0).reset_index()

                # val_pred = val_z["pred_0"].to_numpy()
                # val_gt = val_z["y"].to_numpy()

                # Add typing information for plots
                val_z["ModelID"] = val_z["ind"].apply(
                    lambda x: dataset.idx_to_sample[x]
                )
                val_z = val_z.set_index("ModelID")

                # Add type info
                val_z = pd.merge(
                    val_z,
                    self.sample_info["OncotreeLineage"],
                    left_index=True,
                    right_index=True,
                    how="left",
                )

            lineages = list(set(test_z["OncotreeLineage"]))
            if fe != "nn":
                lineages_val = list(set(val_z["OncotreeLineage"]))

            plot_lins = []
            for lineage in lineages:
                test_z_type = test_z[test_z["OncotreeLineage"] == lineage]
                test_pred_type = test_z_type["pred_0"].to_numpy()
                test_gt_type = test_z_type["y"].to_numpy()

                if len(test_pred_type) > 10:
                    plot_lins.append(lineage)

                nas = np.logical_or(np.isnan(test_pred_type), np.isnan(test_gt_type))
                test_pred_type = test_pred_type[~nas]
                test_gt_type = test_gt_type[~nas]
                if len(test_pred_type) > 1:
                    rmse = np.sqrt(np.mean(np.square(test_pred_type - test_gt_type)))
                    pearson_r = pearsonr(test_pred_type, test_gt_type)[0]
                    spearman_r = spearmanr(test_pred_type, test_gt_type)[0]
                else:
                    continue

                if fe == "nn":
                    pred_dict_list.append(
                        {
                            "target": target,
                            "model": model,
                            "fe": fe,
                            "lineage": lineage,
                            "n_genes": np.nan,
                            "rmse": rmse,
                            "pearson_r": pearson_r,
                            "spearman_r": spearman_r,
                            "rmse_val": np.nan,
                            "pearson_r_val": np.nan,
                            "spearman_r_val": np.nan,
                            "seed": seed,
                        }
                    )
                else:
                    pred_dict_list.append(
                        {
                            "target": target,
                            "model": model,
                            "fe": fe,
                            "lineage": lineage,
                            "n_genes": n_genes,
                            "rmse": rmse,
                            "pearson_r": pearson_r,
                            "spearman_r": spearman_r,
                            "rmse_val": np.nan,
                            "pearson_r_val": np.nan,
                            "spearman_r_val": np.nan,
                            "seed": seed,
                        }
                    )

            if fe != "nn":
                for lineage in lineages_val:
                    val_z_type = val_z[val_z["OncotreeLineage"] == lineage]
                    val_pred_type = val_z_type["pred_0"].to_numpy()
                    val_gt_type = val_z_type["y"].to_numpy()

                    if len(val_pred_type) > 10:
                        plot_lins.append(lineage)

                    nas = np.logical_or(np.isnan(val_pred_type), np.isnan(val_gt_type))
                    val_pred_type = val_pred_type[~nas]
                    val_gt_type = val_gt_type[~nas]
                    if len(val_pred_type) > 1:
                        rmse = np.sqrt(np.mean(np.square(val_pred_type - val_gt_type)))
                        pearson_r = pearsonr(val_pred_type, val_gt_type)[0]
                        spearman_r = spearmanr(val_pred_type, val_gt_type)[0]
                    else:
                        continue

                    if fe == "nn":
                        pass
                    else:
                        pred_dict_list.append(
                            {
                                "target": target,
                                "model": model,
                                "fe": fe,
                                "lineage": lineage,
                                "n_genes": n_genes,
                                "rmse": np.nan,
                                "pearson_r": np.nan,
                                "spearman_r": np.nan,
                                "rmse_val": rmse,
                                "pearson_r_val": pearson_r,
                                "spearman_r_val": spearman_r,
                                "seed": seed,
                            }
                        )

            # Plot scatter plots for each type
            if plot and (seed == 10):
                palette_lm = {"ccvae": palette[1], "vae": palette[0], "nn": palette[2]}
                g = sns.lmplot(
                    data=test_z[test_z["OncotreeLineage"].isin(plot_lins)],
                    row="OncotreeLineage",
                    x="pred_0",
                    y=target.upper(),
                    height=1.5,
                    aspect=1.2,
                    ci=None,
                    scatter_kws={"color": "lightgrey"},
                    line_kws={"color": palette_lm[fe]},
                    facet_kws={"sharex": True, "sharey": True},
                )
                g.set_titles(
                    "{row_name}",
                    wrap=True,
                    x=0.1,
                    y=0.76,
                    fontdict={
                        "fontsize": 12,
                        "verticalalignment": "baseline",
                        "horizontalalignment": "left",
                    },
                )
                g.set_axis_labels("", "")
                g.fig.suptitle(f"{target}\n{fe}", x=0.56, y=0.99, fontsize=10)
                # g.set_axis_labels(f"{drug}\npred. ln(IC50)", f"{drug} ln(IC50)")
                g.fig.tight_layout()

                g.map_dataframe(self._annotate, target=target)
                sns.despine()
                plt.savefig(
                    f"./figures/type_{target}_{fe}_{model}_{seed}.png",
                    bbox_inches="tight",
                    dpi=400,
                )

        return pred_dict_list

    @staticmethod
    def _annotate(data, target, **kws):
        r, p = pearsonr(data["pred_0"], data[target.upper()])
        ax = plt.gca()
        ax.text(
            0.1,
            0.71,
            "r = {:.2f}".format(
                r,
            ),
            transform=ax.transAxes,
        )

    def calculate_perf(
        self, fes, models, plot_iqr=False, by_type=False, use_cache=True, wd_path=None
    ):
        perf_dict_list = []
        for target in self.targets:
            for fe in fes:
                for model in models:
                    # If cache is present and we want to use it then don't load new data
                    if by_type:
                        if (self.perf_df_type is not None) and use_cache:
                            perf_df_cache = self.perf_df_type.copy()
                            if (
                                (perf_df_cache["target"] == target)
                                & (perf_df_cache["fe"] == fe)
                                & (perf_df_cache["model"] == model)
                            ).any():
                                print(f"Using cache: {target}, {fe}, {model}\n")
                                continue
                            else:
                                pass
                    else:
                        if (self.perf_df is not None) and use_cache:
                            perf_df_cache = self.perf_df.copy()
                            if (
                                (perf_df_cache["target"] == target)
                                & (perf_df_cache["fe"] == fe)
                                & (perf_df_cache["model"] == model)
                            ).any():
                                print(f"Using cache: {target}, {fe}, {model}\n")
                                continue
                            else:
                                pass
                    try:
                        if by_type:
                            curr_perf_dict_list = self._load_preds_type(
                                fe,
                                model,
                                target,
                                dataset_name=self.dataset,
                                experiment=self.experiment,
                                seeds=self.seeds,
                            )
                        else:
                            curr_perf_dict_list = self._load_preds(
                                fe,
                                model,
                                target,
                                self.dataset,
                                self.experiment,
                                self.seeds,
                                plot_iqr=plot_iqr,
                            )
                    except UserWarning(f"Missing: {target}, {fe}, {model}\n"):
                        continue
                    perf_dict_list.extend(curr_perf_dict_list)

        # Create perf_df from newly loaded data

        # If new data loaded, create perf_df
        # If not, just return cached perf_df
        if len(perf_dict_list) > 0:
            perf_df = pd.DataFrame.from_dict(perf_dict_list).sort_values("target")
        elif (self.perf_df is not None) and use_cache and not by_type:
            return self.perf_df
        elif (self.perf_df_type is not None) and use_cache and by_type:
            return self.perf_df_type
        else:
            raise ValueError("No data loaded.")

        # If cache exists, concat newly loaded perf_df and cached version
        if by_type:
            if (self.perf_df_type is not None) and use_cache:
                self.perf_df_type = pd.concat([self.perf_df_type, perf_df], axis=0)
            else:
                self.perf_df_type = perf_df
        else:
            if (self.perf_df is not None) and use_cache:
                self.perf_df = pd.concat([self.perf_df, perf_df], axis=0)
            else:
                self.perf_df = perf_df

        return perf_df

    def plot_perf(self):
        pass

    @staticmethod
    def _plot_iqr(iqr_df, drug, n_dims=8):
        """Plots feature effect IQR for n_dims features, ordered by effect IQR"""
        pal = sns.color_palette("Set2")
        q_df_long = iqr_df.melt(var_name="z_dim")
        plot_order = (
            q_df_long.groupby("z_dim")
            .mean()
            .abs()
            .sort_values(by="value", ascending=False)
            .index.tolist()
        )
        # Enable subscript labelling
        q_df_long = q_df_long[q_df_long["z_dim"].isin(plot_order[:n_dims])]
        fig, ax = plt.subplots(1, 1, figsize=(n_dims / 3, 3), sharex=True, sharey=True)
        plot_order_lab = [x.split("_")[1] for x in plot_order]
        plot_order_lab = [f"z_{{{x}}}" for x in plot_order_lab]
        plot_order_lab = [rf"$\it{{{x}}}$" for x in plot_order_lab]
        is_const = [x.split("_")[1].isnumeric() for x in plot_order]
        pal_plot = {False: pal[2], True: "lightgrey"}
        pal = {plot_order[i]: pal_plot[x] for i, x in enumerate(is_const)}

        sns.barplot(
            data=q_df_long,
            y="z_dim",
            x="value",
            ax=ax,
            hue="z_dim",
            palette=pal,
            order=plot_order[:n_dims],
            linewidth=1,
        )
        ax.set_yticks(
            plot_order[:n_dims], plot_order_lab[:n_dims], rotation=0, fontsize=12
        )
        # ax.set_xticklabels(plot_order_lab[:n_dims], rotation=90)
        ax.set_facecolor((0, 0, 0, 0))
        ax.set_ylabel("")
        ax.set_xlabel("Effect IQR")
        ax.set_title(drug)
        sns.despine()
        ax.tick_params(left=False)

    def plot_coeffs(self, model, regressor, z_cols):
        sns.set_palette("colorblind")
        sns.set_context("talk")
        sns.set_style("whitegrid")

        z_coefs_plot = sorted(
            list(zip(z_cols, regressor.coef_)),
            key=lambda tup: abs(tup[1]),
            reverse=True,
        )
        z_coefs_names = [tup[0] for tup in z_coefs_plot[:10]]
        z_coefs_heights = [tup[1] for tup in z_coefs_plot[:10]]
        colors = [
            self.c0 if "_".split(x)[1].isdigit() else "#adadad" for x in z_coefs_names
        ]
        plt.figure(figsize=(1.5, 5))
        plt.barh(z_coefs_names, z_coefs_heights, color=colors)
        plt.xticks(rotation=90, ha="center", va="center")
        plt.title(f"Lasso ({model})\nweights", fontweight="regular", fontsize="12")
        plt.grid(True)
        plt.gca().invert_yaxis()

        sns.despine(left=True, bottom=True)


def calculate_feat_imps(
    enc,
    reg,
    model_path,
    target,
    seeds,
    return_z=False,
    nn=False,
    train=False,
):
    # Loads a single prediction file and calculates metrics, returning a dictionary
    pred_dict_list = []
    perm_imp_list = []
    for i, seed in enumerate(seeds):
        # Load resp and targets
        # Only need to load resp once -- same for each seed
        # if i == 0:
        #     resp = pd.read_csv(
        #         f"{curr_folder}/response_{dataset.lower()}_{drug.upper()}_{target}.csv",
        #         header=0,
        #     )

        # LOAD ARGUMENTS
        with open(f"{model_path}/args_best_s{seed}.txt", "r") as f:
            args = json.load(f)

        print(args)

        # Load constraints and feature extractor args if applicable
        if not nn:
            constraints = args["constraints"]
            n_constraints = len(constraints)
            confounders = args["confounders"]

        # Load predictions
        if train:
            preds_path = f"pred_train_s{seed}.csv"
        else:
            preds_path = f"pred_test_s{seed}.csv"
        if nn:
            test_z = pd.read_csv(f"{model_path}/{preds_path}")
        else:
            test_z = pd.read_csv(f"{model_path}/z_{preds_path}")

        test_pred = test_z["pred_0"].to_numpy()
        test_gt = test_z["y"].to_numpy()

        # Add drug response and typing information for plots
        # test_z = pd.merge(
        #     test_z,
        #     self.sample_info[
        #         ["OncotreeLineage", "ModelID", "LegacySubSubtype"]
        #     ],
        #     left_on="COSMIC_ID",
        #     right_on="COSMICID",
        #     how="left",
        # )
        # if merge_muts is not None:
        #     merged_df_full = pd.merge(
        #         test_z[["ModelID"]],
        #         self.mut_hs,
        #         left_on="ModelID",
        #         right_index=True,
        #         how="left",
        #     )
        #     merged_df_full = pd.merge(
        #         merged_df_full,
        #         self.mut_dam,
        #         left_on="ModelID",
        #         right_index=True,
        #         how="left",
        #         suffixes=["_hs", "_dam"],
        #     ).drop(["ModelID"], axis=1)
        #     mut_count = print(
        #         merged_df_full.sum(axis=0).T.sort_values(ascending=False).head(20)
        #     )
        #     test_z = pd.merge(
        #         test_z, self.mut_hs, left_on="ModelID", right_index=True, how="left"
        #     )
        #     test_z = pd.merge(
        #         test_z,
        #         self.mut_dam,
        #         left_on="ModelID",
        #         right_index=True,
        #         how="left",
        #         suffixes=["_hs", "_dam"],
        #     ).drop(["ModelID"], axis=1)

        nas = np.logical_or(np.isnan(test_pred), np.isnan(test_gt))

        if reg == "LogisticRegression":
            fpr, tpr, threshold = roc_curve(test_gt[~nas], test_pred[~nas], pos_label=1)
            auroc = auc(fpr, tpr)
        else:
            rmse = np.sqrt(np.mean(np.square(test_pred[~nas] - test_gt[~nas])))
            pearson_r = pearsonr(test_pred[~nas], test_gt[~nas])[0]
            spearman_r = spearmanr(test_pred[~nas], test_gt[~nas])[0]

        # test_metrics = pd.read_csv(f"{curr_folder}/test_metrics.csv", header=0)

        if reg == "transfer":
            with torch.no_grad():
                reg_model = torch.load(
                    f"{model_path}/regressor_s{seed}.pt",
                    map_location=torch.device("cpu"),
                )
                reg_weights = reg_model.loc.weight.numpy()[0]
                reg_bias = reg_model.loc.bias.numpy()
        elif reg == "ElasticNet":
            with open(f"{model_path}/regressor_s{seed}.txt") as reg_model:
                data = json.load(reg_model)
                reg_weights = np.array(data["coeffs"])
                reg_bias = np.array(data["intercept"])
        elif reg == "LogisticRegression":
            with open(f"{model_path}/regressor_s{seed}.txt") as reg_model:
                data = json.load(reg_model)
                reg_weights = np.array(data["coeffs"])
                reg_bias = np.array(data["intercept"])
        elif reg == "SVR":
            with open(f"{model_path}/regressor_s{seed}.txt") as reg_model:
                data = json.load(reg_model)
                reg_weights = np.array(data["coeffs"][0])
                reg_bias = np.array(data["intercept"])

        test_z_rep_z = test_z.iloc[:, test_z.columns.str.startswith("z")]
        test_z_rep_c = test_z.iloc[:, test_z.columns.str.startswith("c")]
        test_z_rep_z = test_z_rep_z.rename(
            mapper=partial(rep_renamer, constraints=constraints, prefix="z"), axis=1
        )
        test_z_rep_c = test_z_rep_c.rename(
            mapper=partial(rep_renamer, constraints=confounders, prefix="c"), axis=1
        )

        test_z_rep = pd.concat([test_z_rep_z, test_z_rep_c], axis=1)

        for col in test_z_rep.columns:
            # Do perturbation 100 times for each column
            for j in range(100):
                test_z_rep_perm = test_z_rep.copy()
                test_z_rep_perm[col] = test_z_rep_perm.sample(frac=1).reset_index(
                    drop=True
                )[col]
                test_z_pred_perm = (test_z_rep_perm * reg_weights).sum(
                    axis=1
                ) + reg_bias

                if reg == "LogisticRegression":
                    test_z_pred_perm = 1.0 / (1.0 + np.exp(-test_z_pred_perm))
                    test_z_pred_perm = test_z_pred_perm / test_z_pred_perm.sum()
                    nas = np.logical_or(np.isnan(test_z_pred_perm), np.isnan(test_gt))
                    fpr_val_perm, tpr_val_perm, threshold_val_perm = roc_curve(
                        test_gt[~nas], test_z_pred_perm[~nas], pos_label=1
                    )
                    auc_val_perm = auc(fpr_val_perm, tpr_val_perm)

                    perm_imp_list.append(
                        {
                            "dim": col,
                            "auroc_perm": auc_val_perm,
                            "auroc": auroc,
                            "seed": seed,
                            "iter": j,
                        }
                    )

                else:
                    nas = np.logical_or(np.isnan(test_z_pred_perm), np.isnan(test_gt))
                    r_perm = pearsonr(test_z_pred_perm[~nas], test_gt[~nas])[0]
                    s_perm = spearmanr(test_z_pred_perm[~nas], test_gt[~nas])[0]
                    rmse_perm = np.sqrt(
                        np.mean(np.square(test_z_pred_perm[~nas] - test_gt[~nas]))
                    )
                    perm_imp_list.append(
                        {
                            "dim": col,
                            "r_perm": r_perm,
                            "s_perm": s_perm,
                            "rmse_perm": rmse_perm,
                            "r": pearson_r,
                            "s": spearman_r,
                            "rmse": rmse,
                            "seed": seed,
                            "iter": j,
                        }
                    )

        if i == 0:
            pi_df = pd.DataFrame(perm_imp_list)
        else:
            pi_df = pd.concat([pi_df, pd.DataFrame(perm_imp_list)], axis=0)

        if nn:
            pred_dict_list.append(
                {
                    "target": target,
                    "reg": reg,
                    "enc": enc,
                    "zdim": np.nan,
                    "n_constraints": np.nan,
                    "rmse": rmse,
                    "pearson_r": pearson_r,
                    "spearman_r": spearman_r,
                }
            )
        else:
            if reg == "LogisticRegression":
                pred_dict_list.append(
                    {
                        "target": target,
                        "reg": reg,
                        "enc": enc,
                        "n_constraints": n_constraints,
                        "auroc": auroc,
                    }
                )
            else:
                pred_dict_list.append(
                    {
                        "target": target,
                        "reg": reg,
                        "enc": enc,
                        "n_constraints": n_constraints,
                        "rmse": rmse,
                        "pearson_r": pearson_r,
                        "spearman_r": spearman_r,
                    }
                )

    if return_z:
        return pred_dict_list, test_z, constraints, confounders, pi_df
    else:
        return pred_dict_list, constraints, confounders, pi_df


def plot_feat_imps_v2(
    pi_df,
    target,
    constraints,
    confounders,
    zdim,
    enc,
    reg,
    experiment,
    save_path,
    metric="r",
    names_map=None,
    norep=False,
    sort_feats=False,
    top_k=16,
):
    # PERMUTATION FEATURE IMPORTANCE
    from matplotlib.colors import rgb_to_hsv

    n_feats = zdim

    n_feats_plot = top_k if sort_feats else zdim

    pal = sns.color_palette("colorblind")
    f, ax = plt.subplots(
        2,
        2,
        figsize=(3, n_feats_plot / 5),
        sharey=False,
        gridspec_kw={"width_ratios": [6, 4], "height_ratios": [40, 1]},
    )
    pi_df_plot = pi_df.copy()
    if metric in ["r", "s", "rmse"]:
        pi_df_plot["fi_r"] = (
            100 * (pi_df_plot["r_perm"] - pi_df_plot["r"]) / (pi_df_plot["r"])
        )
        pi_df_plot["fi_s"] = (
            100 * (pi_df_plot["s_perm"] - pi_df_plot["s"]) / (pi_df_plot["s"])
        )
        pi_df_plot["fi_rmse"] = (
            100 * (pi_df_plot["rmse_perm"] - pi_df_plot["rmse"]) / pi_df_plot["rmse"]
        )
    else:
        pi_df_plot["fi_auroc"] = pi_df_plot["auroc_perm"] - pi_df_plot["auroc"]
    pi_df_plot["dim"] = pi_df_plot["dim"].apply(lambda x: x.split("_")[1])
    pi_df_plot["dim"] = pi_df_plot["dim"].apply(
        lambda x: rf"$z_{{{x}}}$" if x not in confounders else x
    )
    # Rename features in names_map if provided
    if names_map is not None:
        pi_df_plot["dim"] = pi_df_plot["dim"].apply(
            lambda x: names_map[x] if x in names_map.keys() else x
        )

    # Group by dim and seed and take mean over perturbation iterations
    pi_df_plot = pi_df_plot.groupby(["dim", "seed"]).mean().reset_index()

    plot_order = (
        pi_df_plot.groupby("dim")
        .mean()
        .sort_values(by=f"{metric}_perm", ascending=(metric != "rmse"))
        .index.tolist()
    )
    pi_df_plot = pi_df_plot[
        pi_df_plot["dim"].isin(plot_order[: n_feats + len(confounders)])
    ]

    pi_df_plot_hm = (
        pd.melt(
            pi_df_plot[["dim", f"fi_{metric}", "seed"]],
            id_vars=["dim", "seed"],
            value_vars=[f"fi_{metric}"],
        )
        .drop("variable", axis=1)
        .sort_values("value")
        .reset_index()
    )

    pi_df_plot_hm = pi_df_plot_hm.pivot_table(
        index="dim", columns="seed", values="value", sort=True
    )

    if sort_feats:
        plot_order = plot_order[:top_k]
    else:
        plot_order = []
        if not norep:
            plot_order = [rf"$z_{{{constraint}}}$" for constraint in constraints]
            plot_order = plot_order + [
                rf"$z_{{{i + len(constraints)}}}$"
                for i in range(n_feats - len(constraints))
            ]
        plot_order = plot_order + confounders

    pi_df_plot_hm = pi_df_plot_hm.loc[plot_order]

    pi_df_plot = pi_df_plot.set_index("dim").loc[plot_order].reset_index()

    print(
        pi_df_plot.groupby(["dim"])
        .mean()
        .sort_values(f"fi_{metric}", ascending=(metric != "rmse"))
    )

    if metric == "rmse":
        pal_cm = sns.diverging_palette(
            rgb_to_hsv(pal[1])[0] * 360,
            rgb_to_hsv(pal[0])[0] * 360,
            s=100,
            center="light",
            as_cmap=True,
        )
    else:
        pal_cm = sns.diverging_palette(
            rgb_to_hsv(pal[0])[0] * 360,
            rgb_to_hsv(pal[1])[0] * 360,
            s=100,
            center="light",
            as_cmap=True,
        )

    # HEATMAP
    cbar_labels = {
        "r": r"$\Delta \% \rho_{{p}}$",
        "s": r"$\Delta \% \rho_{{s}}$",
        "rmse": r"$\Delta\%$ RMSE",
        "auroc": r"$\Delta$AUROC",
    }
    # sns.barplot(data=pi_df_plot, y="dim", x="fi_r", order=plot_order[:n_feats], estimator="mean", errorbar=("sd", 1), palette=pal)
    sns.heatmap(
        pi_df_plot_hm,
        yticklabels=True,
        square=False,
        linewidths=1,
        cmap=pal_cm,
        center=0,
        ax=ax[0, 0],
        cbar_ax=ax[1, 0],
        cbar_kws={
            "orientation": "horizontal",
            "fraction": 0.02,
            "label": f"Permutation FI ({cbar_labels[metric]})",
            "use_gridspec": False,
        },
    )
    sns.despine(ax=ax[0, 0], bottom=True, left=True)
    # ax.set_xlabel("Permutation feature importance")
    ax[0, 0].set_ylabel("")
    ax[0, 0].set_xlabel("")
    # ax.set_yticks(plot_order[:n_feats], plot_order[:n_feats], rotation=0, fontsize=10)
    # ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
    ax[0, 0].set_xticks([], [])
    ax[0, 0].tick_params(axis="y", which="major", labelsize=12)
    # ax[0,0].set_xlim(0.0, 10.25)
    # if drug.startswith("AZD"):
    #     ax[0,0].set_title(drug.upper(), fontweight="bold")
    # else:
    #     ax[0,0].set_title(drug.capitalize(), fontweight="bold")
    ax[0, 0].set_title("")

    # BARPLOT
    sns.barplot(
        data=pi_df_plot,
        y="dim",
        x=f"fi_{metric}",
        estimator="mean",
        errorbar=("sd", 1),
        ax=ax[0, 1],
        capsize=0.25,
        dodge=False,
        err_kws={"linewidth": 1, "alpha": 0.5},
    )
    ax[0, 1].axvline(0, linestyle="--", color="grey", alpha=0.5)
    ax[0, 1].set_ylim(len(plot_order) - 0.5, -0.5)
    ax[0, 1].set_yticks([], [])
    ax[0, 1].set_ylabel("")
    ax[0, 1].set_xlabel("FI")

    for i, bar in enumerate(ax[0, 1].patches):
        if bar.get_height() > 0:
            if (ax[0, 1].lines[i].get_xdata()[-1] > 0) and (
                ax[0, 1].lines[i].get_xdata()[0] > 0
            ):
                bar.set_color(pal_cm(0.75))
            elif (ax[0, 1].lines[i].get_xdata()[-1] < 0) and (
                ax[0, 1].lines[i].get_xdata()[0] < 0
            ):
                bar.set_color(pal_cm(0.25))
            else:
                bar.set_color("lightgrey")

    if metric != "rmse":
        ax[0, 1].invert_xaxis()

    # y_tick_pos = [i+0.5 for i in range(len(plot_order))]
    # ax[0,1].yticks
    sns.despine(ax=ax[0, 1], bottom=True, left=True)
    sns.despine(ax=ax[1, 0], bottom=True, left=True)
    ax[1, 1].axis("off")

    plt.subplots_adjust(wspace=0.00, hspace=0.05)

    plt.savefig(f"./figures/fi_{save_path}.svg", bbox_inches="tight")
    plt.savefig(f"./figures/fi_{save_path}.png", bbox_inches="tight", dpi=600)
