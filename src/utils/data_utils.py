import shutil
import sys
import time
import urllib.request
import zipfile

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import pandas as pd
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from scipy.stats import pearsonr, ttest_ind

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statsmodels.formula.api import ols
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection

from sklearn.model_selection import KFold
from multiprocessing.pool import ThreadPool

import seaborn as sns

# from adjustText import adjust_text


def ensg_column_renamer(name):
    """Strip version suffix from ENSEMBL IDs (e.g., ENSG... .1 -> ENSG...)."""
    return name.split(".")[0]


def type_renamer(x):
    """Normalize lineage/type strings by replacing spaces and slashes with underscores."""
    x = "_".join((x.split(" ")))
    return "_".join(x.split("/"))


def normalise(x):
    """Normalise globally"""
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def scanb_row_renamer(name):
    """Strip version suffix from ScanB row identifiers."""
    return name.split(".")[0]


def extract_all(archives, extract_path):
    """Extract a list of archive files to a target directory."""
    for filename in archives:
        shutil.unpack_archive(filename, extract_path)


def optimize_dtype(column):
    """Downcast a pandas Series to int for binary-like data, else float32."""
    # Check if unique values are [0, 1] (binary-like values)
    unique_values = column.unique()
    if set(np.round(unique_values, 6)) <= {0, 1}:
        return column.astype(int)

    # Alternatively, check if all values are either 0 or non-zero
    if (column == 0).sum() == 0 or (column != 0).sum() == len(column):
        return column.astype(int)

    return column.astype(np.float32)


def feature_selection_var(x_train, x, num_features=1000):
    """Return num_features features with greatest variance in x_train, for x"""
    variances = np.var(x_train, axis=0)
    # plt.figure(figsize=(15, 10))
    # plt.hist(variances, bins=20)
    # plt.xlabel("Variance")
    # plt.ylabel("Frequency")
    # plt.savefig("var_dist.png", dpi=200, bbox_inches="tight")
    feature_inds = np.argsort(variances)[-num_features:]
    x = x[:, feature_inds]
    return x


def feature_selection_pca(x_train, x, num_features=1000):
    """Return highest num_features variance PCA components"""
    pca = PCA(n_components=num_features, svd_solver="auto")
    pca = pca.fit(x_train)
    x = pca.transform(x)
    return x


class ConstraintSelectorV2:
    """
    Utility for selecting gene constraints using linear-model screens.

    Loads the requested dataset, assembles confounders, and provides
    cross-validation or linear-model based selection with optional caching.

    Args:
        dataset_name: Dataset identifier (e.g. `depmap_gdsc`, `depmap_ctrp`).
        experiment: Optional experiment suffix for dataset paths.
        wd_path: Working directory root for data files and outputs.
    """

    def __init__(self, dataset_name="depmap_gdsc", experiment=None, wd_path=None):
        self.joined_df = None

        # Functionality to add random effects later
        self.confounders_fixed = [
            "GrowthPattern",
            "ScreenROCAUC",
            "ScreenMedianEssentialDepletion",
        ]
        self.confounders_random = ["OncotreeLineage"]
        self.confounders = self.confounders_fixed + self.confounders_random

        self.x, self.s, _, self.y, self.test_samples = process_data(
            dataset=dataset_name, wd_path=wd_path, experiment=experiment
        )

        self.wd_path = wd_path

        self.dataset_name = dataset_name

        if self.dataset_name in ["depmap_gdsc", "depmap_ctrp"]:
            self.c = pd.read_csv(
                f"{self.wd_path}/data/depmap23q2/AchillesScreenQCReport.csv"
            ).set_index("ModelID")
            mc = pd.read_csv(f"{self.wd_path}/data/depmap23q2/Model.csv").set_index(
                "ModelID"
            )
            self.c = pd.merge(
                self.c,
                mc[["GrowthPattern", "OncotreeLineage"]],
                right_index=True,
                left_index=True,
            )

        # Make folder for constraints if it doesnt exist
        if not os.path.exists(f"{self.wd_path}/data/constraints"):
            os.makedirs(f"{self.wd_path}/data/constraints")

    def _make_screen_df(self, drug):
        """Build a merged dataframe of constraints, confounders, and targets."""
        self.drug = drug

        dataset = Manual(
            x=self.x,
            s=self.s,
            y=self.y,
            c=self.c,
            constraints=self.s.columns,
            target=self.drug,
            confounders=self.confounders,
            drop_na=False,
        )

        # train_i_x_only, train_i_x_s, val_i, train_p, val_p, test = split_dataset(data=dataset, test_samples=self.test_samples, pretraining="i", verbose=True)

        # train_i_x_s contains all samples with s, so get samples with x and y
        # dataset = Subset(train_i_x_s, dataset.x_s_y_idx)

        # Make this back into a dataframe
        joined_df = pd.concat(
            [
                pd.DataFrame(dataset.s[dataset.x_s_c_y_idx]),
                pd.DataFrame(dataset.c[dataset.x_s_c_y_idx]),
                pd.DataFrame(dataset.y[dataset.x_s_c_y_idx]),
            ],
            axis=1,
        )
        joined_df.columns = dataset.s_features + dataset.c_features + dataset.y_features
        # Change continuous values to float
        if self.dataset_name in ["depmap_gdsc", "depmap_ctrp"]:
            joined_df[["SCREENROCAUC_c", "SCREENMEDIANESSENTIALDEPLETION_c"]] = (
                joined_df[
                    ["SCREENROCAUC_c", "SCREENMEDIANESSENTIALDEPLETION_c"]
                ].astype(float)
            )

        # Check for drugs with a dash and a number -- lm formula interprets dash as a
        if "-" in self.drug:
            # Collapse drug name by removing dashes
            joined_df = joined_df.rename(
                columns={f"{self.drug}_y": f"A{''.join(self.drug.split('-'))}_y"}
            )
            self.drug = f"A{''.join(self.drug.split('-'))}"
            print(f"Correcting drug name to {self.drug}...")

        return joined_df

    def _load_prism(self):
        """Placeholder for PRISM data loading (not implemented)."""
        raise NotImplementedError()

    def _test_constraint_cv(self, gene):
        """Evaluate a single gene constraint in CV using OLS and return metrics."""
        model_formula = f"{self.drug}_y ~ {str(gene)}_s + {self.cov_formula}"
        try:
            mod1 = ols(model_formula, data=self.train_df)
            res1 = mod1.fit()
            # print(res1.summary())

            curr_test_df = self.test_df[
                [
                    gene,
                    "GrowthPattern",
                    "ScreenROCAUC",
                    "ScreenMedianEssentialDepletion",
                    "ScreenDoublingTime",
                    "CasActivity",
                    "ScreenMADEssentials",
                ]
            ]
            curr_test_df = sm.add_constant(curr_test_df)
            test_pred = res1.predict(curr_test_df)
            nas = np.logical_or(np.isnan(self.test_df[self.drug]), np.isnan(test_pred))
            r, _ = pearsonr(self.test_df[self.drug][~nas], test_pred[~nas])
            s, _ = spearmanr(self.test_df[self.drug][~nas], test_pred[~nas])
            return {"val_pearson": r, "val_spearman": s, "gene": gene}
        except UserWarning(f"Fit failed for {gene}"):
            return {"val_pearson": np.nan, "val_spearman": np.nan, "gene": gene}

    def _test_constraint_lm(self, gene, print_summary=False):
        """Evaluate a single gene with a likelihood ratio test vs base model."""
        model_formula = f"{self.drug}_y ~ {str(gene)} + {self.cov_formula}"
        # Create base model should use same data as above, to deal with missing values
        base_formula = f"{self.drug}_y ~ {self.cov_formula}"

        curr_df = self.train_df.dropna(subset=[f"{self.drug}_y", gene])

        try:
            mod1 = ols(model_formula, data=curr_df)
            res1 = mod1.fit()

            bm = ols(base_formula, data=curr_df)
            res_bm = bm.fit()

            lr, p, ddf = res1.compare_lr_test(res_bm)

            return {"gene": gene, "lr": lr, "p": p, "ddf": ddf}
        except UserWarning(f"Fit failed for {gene}"):
            return {"gene": gene, "lr": np.nan, "p": np.nan, "ddf": np.nan}

    def select_cv(
        self,
        drug,
        c_type="all",
        covariates=None,
        cv_num=5,
        use_cache=False,
        save_path=None,
    ):
        """Run CV-based constraint selection and return a results dataframe."""
        if (self.joined_df is None) or (not use_cache):
            self.joined_df = self._make_screen_df(
                drug=drug, c_type=c_type, covariates=covariates
            )
            df = self.joined_df.copy()
        else:
            df = self.joined_df.copy()

        self.cov_formula = " + ".join(self.covariates)
        base_model_formula = f"{self.drug} ~ {self.cov_formula}"

        cov_list = self.covariates
        cov_list.append(self.drug)
        gene_list = df.drop(cov_list, axis=1).columns
        num_genes = len(gene_list)

        kf = KFold(n_splits=cv_num, shuffle=True, random_state=2)
        split = kf.split(df)
        results_dict_list = []

        for fold, (train_index, test_index) in enumerate(split):
            # added some parameters
            print(f"[Fold {fold}]")

            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

            self.train_df = train_df
            self.test_df = test_df

            with ThreadPool(1) as p:
                results_dict_list = list(
                    tqdm(p.imap(self._test_constraint_cv, gene_list), total=num_genes)
                )

            try:
                mod_base = ols(base_model_formula, data=train_df)
                res_base = mod_base.fit()
                curr_test_df = self.test_df[self.covariates]
                # print(curr_test_df[curr_test_df.index.duplicated()])
                curr_test_df = sm.add_constant(curr_test_df)
                test_pred = res_base.predict(curr_test_df)
                nas = np.logical_or(np.isnan(test_df[self.drug]), np.isnan(test_pred))
                r, _ = pearsonr(test_df[self.drug][~nas], test_pred[~nas])
                s, _ = spearmanr(test_df[self.drug][~nas], test_pred[~nas])
                results_dict_list.append(
                    {"val_pearson": r, "val_spearman": s, "gene": "base"}
                )
            except UserWarning(f"Fit failed for {drug} in fold {fold}..."):
                results_dict_list.append(
                    {"val_pearson": np.nan, "val_spearman": np.nan, "gene": "base"}
                )

        self.res_df = pd.DataFrame.from_dict(results_dict_list)

        if save_path is not None:
            self.res_df.to_csv(save_path)

        return self.res_df

    def select_lm(
        self,
        drug,
        use_cache=False,
        save_path=None,
    ):
        """Run linear-model constraint selection and return a results dataframe."""
        if (self.joined_df is None) or (not use_cache):
            self.joined_df = self._make_screen_df(
                drug=drug,
            )
            df = self.joined_df.copy()
        else:
            df = self.joined_df.copy()

        # Random confounders are also used as a fixed effect here, but can be considered as random later
        self.cov_formula = " + ".join(
            [f"{confounder.upper()}_c" for confounder in self.confounders_fixed]
        )
        self.cov_formula = (
            self.cov_formula
            + " + "
            + " + ".join(
                [f"{confounder.upper()}_c" for confounder in self.confounders_random]
            )
        )
        base_model_formula = f"{self.drug.upper()}_y ~ {self.cov_formula}"

        cov_list = self.confounders.copy()
        cov_list = [f"{cov.upper()}_c" for cov in cov_list]
        cov_list.append(f"{self.drug.upper()}_y")
        gene_list = df.drop(cov_list, axis=1).columns
        num_genes = len(gene_list)

        mod_base = ols(base_model_formula, data=df)
        res_base = mod_base.fit()

        self.bm_res = res_base

        self.train_df = df

        print(f"[INFO] Selecting constraints for {drug}...")

        with ThreadPool(1) as p:
            results_dict_list = list(
                tqdm(p.imap(self._test_constraint_lm, gene_list), total=num_genes)
            )

        self.res_df = pd.DataFrame.from_dict(results_dict_list)

        # drop nas
        self.res_df = self.res_df.dropna()

        # Rename columns again
        self.res_df["gene"] = self.res_df["gene"].apply(lambda x: x.split("_")[0])

        # BH correction
        _, self.res_df["p_corrected"] = fdrcorrection(self.res_df["p"])

        if save_path is not None:
            self.res_df.to_csv(save_path)

        return self.res_df

    def filter_cv(self, res_df, method="corr_filt"):
        """Filter CV results (e.g., correlation-based heuristics)."""
        if method == "corr_filt":
            # Selection by correlation filtering (simple heuristic)
            top_genes = (
                res_df.groupby("gene")
                .mean()
                .sort_values("val_pearson", ascending=False)
                .head(50)
            )
            top_corr = self.joined_df[top_genes.index].corr().abs()

            # if top genes are co-dependencies, then only keep the one most correlated with the target
            filt_genes = []
            for i, gene_a in enumerate(top_genes.index):
                if (top_corr[gene_a].loc[top_genes.index[:i]] < 0.3).all():
                    filt_genes.append(gene_a)

            return res_df[res_df["gene"].isin(filt_genes)].sort_values(
                "val_pearson", ascending=False
            )

        else:
            return None

    def filter_lm(self, res_df, method="corr_filt", thresh=0.5):
        """Filter LM results to reduce collinearity among top hits."""
        if method == "corr_filt":
            # Selection by correlation filtering (simple heuristic)
            top_genes = res_df.sort_values("p_corrected").head(200)
            top_corr = (
                self.joined_df[top_genes["gene"].apply(lambda x: f"{x}_s")].corr().abs()
            )

            # if top genes are co-dependencies, then only keep the one most correlated with the target
            filt_genes = []
            for i, gene_a in enumerate(top_genes["gene"]):
                # FIX!
                if (top_corr[f"{gene_a}_s"].loc[filt_genes] < thresh).all():
                    filt_genes.append(f"{gene_a}_s")

            return res_df[
                res_df["gene"].isin([gene.split("_")[0] for gene in filt_genes])
            ].sort_values("p_corrected")

        else:
            return None

    def plot_single(self, gene):
        """Plot a single gene effect vs drug response with correlation stats."""
        sns.set_context("notebook")
        plt.figure(figsize=(5, 5))
        x_plot = self.joined_df[gene].copy()
        y_plot = self.joined_df[self.drug].copy()
        nas = np.logical_or(np.isnan(x_plot), np.isnan(y_plot))
        plt.scatter(x_plot, y_plot)
        plt.xlabel(f"{gene} effect")
        plt.ylabel(f"{self.drug} LFC")
        sns.despine(bottom=True, left=True)
        r, _ = pearsonr(x_plot[~nas], y_plot[~nas])
        s, _ = spearmanr(x_plot[~nas], y_plot[~nas])
        y_range = np.nanmax(y_plot) - np.nanmin(y_plot)
        plt.text(
            np.nanmax(x_plot), np.nanmax(y_plot) - 0.1 * y_range, f"Pearson r: {r:.4f}"
        )
        plt.text(np.nanmax(x_plot), np.nanmax(y_plot), f"Spearman r: {s:.4f}")


def get_constraints(
    drug, dataset_name, zdim, experiment=None, col_thresh=0.7, wd_path=None
):
    """Get constraints for a drug in a dataset using linear model method."""
    if experiment is not None:
        path = f"{wd_path}/data/constraints/univar_lm_{drug}_lrt_IC50_{dataset_name}_{experiment}_v2.csv"
    else:
        path = f"{wd_path}/data/constraints/univar_lm_{drug}_lrt_IC50_{dataset_name}_v2.csv"
    # Check if constraints file exists for this setup
    if os.path.isfile(path):
        # Load res_df but create joined_df for filtering
        selector = ConstraintSelectorV2(
            dataset_name=dataset_name,
            experiment=experiment,
            wd_path=wd_path,
        )
        res_df = pd.read_csv(path)
        selector.joined_df = selector._make_screen_df(drug=drug)
    else:
        # Create selector
        selector = ConstraintSelectorV2(
            dataset_name=dataset_name,
            experiment=experiment,
            wd_path=wd_path,
        )
        # Run selection screen using linear model method, save results to path
        res_df = selector.select_lm(drug=drug, save_path=path)
    # Filter to remove strong collinearity
    res_df_filt = selector.filter_lm(res_df, thresh=col_thresh)
    # print(res_df_filt.sort_values("p_corrected"))
    # Select significant constraints (p<0.01)
    res_df_filt = res_df_filt[res_df_filt["p_corrected"] < 0.05].sort_values(
        "p_corrected"
    )
    # k = min(n_sig, L//2)
    if len(res_df_filt) < (zdim // 2):
        return res_df_filt["gene"].tolist()
    else:
        return res_df_filt["gene"][: (zdim // 2)].tolist()


def is_multidata(dataB):
    """Check if a data batch is a list/tuple container."""
    return (isinstance(dataB, list)) or isinstance(dataB, tuple)


def unpack_data(dataB, device="cuda"):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """Unpacks the data batch object in an appropriate manner to extract data"""
    if is_multidata(dataB):
        if len(dataB) == 1:
            return dataB[0].to(device)
        elif torch.is_tensor(dataB[0]):
            if torch.is_tensor(dataB[1]):
                return dataB[0].to(device)  # mnist, svhn, cubI
            elif is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
            else:
                raise RuntimeError(
                    "Invalid data format {} -- check your dataloader!".format(
                        type(dataB[1])
                    )
                )

        elif is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-svhn, cubIS
        else:
            raise RuntimeError(
                "Invalid data format {} -- check your dataloader!".format(
                    type(dataB[0])
                )
            )
    elif torch.is_tensor(dataB):
        return dataB.to(device)
    else:
        raise RuntimeError(
            "Invalid data format {} -- check your dataloader!".format(type(dataB))
        )


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    """
    Simple stdout tee that writes to both terminal and a file.

    Args:
        filename: Path to the log file.
        mode: File mode for the log file.
    """
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    """
    Context manager that prints elapsed wall time on exit.

    Args:
        name: Label shown in the timing output.
    """
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print(
            "====> [{}] Time: {:7.3f}s or {}".format(
                self.name, self.elapsed, time.strftime("%H:%M:%S", self.elapsedH)
            )
        )


def plot_mut_wt_boxplot(
    df_eff, df_mut, gene_mut="TP53", gene_eff="TP53", pathway_genes=None
):
    """Plot mutation vs wild-type effect distributions with optional pathway grouping."""
    df_eff_gene = df_eff[gene_eff]
    if pathway_genes is None:
        filt_muts = df_mut.loc[:, gene_mut].copy()
        filt_gene = df_eff_gene[filt_muts == 1]
        num_muts = len(filt_gene)
        filt_none = df_eff_gene[filt_muts == 0]

        t_val, p_val = ttest_ind(filt_gene, filt_none, equal_var=False)

        labels = ["wild-type", f"{gene_mut}\nmutant\n(n={num_muts})"]
        data = [filt_none, filt_gene]
        data_flat = [item for sublist in data for item in sublist]
        x = []
        for i in range(len(labels)):
            x.append([labels[i]] * len(data[i]))
        x = [item for sublist in x for item in sublist]
    else:
        filt_muts = df_mut.loc[
            :, df_mut.columns.str.startswith(tuple(pathway_genes))
        ].copy()
        filt_muts["Pathway"] = filt_muts.sum(axis=1) >= 1
        filt_gene = df_eff_gene[df_mut[gene_mut] == 1]
        num_muts = len(filt_gene)
        filt_pathway = df_eff_gene[(filt_muts["Pathway"]) & (df_mut[gene_mut] == 0)]
        num_muts_path = len(filt_pathway)
        filt_none = df_eff_gene[(not filt_muts["Pathway"]) & (df_mut[gene_mut] == 0)]
        labels = [
            "wild-type",
            f"{gene_mut}\nmutant\n(n={num_muts})",
            f"{pathway_genes[0]}\nmutant\n(n={num_muts_path})",
        ]
        data = [filt_none, filt_gene, filt_pathway]
        data_flat = [item for sublist in data for item in sublist]
        x = []
        for i in range(len(labels)):
            x.append([labels[i]] * len(data[i]))
        x = [item for sublist in x for item in sublist]

    sns.set_theme(style="whitegrid", palette="Set2")
    sns.set_context("talk")
    fig, ax = plt.subplots(1, 1, figsize=(4, 5))
    sns.boxplot(x=x, y=data_flat, width=0.25)
    sns.swarmplot(x=x, y=data_flat, color=".25", size=1.5)
    sns.despine(top=True, bottom=True, left=True, right=True)
    if pathway_genes is None:
        plt.text(0.5, 0.9, rf"$p$ = {p_val:.2e}", size="medium", transform=ax.transAxes)
    plt.xlabel(None)
    plt.ylabel(f"{gene_eff} Chronos")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    # plt.title(f"Depletion effect of {gene_eff} knockout")


def plot_by_cancer_lineage(data, genes):
    """Plot per-lineage Pearson correlations for predicted vs observed values."""
    cancer_types = list(set(data["lineage"]))
    # Calculate correlations by type
    corrs = []
    cancer_types_plot = []
    genes_plot = []
    for gene in genes:
        for type in cancer_types:
            y_current = data[data["lineage"] == type]
            if len(y_current) > 2:
                print(f"{type}: {len(y_current)}")
                corr = pearsonr(y_current[f"y_{gene}"], y_current[f"y_pred_{gene}"])
                corrs.append(corr[0])
                cancer_types_plot.append(type)
                genes_plot.append(gene)

    corr_df = pd.DataFrame(
        {"cancer_type": cancer_types_plot, "corr": corrs, "gene": genes_plot}
    )
    print(corr_df)

    sns.set_theme(style="white")
    sns.set_style("ticks")

    # Initialize the figure
    f, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.despine(right=True, top=True, left=True)

    # Pointplot the correlations
    sns.pointplot(
        x="cancer_type",
        y="corr",
        hue="gene",
        data=corr_df,
        dodge=0,
        join=False,
        palette="RdBu",
        markers="o",
        scale=0.75,
        ci=None,
    )
    ax.legend()
    plt.xticks(rotation=90)


def gene_column_renamer(name):
    """Extract HGNC gene symbol from a composite column name."""
    return name.split(" ")[0]


def gene_column_renamer_ncbi(name):
    """Extract NCBI/Entrez ID from a name containing a trailing '(ID)'."""
    ncbi_id = name.split("(")[-1].split(")")[0]
    return ncbi_id


def lineage_renamer(x):
    """Normalize lineage strings by replacing spaces and slashes with underscores."""
    x = "_".join((x.split(" ")))
    return "_".join(x.split("/"))


class Manual(Dataset):
    """
    A custom dataset class for handling and preprocessing data with features, constraints, and targets.

    Attributes:
        x (pd.DataFrame): DataFrame containing feature data.
        s (pd.DataFrame): DataFrame containing constraint data.
        y (pd.DataFrame): DataFrame containing target data.
        params (dict): Dictionary containing parameters for variance filtering.
        x_features (list): List of feature column names in x.
        x_samples (Index): Index of samples in x.
        s_features (list): List of constraint column names in s.
        s_samples (Index): Index of samples in s.
        y_features (list): List of target column names in y.
        y_samples (Index): Index of samples in y.
        x_s_y_samples (list): List of samples present in x, s, and y.
        x_y_samples (list): List of samples present in x and y but not in s.
        x_s_samples (list): List of samples present in x and s but not in y.
        x_only_samples (list): List of samples present only in x.
        sample_types (dict): Dictionary mapping samples to their types based on data availability.
        x_s_y (pd.DataFrame): Merged DataFrame containing x, s, and y data.
        sample_to_idx (dict): Dictionary mapping samples to their indices in the merged DataFrame.
        idx_to_sample (dict): Dictionary mapping indices to their samples in the merged DataFrame.
        idx (np.ndarray): Array of indices for samples in the merged DataFrame.
        st (np.ndarray): Array of sample types for samples in the merged DataFrame.
        x_s_y_idx (list): List of indices for samples present in x, s, and y.
        x_y_idx (list): List of indices for samples present in x and y but not in s.
        x_s_idx (list): List of indices for samples present in x and s but not in y.
        x_only_idx (list): List of indices for samples present only in x.
        s_prior_loc (None): Placeholder for prior location of s.
        s_prior_scale (None): Placeholder for prior scale of s.

    Methods:
        mean_impute_x(train_idx):
            Imputes missing values in x using the mean of the training data.

        s_prior_fn_loc():
            Returns the prior location for s.

        s_prior_fn_scale():
            Returns the prior scale for s.

        __len__():
            Returns the number of samples in the dataset.

        __getitem__(idx):
            Returns the features, constraints, targets, index, and sample type for a given index.
    """

    def __init__(
        self,
        x: pd.DataFrame,
        s: pd.DataFrame,
        y: pd.DataFrame,
        constraints: list,
        target: str,
        params={"var_filt_x": None, "var_filt_s": None},
        c=None,
        confounders=None,
        verbose=True,
        drop_na=True,
        duration_event=None,
    ):
        self.x = x
        self.s = s
        self.y = y
        self.c = c
        self.params = params
        self.use_c = (c is not None) and (confounders is not None)

        # Rename columns to track features in each dataset (since columns may have same names)
        # Don't drop NAs from x at this stage
        self.x = self.x.rename(lambda x: f"{x.upper()}_x", axis=1)
        # Drop NAs from s and y (samples without data for the chosen constraints/targets)
        # Constraints input should be in format CONSTRAINT_s
        # Input empty list to use all constraints
        if len(constraints) == 0:
            constraints = self.s.columns
        self.s = self.s.rename(lambda x: f"{x.upper()}_s", axis=1)[
            [f"{constraint.upper()}_s" for constraint in constraints]
        ]

        # print(self.y)
        if duration_event is None:
            if isinstance(target, list):
                self.y = self.y.rename(lambda x: f"{x.upper()}_y", axis=1)[
                    [f"{target_.upper()}_y" for target_ in target]
                ]
            else:
                self.y = self.y.rename(lambda x: f"{x.upper()}_y", axis=1)[
                    [f"{target.upper()}_y"]
                ]
        else:
            target_root = target.split("_")[0]
            self.y = self.y.rename(lambda x: f"{x.upper()}_y", axis=1)[
                [
                    f"{target_root.upper()}_{duration_event[0]}_y",
                    f"{target_root.upper()}_{duration_event[1]}_y",
                ]
            ]

        # This is done by default
        if drop_na:
            self.s = self.s.dropna(axis=0)
            self.x = self.x.dropna(axis=0)
            self.y = self.y.dropna(axis=0)

        # If used, this should be available for all samples with y
        if self.use_c:
            self.c = self.c.rename(lambda x: f"{x.upper()}_c", axis=1)[
                [f"{confounder.upper()}_c" for confounder in confounders]
            ]
            if drop_na:
                self.c = self.c.dropna(axis=0)

        # x should be of shape (num_samples, num_features)
        # if self.params["var_filt_x"] is not None:
        #     x_var = np.nanvar(self.x, axis=0)
        #     x_features = np.argwhere(x_var > self.params["var_filt_x"]).squeeze()
        #     self.x = self.x.iloc[:, x_features]
        self.x_features = self.x.columns.tolist()
        self.x_samples = self.x.index
        # s should be of shape (num_samples, num_constraints)
        # if self.params["var_filt_s"] is not None:
        #     s_var = np.nanvar(s, axis=0)
        #     s_features = np.argwhere(s_var > self.params["var_filt_s"]).squeeze()
        #     self.s = self.s.iloc[:, s_features]
        self.s_features = self.s.columns.tolist()
        self.s_samples = self.s.index
        # y should be of shape (num_samples, num_targets)
        self.y_features = self.y.columns.tolist()
        self.y_samples = self.y.index
        if self.use_c:
            self.c_features = self.c.columns.tolist()
            self.c_samples = self.c.index

        # Get intersecting samples
        if self.use_c:
            # Have x, s and y but not c
            self.x_s_y_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.s_samples))
                    .intersection(set(self.y_samples))
                    .difference(set(self.c_samples))
                )
            )
            # Have x and y but not s or c
            self.x_y_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.y_samples))
                    .difference(set(self.s_samples))
                    .difference(set(self.c_samples))
                )
            )
            # Have only x and s but not y or c
            self.x_s_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.s_samples))
                    .difference(set(self.y_samples))
                    .difference(set(self.c_samples))
                )
            )
            # Have only x
            self.x_only_samples = sorted(
                list(
                    set(self.x_samples)
                    .difference(set(self.s_samples))
                    .difference(set(self.y_samples))
                    .difference(set(self.c_samples))
                )
            )
            # Have x, c, and y but not s - PiCo training samples if using s
            self.x_c_y_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.c_samples))
                    .intersection(set(self.y_samples))
                    .difference(set(self.s_samples))
                )
            )
            # Have x, c but not s or y
            self.x_c_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.c_samples))
                    .difference(set(self.y_samples))
                    .difference(set(self.s_samples))
                )
            )
            # Don't care about x_s_c or x_s_y_c since we don't expect any samples, but get sets anyway
            self.x_s_c_y_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.c_samples))
                    .intersection(set(self.y_samples))
                    .intersection(set(self.s_samples))
                )
            )
            # x,s,c but not y
            self.x_s_c_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.c_samples))
                    .difference(set(self.y_samples))
                    .intersection(set(self.s_samples))
                )
            )
        else:
            # Have x, s and y
            self.x_s_y_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.s_samples))
                    .intersection(set(self.y_samples))
                )
            )
            # Have only x and y but not s
            self.x_y_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.y_samples))
                    .difference(set(self.s_samples))
                )
            )
            # Have only x and s but not y
            self.x_s_samples = sorted(
                list(
                    set(self.x_samples)
                    .intersection(set(self.s_samples))
                    .difference(set(self.y_samples))
                )
            )
            # Have only x
            self.x_only_samples = sorted(
                list(
                    set(self.x_samples)
                    .difference(set(self.s_samples))
                    .difference(set(self.y_samples))
                )
            )

        # Make a dict of sample types
        self.sample_types = {sample: "x_s_y" for sample in self.x_s_y_samples}
        self.sample_types.update({sample: "x_y" for sample in self.x_y_samples})
        self.sample_types.update({sample: "x_s" for sample in self.x_s_samples})
        self.sample_types.update({sample: "x_only" for sample in self.x_only_samples})
        if self.use_c:
            # Add extra sample types
            self.sample_types.update(
                {sample: "x_s_c_y" for sample in self.x_s_c_y_samples}
            )
            self.sample_types.update({sample: "x_c_y" for sample in self.x_c_y_samples})
            self.sample_types.update({sample: "x_s_c" for sample in self.x_s_c_samples})
            self.sample_types.update({sample: "x_c" for sample in self.x_c_samples})

        # Merge left so keep only samples that have x
        self.x_s_y = pd.merge(
            self.x, self.s, how="left", left_index=True, right_index=True
        )
        self.x_s_y = pd.merge(
            self.x_s_y, self.y, how="left", left_index=True, right_index=True
        )
        if self.use_c:
            # Add c but keep name the same
            self.x_s_y = pd.merge(
                self.x_s_y, self.c, how="left", left_index=True, right_index=True
            )
        # Create sample to indices map from samples in this joint dataset
        self.sample_to_idx = {self.x_s_y.index[i]: i for i in range(len(self.x_s_y))}
        self.idx_to_sample = {i: self.x_s_y.index[i] for i in range(len(self.x_s_y))}

        # Put joined data and idx back
        self.x = self.x_s_y.loc[:, self.x_features].to_numpy().astype(np.float32)
        self.s = self.x_s_y.loc[:, self.s_features].to_numpy().astype(np.float32)
        self.y = self.x_s_y.loc[:, self.y_features].to_numpy().astype(np.float32)
        self.idx = (
            self.x_s_y.index.map(lambda x: self.sample_to_idx[x]).to_numpy().astype(int)
        )
        # Data availability for a loaded sample -- quicker than checking for NaN when loading
        self.st = (
            self.x_s_y.index.map(lambda x: self.sample_types[x]).to_numpy().astype(str)
        )
        if self.use_c:
            self.c = self.x_s_y.loc[:, self.c_features]
            # for column in self.c_features:
            # If features are binary then store as bool, if not store as np.float32
            # self.c.loc[:, column] = optimize_dtype(self.c.loc[:, column])
            # If dropping nas we are using in PiCo, if not in something else
            if drop_na:
                self.c = self.c.to_numpy().astype(np.float32)
            else:
                self.c = self.c.to_numpy()

        # Get idx for different sample data availabilities
        self.x_s_y_idx = [self.sample_to_idx[x] for x in self.x_s_y_samples]
        self.x_y_idx = [self.sample_to_idx[x] for x in self.x_y_samples]
        self.x_s_idx = [self.sample_to_idx[x] for x in self.x_s_samples]
        self.x_only_idx = [self.sample_to_idx[x] for x in self.x_only_samples]

        print(f"x_s_y: {len(self.x_s_y_idx)}")
        print(f"x_y: {len(self.x_y_idx)}")
        print(f"x_s: {len(self.x_s_idx)}")
        print(f"x_only: {len(self.x_only_idx)}")

        if self.use_c:
            self.x_s_c_y_idx = [self.sample_to_idx[x] for x in self.x_s_c_y_samples]
            self.x_c_y_idx = [self.sample_to_idx[x] for x in self.x_c_y_samples]
            self.x_s_c_idx = [self.sample_to_idx[x] for x in self.x_s_c_samples]
            self.x_c_idx = [self.sample_to_idx[x] for x in self.x_c_samples]

        # Print
        if verbose:
            print(f"\n{'Dataset created:':<35}")
            print(f"{'-' * 50}")
            print(f"{'Data type':<35}{'Size'}")
            print(f"{'-' * 50}")
            print(f"{'x':<35}{self.x.shape}")
            print(f"{'s':<35}{self.s.shape}")
            print(f"{'y':<35}{self.y.shape}")
            if self.use_c:
                print(f"{'c':<35}{self.c.shape}")
            print(f"{'-' * 50}")

        # Placeholder priors for s
        self.s_prior_loc = None
        self.s_prior_scale = None

    def mean_impute_x(self, train_idx):
        """Impute missing values in x using column means from train_idx."""
        means = np.nanmean(self.x[train_idx], axis=0)

        # Find indices that you need to replace
        inds = np.where(np.isnan(self.x))

        # Place column means in the indices. Align the arrays using take
        self.x[inds] = np.take(means, inds[1])

    def var_filter_x(self, train_idx):
        """Variance-filter x columns using train_idx and update feature list."""
        # Variance filter on x using training data only
        variances = np.nanvar(self.x[train_idx], axis=0)
        feature_inds = np.argsort(variances)[-self.params["var_filt_x"] :]
        self.x = self.x[:, feature_inds]
        self.x_features = [self.x_features[i] for i in feature_inds]
        print(
            f"[INFO] Number of features in x reduced to {self.x.shape[1]} after variance filtering"
        )

    def mean_impute_s(self, train_idx):
        """Impute missing values in s using column means from train_idx."""
        means = np.nanmean(self.s[train_idx], axis=0)

        # Find indices that you need to replace
        inds = np.where(np.isnan(self.s))

        # Place column means in the indices. Align the arrays using take
        self.s[inds] = np.take(means, inds[1])

    def pca_s(self, train_idx):
        """Fit PCA on s (train_idx) and replace s with PCA scores."""
        # Scale all features
        scaler = StandardScaler()
        scaler = scaler.fit(self.s[train_idx])
        s_scaled = scaler.transform(self.s)
        self.pca = PCA(n_components=128, whiten=True)
        # Fit on training samples with s
        self.pca.fit(s_scaled[train_idx])
        # Transform all samples with s, cannot apply to others (NAs)
        # Need to reshape s since samples with NA have more columns
        s_pca = np.nan * np.zeros((len(self.s), 128))
        s_pca[self.x_s_y_idx] = self.pca.transform(s_scaled[self.x_s_y_idx])
        s_pca[self.x_s_idx] = self.pca.transform(s_scaled[self.x_s_idx])
        self.s = s_pca

    def s_prior_fn_loc(self):
        """Return the stored prior mean for s."""
        return self.s_prior_loc

    def s_prior_fn_scale(self):
        """Return the stored prior scale for s."""
        return self.s_prior_scale

    def y_prior_fn_loc(self):
        """Return the stored prior mean for y."""
        return self.y_prior_loc

    def y_prior_fn_scale(self):
        """Return the stored prior scale for y."""
        return self.y_prior_scale

    def __len__(self):
        """Return the number of samples."""
        return len(self.x)

    def __getitem__(self, idx):
        """Return a single sample tuple (x, s, c, y, idx, st)."""
        if self.use_c:
            return (
                self.x[idx],
                self.s[idx],
                self.c[idx],
                self.y[idx],
                self.idx[idx],
                self.st[idx],
            )
        else:
            return (
                self.x[idx],
                self.s[idx],
                np.ones_like(self.s[idx]) * np.nan,
                self.y[idx],
                self.idx[idx],
                self.st[idx],
            )


def split_dataset(
    data: Manual,
    test_samples=[],
    fold=0,
    val_split=0.2,
    seed=4563,
    pretraining=None,
    var_filter=True,
    verbose=False,
):
    """
    Splits a dataset into training, validation, and test sets for each stage.
    Parameters:
    data (Manual): The dataset to be split.
    test_samples (list, optional): List of samples to be used as the test set. Defaults to an empty list.
    fold (int, optional): The fold index for KFold splitting. Defaults to 0.
    val_split (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.2.
    seed (int, optional): Random seed for reproducibility. Defaults to 4563.
    pretraining (str, optional): Specifies the pretraining stage ('i' for iCoVAE, 'p' for PiCo). Defaults to None.
    verbose (bool, optional): If True, prints detailed information about the dataset splits. Defaults to False.
    Returns:
    tuple: A tuple containing six Subset objects:
        - Subset for iCoVAE training set (x only, x_s)
        - Subset for iCoVAE validation set (x_s)
        - Empty Subset (for compatibility)
        - Subset for PiCo training set
        - Subset for PiCo validation set
        - Subset for test set
    Notes:
    - Ensures no samples with 's' end up in the PiCo test set (they are all used for training the iCoVAE).
    - Ensures no samples with 's' end up in the validation set for PiCo (no representations from iCoVAE training set in PiCo validation set).
    - If `test_samples` is provided, it defines the test set; otherwise, the test set is derived from samples with only x and y.
    - Uses KFold for splitting samples with x and s.
    - Performs mean imputation on all of x using the iCoVAE training set.
    - Sets priors for 's' based on the training set mean and standard deviation.
    """
    k = int(1 / val_split)
    if len(test_samples) > 0:
        test_idx = [data.sample_to_idx[sample] for sample in test_samples]
        # No test samples
        train_val_idx = [idx for idx in data.idx if idx not in test_idx]
        # No samples with only x and y for iCoVAE (i) training and validation
        train_val_idx_i = [idx for idx in train_val_idx if idx not in data.x_y_idx]
        if data.use_c:
            # No samples with x, c and y in iCoVAE train+val - these are for PiCo train val. Samples with x, s, c, y will stay but these are not typically available
            train_val_idx_i = [
                idx for idx in train_val_idx_i if idx not in data.x_c_y_idx
            ]
            # No samples with x and c in iCoVAE train+val (can make new predictions)
            train_val_idx_i = [
                idx for idx in train_val_idx_i if idx not in data.x_c_idx
            ]
            # Test samples stay the same, train and val for PiCo (p) has x and y
            train_val_idx_p = [
                idx
                for idx in train_val_idx
                if (idx in data.x_s_c_y_idx) or (idx in data.x_c_y_idx)
            ]
        else:
            train_val_idx_p = [
                idx
                for idx in train_val_idx
                if (idx in data.x_s_y_idx) or (idx in data.x_y_idx)
            ]
    else:
        # No samples with only x and y for iCoVAE training and validation
        train_val_idx_i = [idx for idx in data.idx if idx not in data.x_y_idx]
        # All samples from iCoVAE training and validation are always in PiCo training set (refitted model has all of these in train set)
        # All samples with x,y,s or just x,y
        train_val_idx_p = [
            idx
            for idx in data.idx
            if ((idx in data.x_s_y_idx) or (idx in data.x_y_idx))
        ]
        # We do train val test split later, so return an empty test set here
        test_idx = []

    # Put all x only samples not in test set in training set for iCoVAE
    train_idx_i1 = [idx for idx in train_val_idx_i if idx in data.x_only_idx]
    # Split samples with x and s
    if data.use_c:
        train_val_idx_i2 = [
            idx
            for idx in train_val_idx_i
            if (idx in data.x_s_y_idx)
            or (idx in data.x_s_idx)
            or (idx in data.x_s_c_idx)
            or (idx in data.x_s_c_y_idx)
        ]
    else:
        train_val_idx_i2 = [
            idx
            for idx in train_val_idx_i
            if (idx in data.x_s_y_idx) or (idx in data.x_s_idx)
        ]

    # Split samples with x and s using KFold
    kf_i2 = KFold(n_splits=k, random_state=seed, shuffle=True)
    # Randomly split samples with x and s
    for i, (fold_train_idx, fold_val_idx) in enumerate(kf_i2.split(train_val_idx_i2)):
        if i == fold:
            train_idx_i2 = [train_val_idx_i2[j] for j in fold_train_idx]
            val_idx_i = [train_val_idx_i2[j] for j in fold_val_idx]
            break

    train_idx_i = train_idx_i1 + train_idx_i2

    if var_filter:
        # Use train_idx_i to do variance filtering on x
        data.var_filter_x(train_idx_i)

    # Use train_idx_i to do mean imputation on all of x
    data.mean_impute_x(train_idx_i)

    # Put all x_s_y samples in PiCo training set, since these were seen by pretrained iCoVAE
    if data.use_c:
        train_idx_p1 = [
            idx
            for idx in train_val_idx_p
            if (idx in data.x_s_y_idx) or (idx in data.x_s_c_y_idx)
        ]
        train_val_idx_p2 = [idx for idx in train_val_idx_p if (idx in data.x_c_y_idx)]
    else:
        train_idx_p1 = [idx for idx in train_val_idx_p if idx in data.x_s_y_idx]
        # Split remaining samples by val_split
        train_val_idx_p2 = [idx for idx in train_val_idx_p if idx in data.x_y_idx]

    # If we need to create a test set split k+1 ways
    if len(test_samples) == 0:
        k += 1

    # Split samples with x and y using KFold
    kf_p2 = KFold(n_splits=k, random_state=seed, shuffle=True)
    # Randomly split samples with x and y
    if len(train_val_idx_p2) > 0:
        for i, (fold_train_idx, fold_val_idx) in enumerate(
            kf_p2.split(train_val_idx_p2)
        ):
            if i == fold:
                train_idx_p2 = [train_val_idx_p2[j] for j in fold_train_idx]
                val_idx_p = [train_val_idx_p2[j] for j in fold_val_idx]
                break
    else:
        train_idx_p2 = []
        val_idx_p = []

    train_idx_p = train_idx_p1 + train_idx_p2

    # Split datasets according to sample type
    # print(train_data_i1.dataset.prior_loc.shape)
    # print(train_data_i1.dataset.prior_scale.shape)

    # Set priors for s based on training set mean and std
    # Store prior in dataset
    data.s_prior_loc = np.expand_dims(np.nanmean(data.s[train_idx_i2], axis=0), axis=0)
    data.s_prior_scale = np.expand_dims(np.nanstd(data.s[train_idx_i2], axis=0), axis=0)

    if verbose:
        print(f"{'Data split summary:':<35}")
        print(f"{'-' * 50}")
        print(f"{'Dataset':<35}{'Size'}")
        print(f"{'-' * 50}")
        print(
            f"{'Train iCoVAE (x only, x_s)':<35}{len(train_idx_i1)}, {len(train_idx_i2)}"
        )
        print(f"{'Val iCoVAE (x_s)':<35}{len(val_idx_i)}")
        print(f"{'Train PiCo':<35}{len(train_idx_p)}")
        print(f"{'Val PiCo':<35}{len(val_idx_p)}")
        print(f"{'Test':<35}{len(test_idx)}")
        print(f"{'-' * 50}")

    if pretraining == "i":
        print("Pretraining iCoVAE: Train and validation sets combined")
        return (
            Subset(data, train_idx_i1),
            Subset(data, train_idx_i2 + val_idx_i),
            Subset(data, []),
            Subset(data, train_idx_p),
            Subset(data, val_idx_p),
            Subset(data, test_idx),
        )
    elif pretraining == "p":
        print("Pretraining PiCo: Train and validation sets combined")
        return (
            Subset(data, train_idx_i1),
            Subset(data, train_idx_i2 + val_idx_i),
            Subset(data, []),
            Subset(data, train_idx_p + val_idx_p),
            Subset(data, []),
            Subset(data, test_idx),
        )
    else:
        return (
            Subset(data, train_idx_i1),
            Subset(data, train_idx_i2),
            Subset(data, val_idx_i),
            Subset(data, train_idx_p),
            Subset(data, val_idx_p),
            Subset(data, test_idx),
        )


def process_depmap_gdsc(wd_path, experiment, **kwargs):
    """Load (and download if needed) DepMap + GDSC data and select test samples."""
    # DOWNLOADING DATA
    # If data not present in specified folders, download it
    if not os.path.exists(f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv"):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40448555",
            f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv",
        )
    if not os.path.exists(
        f"{wd_path}/data/depmap23q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    ):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40449128",
            f"{wd_path}/data/depmap23q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        )
    if not os.path.exists(
        f"{wd_path}/data/depmap23q2/OmicsSomaticMutationsMatrixDamaging.csv"
    ):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40449647",
            f"{wd_path}/data/depmap23q2/OmicsSomaticMutationsMatrixDamaging.csv",
        )
    if not os.path.exists(
        f"{wd_path}/data/depmap23q2/OmicsSomaticMutationsMatrixHotspot.csv"
    ):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40449650",
            f"{wd_path}/data/depmap23q2/OmicsSomaticMutationsMatrixHotspot.csv",
        )
    if not os.path.exists(f"{wd_path}/data/depmap23q2/Model.csv"):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40448834",
            f"{wd_path}/data/depmap23q2/Model.csv",
        )
    if not os.path.exists(f"{wd_path}/data/depmap23q2/ModelCondition.csv"):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40448837",
            f"{wd_path}/data/depmap23q2/ModelCondition.csv",
        )
    if not os.path.exists(
        f"{wd_path}/data/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx"
    ):
        urllib.request.urlretrieve(
            "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx",
            f"{wd_path}/data/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx",
        )

    # LOADING DATA
    # Load gene expression (indexed by DepMapID)
    exp = pd.read_csv(
        f"{wd_path}/data/depmap23q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        index_col=0,
    )
    # Load sample details (contains COSMICID to DepMapID map)
    model = pd.read_csv(f"{wd_path}/data/depmap23q2/Model.csv", index_col=0)
    # Reformat cancer types
    model["OncotreeLineage"] = model["OncotreeLineage"].map(type_renamer)
    # Load gene effect data (indexed by DepMapID)
    eff = pd.read_csv(f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv", index_col=0)
    # Also include confounders in eff
    # See how this is processed for crisprconverter
    conf = pd.read_csv(f"{wd_path}/data/depmap23q2/AchillesScreenQCReport.csv")
    conf = conf[conf["Library"] == "Avana"].groupby("ModelID").mean()
    # eff = pd.merge(eff, conf, left_index=True, right_index=True, how="outer")

    # Load GDSC (indexed by COSMICID)
    gdsc_scores = pd.read_excel(
        f"{wd_path}/data/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx"
    )
    gdsc = gdsc_scores.pivot_table(
        index="COSMIC_ID", columns="DRUG_NAME", values="LN_IC50"
    )

    # Rename columns to be just HGNC name
    eff = eff.rename(mapper=gene_column_renamer, axis=1)
    exp = exp.rename(mapper=gene_column_renamer, axis=1)

    # Make mapping from COSMICID to DepMapID
    cosmic_to_depmapid = model["COSMICID"].reset_index()
    cosmic_to_depmapid = {row["COSMICID"]: row.name for i, row in model.iterrows()}

    # Map GDSC samples to DepMap_ID, leave COSMICID if not present
    gdsc = gdsc.rename(cosmic_to_depmapid, axis=0)
    # Convert all columns to upper case and take target.upper()
    gdsc = gdsc.rename(lambda x: x.upper(), axis=1)

    # TEST SAMPLE SELECTION/EXPERIMENT DEFINITION
    if experiment is None:
        test_samples = []
    elif experiment == "h16":
        experiment = [
            "Vulva_Vagina",
            "Testis",
            "Ampulla_of_Vater",
            "Prostate",
            "Thyroid",
            "Eye",
            "Cervix",
            "Pleura",
            "Liver",
            "Bladder_Urinary_Tract",
            "Kidney",
            "Bone",
            "Peripheral_Nervous_System",
            "Fibroblast",
            "Uterus",
            "Biliary_Tract",
        ]
        test_samples = set(
            model[model["OncotreeLineage"].isin(experiment)].index.tolist()
        )
        # Need to be samples with gene exp
        test_samples = test_samples.intersection(set(exp.index.tolist()))
        test_samples = sorted(list(test_samples))
    elif experiment.split("_")[-1] == "only":
        test_samples = set(
            model[model["OncotreeLineage"] != experiment.split("_")[0]].index.tolist()
        )
        test_samples = test_samples.intersection(set(exp.index.tolist()))
        test_samples = sorted(list(test_samples))
    else:
        test_samples = set(model[model["OncotreeLineage"] == experiment].index.tolist())
        # Need to be samples with gene exp
        test_samples = test_samples.intersection(set(exp.index.tolist()))
        test_samples = sorted(list(test_samples))

    return exp, eff, None, gdsc, test_samples


def process_depmap_gdsc_transneo(wd_path, experiment, **kwargs):
    """Load DepMap + TransNEO data, align features, and select test samples."""
    # DOWNLOADING DATA
    # If data not present in specified folders, download it
    if not os.path.exists(f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv"):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40448555",
            f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv",
        )
    if not os.path.exists(
        f"{wd_path}/data/depmap23q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    ):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40449128",
            f"{wd_path}/data/depmap23q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        )
    if not os.path.exists(f"{wd_path}/data/depmap23q2/Model.csv"):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40448834",
            f"{wd_path}/data/depmap23q2/Model.csv",
        )
    if not os.path.exists(
        f"{wd_path}/data/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx"
    ):
        urllib.request.urlretrieve(
            "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx",
            f"{wd_path}/data/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx",
        )
    # Download L1000 gene list
    if not os.path.exists(f"{wd_path}/data/transneo/lincs_l1000_genes.txt"):
        urllib.request.urlretrieve(
            "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt?AWSAccessKeyId=AKIATAWTSI6KFUAMHHXN&Expires=1764619768&Signature=GE7TNTgkkQ3mWuOU7mnZbsmTW7I%3D",
            f"{wd_path}/data/transneo/lincs_l1000_genes.txt",
        )

    # LOADING DATA
    # Load gene expression (indexed by DepMapID)
    exp_depmap = pd.read_csv(
        f"{wd_path}/data/depmap23q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        index_col=0,
    )
    exp_transneo = pd.read_csv(
        f"{wd_path}/data/transneo/transneo_exp_filt.csv", index_col=0
    )
    exp_transneo_val = pd.read_csv(
        f"{wd_path}/data/transneo/transneo_exp_filt_val.csv", index_col=0
    )
    # Load sample details (contains COSMICID to DepMapID map)
    model = pd.read_csv(f"{wd_path}/data/depmap23q2/Model.csv", index_col=0)
    # Load gene effect data (indexed by DepMapID)
    eff = pd.read_csv(f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv", index_col=0)
    transneo_response = pd.read_csv(
        f"{wd_path}/data/transneo/training_df.csv"
    ).set_index("Trial.ID")
    transneo_response_val1 = pd.read_csv(
        f"{wd_path}/data/transneo/testing_her2neg_df.csv"
    ).set_index("Trial.ID")
    transneo_response_val2 = pd.read_csv(
        f"{wd_path}/data/transneo/testing_her2pos_df.csv"
    ).set_index("Trial.ID")

    transneo_response = pd.concat(
        [transneo_response, transneo_response_val1, transneo_response_val2], axis=0
    )

    # Get additional features to choose from for model. Features are selected from this downstream
    transneo_features = transneo_response

    # Load GDSC (indexed by COSMICID)
    gdsc_scores = pd.read_excel(
        f"{wd_path}/data/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx"
    )
    gdsc = gdsc_scores.pivot_table(
        index="COSMIC_ID", columns="DRUG_NAME", values="LN_IC50"
    )

    # Rename columns to be HGNC name for knockouts
    eff = eff.rename(mapper=gene_column_renamer, axis=1)
    # Rename columns to be Entrez/NCBI ID for expression
    exp_depmap = exp_depmap.rename(mapper=gene_column_renamer_ncbi, axis=1)
    # Map columns to ENSG from NCBI
    ncbi_to_ensg = pd.read_csv(f"{wd_path}/data/transneo/biomart_ncbi_to_ensg.csv")
    ncbi_to_ensg = ncbi_to_ensg[
        ~ncbi_to_ensg["NCBI gene (formerly Entrezgene) ID"].isna()
    ]
    ncbi_to_ensg_map = {
        str(int(row["NCBI gene (formerly Entrezgene) ID"])): row["Gene stable ID"]
        for ind, row in ncbi_to_ensg.iterrows()
    }
    exp_depmap = exp_depmap.rename(mapper=ncbi_to_ensg_map, axis=1)

    # Deal with duplicate genes by taking mean
    exp_depmap = (
        exp_depmap.transpose().reset_index().groupby("index").mean().transpose()
    )

    # Make mapping from COSMICID to DepMapID
    cosmic_to_depmapid = model["COSMICID"].reset_index()
    cosmic_to_depmapid = {row["COSMICID"]: row.name for i, row in model.iterrows()}

    # Map GDSC samples to DepMap_ID, leave COSMICID if not present
    gdsc = gdsc.rename(cosmic_to_depmapid, axis=0)
    # Convert all columns to upper case and take target.upper()
    gdsc = gdsc.rename(lambda x: x.upper(), axis=1)

    # TEST SAMPLE SELECTION/EXPERIMENT DEFINITION
    if experiment == "artemis_pbcp":
        test_samples = sorted(list(set(exp_transneo_val.index.tolist())))
    elif experiment == "tnbc":
        test_samples = transneo_features[
            (transneo_features["ER.status"] == 0)
            & (transneo_features["HER2.status"] == 0)
        ].index.tolist()
    else:
        ValueError("Unrecognised experiment. Please use artemis_pbcp or tnbc.")

    # Concatenate all expression data and filter for shared genes
    shared_cols = sorted(
        list(set(exp_transneo.columns).intersection(set(exp_depmap.columns)))
    )
    print(f"Number of shared features in x: {len(shared_cols)}")

    # We prefilter genes here to LINCS L1000 to reduce dimensionality and avoid a very biased variance filter
    lincs_genes = pd.read_csv(
        f"{wd_path}/data/transneo/lincs_l1000_genes.txt", header=0, sep="\t"
    )[["ensembl_id", "feature_space"]]
    lincs_genes = lincs_genes[lincs_genes["feature_space"] == "landmark"][
        "ensembl_id"
    ].values
    shared_cols = [gene for gene in lincs_genes if gene in shared_cols]
    print(f"Number of LINCS L1000 genes in shared features: {len(shared_cols)}")

    exp_transneo = exp_transneo.loc[:, shared_cols]
    exp_transneo_val = exp_transneo_val.loc[:, shared_cols]
    exp_depmap = exp_depmap.loc[:, shared_cols]

    # Concatenate by samples, all should have same columns
    exp = pd.concat([exp_depmap, exp_transneo, exp_transneo_val], axis=0)

    # Concatenate GDSC response and DepMap effect, concatenate by feature, so pd.merge
    eff_gdsc = pd.merge(eff, gdsc, right_index=True, left_index=True, how="outer")

    return exp, eff_gdsc, transneo_features, transneo_response, test_samples


def process_depmap_ctrp(wd_path, experiment, **kwargs):
    """Load (and download if needed) DepMap + CTRP data and select test samples."""
    # DOWNLOADING DATA
    # If data not present in specified folders, download it
    if not os.path.exists(f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv"):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40448555",
            f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv",
        )
    if not os.path.exists(
        f"{wd_path}/data/depmap23q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    ):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40449128",
            f"{wd_path}/data/depmap23q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        )
    if not os.path.exists(f"{wd_path}/data/depmap23q2/Model.csv"):
        urllib.request.urlretrieve(
            "https://figshare.com/ndownloader/files/40448834",
            f"{wd_path}/data/depmap23q2/Model.csv",
        )
    if not os.path.exists(
        f"{wd_path}/data/ctrp/CTRPv2.0_2015_ctd2_ExpandedDataset.zip"
    ):
        urllib.request.urlretrieve(
            "https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip",
            f"{wd_path}/data/ctrp/CTRPv2.0_2015_ctd2_ExpandedDataset.zip",
        )
        # Extract files from CTRP zip
        with zipfile.ZipFile(
            f"{wd_path}/data/ctrp/CTRPv2.0_2015_ctd2_ExpandedDataset.zip", "r"
        ) as zip_ref:
            zip_ref.extractall(f"{wd_path}/data/ctrp")

    # LOADING DATA
    # Load gene expression (indexed by DepMapID)
    exp = pd.read_csv(
        f"{wd_path}/data/depmap23q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        index_col=0,
    )
    # Load sample details (contains COSMICID to DepMapID map)
    model = pd.read_csv(f"{wd_path}/data/depmap23q2/Model.csv", index_col=0)
    # Load gene effect data (indexed by DepMapID)
    eff = pd.read_csv(f"{wd_path}/data/depmap23q2/CRISPRGeneEffect.csv", index_col=0)

    # Load necessary files for CTRP
    ctrp_scores = pd.read_csv(
        f"{wd_path}/data/ctrp/v20.data.curves_post_qc.txt", delimiter="\t"
    )
    ctrp_experiments = pd.read_csv(
        f"{wd_path}/data/ctrp/v20.meta.per_experiment.txt", delimiter="\t"
    )
    ctrp_compounds = pd.read_csv(
        f"{wd_path}/data/ctrp/v20.meta.per_compound.txt", delimiter="\t"
    )
    ctrp_cells = pd.read_csv(
        f"{wd_path}/data/ctrp/v20.meta.per_cell_line.txt", delimiter="\t"
    )
    ctrp_wells = pd.read_csv(
        f"{wd_path}/data/ctrp/v20.data.per_cpd_well.txt",
        delimiter="\t",
        usecols=["master_cpd_id", "cpd_conc_umol"],
    )

    ctrp_wells_min = (
        ctrp_wells.groupby("master_cpd_id")
        .min()
        .rename({"cpd_conc_umol": "min_conc_umol"}, axis=1)
    )
    ctrp_wells_max = (
        ctrp_wells.groupby("master_cpd_id")
        .max()
        .rename({"cpd_conc_umol": "max_conc_umol"}, axis=1)
    )
    ctrp_wells = pd.merge(
        ctrp_wells_max, ctrp_wells_min, left_index=True, right_index=True
    ).reset_index()

    model["ccl_name"] = model["CCLEName"].apply(lambda x: str(x).split("_")[0])
    ctrp_scores = pd.merge(
        ctrp_scores[["experiment_id", "area_under_curve", "master_cpd_id"]],
        ctrp_experiments,
        left_on="experiment_id",
        right_on="experiment_id",
    )
    ctrp_scores = pd.merge(
        ctrp_scores,
        ctrp_wells[["master_cpd_id", "min_conc_umol", "max_conc_umol"]],
        left_on="master_cpd_id",
        right_on="master_cpd_id",
    )
    ctrp_scores = pd.merge(
        ctrp_scores,
        ctrp_compounds,
        left_on="master_cpd_id",
        right_on="master_cpd_id",
    )
    ctrp_scores = pd.merge(
        ctrp_scores,
        ctrp_cells[["master_ccl_id", "ccl_name"]],
        left_on="master_ccl_id",
        right_on="master_ccl_id",
    )
    ctrp_scores = pd.merge(
        ctrp_scores, model.reset_index(), left_on="ccl_name", right_on="ccl_name"
    )
    ctrp_scores["auc_norm"] = ctrp_scores["area_under_curve"] / (
        np.log2(ctrp_scores["max_conc_umol"]) - np.log2(ctrp_scores["min_conc_umol"])
    )
    # Index by ModelID, columns are compound names
    ctrp_df = ctrp_scores.pivot_table(
        index="ModelID", columns="cpd_name", values="auc_norm"
    )
    # Make all drug names upper case for consistency
    ctrp = ctrp_df.rename(columns=lambda x: str(x).upper()).astype(float)

    # Rename columns to be just HGNC name
    eff = eff.rename(mapper=gene_column_renamer, axis=1)
    exp = exp.rename(mapper=gene_column_renamer, axis=1)

    # TEST SAMPLE SELECTION/EXPERIMENT DEFINITION
    if experiment is None:
        test_samples = []
    elif experiment == "h16":
        experiment = [
            "Vulva_Vagina",
            "Testis",
            "Ampulla_of_Vater",
            "Prostate",
            "Thyroid",
            "Eye",
            "Cervix",
            "Pleura",
            "Liver",
            "Bladder_Urinary_Tract",
            "Kidney",
            "Bone",
            "Peripheral_Nervous_System",
            "Fibroblast",
            "Uterus",
            "Biliary_Tract",
        ]
        test_samples = set(
            model[model["OncotreeLineage"].isin(experiment)].index.tolist()
        )
        # Need to be samples with gene exp
        test_samples = test_samples.intersection(set(exp.index.tolist()))
        test_samples = sorted(list(test_samples))
    elif experiment.split("_")[-1] == "only":
        test_samples = set(
            model[model["OncotreeLineage"] != experiment.split("_")[0]].index.tolist()
        )
        test_samples = test_samples.intersection(set(exp.index.tolist()))
        test_samples = sorted(list(test_samples))
    else:
        test_samples = set(
            model[model["OncotreeLineage"].isin(experiment)].index.tolist()
        )
        # Need to be samples with gene exp
        test_samples = test_samples.intersection(set(exp.index.tolist()))
        test_samples = sorted(list(test_samples))

    return exp, eff, None, ctrp, test_samples


def process_data(dataset, wd_path, experiment, **kwargs):
    """Dispatch to the appropriate dataset loader based on `dataset` name."""
    if dataset == "depmap_gdsc":
        x, s, c, y, test_samples = process_depmap_gdsc(wd_path, experiment, **kwargs)
    elif dataset == "depmap_gdsc_transneo":
        x, s, c, y, test_samples = process_depmap_gdsc_transneo(
            wd_path, experiment, **kwargs
        )
    elif dataset == "depmap_ctrp":
        x, s, c, y, test_samples = process_depmap_ctrp(wd_path, experiment, **kwargs)
    else:
        raise ValueError(
            "Unrecoginised dataset. Please choose from: depmap_gdsc, depmap_gdsc_transneo, depmap_ctrp, depmap_gdsc_scanb_tcga"
        )

    return x, s, c, y, test_samples


def get_data_loaders(
    dataset: Dataset,
    test_samples: str,
    batch_size: int,
    fold: int,
    seed: int,
    val_split: float,
    stage: str,
    hopt: bool,
    verbose: bool = False,
    **kwargs,
):
    """Build DataLoader objects for iCoVAE/PiCo stages with optional HPO settings."""
    if "num_workers" not in kwargs:
        kwargs = {"num_workers": 4, "pin_memory": True}

    loaders = {}

    pretraining = stage if not hopt else None
    train_i_x_only, train_i_x_s, val_i, train_p, val_p, test = split_dataset(
        data=dataset,
        test_samples=test_samples,
        fold=fold,
        val_split=val_split,
        seed=seed,
        pretraining=pretraining,
        verbose=verbose,
    )
    data = {
        "train_i_x_only": train_i_x_only,
        "train_i_x_s": train_i_x_s,
        "val_i": val_i,
        "train_p": train_p,
        "val_p": val_p,
        "test": test,
    }

    if stage == "i":
        # Combine supervised and unsupervsied samples
        curr_train = (data["train_i_x_only"], data["train_i_x_s"])
    elif stage == "p":
        # Add a dummy array, since all these samples are supervised
        curr_train = (data[f"train_{stage}"], np.array([]))

    curr_val = data[f"val_{stage}"]

    if (len(curr_train[0]) % batch_size == 1) or (len(curr_train[1]) % batch_size == 1):
        drop_last_tr = True
    else:
        drop_last_tr = False

    if len(curr_val) % batch_size == 1:
        drop_last_v = True
    else:
        drop_last_v = False

    if len(test) % batch_size == 1:
        drop_last_te = True
    else:
        drop_last_te = False

    if stage == "i":
        loaders["train_x_only"] = DataLoader(
            curr_train[0],
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_tr,
            **kwargs,
        )
        loaders["train_x_s"] = DataLoader(
            curr_train[1],
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_tr,
            **kwargs,
        )
    elif stage == "p":
        loaders["train"] = DataLoader(
            curr_train[0],
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_tr,
            **kwargs,
        )

    if hopt:
        loaders["val"] = DataLoader(
            curr_val,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_v,
            **kwargs,
        )

    if len(test) > 0:
        loaders["test"] = DataLoader(
            test,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_te,
            **kwargs,
        )

    return loaders
