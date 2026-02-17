import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import os
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, log_loss


from .objective import exp_log_likelihood, compute_kl

# Define model sizes for hyperparameter optimisation
model_sizes = {"linear": [], "1": [512], "2": [512, 256], "3": [512, 256, 128]}

cf_model_sizes = {"linear": [], "1": [512], "2": [512, 256], "3": [512, 256, 128]}


def get_search_spaces(model_type: str) -> Dict[str, list]:
    if (
        (model_type == "ElasticNet")
        or (model_type == "CoxPH")
        or model_type == "LogisticRegression"
    ):
        search_spaces = {
            "alpha": np.logspace(-4, 2, 30),
            "l1_ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    elif model_type == "SVR":
        search_spaces = {
            "eps": np.logspace(-3, 0, 15),
            "c": np.logspace(-3, 3, 30),
            "kernel": ["linear"],
        }
    elif model_type == "RandomForestRegressor":
        search_spaces = {
            "max_depth": [None, 8, 12, 16, 20],
            "max_features": [0.5, 0.7],
            "max_samples": [None, 0.8, 0.9],
            "min_samples_leaf": [1, 4],
            "min_samples_split": [2, 5, 10, 15],
            "n_estimators": [200, 400, 600],
        }
    else:
        raise ValueError(
            "Model type not recognised. Please choose from ElasticNet, SVR, RandomForestRegressor, LogisticRegression"
        )

    return search_spaces


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


# MODEL COMPONENT CLASSES
class TabularEncoder(nn.Module):
    """
    MLP encoder that parameterizes q(z|x) as a diagonal Gaussian.

    Builds `n_layers` fully connected layers with optional LayerNorm and dropout.
    Returns `(loc, scale)` where `scale` is softplus-transformed and clamped.

    Args:
        input_dim: Input feature dimension.
        latent_dim: Latent dimensionality.
        layer_width: Hidden layer width.
        n_layers: Number of hidden layers.
        activation: Activation name (`relu`, `leakyrelu`, or None).
        ln: Whether to apply LayerNorm after each hidden layer.
        dropout: Dropout probability.

    Returns:
        loc: Mean of q(z|x), shape [batch, latent_dim].
        scale: Stddev of q(z|x), shape [batch, latent_dim].
    """

    _modules: Dict[str, nn.Module]

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        layer_width: int,
        n_layers: int,
        activation: str = "leakyrelu",
        ln: bool = True,
        dropout: float = 0.0,
    ):
        super(TabularEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layer_width = layer_width
        self.n_layers = n_layers
        self.layer_widths = [layer_width] * n_layers
        self.activation = activation
        self.ln = ln
        self.dropout = dropout
        self._modules = {}

        # Build the layers for the model architecture (without nn.Sequential)
        for layer_num in range(n_layers):
            if layer_num == 0:
                self._modules[f"f{layer_num}"] = nn.Linear(
                    input_dim, self.layer_widths[layer_num]
                )
            else:
                self._modules[f"f{layer_num}"] = nn.Linear(
                    self.layer_widths[layer_num - 1], self.layer_widths[layer_num]
                )
            if self.ln:
                self._modules[f"ln{layer_num}"] = nn.LayerNorm(
                    self.layer_widths[layer_num]
                )
            if self.dropout > 0:
                self._modules[f"d{layer_num}"] = nn.Dropout(p=self.dropout)

        # Final linear layer for latent space, needs to take input dim if no previous layers
        if len(self.layer_widths) == 0:
            self.locs = nn.Linear(self.input_dim, self.latent_dim)
            self.scales = nn.Linear(self.input_dim, self.latent_dim)
        else:
            self.locs = nn.Linear(self.layer_widths[-1], self.latent_dim)
            self.scales = nn.Linear(self.layer_widths[-1], self.latent_dim)

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        # Build layer order for forward pass, including BN, dropout, activations
        for layer_num in range(self.n_layers):
            x = self._modules[f"f{layer_num}"](x)
            if self.ln:
                x = self._modules[f"ln{layer_num}"](x)
            if self.activation == "relu":
                x = F.relu(x)
            if self.activation == "leakyrelu":
                x = F.leaky_relu(x)
            elif self.activation == None:
                pass
            if self.dropout > 0:
                x = self._modules[f"d{layer_num}"](x)

        # Output learns softplus(scale)
        return self.locs(x), torch.clamp(F.softplus(self.scales(x)), min=1e-3, max=1e3)


class TabularDecoder(nn.Module):
    """
    MLP decoder that reconstructs x from latent z.

    Builds `n_layers` fully connected layers with optional LayerNorm and dropout.
    When `calibrated=True`, learns a single global reconstruction scale.

    Args:
        input_dim: Output feature dimension.
        latent_dim: Latent dimensionality.
        activation: Activation name (`relu`, `leakyrelu`, `sigmoid`, or None).
        layer_width: Hidden layer width.
        n_layers: Number of hidden layers.
        ln: Whether to apply LayerNorm after each hidden layer.
        dropout: Dropout probability.
        calibrated: Whether to learn a global reconstruction scale parameter.
    """

    _modules: Dict[str, nn.Module]

    def __init__(
        self,
        input_dim=1000,
        latent_dim=128,
        activation="leakyrelu",
        layer_width=256,
        n_layers=2,
        ln=True,
        dropout=0,
        calibrated=True,
    ):
        super(TabularDecoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layer_width = layer_width
        self.n_layers = n_layers
        self.layer_widths = [layer_width] * n_layers
        self.activation = activation
        self.ln = ln
        self.calibrated = calibrated
        # Learn a single variance for a calibrated decoder
        if self.calibrated:
            self.scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("scale", torch.zeros(1))
        self.dropout = dropout
        self._modules = {}

        # Build the layers for the model architecture (without nn.Sequential)
        for layer_num in range(self.n_layers):
            if layer_num == 0:
                self._modules[f"f{layer_num}"] = nn.Linear(
                    self.latent_dim, self.layer_widths[layer_num]
                )
            else:
                self._modules[f"f{layer_num}"] = nn.Linear(
                    self.layer_widths[layer_num - 1], self.layer_widths[layer_num]
                )
            if self.ln:
                self._modules[f"ln{layer_num}"] = nn.LayerNorm(
                    self.layer_widths[layer_num]
                )
            if self.dropout > 0:
                self._modules[f"d{layer_num}"] = nn.Dropout(p=self.dropout)

        # Final linear layer for output, if no previous layers this needs latent dim input size
        if len(self.layer_widths) == 0:
            self._modules["fd"] = nn.Linear(self.latent_dim, self.input_dim)
        else:
            self._modules["fd"] = nn.Linear(self.layer_widths[-1], self.input_dim)

    def forward(self, z: torch.tensor) -> torch.tensor:
        # Build layer order for forward pass, including LN, dropout, activations
        for layer_num in range(self.n_layers):
            z = self._modules[f"f{layer_num}"](z)
            if self.ln:
                z = self._modules[f"ln{layer_num}"](z)
            if self.activation == "relu":
                z = F.relu(z)
            if self.activation == "leakyrelu":
                z = F.leaky_relu(z)
            elif self.activation == "sigmoid":
                z = F.sigmoid(z)
            elif self.activation == None:
                pass
            if self.dropout > 0:
                z = self._modules[f"d{layer_num}"](z)

        # Runs through last linear layer for reconstruction
        x_recon = self._modules["fd"](z)
        return x_recon


class Diagonal(nn.Module):
    """
    Learnable per-dimension affine transform: y = x * weight + bias.

    Args:
        dim: Number of dimensions.
    """
    def __init__(self, dim):
        super(Diagonal, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x * self.weight + self.bias


class DiagonalRegressor(nn.Module):
    """
    Diagonal Gaussian regressor with learnable mean and variance.

    The mean is modeled as a per-dimension affine transform. Variance can be:
    - `constraint`: per-dimension variance
    - `single`: one shared variance
    - `fixed`: fixed variance buffer (no learning)

    Args:
        dim: Number of output dimensions.
        var_type: Variance mode (`constraint`, `single`, or `fixed`).
    """
    # Changed to initialize weight mean as random
    # Added bias to variance term
    def __init__(self, dim: int, var_type="single"):
        super(DiagonalRegressor, self).__init__()
        self.dim = dim
        self.var_type = var_type
        self.mean_coeff = nn.Parameter(torch.ones(self.dim))
        self.mean_bias = nn.Parameter(torch.zeros(self.dim))
        # should we fit a bias?
        # self.mean_bias = nn.Parameter(torch.zeros(self.dim))
        # Should variance scale with prediction? Will likely mean that higher predictions have lower prob
        # Variance should be a fixed parameter for each target or should have its own dimension associated with it
        # This would give a certainty for each sample, not just based on the value of the prediction
        # For now, model variance
        # self.var_coeff = nn.Parameter(torch.zeros(self.dim))
        # self.var_bias = nn.Parameter(torch.zeros(self.dim))
        # Variance should be learned but not depend on x
        # Below: Learning a variance parameter for each constraint
        if var_type == "constraint":
            self.scale = nn.Parameter(torch.zeros(self.dim))
        # Below: Learning a single variance parameter for all targets
        elif var_type == "single":
            self.scale = nn.Parameter(torch.zeros(1))
        elif var_type == "fixed":
            self.register_buffer("scale", torch.zeros(1))
        # Vector of ones like x to preserve shape for variance, Parameter for pushing to GPU
        # This starts higher to avoid -inf at the start of training due to initialization of mean_coeff as 1

    def forward(self, x: torch.tensor) -> torch.tensor:
        # return x * self.mean_coeff + self.mean_bias, torch.clamp((x * self.var_coeff + self.var_bias).div(2).exp(),
        # min=1e-8)
        # Below: per-constraint variance
        if self.var_type == "constraint":
            scale = torch.clamp(self.scale.div(2).exp(), min=1e-3).repeat(
                (x.shape[0], 1)
            )
        # Below: single variance
        elif self.var_type in ["single", "fixed"]:
            scale = torch.clamp(self.scale.div(2).exp(), min=1e-3).repeat(
                (x.shape[0], self.dim)
            )

        # Test fixing variance for predictions to 1
        return x * self.mean_coeff + self.mean_bias, scale


class DiagonalClassifier(nn.Module):
    """
    Lightweight classifier wrapper using a Diagonal affine transform.

    Args:
        dim: Number of dimensions.
    """
    def __init__(self, dim):
        super(DiagonalClassifier, self).__init__()
        self.dim = dim
        self.diag = Diagonal(self.dim)

    def forward(self, x):
        return self.diag(x)


class RegressionModel(nn.Module):
    """
    Wrapper for a diagonal Gaussian regressor.

    Args:
        dim: Number of output dimensions.
        zdim: Optional latent dimension (kept for interface consistency).
        var_type: Variance mode passed to DiagonalRegressor.
    """
    def __init__(self, dim, zdim=None, var_type="fixed"):
        super(RegressionModel, self).__init__()
        self.dim = dim
        self.zdim = zdim
        print("[INFO] Using diagonal regression...")
        self.reg = DiagonalRegressor(self.dim, var_type=var_type)

    def forward(self, x):
        return self.reg(x)


class ClassificationModel(nn.Module):
    """
    Wrapper for a diagonal classifier.

    Args:
        dim: Number of output dimensions.
        fc: Unused flag for compatibility.
        uneven: Unused flag for compatibility.
        target: Unused flag for compatibility.
        zdim: Optional latent dimension (kept for interface consistency).
        mask: Optional mask (unused).
    """
    def __init__(self, dim, fc=False, uneven=False, target=False, zdim=None, mask=None):
        super(ClassificationModel, self).__init__()
        self.dim = dim
        self.zdim = zdim
        self.fc = fc
        self.uneven = uneven
        self.target = target
        print("Using diagonal Classifier...")
        self.reg = DiagonalClassifier(self.dim)

    def forward(self, x):
        return self.reg(x)


class CondPrior(nn.Module):
    """
    Binary conditional prior p(z|s) with separate parameters for s=1 and s=0.

    For each dimension, learns loc and scale for the true and false cases.

    Args:
        dim: Number of conditional dimensions.
    """
    def __init__(self, dim):
        super(CondPrior, self).__init__()
        self.dim = dim
        self.diag_loc_true = nn.Parameter(torch.zeros(self.dim))
        self.diag_loc_false = nn.Parameter(torch.zeros(self.dim))
        self.diag_scale_true = nn.Parameter(torch.ones(self.dim))
        self.diag_scale_false = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
        scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
        return loc, torch.clamp(scale.div(2).exp(), min=1e-3)


class CondPriorCont(nn.Module):
    """
    Continuous conditional prior with per-dimension affine loc and learnable scale.

    Loc is modeled as `loc = weight * s + bias`. Scale can be per-dimension
    (`target`) or shared (`single`).

    Args:
        dim: Number of conditional dimensions.
        var_type: Variance mode (`target` or `single`).
    """

    def __init__(self, dim, var_type="single"):
        super(CondPriorCont, self).__init__()
        self.dim = dim
        self.var_type = var_type
        self.diag_loc_weight = nn.Parameter(torch.ones(self.dim))
        self.diag_loc_bias = nn.Parameter(torch.zeros(self.dim))
        # Scale should be learned but shouldn't depend on x
        # Per-target variance
        if var_type == "target":
            self.diag_scale = nn.Parameter(torch.zeros(self.dim))
        elif var_type == "single":
            self.diag_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        loc = x * self.diag_loc_weight + self.diag_loc_bias
        if self.var_type == "target":
            scale = torch.clamp(self.diag_scale.div(2).exp(), min=1e-3).repeat(
                (x.shape[0], 1)
            )
        elif self.var_type == "single":
            scale = torch.clamp(self.diag_scale.div(2).exp(), min=1e-3).repeat(
                (x.shape[0], x.shape[1])
            )
        return loc, scale


# COMBINED MODEL CLASSES
class VanillaVAE(nn.Module):
    """
    Standard VAE for tabular inputs with a Gaussian latent space.

    Uses a TabularEncoder to parameterize q(z|x) and a TabularDecoder to
    reconstruct x from z. Training is unsupervised via the ELBO.

    Args:
        layer_width: Hidden layer width for encoder/decoder MLPs.
        n_layers: Number of hidden layers in encoder/decoder.
        z_dim: Latent dimensionality.
        input_dim: Input feature dimension.
        dropout: Dropout probability for encoder/decoder MLPs.
        use_cuda: Whether to move model and buffers to CUDA.

    Attributes:
        encoder: TabularEncoder producing q(z|x) parameters.
        decoder: TabularDecoder reconstructing x from z.
        z_dim: Latent dimensionality.
        input_dim: Input feature dimension.
    """


    def __init__(self, layer_width, n_layers, z_dim, input_dim, dropout, use_cuda=True):
        super(VanillaVAE, self).__init__()
        self.layer_width = layer_width
        self.n_layers = n_layers
        self.dropout = dropout
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.use_cuda = use_cuda
        self.ones = torch.ones(1, self.z_dim)
        self.zeros = torch.zeros(1, self.z_dim)

        self.encoder = TabularEncoder(
            input_dim=input_dim,
            latent_dim=self.z_dim,
            layer_width=self.layer_width,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )
        self.decoder = TabularDecoder(
            input_dim=input_dim,
            latent_dim=self.z_dim,
            layer_width=self.layer_width,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        if self.use_cuda:
            self.cuda()
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()

    def unsup(self, x):
        bs = x.shape[0]
        # inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()

        # compute kl
        prior_params = (self.zeros.expand(bs, -1), self.ones.expand(bs, -1))
        kl = compute_kl(*post_params, *prior_params)

        # compute log probs for x and y
        recon = self.decoder(z)
        elbo = (exp_log_likelihood(recon, x, scale=self.decoder.scale) - kl).mean()
        return -elbo

    def reconstruct_input(self, x):
        return self.decoder(dist.Normal(*self.encoder(x)).rsample())

    def recon_error(self, x):
        mu, sigma = self.encoder(x)
        recon = self.decoder(mu)
        log_pxz = exp_log_likelihood(recon, x, scale=self.decoder.scale)
        return log_pxz.mean()

    def encode(self, x):
        """Encodes input into latent. Only to be used in evaluation, one sample at a time"""
        # Use MAP estimate
        mu, sigma = self.encoder(x)
        # pred_locs = pred_locs.view(-1, self.num_targets)
        # pred_scales = pred_scales.view(-1, self.num_targets)
        return mu.detach()

    def save_models(self, path="./data", seed=4563):
        torch.save(self.encoder, os.path.join(path, f"encoder_{seed}.pt"))
        torch.save(self.decoder, os.path.join(path, f"decoder_{seed}.pt"))
        torch.save(self, os.path.join(path, f"best_model_{seed}.pt"))

    def generate_z_pred(self, data_loader, save_dir, suffix=None):
        """Generates latent representation and predictions for all samples in data loader"""
        z_arr = np.zeros((len(data_loader.dataset), self.z_dim))
        self.eval()
        for x, y, ind in data_loader:
            if self.use_cuda:
                x, y, ind = x.cuda(), y, ind
            z = self.encode(x)

            ind = ind.numpy().astype(int).tolist()

            z_arr[ind, :] = z.cpu().numpy()

        z_cols = []
        for i in range(self.z_dim):
            # zc_cols.append(f"zc_{i}")
            z_cols.append(f"z_{i}")

        z_df = pd.DataFrame(z_arr, columns=z_cols)

        z_df.to_csv(f"{save_dir}/z_pred_{suffix}.csv")

        return None


class iCoVAE(nn.Module):
    """
    Conditional VAE variant with continuous constraints (CCVAE-style).

    The latent space is split into:
    - `z_s`: constraint-aligned dimensions (one per constraint)
    - `z_not_s`: remaining unconstrained dimensions

    A diagonal Gaussian regressor predicts constraint values from `z_s`, and a
    conditional prior `p(z_s | s)` is learned via a continuous conditional prior.
    Training supports both unsupervised and supervised ELBO objectives.

    Args:
        layer_width: Hidden layer width for encoder/decoder MLPs.
        n_layers: Number of hidden layers in encoder/decoder.
        z_dim: Total latent dimensionality.
        constraints: List of constraint names (length = number of supervised dims).
        input_dim: Input feature dimension.
        dropout: Dropout probability for encoder/decoder MLPs.
        s_prior_fn_loc: Callable returning prior mean(s) for constraints.
        s_prior_fn_scale: Callable returning prior scale(s) for constraints.
        use_cuda: Whether to move model and buffers to CUDA.

    Attributes:
        num_constraints: Number of constraints (len(constraints)).
        zs_dim: Latent dimensions aligned to constraints.
        znots_dim: Remaining latent dimensions.
        encoder: TabularEncoder producing q(z|x) parameters.
        decoder: TabularDecoder reconstructing x from z.
        regressor: Diagonal Gaussian regressor for q(s|z_s).
        cond_prior: Conditional prior p(z_s|s).
    """


    def __init__(
        self,
        layer_width,
        n_layers,
        z_dim,
        constraints,
        input_dim,
        dropout,
        s_prior_fn_loc,
        s_prior_fn_scale,
        use_cuda=True,
    ):
        super(iCoVAE, self).__init__()
        self.layer_width = layer_width
        self.n_layers = n_layers
        self.constraints = constraints
        self.dropout = dropout
        self.num_constraints = len(constraints)
        self.z_dim = z_dim
        self.zs_dim = self.num_constraints
        self.znots_dim = z_dim - self.num_constraints
        self.input_dim = input_dim
        self.use_cuda = use_cuda
        self.ones = torch.ones(1, self.znots_dim)
        self.zeros = torch.zeros(1, self.znots_dim)
        # prior needs to be defined from training set -- these will not be unit normals
        self.s_prior_locs = torch.as_tensor(s_prior_fn_loc())
        self.s_prior_scales = torch.as_tensor(s_prior_fn_scale())

        self.encoder = TabularEncoder(
            input_dim=input_dim,
            latent_dim=self.z_dim,
            layer_width=self.layer_width,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        self.decoder = TabularDecoder(
            input_dim=input_dim,
            latent_dim=self.z_dim,
            layer_width=self.layer_width,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        self.regressor = RegressionModel(self.num_constraints)

        print("Using diagonal cond prior...")
        self.cond_prior = CondPriorCont(self.num_constraints)

        if self.use_cuda:
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()
            self.s_prior_locs = self.s_prior_locs.cuda()
            self.s_prior_scales = self.s_prior_scales.cuda()
            self.cuda()

    def unsup(self, x):
        bs = x.shape[0]
        # inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        zs, znots = z.split([self.zs_dim, self.znots_dim], 1)
        qszc = dist.Normal(*self.regressor(zs))
        s = qszc.sample()
        log_qs = qszc.log_prob(s).sum(dim=-1)

        # compute kl
        locs_p_zs, scales_p_zs = self.cond_prior(s)
        prior_params = (
            torch.cat([locs_p_zs, self.zeros.expand(bs, -1)], dim=1),
            torch.cat([scales_p_zs, self.ones.expand(bs, -1)], dim=1),
        )
        kl = compute_kl(*post_params, *prior_params)

        # compute log probs for x and y
        recon = self.decoder(z)
        log_ps = (
            dist.Normal(
                self.s_prior_locs.expand(bs, -1), self.s_prior_scales.expand(bs, -1)
            )
            .log_prob(s)
            .sum(dim=-1)
        )
        elbo = (
            exp_log_likelihood(recon, x, scale=self.decoder.scale)
            + log_ps
            - kl
            - log_qs
        ).mean()

        return -elbo

    def sup(self, x, s):
        bs = x.shape[0]
        # inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        zs, znots = z.split([self.zs_dim, self.znots_dim], 1)
        qs_zc = dist.Normal(*self.regressor(zs))
        log_qs_zc = qs_zc.log_prob(s).sum(dim=-1)

        # compute kl
        locs_p_zc, scales_p_zc = self.cond_prior(s)
        prior_params = (
            torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=1),
            torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=1),
        )
        # prior_params = (self.zeros.expand(bs, -1), self.ones.expand(bs, -1))
        kl = compute_kl(*post_params, *prior_params)

        # compute log probs for x and y
        recon = self.decoder(z)
        log_ps = (
            dist.Normal(
                self.s_prior_locs.expand(bs, -1), self.s_prior_scales.expand(bs, -1)
            )
            .log_prob(s)
            .sum(dim=-1)
        )
        # k = 100 for regressor loss
        log_qs_x = self.qs_x_loss(x, s, 100)
        log_px_z = exp_log_likelihood(recon, x, scale=self.decoder.scale)
        # print(f"log_ps: {log_ps}")
        # print(f"log_qs_x: {log_qs_x}")
        # print(f"log_pxz: {log_px_z}")
        # print(f"kl: {kl}")
        # print(self.regressor.reg.scale)

        # we only want gradients wrt to params of qyz, so stop them propogating to qzx
        log_qs_zc_ = dist.Normal(*self.regressor(zs.detach())).log_prob(s).sum(dim=-1)
        # print(f"log_qs_zc_: {log_qs_zc_}")
        w = torch.exp(torch.tanh(log_qs_zc_ - log_qs_x))
        elbo = (w * (log_px_z - kl - log_qs_zc_) + log_ps + log_qs_x).mean()
        return -elbo

    def qs_x_loss(self, x, s, k):
        """
        Computes the regressor loss q(s|x).
        """
        zs, _ = (
            dist.Normal(*self.encoder(x))
            .rsample(torch.tensor([k]))
            .split([self.zs_dim, self.znots_dim], -1)
        )
        d_locs, d_scales = self.regressor(zs.view(-1, self.zs_dim))
        d = dist.Normal(d_locs, d_scales)
        s = s.expand(k, -1, -1).contiguous().view(-1, self.num_constraints)
        lqs_z = d.log_prob(s).view(k, x.shape[0], self.num_constraints).sum(dim=-1)
        lqs_x = torch.logsumexp(lqs_z, dim=0) - np.log(k)
        return lqs_x

    def reconstruct_input(self, x):
        return self.decoder(dist.Normal(*self.encoder(x)).rsample())

    def recon_error(self, x):
        mu, sigma = self.encoder(x)
        recon = self.decoder(mu)
        log_pxz = exp_log_likelihood(recon, x, scale=self.decoder.scale)
        return -log_pxz.mean()

    def regressor_rmse(
        self, x: torch.Tensor, s: torch.Tensor, map_est: bool, k: int
    ) -> Tuple[float, torch.Tensor]:
        if map_est:
            # Use map estimate from dist for validation preds
            mu, sigma = self.encoder(x)
            mu_s, mu_nots = mu.split([self.zs_dim, self.znots_dim], -1)
            pred_locs, pred_scales = self.regressor(mu_s.view(-1, self.zs_dim))
        else:
            zs, _ = (
                dist.Normal(*self.encoder(x))
                .rsample(torch.tensor([k]))
                .split([self.zs_dim, self.znots_dim], -1)
            )
            pred_locs, pred_scales = self.regressor(zs.view(-1, self.zs_dim))
        pred_locs = pred_locs.view(-1, self.num_constraints)
        pred_scales = pred_scales.view(-1, self.num_constraints)
        s = s.expand(k, -1, -1).contiguous().view(-1, self.num_constraints)
        # Take predictions as mean values for RMSE
        preds = pred_locs
        rmse = (s - preds).square().float().mean(dim=0).sqrt()
        return rmse, preds

    def encode_predict(self, x):
        """Encodes input into latent and generates predictions. Only to be used in evaluation, one sample at a time"""
        # Use MAP estimate
        mu, sigma = self.encoder(x)
        mu_s, mu_nots = mu.split([self.zs_dim, self.znots_dim], -1)
        pred_locs, pred_scales = self.regressor(mu_s.view(-1, self.zs_dim))
        # pred_locs = pred_locs.view(-1, self.num_constraints)
        # pred_scales = pred_scales.view(-1, self.num_constraints)
        return mu_s.detach(), mu_nots.detach(), pred_locs.detach(), pred_scales.detach()

    def save_models(self, path="./data", seed=4563):
        torch.save(self.encoder, os.path.join(path, f"encoder_{seed}.pt"))
        torch.save(self.decoder, os.path.join(path, f"decoder_{seed}.pt"))
        torch.save(self.regressor, os.path.join(path, f"regressor_{seed}.pt"))
        torch.save(self.cond_prior, os.path.join(path, f"cond_prior_{seed}.pt"))
        torch.save(self, os.path.join(path, f"best_model_{seed}.pt"))

    def accuracy(self, data_loader, *args, **kwargs):
        acc = 0.0
        for x, y, _ in data_loader:
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            batch_acc = self.classifier_acc(x, y)
            acc += batch_acc
        return acc / len(data_loader)

    def rmse(self, data_loader, num_targets, map_est, k):
        preds = []
        gt = []
        kl = 0
        for x, s, c, y, _, _ in data_loader:
            bs = x.shape[0]
            gt.append(s)
            if self.use_cuda:
                x, s = x.cuda(non_blocking=True), s.cuda(non_blocking=True)
            _, batch_preds = self.regressor_rmse(x, s, map_est, k)
            preds.append(batch_preds.detach())

            # Also compute KL divergence to track posterior collapse
            post_params = self.encoder(x)
            locs_pz_s, scales_pz_s = self.cond_prior(s)
            prior_params = (
                torch.cat([locs_pz_s, self.zeros.expand(bs, -1)], dim=1),
                torch.cat([scales_pz_s, self.ones.expand(bs, -1)], dim=1),
            )
            # prior_params = (self.zeros.expand(bs, -1), self.ones.expand(bs, -1))
            kl += compute_kl(*post_params, *prior_params).mean()

        # print(f"Mean KL divergence: {kl/len(data_loader)}")

        # gt already on cpu
        gt = torch.cat(gt, axis=0).numpy()
        preds = torch.cat(preds, axis=0).cpu().numpy()
        pearson_rs = []
        spearman_rs = []
        rmses = []
        for i in range(num_targets):
            pearson_rs.append(pearsonr(preds[:, i], gt[:, i])[0])
            spearman_rs.append(spearmanr(preds[:, i], gt[:, i])[0])
            rmses.append(
                np.sqrt(np.mean(np.square(np.array(preds[:, i]) - np.array(gt[:, i]))))
            )

        return rmses, pearson_rs, spearman_rs, kl

    def generate_z_pred(self, data_loader, save_dir, suffix=None):
        """Generates latent representation and predictions for all samples in data loader"""
        zs_arr = np.zeros((len(data_loader.dataset), self.num_constraints))
        znots_arr = np.zeros(
            (len(data_loader.dataset), self.z_dim - self.num_constraints)
        )
        pred_loc_arr = np.zeros((len(data_loader.dataset), self.num_constraints))
        pred_scale_arr = np.zeros((len(data_loader.dataset), self.num_constraints))
        self.eval()
        for x, s, c, y, ind, st in data_loader:
            if self.use_cuda:
                x = x.cuda()
            zs, znots, pred_loc, pred_scale = self.encode_predict(x)

            ind = ind.numpy().astype(int).tolist()

            zs_arr[ind, :] = zs.cpu().numpy()
            znots_arr[ind, :] = znots.cpu().numpy()
            pred_loc_arr[ind, :] = pred_loc.cpu().numpy()
            pred_scale_arr[ind, :] = pred_scale.cpu().numpy()

        z_arr = np.concatenate([zs_arr, znots_arr], axis=1)
        pred_arr = np.concatenate([pred_loc_arr, pred_scale_arr], axis=1)

        z_cols = []
        for i in range(self.z_dim):
            # zc_cols.append(f"zc_{i}")
            if i < len(self.constraints):
                z_cols.append(f"z_{self.constraints[i]}")
            else:
                z_cols.append(f"z_{i}")

        pred_cols = []
        for i in range(len(self.constraints)):
            pred_cols.append(f"{self.constraints[i]}_loc")
        for i in range(len(self.constraints)):
            pred_cols.append(f"{self.constraints[i]}_scale")

        z_df = pd.DataFrame(z_arr, columns=z_cols)
        pred_df = pd.DataFrame(pred_arr, columns=pred_cols)

        results_df = pd.concat([z_df, pred_df], axis=1)

        results_df.to_csv(f"{save_dir}/z_pred_{suffix}.csv")

        return None


class TargetRegressorDet(nn.Module):
    """
    Deterministic MLP regressor over latent or input features.

    Supports linear or multi-layer configurations via `model_size` and applies
    optional LayerNorm and dropout in hidden layers.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        activation: Activation name (`relu`, `leakyrelu`, or None).
        model_size: Key into `cf_model_sizes` for layer widths.
        bn: Whether to apply LayerNorm in hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim=1000,
        output_dim=10,
        activation="relu",
        model_size="s",
        bn=True,
        dropout=0.2,
    ):
        super(TargetRegressorDet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_widths = cf_model_sizes[model_size]
        self.activation = activation
        self.bn = bn
        self.dropout = dropout

        if model_size == "linear":
            self._modules["loc"] = nn.Linear(self.input_dim, self.output_dim)

        else:
            # Build the layers for the model architecture (without nn.Sequential)
            for layer_num in range(len(self.layer_widths)):
                if layer_num == 0:
                    self._modules[f"f{layer_num}"] = nn.Linear(
                        input_dim, self.layer_widths[layer_num]
                    )
                else:
                    self._modules[f"f{layer_num}"] = nn.Linear(
                        self.layer_widths[layer_num - 1], self.layer_widths[layer_num]
                    )
                if self.bn:
                    self._modules[f"bn{layer_num}"] = nn.LayerNorm(
                        self.layer_widths[layer_num]
                    )
                if self.dropout > 0:
                    self._modules[f"d{layer_num}"] = nn.Dropout(p=self.dropout)
            # Final linear layer for prediction
            self._modules["loc"] = nn.Linear(self.layer_widths[-1], self.output_dim)

    def forward(self, x):
        # Build layer order for forward pass, including BN, dropout, activations
        for layer_num in range(len(self.layer_widths)):
            x = self._modules[f"f{layer_num}"](x)
            if self.bn:
                x = self._modules[f"bn{layer_num}"](x)
            if self.activation == "relu":
                x = F.relu(x)
            if self.activation == "leakyrelu":
                x = F.leaky_relu(x)
            elif self.activation is None:
                pass
            if self.dropout > 0:
                x = self._modules[f"d{layer_num}"](x)

        # return prediction
        return self._modules["loc"](x)


class PiCoSK(nn.Module):
    """
    Pretrained encoder + scikit-learn regressor/classifier.

    The encoder is frozen and used to generate representations which are then
    fed into a scikit-learn model. Optionally concatenates covariates and can
    skip latent reps (`norep=True`).

    Args:
        encoder: Trained encoder module.
        args: Namespace of hyperparameters for the sklearn model.
        model: Model type (`ElasticNet`, `SVR`, `RandomForestRegressor`,
            `LinearRegression`, `LogisticRegression`, `SVC`).
        use_cuda: Whether to use CUDA for the encoder.
        max_ap: Flag used by callers for metric handling.
        random_state: Random seed for sklearn models.
        feature_inds: Optional feature index subset (unused in this class).
        norep: If True, use covariates only (skip latent representation).

    Attributes:
        metric_type: `classification` or `regression` based on model.
        scaler: StandardScaler fit on representations.
    """

    def __init__(
        self,
        encoder,
        args,
        model="ElasticNet",
        use_cuda=True,
        max_ap=True,
        random_state=0,
        feature_inds=None,
        norep=False,
    ):
        super(PiCoSK, self).__init__()
        self.max_ap = max_ap
        self.args = args
        self.model = model
        self.random_state = random_state
        self.feature_inds = feature_inds
        self.norep = norep
        if use_cuda:
            map_loc = torch.device("cuda:0")
        else:
            map_loc = "cpu"
        self.encoder = encoder

        if self.model == "ElasticNet":
            self.regressor = ElasticNet(
                self.args.alpha,
                l1_ratio=self.args.l1_ratio,
                random_state=self.random_state,
                max_iter=10000,
            )
        elif self.model == "SVR":
            self.regressor = SVR(
                C=self.args.c,
                epsilon=self.args.eps,
                kernel=self.args.kernel,
                max_iter=-1,
            )
        elif self.model == "RandomForestRegressor":
            self.regressor = RandomForestRegressor(
                n_estimators=self.args.n_estimators,
                max_depth=self.args.max_depth,
                min_samples_leaf=self.args.min_samples_leaf,
                min_samples_split=self.args.min_samples_split,
                max_features=self.args.max_features,
                max_samples=self.args.max_samples,
                criterion="mse",
                random_state=self.random_state,
            )
        elif self.model == "LinearRegression":
            self.regressor = LinearRegression()
        elif self.model == "LogisticRegression":
            self.regressor = LogisticRegression(
                penalty="elasticnet",
                C=1 / self.args.alpha,
                l1_ratio=self.args.l1_ratio,
                random_state=self.random_state,
                class_weight="balanced",
                solver="saga",
                max_iter=10000,
            )
        elif self.model == "SVC":
            self.regressor = SVC(
                C=self.args.c,
                kernel=self.args.kernel,
                probability=True,
                max_iter=-1,
                class_weight="balanced",
                random_state=self.random_state,
            )

        # Assign metric type for later calculations
        if self.model in ["SVC", "LogisticRegression"]:
            self.metric_type = "classification"
        else:
            self.metric_type = "regression"

        self.use_cuda = use_cuda

        # Set to fix batchnorm and dropout layers
        self.encoder.eval()

        # freeze params for encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Push encoder to GPU
        if use_cuda:
            self.encoder.cuda()

    def fit_regressor(self, data_loader):
        gt = []
        zs = []
        cs = []
        # Generate representations for all samples
        for x, s, c, y, idx, st in data_loader:
            gt.append(y.numpy())
            if self.use_cuda:
                x = x.cuda(non_blocking=True)
            z_mean, _ = self.encoder(x)
            z_mean = z_mean.detach().cpu().numpy()
            zs.append(z_mean)
            cs.append(c.numpy())

        # Concatenate all representations and targets
        # if len(zs) > 1:
        zs = np.concatenate(zs, axis=0)
        gt = np.concatenate(gt, axis=0)
        cs = np.concatenate(cs, axis=0)
        # else:
        #   zs = np.array(zs[0])
        #  gt = np.array(gt)

        if len(gt.shape) == 1:
            gt = np.expand_dims(gt, axis=1)

        # This can occur if batch size is 1
        if len(zs.shape) == 1:
            zs = np.expand_dims(zs, axis=0)
        if len(cs.shape) == 1:
            cs = np.expand_dims(cs, axis=0)

        self.z_dim = zs.shape[1]
        self.num_targets = gt.shape[1]

        # Fit regressor using reps as input, with c concatenated if not NA
        print(np.isnan(cs).any())
        if not np.isnan(cs).any():
            # If norep, only use c
            if self.norep:
                zs = cs
            else:
                zs = np.concatenate([zs, cs], axis=1)
            self.c_dim = cs.shape[1]
        else:
            self.c_dim = 0

        # STANDARDISE CONTINUOUS FEATURES
        # self.scale_cols = []

        # Iterate through columns to identify np.float32 columns and skip boolean ones
        # for i in range(zs.shape[1]):
        #    if np.issubdtype(zs[:, i].dtype, np.bool_):
        #        continue  # Skip boolean columns
        #    if np.issubdtype(zs[:, i].dtype, np.floating) and zs[:, i].dtype != np.bool_:
        #        # If the column is np.float32 (or other floating types), add it for scaling
        #        self.scale_cols.append(i)

        # Apply StandardScaler only to the np.float32 columns
        self.scaler = StandardScaler()

        # Convert the continuous columns to np.float32 if they aren't already (for consistency)
        # for i in self.scale_cols:
        #    zs[:, i] = zs[:, i].astype(np.float32)

        # Apply StandardScaler to the continuous columns
        # self.scaler.fit(zs[:, self.scale_cols])
        # zs[:, self.scale_cols] = self.scaler.transform(zs[:, self.scale_cols])
        self.scaler.fit(zs)
        zs = self.scaler.transform(zs)

        self.regressor.fit(zs, gt)

    def calculate_metrics(
        self, data_loader, num_targets, return_roc=False, *args, **kwargs
    ):
        preds = []
        gt = []
        for x, s, c, y, idx, st in data_loader:
            # Generate predictions for each batch in dataloader
            if len(y.shape) == 1:
                y = y.unsqueeze(dim=1)
            # Append y before cuda
            gt.append(y)
            if self.use_cuda:
                x = x.cuda(non_blocking=True)
            z, _ = self.encoder(x)
            if self.use_cuda:
                z = z.detach().cpu().numpy()
            else:
                z = z.detach().numpy()

            # If c not nan, concat with z
            if not np.isnan(c.numpy()).any():
                # If norep, only use c
                if self.norep:
                    z = c
                else:
                    z = np.concatenate([z, c], axis=1)

            # Standardise continuous columns
            # z[:, self.scale_cols] = self.scaler.transform(z[:, self.scale_cols])
            z = self.scaler.transform(z)

            if self.metric_type == "classification":
                batch_preds = self.regressor.predict_proba(z)[:, 1]
            else:
                batch_preds = self.regressor.predict(z)
            # Save the predictions
            preds.append(batch_preds)
        # Get array of all ground truth and predictions concatenated
        gt = np.concatenate(gt, axis=0)
        preds = np.concatenate(preds, axis=0)
        # Fix dimensions if there is only one target
        if len(preds.shape) == 1:
            preds = np.expand_dims(preds, axis=1)
        # Calculate metrics
        if self.metric_type == "regression":
            pearson_rs = []
            spearman_rs = []
            rmses = []
            for i in range(num_targets):
                curr_pred = preds[:, i]
                curr_gt = gt[:, i]
                curr_pred = np.squeeze(curr_pred)
                curr_gt = np.squeeze(curr_gt)

                nas = np.logical_or(np.isnan(curr_pred), np.isnan(curr_gt))

                pearson_rs.append(pearsonr(curr_pred[~nas], curr_gt[~nas])[0])
                spearman_rs.append(spearmanr(curr_pred[~nas], curr_gt[~nas])[0])
                rmses.append(
                    np.sqrt(np.mean(np.square(curr_pred[~nas] - curr_gt[~nas])))
                )
            return (rmses, pearson_rs, spearman_rs)
        elif self.metric_type == "classification":
            bces = []
            f1s = []
            auprs = []
            aurocs = []
            # Expect num_targets to be 1
            for i in range(num_targets):
                curr_pred = preds[:, i]
                curr_gt = gt[:, i]
                curr_pred = np.squeeze(curr_pred)
                curr_gt = np.squeeze(curr_gt)

                nas = np.logical_or(np.isnan(curr_pred), np.isnan(curr_gt))

                bces.append(log_loss(curr_gt[~nas], curr_pred[~nas], eps=1e-8))

                fpr, tpr, threshold = roc_curve(
                    curr_gt[~nas], curr_pred[~nas], pos_label=1
                )

                aurocs.append(auc(fpr, tpr))
                precision, recall, thresholds = precision_recall_curve(
                    curr_gt[~nas], curr_pred[~nas], pos_label=1
                )
                auprs.append(auc(recall, precision))
                f1s.append(f1_score(curr_gt[~nas], (curr_pred[~nas] > 0.5)))

            if return_roc:
                return (bces, f1s, auprs, aurocs, fpr, tpr, threshold)
            else:
                return (bces, f1s, auprs, aurocs)

    def generate_predictions(self, data_loader, save_dir, suffix="val"):
        """Generates latent representation and predictions for all samples in data loader"""
        # z_arr = np.zeros((len(data_loader), self.z_dim))
        # pred_arr = np.zeros((len(data_loader), self.num_targets))
        # ind_arr = np.zeros(len(data_loader), 1)
        zs = []
        preds = []
        inds = []
        ys = []
        self.eval()
        self.encoder.eval()
        # counter for filling results array
        # i = 0
        for x, s, c, y, idx, st in data_loader:
            if self.use_cuda:
                x = x.cuda()
            z, _ = self.encoder(x)
            if self.use_cuda:
                z = z.detach().cpu().numpy()
            else:
                z = z.detach().numpy()

            # Concatenate z with c if c is not nan
            if not np.isnan(c.numpy()).any():
                # If norep, only use c
                if self.norep:
                    z = c
                else:
                    z = np.concatenate([z, c], axis=1)

            # z[:, self.scale_cols] = self.scaler.transform(z[:, self.scale_cols])
            z = self.scaler.transform(z)

            if self.metric_type == "classification":
                pred = self.regressor.predict_proba(z)[:, 1]
            else:
                pred = self.regressor.predict(z)

            zs.append(z)
            preds.append(pred)
            ys.append(y.numpy())
            inds.append(idx.numpy())

            # ind_arr[i, :] = ind.numpy()
            # z_arr[i, :] = z_params[0].cpu().numpy()
            # pred_arr[i, :] = pred.cpu().numpy()
            # Update counter, accounts for variable batch sizes and shuffled inds when input dataset is split version
            # of another
            # i = i + 1

        z_arr = np.concatenate(zs, axis=0)
        pred_arr = np.concatenate(preds, axis=0)
        ind_arr = np.concatenate(inds, axis=0)
        y_arr = np.concatenate(ys, axis=0)

        z_cols = []
        if not self.norep:
            for i in range(self.z_dim):
                # zc_cols.append(f"zc_{i}")
                z_cols.append(f"z_{i}")
        for i in range(self.c_dim):
            z_cols.append(f"c_{i}")

        pred_cols = []
        for i in range(self.num_targets):
            pred_cols.append(f"pred_{i}")

        ind_df = pd.DataFrame(ind_arr, columns=["ind"])
        z_df = pd.DataFrame(z_arr, columns=z_cols)
        pred_df = pd.DataFrame(pred_arr, columns=pred_cols)
        y_df = pd.DataFrame(y_arr, columns=["y"])

        results_df = pd.concat([ind_df, z_df, pred_df, y_df], axis=1)

        results_df.to_csv(f"{save_dir}/z_pred_{suffix}.csv")

    def forward(self, x, c):
        z, _ = self.encoder(x)
        if self.use_cuda:
            z = z.detach().cpu().numpy()
        else:
            z = z.detach().numpy()

        # Concatenate z with c if c is not nan
        if not np.isnan(c.numpy()).any():
            # If norep, only use c
            if self.norep:
                z = c
            else:
                z = np.concatenate([z, c], axis=1)

        # z[:, self.scale_cols] = self.scaler.transform(z[:, self.scale_cols])
        z = self.scaler.transform(z)

        pred = self.regressor.predict(z)

        return pred

    def save_models(self, seed, path="./data"):
        # SAVE ENCODER
        torch.save(self.encoder, os.path.join(path, "encoder.pt"))
        # SAVE COEFFS
        if self.model == "ElasticNet":
            reg_coeffs = self.regressor.coef_.astype(float).tolist()
            reg_intercept = self.regressor.intercept_.astype(float).tolist()
            reg_dict = {"coeffs": reg_coeffs, "intercept": reg_intercept}
            with open(os.path.join(path, f"regressor_s{seed}.txt"), "w") as f:
                json.dump(reg_dict, f, indent=2)
        elif self.model == "SVR":
            reg_coeffs = self.regressor.coef_.astype(float).tolist()
            reg_intercept = self.regressor.intercept_.astype(float).tolist()
            reg_dict = {"coeffs": reg_coeffs, "intercept": reg_intercept}
            with open(os.path.join(path, f"regressor_s{seed}.txt"), "w") as f:
                json.dump(reg_dict, f, indent=2)
        elif self.model == "LogisticRegression":
            reg_coeffs = self.regressor.coef_.astype(float).tolist()
            reg_intercept = self.regressor.intercept_.astype(float).tolist()
            reg_dict = {"coeffs": reg_coeffs, "intercept": reg_intercept}
            with open(os.path.join(path, f"regressor_s{seed}.txt"), "w") as f:
                json.dump(reg_dict, f, indent=2)
        elif self.model == "SVC":
            reg_coeffs = self.regressor.coef_.astype(float).tolist()
            reg_intercept = self.regressor.intercept_.astype(float).tolist()
            reg_dict = {"coeffs": reg_coeffs, "intercept": reg_intercept}
            with open(os.path.join(path, f"regressor_s{seed}.txt"), "w") as f:
                json.dump(reg_dict, f, indent=2)


class BaselineNN(nn.Module):
    """
    Neural-network baseline regressor without a VAE.

    Wraps a deterministic MLP regressor and provides MSE/RMSE evaluation and
    prediction export utilities.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        dropout: Dropout probability for the regressor.
        n_layers: Number of hidden layers.
        layer_width: Hidden layer width.
        use_cuda: Whether to move model to CUDA.
    """

    def __init__(
        self, input_dim, output_dim, dropout, n_layers, layer_width, use_cuda=True
    ):
        super(BaselineNN, self).__init__()
        if use_cuda:
            map_loc = torch.device("cuda:0")
        else:
            map_loc = "cpu"
        self.dropout = dropout
        self.regressor = TargetRegressorDet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            layer_width=layer_width,
            dropout=self.dropout,
        )
        self.num_targets = output_dim
        self.use_cuda = use_cuda

        if use_cuda:
            self.cuda()

    def regressor_loss(self, x, y, k=100):
        """
        Computes the MSE loss. Sample k times from z.
        Alternative: Use MAP representation or distribution embedding
        """
        y_pred = self.regressor(x)
        y_pred = y_pred.view(-1, self.num_targets)
        y = y.view(-1, self.num_targets)
        mse = (y - y_pred).square().float().mean(dim=0)

        return mse

    def regressor_rmse(self, x, y=None, k=1):
        y_pred = self.regressor(x)
        y_pred = y_pred.view(-1, self.num_targets)
        y = y.view(-1, self.num_targets)
        rmse = (y - y_pred).square().float().mean(dim=0).sqrt()

        return rmse, y_pred

    def calculate_metrics(self, data_loader, num_targets, *args, **kwargs):
        preds = []
        gt = []
        for x, s, c, y, idx, st in data_loader:
            # Generate predictions for each batch in dataloader
            # Append y before cuda
            gt.append(y)
            if self.use_cuda:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if len(y.shape) == 1:
                unsqueeze = True
                y = y.unsqueeze(dim=1)
            else:
                unsqueeze = False
            _, batch_preds = self.regressor_rmse(x, y)
            # Save the predictions
            preds.append(batch_preds.detach())
        # Get array of all ground truth and predictions concatenated
        gt = torch.cat(gt, axis=0).numpy()
        preds = torch.cat(preds, axis=0).cpu().numpy()
        # Fix dimensions if there is only one target
        if unsqueeze:
            gt = np.expand_dims(gt, axis=1)
            # preds = np.expand_dims(preds, axis=1)
        # Calculate metrics
        pearson_rs = []
        spearman_rs = []
        rmses = []
        for i in range(num_targets):
            curr_pred = preds[:, i]
            curr_gt = gt[:, i]
            curr_pred = np.squeeze(curr_pred)
            curr_gt = np.squeeze(curr_gt)

            nas = np.logical_or(np.isnan(curr_pred), np.isnan(curr_gt))

            pearson_rs.append(pearsonr(curr_pred[~nas], curr_gt[~nas])[0])
            spearman_rs.append(spearmanr(curr_pred[~nas], curr_gt[~nas])[0])
            rmses.append(np.sqrt(np.mean(np.square(curr_pred[~nas] - curr_gt[~nas]))))
        return rmses, pearson_rs, spearman_rs

    def generate_predictions(self, data_loader, save_dir, suffix="val"):
        """Generates latent representation and predictions for all samples in data loader"""
        # z_arr = np.zeros((len(data_loader), self.z_dim))
        # pred_arr = np.zeros((len(data_loader), self.num_targets))
        # ind_arr = np.zeros(len(data_loader), 1)
        preds = []
        idxs = []
        ys = []
        self.eval()
        self.regressor.eval()
        # counter for filling results array
        # i = 0
        for x, s, c, y, idx, st in data_loader:
            if self.use_cuda:
                x, y, idx = x.cuda(), y, idx
            pred = self.regressor(x)

            preds.append(pred.cpu().numpy())
            idxs.append(idx.numpy())
            ys.append(y.numpy())

            # ind_arr[i, :] = ind.numpy()
            # z_arr[i, :] = z_params[0].cpu().numpy()
            # pred_arr[i, :] = pred.cpu().numpy()
            # Update counter, accounts for variable batch sizes and shuffled inds when input dataset is split version
            # of another
            # i = i + 1

        pred_arr = np.concatenate(preds, axis=0)
        idx_arr = np.concatenate(idxs, axis=0)
        y_arr = np.concatenate(ys, axis=0)

        pred_cols = []
        for i in range(self.num_targets):
            pred_cols.append(f"pred_{i}")

        idx_df = pd.DataFrame(idx_arr, columns=["ind"])
        pred_df = pd.DataFrame(pred_arr, columns=pred_cols)
        y_df = pd.DataFrame(y_arr, columns=["y"])

        results_df = pd.concat([idx_df, pred_df, y_df], axis=1)

        results_df.to_csv(f"{save_dir}/pred_{suffix}.csv")

    def forward(self, x):
        pred = self.regressor(x)
        return pred

    def save_models(self, path="./data", seed=4536):
        torch.save(self.regressor, os.path.join(path, f"regressor_{seed}.pt"))
        torch.save(self, os.path.join(path, f"best_model_{seed}.pt"))


class BaselineSK(nn.Module):
    """
    scikit-learn baseline regressor with optional PCA.

    Fits a scaler (and optional PCA) on raw inputs, then trains an sklearn
    regressor. Provides metric computation and prediction export utilities.

    Args:
        args: Namespace with model hyperparameters.
        reg_model: Model type (`ElasticNet`, `Lasso`, `SVR`, `RandomForestRegressor`).
        fit_pca: Whether to apply PCA before fitting.
        pca_dim: PCA dimensionality if `fit_pca=True`.

    Attributes:
        scaler: StandardScaler fit on inputs.
        pca: PCA transformer (only if `fit_pca=True`).
        metric_type: `classification` or `regression` based on model.
    """

    def __init__(self, args, reg_model="ElasticNet", fit_pca=False, pca_dim=64):
        super(BaselineSK, self).__init__()
        self.args = args
        self.reg_model = reg_model
        # Whether to run a PCA on the input data before fitting the regressor
        self.fit_pca = fit_pca
        self.pca_dim = pca_dim
        # Placeholder for PCA object
        self.pca = None

        if self.reg_model == "ElasticNet":
            self.regressor = ElasticNet(
                alpha=self.args.alpha,
                l1_ratio=self.args.l1_ratio,
                random_state=self.args.seed,
            )
        elif self.reg_model == "Lasso":
            self.regressor = Lasso(self.args.alpha)
        elif self.reg_model == "SVR":
            self.regressor = SVR(C=self.args.c, epsilon=self.args.eps, kernel="linear")
        elif self.reg_model == "RandomForestRegressor":
            self.regressor = RandomForestRegressor(
                n_estimators=self.args.n_estimators,
                max_depth=self.args.max_depth,
                min_samples_leaf=self.args.min_samples_leaf,
                min_samples_split=self.args.min_samples_split,
                max_features=self.args.max_features,
                max_samples=self.args.max_samples,
                criterion="mse",
                random_state=self.args.seed,
            )

        # Assign metric type for later calculations
        if self.reg_model in ["SVC", "LogisticRegression"]:
            self.metric_type = "classification"
        else:
            self.metric_type = "regression"

    def fit_regressor(self, data_loader):
        gt = []
        xs = []
        # Generate representations for all samples
        for x, s, c, y, ind, st in data_loader:
            gt.append(y)
            xs.append(x)

        # Concatenate all representations and targets
        xs = np.concatenate(xs, axis=0)
        gt = np.concatenate(gt, axis=0)

        if len(gt.shape) == 1:
            gt = np.expand_dims(gt, axis=1)

        # Standard scaling
        self.scaler = StandardScaler()

        self.scaler.fit(xs)

        xs = self.scaler.transform(xs)

        # Run PCA if specified
        if self.fit_pca:
            print("[INFO] Fitting PCA...")
            self.pca = PCA(n_components=self.pca_dim)
            xs = self.pca.fit_transform(xs)

        self.input_dim = xs.shape[1]
        self.num_targets = gt.shape[1]

        # Fit regressor using reps as input
        self.regressor.fit(xs, gt)

    def calculate_metrics(self, data_loader, num_targets, *args, **kwargs):
        preds = []
        gt = []
        for x, s, c, y, ind, st in data_loader:
            # Generate predictions for each batch in dataloader
            # Append y before cuda
            gt.append(y)
            if len(y.shape) == 1:
                y = y.unsqueeze(dim=1)

            x = self.scaler.transform(x)

            # Transform with PCA if necessary
            if self.fit_pca:
                x = self.pca.transform(x)

            batch_preds = self.regressor.predict(x)
            # Save the predictions
            preds.append(batch_preds)
        # Get array of all ground truth and predictions concatenated
        gt = np.concatenate(gt, axis=0)
        preds = np.concatenate(preds, axis=0)
        # Fix dimensions if there is only one target
        if len(gt.shape) == 1:
            gt = np.expand_dims(gt, axis=1)
        if len(preds.shape) == 1:
            preds = np.expand_dims(preds, axis=1)
        # Calculate metrics
        pearson_rs = []
        spearman_rs = []
        rmses = []
        for i in range(num_targets):
            curr_pred = preds[:, i]
            curr_gt = gt[:, i]
            curr_pred = np.squeeze(curr_pred)
            curr_gt = np.squeeze(curr_gt)

            nas = np.logical_or(np.isnan(curr_pred), np.isnan(curr_gt))

            pearson_rs.append(pearsonr(curr_pred[~nas], curr_gt[~nas])[0])
            spearman_rs.append(spearmanr(curr_pred[~nas], curr_gt[~nas])[0])
            rmses.append(np.sqrt(np.mean(np.square(curr_pred[~nas] - curr_gt[~nas]))))
        return rmses, pearson_rs, spearman_rs

    def generate_predictions(self, data_loader, save_dir, suffix="val"):
        """Generates latent representation and predictions for all samples in data loader"""
        # z_arr = np.zeros((len(data_loader), self.z_dim))
        # pred_arr = np.zeros((len(data_loader), self.num_targets))
        # ind_arr = np.zeros(len(data_loader), 1)
        preds = []
        inds = []
        gts = []
        self.eval()
        # counter for filling results array
        # i = 0
        for x, s, c, y, ind, st in data_loader:
            x = self.scaler.transform(x)
            # Transform with PCA if necessary
            if self.fit_pca:
                x = self.pca.transform(x)

            pred = self.regressor.predict(x)

            preds.append(pred)
            gts.append(y)
            inds.append(ind.numpy())

            # ind_arr[i, :] = ind.numpy()
            # z_arr[i, :] = z_params[0].cpu().numpy()
            # pred_arr[i, :] = pred.cpu().numpy()
            # Update counter, accounts for variable batch sizes and shuffled inds when input dataset is split version
            # of another
            # i = i + 1

        pred_arr = np.concatenate(preds, axis=0)
        gt_arr = np.concatenate(gts, axis=0)
        ind_arr = np.concatenate(inds, axis=0)

        pred_cols = []
        for i in range(self.num_targets):
            pred_cols.append(f"pred_{i}")

        ind_df = pd.DataFrame(ind_arr, columns=["ind"])
        gt_df = pd.DataFrame(gt_arr, columns=["y"])
        pred_df = pd.DataFrame(pred_arr, columns=pred_cols)

        results_df = pd.concat([ind_df, pred_df, gt_df], axis=1)

        results_df.to_csv(f"{save_dir}/z_pred_{suffix}.csv")

    def forward(self, x):
        x = self.scaler.transform(x)
        # Transform with PCA if necessary
        if self.fit_pca:
            x = self.pca.transform(x)

        pred = self.regressor.predict(x)

        return pred

    def save_models(self, path="./data", fold=0, seed=10):
        # SAVE COEFFS
        if self.reg_model == "ElasticNet":
            reg_coeffs = self.regressor.coef_.astype(float).tolist()
            reg_intercept = self.regressor.intercept_.astype(float).tolist()
            reg_dict = {"coeffs": reg_coeffs, "intercept": reg_intercept}
            with open(os.path.join(path, f"regressor_{seed}.txt"), "w") as f:
                json.dump(reg_dict, f, indent=2)
        elif self.reg_model == "SVR":
            reg_coeffs = self.regressor.coef_.astype(float).tolist()
            reg_intercept = self.regressor.intercept_.astype(float).tolist()
            reg_dict = {"coeffs": reg_coeffs, "intercept": reg_intercept}
            with open(os.path.join(path, f"regressor_{seed}.txt"), "w") as f:
                json.dump(reg_dict, f, indent=2)
        if self.fit_pca:
            with open(os.path.join(path, f"pca_components_{seed}.txt"), "w") as f:
                pca_dict = {
                    "components": self.pca.components_.astype(float).tolist(),
                    "mean": self.pca.mean_.astype(float).tolist(),
                    "explained_variance_ratio": self.pca.explained_variance_ratio_.astype(
                        float
                    ).tolist(),
                }
                json.dump(pca_dict, f, indent=2)
