import torch
import torch.nn.functional as F
import torch.distributions as dist
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Optional


# Loss helper functions
def compute_kl(
    locs_q: torch.Tensor,
    scale_q: torch.Tensor,
    locs_p: Optional[torch.Tensor] = None,
    scale_p: Optional[torch.Tensor] = None,
) -> float:
    """
    Computes the KL(q||p)
    """
    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    dist_q = dist.Normal(locs_q, scale_q)
    dist_p = dist.Normal(locs_p, scale_p)
    return dist.kl.kl_divergence(dist_q, dist_p).sum(dim=-1)


def exp_log_likelihood(recon, xs, scale=0.0):
    # Scale is default to 0 but can be learned as a single value
    return (
        dist.Normal(recon, F.softplus(scale) * torch.ones_like(recon))
        .log_prob(xs)
        .sum(dim=1)
    )


def mixed_loss(x, y, x_recon, y_pred, rscale=1, pscale=1, cf_loss="mse"):
    assert x.shape == x_recon.shape
    assert y.shape == y_pred.shape
    recon_loss = ((x - x_recon) ** 2).mean(-1).sum()
    if cf_loss == "bce":
        pred_loss = F.binary_cross_entropy(y_pred, y).sum()
    elif cf_loss == "mse":
        pred_loss = ((y - y_pred) ** 2).mean(-1).sum()

    return rscale * recon_loss + pscale * pred_loss


def mse_1d(y, y_pred):
    assert y.shape == y_pred.shape

    return ((y - y_pred) ** 2).mean()


def mse(y, y_pred):
    assert y.shape == y_pred.shape

    return ((y - y_pred) ** 2).mean()


def diff_sum(y, y_pred):
    y_pred_round = y_pred.round()
    return ((y - y_pred_round).pow(2)).sum(0)


def batch_f1_update(y, y_pred):
    pred_classes = y_pred.round()
    target_classes = y
    target_true = torch.sum(y).float()
    pred_true = torch.sum(pred_classes).float()
    correct_true = torch.sum((pred_classes == target_classes) * pred_classes).float()

    return target_true, pred_true, correct_true


def r_p_f1(target_true, pred_true, correct_true):
    recall = correct_true / target_true
    precision = correct_true / pred_true
    f1_score = 2 * precision * recall / (precision + recall)

    return recall, precision, f1_score


def auc(y, y_pred):
    return roc_auc_score(y, y_pred)


def aupr(y, y_pred):
    return average_precision_score(y, y_pred)
