"""Partially labelled competing-risk DeepHit for delivery time and cord pH.

Observed pH assigns every delivery to one mutually exclusive pH category.  For
missing pH, delivery time remains observed and contributes a marginal delivery
likelihood plus an all-cause delivery ranking loss.  The two ranking losses are
computed on disjoint pH-observed and pH-missing subsets, respectively, so no
sample is assigned both a cause-specific and an all-cause ranking target.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchtuples as tt
from pycox.models import DeepHit
from pycox.models.data import pair_rank_mat
from pycox.models.loss import rank_loss_deephit_single
from torch import nn

class CompetingDeepHitNet(nn.Module):
    """A shared trace encoder with a joint pH-category/delivery-bin head."""

    def __init__(self, encoder, n_causes, n_time_bins, elapsed_time_feature=False):
        """Build a shared encoder with a joint competing-cause/time head.

        ``elapsed_time_feature`` appends one normalized chunk-end time scalar
        after trace encoding and therefore leaves the physiological channels
        and encoder architecture unchanged.
        """
        super().__init__()
        if not hasattr(encoder, "forward_features"):
            raise TypeError("encoder must expose forward_features(x)")
        feature_dim = getattr(encoder, "feature_dim", None)
        if feature_dim is None and hasattr(encoder, "fc"):
            feature_dim = encoder.fc.in_features
        if feature_dim is None:
            raise TypeError("encoder must expose feature_dim or an fc layer")
        self.encoder = encoder
        self.n_causes = int(n_causes)
        self.n_time_bins = int(n_time_bins)
        self.elapsed_time_feature = bool(elapsed_time_feature)
        self.head = nn.Linear(
            feature_dim + int(self.elapsed_time_feature),
            self.n_causes * self.n_time_bins,
        )

    def forward(self, x, elapsed_time=None):
        """Return joint cause/time logits shaped ``[B, causes, bins]``."""
        features = self.encoder.forward_features(x)
        if self.elapsed_time_feature:
            if elapsed_time is None:
                raise ValueError(
                    "elapsed_time is required when elapsed_time_feature is enabled"
                )
            elapsed_time = elapsed_time.reshape(features.shape[0], 1).to(features.dtype)
            features = torch.cat([features, elapsed_time], dim=1)
        return self.head(features).reshape(-1, self.n_causes, self.n_time_bins)


def _validate_hidden_dims(hidden_dims):
    """Validate a strictly shrinking sequence of MLP hidden widths."""
    try:
        hidden_dims = tuple(int(width) for width in hidden_dims)
    except (TypeError, ValueError) as exc:
        raise ValueError("hidden_dims must be a sequence of positive integers") from exc
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("hidden_dims must contain at least one positive width")
    if any(next_width >= width for width, next_width in zip(hidden_dims, hidden_dims[1:])):
        raise ValueError("hidden_dims must strictly decrease at every layer")
    return hidden_dims


class CauseSpecificCompetingDeepHitNet(nn.Module):
    """Shared encoder with one independent MLP duration head per cause.

    This is the competing-risk architecture used by the trainer.
    The encoder is shared, but each cause has its own MLP parameters and emits
    one logit per delivery-time bin.  The output remains compatible with
    :class:`PartialLabelDeepHit`: ``[batch, causes, time_bins]``.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder exposing ``forward_features(x)`` and ``feature_dim``.
    n_causes, n_time_bins : int
        Number of competing causes and discrete delivery-time bins.
    hidden_dims : sequence of int, default=(128, 64)
        Widths of the independent hidden layers.  They must strictly decrease
        so the default is a compact two-hidden-layer narrowing pattern.
    batch_norm : bool, default=True
        Add batch normalization to each hidden layer.
    dropout : float, default=0.2
        Dropout probability in each hidden layer.
    elapsed_time_feature : bool, default=False
        Append the normalized elapsed-time scalar before every cause head.
    """

    def __init__(
        self,
        encoder,
        n_causes,
        n_time_bins,
        hidden_dims=(128, 64),
        batch_norm=True,
        dropout=0.2,
        elapsed_time_feature=False,
    ):
        super().__init__()
        if not hasattr(encoder, "forward_features"):
            raise TypeError("encoder must expose forward_features(x)")
        feature_dim = getattr(encoder, "feature_dim", None)
        if feature_dim is None and hasattr(encoder, "fc"):
            feature_dim = encoder.fc.in_features
        if feature_dim is None:
            raise TypeError("encoder must expose feature_dim or an fc layer")
        if int(n_causes) < 1 or int(n_time_bins) < 1:
            raise ValueError("n_causes and n_time_bins must be positive")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.encoder = encoder
        self.feature_dim = int(feature_dim)
        self.n_causes = int(n_causes)
        self.n_time_bins = int(n_time_bins)
        self.hidden_dims = _validate_hidden_dims(hidden_dims)
        self.elapsed_time_feature = bool(elapsed_time_feature)
        head_input_dim = self.feature_dim + int(self.elapsed_time_feature)
        self.heads = nn.ModuleDict({
            str(cause): tt.practical.MLPVanilla(
                in_features=head_input_dim,
                num_nodes=list(self.hidden_dims),
                out_features=self.n_time_bins,
                batch_norm=bool(batch_norm),
                dropout=float(dropout),
            )
            for cause in range(self.n_causes)
        })

    def forward(self, x, elapsed_time=None):
        """Return independent cause-head logits shaped ``[B, causes, bins]``."""
        features = self.encoder.forward_features(x)
        if self.elapsed_time_feature:
            if elapsed_time is None:
                raise ValueError(
                    "elapsed_time is required when elapsed_time_feature is enabled"
                )
            elapsed_time = elapsed_time.reshape(features.shape[0], 1).to(features.dtype)
            features = torch.cat([features, elapsed_time], dim=1)
        return torch.stack(
            [self.heads[str(cause)](features) for cause in range(self.n_causes)],
            dim=1,
        )


class PartialLabelDeepHitDataset(tt.data.DatasetTuple):
    """Dataset retaining the pH-observed flag for the custom partial loss."""


def _joint_log_pmf(logits):
    """PyCox-compatible log PMF over cause/bin cells, excluding survival tail."""
    batch_size = logits.shape[0]
    flattened = logits.reshape(batch_size, -1)
    # PyCox pads PMF logits with a zero-valued survival-after-grid cell.
    padded = F.pad(flattened, (0, 1))
    return F.log_softmax(padded, dim=1)[:, :-1].reshape_as(logits)


def marginal_delivery_nll(logits, duration_idx):
    """Compute delivery-bin NLL while marginalizing an unknown pH cause.

    Parameters
    ----------
    logits : torch.Tensor
        Joint cause/time logits shaped ``[B, R, T]``.
    duration_idx : torch.Tensor
        Observed delivery-bin index per row.

    Returns
    -------
    torch.Tensor
        Mean negative log likelihood for delivery time alone.
    """
    log_pmf = _joint_log_pmf(logits)
    index = torch.arange(logits.shape[0], device=logits.device)
    selected = log_pmf[index, :, duration_idx.long()]
    return -torch.logsumexp(selected, dim=1).mean()


def marginal_delivery_rank_loss(logits, duration_idx, sigma):
    """Rank all-cause delivery time within one partially labelled subset.

    Summing the joint cause/time PMF over causes is exactly equivalent to a
    single-risk DeepHit model with per-time logits ``logsumexp(cause_logits)``:
    both retain the same zero-logit survival-tail cell.  Restricting callers to
    one label-observation subset prevents rows from serving as comparators in
    both the cause-specific and all-cause ranking objectives.
    """
    if len(duration_idx) < 2:
        return logits.sum() * 0.0
    rank_mat = torch.as_tensor(
        pair_rank_mat(
            duration_idx.detach().cpu().numpy(),
            np.ones(len(duration_idx), dtype=np.int64),
        ),
        device=logits.device,
    )
    marginal_logits = torch.logsumexp(logits, dim=1)
    return rank_loss_deephit_single(
        marginal_logits,
        duration_idx,
        torch.ones_like(duration_idx),
        rank_mat,
        sigma,
    )


class PartialLabelDeepHit(DeepHit):
    """Competing DeepHit with a disjoint marginal rank loss for missing pH."""

    def __init__(self, *args, missing_delivery_loss_weight=1.0, **kwargs):
        """Initialize partially labeled DeepHit and its delivery-only loss weight."""
        self.missing_delivery_loss_weight = float(missing_delivery_loss_weight)
        super().__init__(*args, **kwargs)

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        """Create a loader retaining pH-observed flags for the custom loss."""
        return super(DeepHit, self).make_dataloader(
            data,
            batch_size,
            shuffle,
            num_workers,
            make_dataset=PartialLabelDeepHitDataset,
        )

    def compute_metrics(self, data, metrics=None):
        """Use cause ranks for observed pH and all-cause ranks for missing pH."""
        input_, target = data
        input_, target = self._to_device(input_), self._to_device(target)
        logits = self.net(*input_)
        duration_idx, event_type, ph_observed = target
        observed = ph_observed.bool().reshape(-1)
        missing = ~observed
        total_count = len(observed)

        if observed.any():
            observed_duration = duration_idx[observed]
            observed_events = event_type[observed]
            observed_logits = logits[observed]
            rank_mat = torch.as_tensor(
                pair_rank_mat(
                    observed_duration.detach().cpu().numpy(),
                    observed_events.detach().cpu().numpy(),
                ),
                device=logits.device,
            )
            observed_loss = self.loss(
                observed_logits,
                observed_duration,
                observed_events,
                rank_mat,
            )
        else:
            observed_loss = logits.sum() * 0.0

        if missing.any():
            missing_nll = marginal_delivery_nll(logits[missing], duration_idx[missing])
            missing_rank = marginal_delivery_rank_loss(
                logits[missing], duration_idx[missing], self.loss.sigma
            )
            # Preserve the pre-existing marginal NLL scale.  The ranking term
            # uses DeepHit's configured rank coefficient, matching the
            # observed-pH objective without making a row enter both rank sets.
            missing_loss = missing_nll + (1.0 - self.loss.alpha) * missing_rank
        else:
            missing_loss = logits.sum() * 0.0
            missing_nll = logits.sum() * 0.0
            missing_rank = logits.sum() * 0.0

        total = (
            observed.sum() / total_count * observed_loss
            + self.missing_delivery_loss_weight
            * missing.sum()
            / total_count
            * missing_loss
        )
        return {
            "loss": total,
            "observed_deephit_loss": observed_loss.detach(),
            "missing_delivery_nll": missing_nll.detach(),
            "missing_delivery_rank_loss": missing_rank.detach(),
        }


def competing_pmf(logits):
    """Convert competing-risk logits into a joint cause/time PMF.

    Parameters
    ----------
    logits : numpy.ndarray | torch.Tensor
        Joint logits shaped ``[B, R, T]``.

    Returns
    -------
    numpy.ndarray | torch.Tensor
        Cause/time PMF shaped ``[B, R, T]`` without the survival-tail cell.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
        return _joint_log_pmf(logits).exp().numpy()
    return _joint_log_pmf(logits).exp()


def delivery_pmf_from_competing(logits):
    """Marginalize a cause/time PMF into a delivery-time PMF.

    Parameters
    ----------
    logits : numpy.ndarray | torch.Tensor
        Joint cause/time logits shaped ``[B, R, T]``.

    Returns
    -------
    numpy.ndarray | torch.Tensor
        Delivery PMF shaped ``[B, T]``.
    """
    return (
        competing_pmf(logits).sum(axis=1)
        if isinstance(logits, np.ndarray)
        else competing_pmf(logits).sum(1)
    )


def threshold_joint_risks(logits, threshold_index):
    """Sum causes corresponding to one cumulative pH threshold.

    Parameters
    ----------
    logits : numpy.ndarray | torch.Tensor
        Joint cause/time logits shaped ``[B, R, T]``.
    threshold_index : int
        Zero-based cumulative threshold index.

    Returns
    -------
    numpy.ndarray | torch.Tensor
        Joint threshold/delivery-bin risks shaped ``[B, T]``.
    """
    pmf = competing_pmf(logits)
    return (
        pmf[:, : threshold_index + 1, :].sum(axis=1)
        if isinstance(pmf, np.ndarray)
        else pmf[:, : threshold_index + 1, :].sum(1)
    )


def threshold_cumulative_risks(logits, threshold_index):
    """Accumulate one threshold's joint risks over delivery-time bins.

    Parameters
    ----------
    logits : numpy.ndarray | torch.Tensor
        Joint cause/time logits shaped ``[B, R, T]``.
    threshold_index : int
        Zero-based cumulative threshold index.

    Returns
    -------
    numpy.ndarray | torch.Tensor
        ``P(delivery by t, pH below threshold | X)`` shaped ``[B, T]``.
    """
    risks = threshold_joint_risks(logits, threshold_index)
    return risks.cumsum(axis=1) if isinstance(risks, np.ndarray) else risks.cumsum(1)
