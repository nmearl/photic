from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn.functional as F

from .batch import NPBatch
from .config import JointLossConfig
from .model import JointModelOutput


@dataclass(slots=True)
class JointLosses:
    total: torch.Tensor
    recon: torch.Tensor
    cls: torch.Tensor
    interesting: torch.Tensor
    morph: torch.Tensor
    kl: torch.Tensor


def _peak_weight_map(batch: NPBatch, cfg: JointLossConfig) -> torch.Tensor:
    weights = torch.ones_like(batch.target_y)
    if batch.metadata is None:
        return weights * batch.target_mask

    if cfg.peak_weight_snr_boost > 0.0 and "obs_snr" in batch.metadata:
        obs_snr = batch.metadata["obs_snr"].to(batch.target_y.device)
        weights = weights + cfg.peak_weight_snr_boost * (obs_snr >= cfg.peak_weight_snr_threshold).float()

    if cfg.peak_weight_flux_boost > 0.0:
        y = batch.target_y
        m = batch.target_mask > 0
        high_flux = torch.zeros_like(y)
        q = float(cfg.peak_weight_flux_quantile)
        q = min(max(q, 0.0), 1.0)
        for i in range(y.shape[0]):
            yi = y[i][m[i]]
            if yi.numel() == 0:
                continue
            thr = torch.quantile(yi, q)
            high_flux[i] = ((y[i] >= thr) & (y[i] > 0)).float()
        weights = weights + cfg.peak_weight_flux_boost * high_flux

    return weights * batch.target_mask


def gaussian_reconstruction_nll(out: JointModelOutput, batch: NPBatch, cfg: JointLossConfig | None = None) -> torch.Tensor:
    total_var = out.pred_var + batch.target_yerr.square()
    nll = 0.5 * (
        torch.log(total_var.clamp_min(1e-6))
        + math.log(2.0 * math.pi)
        + (batch.target_y - out.pred_mean).square() / total_var.clamp_min(1e-6)
    )
    weights = batch.target_mask if cfg is None else _peak_weight_map(batch, cfg)
    return (nll * weights).sum() / weights.sum().clamp_min(1.0)


def latent_kl_standard_normal(mu: torch.Tensor | None, logvar: torch.Tensor | None) -> torch.Tensor:
    if mu is None or logvar is None:
        return torch.tensor(0.0, device=mu.device if mu is not None else "cpu")
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
    return kl.sum(dim=-1).mean()


def binary_classification_loss(logits: torch.Tensor | None, labels: torch.Tensor | None, cfg: JointLossConfig) -> torch.Tensor:
    if logits is None or labels is None:
        return torch.tensor(0.0, device=logits.device if logits is not None else labels.device if labels is not None else "cpu")
    if cfg.focal_gamma is None:
        pos_weight = None
        if cfg.pos_weight is not None:
            pos_weight = torch.tensor([cfg.pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
    focal = (1.0 - p_t).pow(cfg.focal_gamma) * ce
    if cfg.pos_weight is not None:
        alpha = torch.where(labels > 0.5, torch.full_like(labels, cfg.pos_weight), torch.ones_like(labels))
        focal = alpha * focal
    return focal.mean()


def morphology_loss(pred: torch.Tensor | None, targets: torch.Tensor | None) -> torch.Tensor:
    if pred is None or targets is None:
        dev = pred.device if pred is not None else targets.device if targets is not None else 'cpu'
        return torch.tensor(0.0, device=dev)
    return F.smooth_l1_loss(pred, targets)


def interestingness_loss(logits: torch.Tensor | None, labels: torch.Tensor | None, cfg: JointLossConfig) -> torch.Tensor:
    if logits is None or labels is None:
        return torch.tensor(0.0, device=logits.device if logits is not None else labels.device if labels is not None else "cpu")
    pos_weight = None
    if cfg.interesting_pos_weight is not None:
        pos_weight = torch.tensor([cfg.interesting_pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def joint_loss(out: JointModelOutput, batch: NPBatch, cfg: JointLossConfig) -> JointLosses:
    recon = gaussian_reconstruction_nll(out, batch, cfg)
    cls = binary_classification_loss(out.class_logits, batch.labels, cfg)
    interesting = interestingness_loss(out.interesting_logits, batch.interesting_labels, cfg)
    morph = morphology_loss(out.morph_pred, batch.morph_targets)
    kl = latent_kl_standard_normal(out.latent_mu, out.latent_logvar).to(recon.device)
    total = cfg.lambda_recon * recon + cfg.lambda_cls * cls + cfg.lambda_interesting * interesting + cfg.lambda_morph * morph + cfg.beta_kl * kl
    return JointLosses(total=total, recon=recon, cls=cls, interesting=interesting, morph=morph, kl=kl)
