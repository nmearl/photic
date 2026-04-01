from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch import NPBatch
from .config import ConvGNPJointConfig
from .modules import ConvBackbone1D, FourierTimeEmbedding, GaussianSetConv1D, GlobalLatentEncoder, MLP


@dataclass(slots=True)
class JointModelOutput:
    pred_mean: torch.Tensor
    pred_var: torch.Tensor
    class_logits: torch.Tensor | None
    morph_pred: torch.Tensor | None
    grid_features: torch.Tensor
    latent_mu: torch.Tensor | None
    latent_logvar: torch.Tensor | None
    latent_sample: torch.Tensor | None


class ConvGNPJointModel(nn.Module):
    def __init__(self, cfg: ConvGNPJointConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embed = FourierTimeEmbedding(cfg.time_fourier_dim)
        self.band_embed = nn.Embedding(cfg.num_bands, cfg.band_emb_dim)

        point_in = 1 + 1 + cfg.band_emb_dim + 2 * cfg.time_fourier_dim
        if cfg.use_redshift:
            point_in += 1

        self.point_encoder = MLP(point_in, cfg.point_feat_dim, cfg.point_feat_dim, depth=3)
        self.setconv = GaussianSetConv1D(cfg.setconv_sigma)

        self.grid_proj = nn.Conv1d(cfg.point_feat_dim + 1, cfg.grid_feat_dim, kernel_size=1)
        self.grid_backbone = ConvBackbone1D(
            cfg.grid_feat_dim,
            layers=cfg.conv_layers,
            kernel_size=cfg.conv_kernel_size,
            dropout=cfg.conv_dropout,
        )

        self.latent_encoder = GlobalLatentEncoder(
            cfg.point_feat_dim,
            cfg.latent_hidden_dim,
            cfg.latent_dim,
        )

        decoder_in = cfg.grid_feat_dim + cfg.band_emb_dim + 2 * cfg.time_fourier_dim
        if cfg.use_latent:
            decoder_in += cfg.latent_dim
        if cfg.use_redshift:
            decoder_in += 1

        self.decoder = MLP(decoder_in, cfg.decoder_hidden_dim, 2, depth=3)

        head_in = 2 * cfg.grid_feat_dim
        if cfg.use_latent:
            head_in += cfg.latent_dim + cfg.latent_dim
        if cfg.use_redshift:
            head_in += 1
        self.classifier = MLP(head_in, cfg.classifier_hidden_dim, 1, depth=3, dropout=0.1)
        self.morphology_head = None
        if cfg.use_morphology_head and cfg.num_morph_targets > 0:
            self.morphology_head = MLP(head_in, cfg.morphology_hidden_dim, cfg.num_morph_targets, depth=3, dropout=0.1)

        grid = torch.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_size)
        self.register_buffer("grid_x", grid)

    def _encode_points(self, x: torch.Tensor, y: torch.Tensor, yerr: torch.Tensor, band: torch.Tensor, mask: torch.Tensor, redshift: torch.Tensor | None) -> torch.Tensor:
        tfeat = self.time_embed(x)
        bfeat = self.band_embed(band)
        feat_parts = [y.unsqueeze(-1), torch.log(yerr.clamp_min(1e-6)).unsqueeze(-1), bfeat, tfeat]
        if self.cfg.use_redshift:
            if redshift is None:
                red = torch.zeros_like(x)
            else:
                red = redshift.unsqueeze(-1).expand_as(x)
            feat_parts.append(red.unsqueeze(-1))
        feat = torch.cat(feat_parts, dim=-1)
        feat = self.point_encoder(feat)
        return feat * mask.unsqueeze(-1)

    def _sample_latent(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _interpolate_grid(self, grid_features: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        bsz, channels, gsz = grid_features.shape
        x_norm = xq.clamp(self.cfg.grid_min, self.cfg.grid_max)
        pos = (x_norm - self.cfg.grid_min) / (self.cfg.grid_max - self.cfg.grid_min) * (gsz - 1)
        left = pos.floor().long().clamp(0, gsz - 1)
        right = (left + 1).clamp(0, gsz - 1)
        alpha = (pos - left.float()).unsqueeze(-1)

        gf = grid_features.transpose(1, 2)
        left_feat = torch.gather(gf, 1, left.unsqueeze(-1).expand(-1, -1, channels))
        right_feat = torch.gather(gf, 1, right.unsqueeze(-1).expand(-1, -1, channels))
        return left_feat * (1.0 - alpha) + right_feat * alpha

    def encode_shared(self, batch: NPBatch):
        point_feat = self._encode_points(
            batch.context_x,
            batch.context_y,
            batch.context_yerr,
            batch.context_band,
            batch.context_mask,
            batch.redshift,
        )
        agg, density = self.setconv(batch.context_x, point_feat, batch.context_mask, self.grid_x)
        grid_in = torch.cat([agg, density.unsqueeze(-1)], dim=-1).transpose(1, 2)
        grid_features = self.grid_backbone(self.grid_proj(grid_in))

        latent_mu = latent_logvar = latent_z = None
        if self.cfg.use_latent:
            latent_mu, latent_logvar = self.latent_encoder(point_feat, batch.context_mask)
            latent_z = self._sample_latent(latent_mu, latent_logvar)
        return grid_features, latent_mu, latent_logvar, latent_z

    def _head_input(self, batch: NPBatch, grid_features: torch.Tensor, latent_mu=None, latent_logvar=None):
        gf = grid_features.transpose(1, 2)
        gmean = gf.mean(dim=1)
        gmax = gf.max(dim=1).values
        parts = [gmean, gmax]
        if self.cfg.use_latent and latent_mu is not None and latent_logvar is not None:
            parts.extend([latent_mu, latent_logvar])
        if self.cfg.use_redshift:
            red = torch.zeros(gmean.shape[0], 1, device=gmean.device) if batch.redshift is None else batch.redshift.unsqueeze(-1)
            parts.append(red)
        return torch.cat(parts, dim=-1)

    def decode_from_shared(self, batch: NPBatch, grid_features: torch.Tensor, latent_mu=None, latent_logvar=None, latent_z=None):
        query_feat = self._interpolate_grid(grid_features, batch.target_x)
        tfeat = self.time_embed(batch.target_x)
        bfeat = self.band_embed(batch.target_band)

        dec_parts = [query_feat, bfeat, tfeat]
        if self.cfg.use_latent and latent_z is not None:
            dec_parts.append(latent_z.unsqueeze(1).expand(-1, batch.target_x.shape[1], -1))
        if self.cfg.use_redshift:
            red = torch.zeros_like(batch.target_x) if batch.redshift is None else batch.redshift.unsqueeze(-1).expand_as(batch.target_x)
            dec_parts.append(red.unsqueeze(-1))
        dec_in = torch.cat(dec_parts, dim=-1)
        dec_out = self.decoder(dec_in)
        pred_mean = dec_out[..., 0]
        pred_std = F.softplus(dec_out[..., 1]) + self.cfg.min_std
        pred_var = pred_std.square()

        head_in = self._head_input(batch, grid_features, latent_mu, latent_logvar)
        class_logits = self.classifier(head_in).squeeze(-1)
        morph_pred = self.morphology_head(head_in) if self.morphology_head is not None else None

        return JointModelOutput(
            pred_mean=pred_mean,
            pred_var=pred_var,
            class_logits=class_logits,
            morph_pred=morph_pred,
            grid_features=grid_features,
            latent_mu=latent_mu,
            latent_logvar=latent_logvar,
            latent_sample=latent_z,
        )

    def forward(self, batch: NPBatch) -> JointModelOutput:
        grid_features, latent_mu, latent_logvar, latent_z = self.encode_shared(batch)
        return self.decode_from_shared(batch, grid_features, latent_mu, latent_logvar, latent_z)

    @torch.no_grad()
    def sample_predictions(self, batch: NPBatch, num_samples: int = 8):
        self.eval()
        grid_features, latent_mu, latent_logvar, _ = self.encode_shared(batch)
        outs = []
        if not self.cfg.use_latent or latent_mu is None or latent_logvar is None or num_samples <= 1:
            out = self.decode_from_shared(batch, grid_features, latent_mu, latent_logvar, latent_mu if self.cfg.use_latent else None)
            return out.pred_mean.unsqueeze(0), out.pred_var.unsqueeze(0)
        for _ in range(num_samples):
            z = self._sample_latent(latent_mu, latent_logvar)
            out = self.decode_from_shared(batch, grid_features, latent_mu, latent_logvar, z)
            outs.append((out.pred_mean, out.pred_var))
        means = torch.stack([m for m, _ in outs], dim=0)
        vars_ = torch.stack([v for _, v in outs], dim=0)
        return means, vars_
