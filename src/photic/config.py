from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConvGNPJointConfig:
    num_bands: int = 6
    grid_size: int = 256
    grid_min: float = 0.0
    grid_max: float = 1.0

    band_emb_dim: int = 8
    time_fourier_dim: int = 8
    point_feat_dim: int = 64
    grid_feat_dim: int = 128
    conv_hidden_dim: int = 128
    conv_layers: int = 6
    conv_kernel_size: int = 5
    conv_dropout: float = 0.0

    latent_dim: int = 16
    latent_hidden_dim: int = 128
    classifier_hidden_dim: int = 128
    interestingness_hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    morphology_hidden_dim: int = 128
    num_morph_targets: int = 4

    setconv_sigma: float = 0.03
    min_std: float = 1e-3
    use_redshift: bool = True
    use_latent: bool = True
    use_interestingness_head: bool = True
    use_morphology_head: bool = True


@dataclass(slots=True)
class AttentiveNPJointConfig:
    num_bands: int = 6

    band_emb_dim: int = 8
    time_fourier_dim: int = 8
    point_feat_dim: int = 128
    latent_dim: int = 16
    latent_hidden_dim: int = 128
    classifier_hidden_dim: int = 128
    interestingness_hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    morphology_hidden_dim: int = 128
    num_morph_targets: int = 4

    attn_hidden_dim: int = 128
    attn_heads: int = 4
    attn_layers: int = 2
    attn_dropout: float = 0.1

    min_std: float = 1e-3
    use_redshift: bool = True
    use_latent: bool = True
    use_interestingness_head: bool = True
    use_morphology_head: bool = True


@dataclass(slots=True)
class JointLossConfig:
    lambda_recon: float = 1.0
    lambda_cls: float = 1.0
    lambda_interesting: float = 0.0
    lambda_morph: float = 0.15
    beta_kl: float = 1e-3
    focal_gamma: float | None = None
    pos_weight: float | None = None
    interesting_pos_weight: float | None = None

    peak_weight_snr_boost: float = 0.0
    peak_weight_flux_boost: float = 0.0
    peak_weight_snr_threshold: float = 5.0
    peak_weight_flux_quantile: float = 0.90
