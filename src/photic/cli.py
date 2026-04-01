from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from .config import ConvGNPJointConfig, JointLossConfig
from .data import MallornCollateConfig, compute_z_norm_stats, compute_morphology_norm_stats, make_mallorn_collate, prepare_mallorn_datasets
from .model import ConvGNPJointModel
from .train import evaluate_epoch, fit_epoch, evaluate_mallorn_epoch


@click.group()
def main():
    """photic command line interface."""


@main.command("train-mallorn")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--epochs", type=int, default=300, show_default=True)
@click.option("--patience", type=int, default=60, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--num-workers", type=int, default=4, show_default=True)
@click.option("--val-frac", type=float, default=0.15, show_default=True)
@click.option("--mask-prob", type=float, default=0.50, show_default=True)
@click.option("--block-mask-prob", type=float, default=0.50, show_default=True)
@click.option("--block-mask-frac", type=float, default=0.35, show_default=True)
@click.option("--min-ctx-per-band", type=int, default=2, show_default=True)
@click.option("--grid-size", type=int, default=256, show_default=True)
@click.option("--grid-feat-dim", type=int, default=128, show_default=True)
@click.option("--conv-layers", type=int, default=6, show_default=True)
@click.option("--latent-dim", type=int, default=16, show_default=True)
@click.option("--beta-kl", type=float, default=1e-3, show_default=True)
@click.option("--lambda-recon", type=float, default=1.0, show_default=True)
@click.option("--lambda-cls", type=float, default=1.0, show_default=True)
@click.option("--lambda-morph", type=float, default=0.15, show_default=True)
@click.option("--focal-gamma", type=float, default=None)
@click.option("--peak-weight-snr-boost", type=float, default=1.0, show_default=True)
@click.option("--peak-weight-flux-boost", type=float, default=0.5, show_default=True)
@click.option("--peak-weight-snr-threshold", type=float, default=5.0, show_default=True)
@click.option("--peak-weight-flux-quantile", type=float, default=0.90, show_default=True)
@click.option("--max-obs", type=int, default=200, show_default=True)
@click.option("--keep-all-snr-gt", type=float, default=5.0, show_default=True)
@click.option("--device", type=str, default=None)
def train_mallorn(
    data_dir: Path,
    out_dir: Path,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    num_workers: int,
    val_frac: float,
    mask_prob: float,
    block_mask_prob: float,
    block_mask_frac: float,
    min_ctx_per_band: int,
    grid_size: int,
    grid_feat_dim: int,
    conv_layers: int,
    latent_dim: int,
    beta_kl: float,
    lambda_recon: float,
    lambda_cls: float,
    lambda_morph: float,
    focal_gamma: float | None,
    peak_weight_snr_boost: float,
    peak_weight_flux_boost: float,
    peak_weight_snr_threshold: float,
    peak_weight_flux_quantile: float,
    max_obs: int,
    keep_all_snr_gt: float,
    device: str | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lc, log, train_ds, val_ds, train_ids, val_ids = prepare_mallorn_datasets(
        data_dir=data_dir,
        seed=seed,
        val_frac=val_frac,
        max_obs=max_obs,
        keep_all_snr_gt=keep_all_snr_gt,
    )
    z_min, z_max = compute_z_norm_stats(train_ds)
    morph_mean, morph_std = compute_morphology_norm_stats(train_ds)
    base_collate_cfg = MallornCollateConfig(
        seed=seed,
        mask_prob=mask_prob,
        block_mask_prob=block_mask_prob,
        block_mask_frac=block_mask_frac,
        min_ctx_per_band=min_ctx_per_band,
        z_min=z_min,
        z_max=z_max,
        morph_mean=morph_mean,
        morph_std=morph_std,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=make_mallorn_collate(training=False, cfg=base_collate_cfg),
    )

    model_cfg = ConvGNPJointConfig(grid_size=grid_size, grid_feat_dim=grid_feat_dim, conv_layers=conv_layers, latent_dim=latent_dim)
    model = ConvGNPJointModel(model_cfg).to(device)
    pos_count = float((log[log["object_id"].isin(train_ids)]["target"] == 1).sum())
    neg_count = float((log[log["object_id"].isin(train_ids)]["target"] == 0).sum())
    pos_weight = neg_count / max(pos_count, 1.0)
    loss_cfg = JointLossConfig(
        lambda_recon=lambda_recon,
        lambda_cls=lambda_cls,
        lambda_morph=lambda_morph,
        beta_kl=beta_kl,
        focal_gamma=focal_gamma,
        pos_weight=pos_weight,
        peak_weight_snr_boost=peak_weight_snr_boost,
        peak_weight_flux_boost=peak_weight_flux_boost,
        peak_weight_snr_threshold=peak_weight_snr_threshold,
        peak_weight_flux_quantile=peak_weight_flux_quantile,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=2, eta_min=1.5e-6)

    click.echo(f"[data] train={len(train_ds)} val={len(val_ds)} z_norm=[{z_min:.3f}, {z_max:.3f}] pos_weight={pos_weight:.3f}")
    click.echo(f"[morph] lambda_morph={lambda_morph:.3f} mean={morph_mean.tolist()} std={morph_std.tolist()}")
    click.echo(f"[loss] peak_weight_snr_boost={peak_weight_snr_boost:.3f} peak_weight_flux_boost={peak_weight_flux_boost:.3f} threshold={peak_weight_snr_threshold:.2f} flux_q={peak_weight_flux_quantile:.2f}")
    click.echo(f"[model] parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,} device={device}")
    click.echo(f"{'Ep':>4} {'train_total':>11} {'val_total':>10} {'bestF1':>8} {'rmse_pk':>8} {'val_nll':>8} {'comp':>8}")
    click.echo('-' * 72)

    history = []
    best_comp = float('inf')
    patience_count = 0
    ckpt_path = out_dir / 'best_checkpoint.pt'

    for epoch in range(1, epochs + 1):
        train_collate_cfg = MallornCollateConfig(
            **{**asdict(base_collate_cfg), "morph_mean": base_collate_cfg.morph_mean, "morph_std": base_collate_cfg.morph_std, "epoch": epoch}
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            collate_fn=make_mallorn_collate(training=True, cfg=train_collate_cfg),
        )
        train_metrics = fit_epoch(model, train_loader, optimizer, loss_cfg, device)
        val_metrics = evaluate_mallorn_epoch(model, val_loader, loss_cfg, device)
        scheduler.step()
        rmse_pk = val_metrics.get('rmse_tde_peak', float('nan'))
        rmse_n = rmse_pk if not math.isnan(rmse_pk) else 2.0
        comp = 0.35 * val_metrics.get('recon', 0.0) + 0.20 * rmse_n - 0.45 * val_metrics.get('best_f1', 0.0)
        improved = comp < best_comp
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'composite': comp, **train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or improved:
            mark = ' ✓' if improved else ''
            click.echo(f"{epoch:4d} {train_metrics['total']:11.4f} {val_metrics['total']:10.4f} {val_metrics.get('best_f1', float('nan')):8.4f} {rmse_pk:8.4f} {val_metrics.get('val_nll', float('nan')):8.4f} {comp:8.4f}{mark}")
        if improved:
            best_comp = comp
            patience_count = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'model_cfg': asdict(model_cfg), 'loss_cfg': asdict(loss_cfg), 'z_min': z_min, 'z_max': z_max, 'morph_mean': morph_mean.tolist(), 'morph_std': morph_std.tolist(), 'metrics': val_metrics, 'composite': comp}, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                click.echo(f"[early stop] No improvement for {patience} epochs.")
                break
        with open(out_dir / 'training_log.json', 'w') as f:
            json.dump(history, f, indent=2)

    click.echo(f"[done] best checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
