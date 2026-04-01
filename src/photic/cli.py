from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from .alerts import AlerceBrokerClient, ForecastResult, JointAlertForecaster
from .config import ConvGNPJointConfig, JointLossConfig
from .data import MallornCollateConfig, compute_z_norm_stats, compute_morphology_norm_stats, make_mallorn_collate, prepare_mallorn_datasets
from .gui import run_forecast_viewer
from .model import ConvGNPJointModel
from .train import evaluate_epoch, fit_epoch, evaluate_mallorn_epoch


@click.group()
def main():
    """photic command line interface."""


def _build_train_loader(train_ds, batch_size: int, num_workers: int, base_collate_cfg: MallornCollateConfig, epoch: int) -> DataLoader:
    train_collate_cfg = MallornCollateConfig(
        **{**asdict(base_collate_cfg), "morph_mean": base_collate_cfg.morph_mean, "morph_std": base_collate_cfg.morph_std, "epoch": epoch}
    )
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=make_mallorn_collate(training=True, cfg=train_collate_cfg),
    )


def _checkpoint_payload(
    *,
    epoch: int,
    phase: str,
    model: ConvGNPJointModel,
    model_cfg: ConvGNPJointConfig,
    loss_cfg: JointLossConfig,
    z_min: float,
    z_max: float,
    morph_mean,
    morph_std,
    metrics: dict[str, float],
    composite: float,
    history: list[dict],
) -> dict:
    return {
        "epoch": epoch,
        "phase": phase,
        "model_state": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "loss_cfg": asdict(loss_cfg),
        "z_min": z_min,
        "z_max": z_max,
        "morph_mean": morph_mean.tolist(),
        "morph_std": morph_std.tolist(),
        "metrics": metrics,
        "composite": composite,
        "history": history,
    }


def _save_json(path: Path, payload: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_checkpoint_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    return {
        "path": path,
        "phase": ckpt.get("phase", "unknown"),
        "epoch": ckpt.get("epoch"),
        "metrics": ckpt.get("metrics", {}),
        "composite": ckpt.get("composite"),
    }


def _fmt_metric(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "nan"
    return str(value)


def _load_stream_state(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_stream_state(path: Path, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _forecast_to_dict(result: ForecastResult) -> dict:
    return {
        "object_id": result.object_id,
        "prob_tde": result.prob_tde,
        "class_logit": result.class_logit,
        "redshift": result.redshift,
        "context_points": result.context_points,
        "flux_mean": result.flux_mean,
        "flux_std": result.flux_std,
        "t_min": result.t_min,
        "t_span": result.t_span,
        "bands": {
            band: {
                "mjd": curve.mjd.tolist(),
                "flux_mean": curve.flux_mean.tolist(),
                "flux_sigma": curve.flux_sigma.tolist(),
                "latent_flux_samples": None if curve.latent_flux_samples is None else curve.latent_flux_samples.tolist(),
            }
            for band, curve in result.bands.items()
        },
    }


def _photometry_to_dict(alert_object) -> list[dict]:
    return [
        {
            "mjd": point.mjd,
            "band": point.band,
            "flux": point.flux,
            "flux_err": point.flux_err,
        }
        for point in alert_object.photometry
    ]


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
@click.option("--checkpoint-metric", type=click.Choice(["best_f1", "ap", "composite"]), default="best_f1", show_default=True, help="Primary metric used for the final stage-1 checkpoint alias.")
@click.option("--stage2-epochs", type=int, default=0, show_default=True, help="Optional classification-focused fine-tuning epochs from the best stage-1 checkpoint.")
@click.option("--stage2-patience", type=int, default=20, show_default=True)
@click.option("--stage2-from", "stage2_from_metric", type=click.Choice(["best_f1", "ap", "composite"]), default="best_f1", show_default=True, help="Stage-1 checkpoint used to start stage 2.")
@click.option("--stage2-lr", type=float, default=None, help="Stage-2 learning rate. Defaults to lr / 5.")
@click.option("--stage2-lambda-recon", type=float, default=None, help="Stage-2 reconstruction weight. Defaults to lambda_recon / 10.")
@click.option("--stage2-lambda-morph", type=float, default=None, help="Stage-2 morphology weight. Defaults to current lambda_morph.")
@click.option("--stage2-focal-gamma", type=float, default=None, help="Optional focal loss gamma for stage 2.")
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
    checkpoint_metric: str,
    stage2_epochs: int,
    stage2_patience: int,
    stage2_from_metric: str,
    stage2_lr: float | None,
    stage2_lambda_recon: float | None,
    stage2_lambda_morph: float | None,
    stage2_focal_gamma: float | None,
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
    click.echo(f"[selection] primary_checkpoint={checkpoint_metric} stage2_from={stage2_from_metric} stage2_epochs={stage2_epochs}")
    click.echo(f"[model] parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,} device={device}")
    click.echo(f"{'Ph':>2} {'Ep':>4} {'train_total':>11} {'val_total':>10} {'bestF1':>8} {'AP':>8} {'rmse_pk':>8} {'val_nll':>8} {'comp':>8}")
    click.echo("-" * 84)

    history = []
    best_comp = float("inf")
    best_f1 = float("-inf")
    best_ap = float("-inf")
    best_paths = {
        "composite": out_dir / "best_checkpoint.pt",
        "best_f1": out_dir / "best_f1_checkpoint.pt",
        "ap": out_dir / "best_ap_checkpoint.pt",
    }
    patience_count = 0
    global_epoch = 0

    for epoch in range(1, epochs + 1):
        global_epoch += 1
        train_loader = _build_train_loader(train_ds, batch_size, num_workers, base_collate_cfg, global_epoch)
        train_metrics = fit_epoch(model, train_loader, optimizer, loss_cfg, device)
        val_metrics = evaluate_mallorn_epoch(model, val_loader, loss_cfg, device)
        scheduler.step()
        rmse_pk = val_metrics.get("rmse_tde_peak", float("nan"))
        rmse_n = rmse_pk if not math.isnan(rmse_pk) else 2.0
        comp = 0.35 * val_metrics.get("recon", 0.0) + 0.20 * rmse_n - 0.45 * val_metrics.get("best_f1", 0.0)
        improved = comp < best_comp
        row = {
            "phase": "s1",
            "epoch": epoch,
            "global_epoch": global_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "composite": comp,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        stage1_f1 = val_metrics.get("best_f1", float("-inf"))
        stage1_ap = val_metrics.get("ap", float("-inf"))
        improved_f1 = stage1_f1 > best_f1
        improved_ap = stage1_ap > best_ap
        if epoch == 1 or epoch % 10 == 0 or improved or improved_f1 or improved_ap:
            marks = []
            if improved:
                marks.append("comp")
            if improved_f1:
                marks.append("f1")
            if improved_ap:
                marks.append("ap")
            mark = f" ✓[{','.join(marks)}]" if marks else ""
            click.echo(
                f"{'s1':>2} {epoch:4d} {train_metrics['total']:11.4f} {val_metrics['total']:10.4f} "
                f"{val_metrics.get('best_f1', float('nan')):8.4f} {val_metrics.get('ap', float('nan')):8.4f} "
                f"{rmse_pk:8.4f} {val_metrics.get('val_nll', float('nan')):8.4f} {comp:8.4f}{mark}"
            )
        if improved:
            best_comp = comp
            patience_count = 0
            torch.save(
                _checkpoint_payload(
                    epoch=epoch,
                    phase="stage1",
                    model=model,
                    model_cfg=model_cfg,
                    loss_cfg=loss_cfg,
                    z_min=z_min,
                    z_max=z_max,
                    morph_mean=morph_mean,
                    morph_std=morph_std,
                    metrics=val_metrics,
                    composite=comp,
                    history=history,
                ),
                best_paths["composite"],
            )
        else:
            patience_count += 1
            if patience_count >= patience:
                click.echo(f"[early stop] No improvement for {patience} epochs.")
                break
        if improved_f1:
            best_f1 = stage1_f1
            torch.save(
                _checkpoint_payload(
                    epoch=epoch,
                    phase="stage1",
                    model=model,
                    model_cfg=model_cfg,
                    loss_cfg=loss_cfg,
                    z_min=z_min,
                    z_max=z_max,
                    morph_mean=morph_mean,
                    morph_std=morph_std,
                    metrics=val_metrics,
                    composite=comp,
                    history=history,
                ),
                best_paths["best_f1"],
            )
        if improved_ap:
            best_ap = stage1_ap
            torch.save(
                _checkpoint_payload(
                    epoch=epoch,
                    phase="stage1",
                    model=model,
                    model_cfg=model_cfg,
                    loss_cfg=loss_cfg,
                    z_min=z_min,
                    z_max=z_max,
                    morph_mean=morph_mean,
                    morph_std=morph_std,
                    metrics=val_metrics,
                    composite=comp,
                    history=history,
                ),
                best_paths["ap"],
            )
        _save_json(out_dir / "training_log.json", history)

    primary_src = best_paths[checkpoint_metric]
    primary_dst = out_dir / "best_primary_checkpoint.pt"
    if primary_src.exists():
        primary_dst.write_bytes(primary_src.read_bytes())

    if stage2_epochs > 0:
        stage2_ckpt_path = best_paths[stage2_from_metric]
        if not stage2_ckpt_path.exists():
            raise click.ClickException(f"Stage-2 checkpoint source not found: {stage2_ckpt_path}")
        ckpt = torch.load(stage2_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        stage2_lr = stage2_lr if stage2_lr is not None else lr / 5.0
        stage2_lambda_recon = stage2_lambda_recon if stage2_lambda_recon is not None else lambda_recon / 10.0
        stage2_lambda_morph = stage2_lambda_morph if stage2_lambda_morph is not None else lambda_morph
        stage2_loss_cfg = JointLossConfig(
            lambda_recon=stage2_lambda_recon,
            lambda_cls=lambda_cls,
            lambda_morph=stage2_lambda_morph,
            beta_kl=beta_kl,
            focal_gamma=stage2_focal_gamma,
            pos_weight=pos_weight,
            peak_weight_snr_boost=peak_weight_snr_boost,
            peak_weight_flux_boost=peak_weight_flux_boost,
            peak_weight_snr_threshold=peak_weight_snr_threshold,
            peak_weight_flux_quantile=peak_weight_flux_quantile,
        )
        stage2_optimizer = torch.optim.AdamW(model.parameters(), lr=stage2_lr, weight_decay=weight_decay)
        stage2_best_f1 = float(ckpt.get("metrics", {}).get("best_f1", float("-inf")))
        stage2_best_ap = float(ckpt.get("metrics", {}).get("ap", float("-inf")))
        stage2_patience_count = 0
        click.echo(
            f"[stage2] from={stage2_from_metric} lr={stage2_lr:.6g} "
            f"lambda_recon={stage2_lambda_recon:.4f} lambda_morph={stage2_lambda_morph:.4f} "
            f"focal_gamma={stage2_focal_gamma}"
        )
        for epoch in range(1, stage2_epochs + 1):
            global_epoch += 1
            train_loader = _build_train_loader(train_ds, batch_size, num_workers, base_collate_cfg, global_epoch)
            train_metrics = fit_epoch(model, train_loader, stage2_optimizer, stage2_loss_cfg, device)
            val_metrics = evaluate_mallorn_epoch(model, val_loader, stage2_loss_cfg, device)
            rmse_pk = val_metrics.get("rmse_tde_peak", float("nan"))
            rmse_n = rmse_pk if not math.isnan(rmse_pk) else 2.0
            comp = 0.35 * val_metrics.get("recon", 0.0) + 0.20 * rmse_n - 0.45 * val_metrics.get("best_f1", 0.0)
            phase_epoch = epochs + epoch
            row = {
                "phase": "s2",
                "epoch": phase_epoch,
                "phase_epoch": epoch,
                "global_epoch": global_epoch,
                "lr": stage2_optimizer.param_groups[0]["lr"],
                "composite": comp,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            current_f1 = val_metrics.get("best_f1", float("-inf"))
            current_ap = val_metrics.get("ap", float("-inf"))
            improved_f1 = current_f1 > stage2_best_f1
            improved_ap = current_ap > stage2_best_ap
            if epoch == 1 or epoch % 10 == 0 or improved_f1 or improved_ap:
                marks = []
                if improved_f1:
                    marks.append("f1")
                if improved_ap:
                    marks.append("ap")
                mark = f" ✓[{','.join(marks)}]" if marks else ""
                click.echo(
                    f"{'s2':>2} {phase_epoch:4d} {train_metrics['total']:11.4f} {val_metrics['total']:10.4f} "
                    f"{val_metrics.get('best_f1', float('nan')):8.4f} {val_metrics.get('ap', float('nan')):8.4f} "
                    f"{rmse_pk:8.4f} {val_metrics.get('val_nll', float('nan')):8.4f} {comp:8.4f}{mark}"
                )
            if improved_f1:
                stage2_best_f1 = current_f1
                stage2_patience_count = 0
                torch.save(
                    _checkpoint_payload(
                        epoch=phase_epoch,
                        phase="stage2",
                        model=model,
                        model_cfg=model_cfg,
                        loss_cfg=stage2_loss_cfg,
                        z_min=z_min,
                        z_max=z_max,
                        morph_mean=morph_mean,
                        morph_std=morph_std,
                        metrics=val_metrics,
                        composite=comp,
                        history=history,
                    ),
                    out_dir / "best_stage2_f1_checkpoint.pt",
                )
                torch.save(
                    _checkpoint_payload(
                        epoch=phase_epoch,
                        phase="stage2",
                        model=model,
                        model_cfg=model_cfg,
                        loss_cfg=stage2_loss_cfg,
                        z_min=z_min,
                        z_max=z_max,
                        morph_mean=morph_mean,
                        morph_std=morph_std,
                        metrics=val_metrics,
                        composite=comp,
                        history=history,
                    ),
                    out_dir / "best_primary_checkpoint.pt",
                )
            else:
                stage2_patience_count += 1
            if improved_ap:
                stage2_best_ap = current_ap
                torch.save(
                    _checkpoint_payload(
                        epoch=phase_epoch,
                        phase="stage2",
                        model=model,
                        model_cfg=model_cfg,
                        loss_cfg=stage2_loss_cfg,
                        z_min=z_min,
                        z_max=z_max,
                        morph_mean=morph_mean,
                        morph_std=morph_std,
                        metrics=val_metrics,
                        composite=comp,
                        history=history,
                    ),
                    out_dir / "best_stage2_ap_checkpoint.pt",
                )
            _save_json(out_dir / "training_log.json", history)
            if stage2_patience_count >= stage2_patience:
                click.echo(f"[stage2 early stop] No F1 improvement for {stage2_patience} epochs.")
                break

    composite_summary = _load_checkpoint_summary(best_paths["composite"])
    f1_summary = _load_checkpoint_summary(best_paths["best_f1"])
    ap_summary = _load_checkpoint_summary(best_paths["ap"])
    primary_summary = _load_checkpoint_summary(out_dir / "best_primary_checkpoint.pt")

    click.echo(f"[done] composite checkpoint: {best_paths['composite']}")
    click.echo(f"[done] best F1 checkpoint: {best_paths['best_f1']}")
    click.echo(f"[done] best AP checkpoint: {best_paths['ap']}")
    click.echo(f"[done] primary checkpoint: {out_dir / 'best_primary_checkpoint.pt'}")
    click.echo("[summary]")
    for label, summary in [
        ("composite", composite_summary),
        ("best_f1", f1_summary),
        ("best_ap", ap_summary),
        ("primary", primary_summary),
    ]:
        if summary is None:
            click.echo(f"  {label:10s}: missing")
            continue
        metrics = summary["metrics"]
        click.echo(
            f"  {label:10s}: phase={summary['phase']} epoch={summary['epoch']} "
            f"best_f1={_fmt_metric(metrics.get('best_f1'))} "
            f"ap={_fmt_metric(metrics.get('ap'))} "
            f"val_nll={_fmt_metric(metrics.get('val_nll'))} "
            f"rmse_pk={_fmt_metric(metrics.get('rmse_tde_peak'))} "
            f"comp={_fmt_metric(summary.get('composite'))}"
        )


@main.command("forecast-alert-stream")
@click.option("--broker", type=click.Choice(["alerce"]), default="alerce", show_default=True)
@click.option("--survey", type=str, default="lsst", show_default=True)
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--device", type=str, default=None)
@click.option("--forecast-days", type=float, default=365.0, show_default=True)
@click.option("--grid-points-per-band", type=int, default=256, show_default=True)
@click.option("--latent-samples", type=int, default=0, show_default=True)
@click.option("--since-mjd", type=float, default=None, help="Initial last-mjd watermark if no state file exists.")
@click.option("--page-size", type=int, default=100, show_default=True)
@click.option("--max-objects", type=int, default=100, show_default=True, help="Maximum objects to process per polling cycle.")
@click.option("--min-context-points", type=int, default=2, show_default=True, help="Skip objects with fewer observed photometry points than this.")
@click.option("--min-context-bands", type=int, default=1, show_default=True, help="Skip objects with fewer distinct observed bands than this.")
@click.option("--poll-interval", type=float, default=60.0, show_default=True, help="Seconds between polling cycles.")
@click.option("--max-polls", type=int, default=1, show_default=True, help="Set >1 for repeated polling, or 0 to run forever.")
@click.option("--state-file", type=click.Path(dir_okay=False, path_type=Path), default=None, help="JSON file storing the polling watermark and processed object metadata.")
def forecast_alert_stream(
    broker: str,
    survey: str,
    checkpoint: Path,
    out_dir: Path,
    device: str | None,
    forecast_days: float,
    grid_points_per_band: int,
    latent_samples: int,
    since_mjd: float | None,
    page_size: int,
    max_objects: int,
    min_context_points: int,
    min_context_bands: int,
    poll_interval: float,
    max_polls: int,
    state_file: Path | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state_path = state_file or (out_dir / "stream_state.json")
    forecasts_dir = out_dir / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    if broker != "alerce":
        raise click.ClickException(f"Unsupported broker: {broker}")
    broker_client = AlerceBrokerClient(survey=survey)
    forecaster = JointAlertForecaster.from_checkpoint(checkpoint, device=device)

    state = _load_stream_state(state_path)
    watermark = state.get("last_mjd", since_mjd)
    print(watermark, since_mjd)
    poll_idx = 0
    click.echo(
        f"[stream] broker={broker} survey={survey} device={device} "
        f"forecast_days={forecast_days:.1f} watermark={watermark} "
        f"min_points={min_context_points} min_bands={min_context_bands}"
    )

    while max_polls == 0 or poll_idx < max_polls:
        poll_idx += 1
        click.echo(f"[poll {poll_idx}] querying recent objects")
        updates = broker_client.poll_recent_objects(
            since_mjd=watermark,
            page=1,
            page_size=page_size,
        )
        if watermark is not None:
            updates = [u for u in updates if u.last_mjd is not None and u.last_mjd >= watermark]
        updates = updates[:max_objects]
        processed = 0
        skipped = 0
        max_seen_mjd = watermark
        for update in updates:
            forecast_path = forecasts_dir / f"{update.object_id}.json"
            try:
                alert_object = broker_client.fetch_alert_object(update.object_id)
                n_points = len(alert_object.photometry)
                n_bands = len({point.band for point in alert_object.photometry})
                if n_points < min_context_points or n_bands < min_context_bands:
                    skipped += 1
                    click.echo(
                        f"[skip] {update.object_id} n_points={n_points} n_bands={n_bands} "
                        f"(min_points={min_context_points}, min_bands={min_context_bands})"
                    )
                    if update.last_mjd is not None:
                        max_seen_mjd = update.last_mjd if max_seen_mjd is None else max(max_seen_mjd, update.last_mjd)
                    continue
                result = forecaster.forecast(
                    alert_object,
                    forecast_days=forecast_days,
                    grid_points_per_band=grid_points_per_band,
                    latent_samples=latent_samples,
                )
            except Exception as exc:
                click.echo(f"[warn] {update.object_id}: {exc}")
                continue
            payload = {
                "broker": broker,
                "survey": survey,
                "checkpoint": str(checkpoint),
                "update": {
                    "object_id": update.object_id,
                    "first_mjd": update.first_mjd,
                    "last_mjd": update.last_mjd,
                    "ndet": update.ndet,
                },
                "alert_object": {
                    "object_id": alert_object.object_id,
                    "redshift": alert_object.redshift,
                    "photometry": _photometry_to_dict(alert_object),
                },
                "forecast": _forecast_to_dict(result),
            }
            with open(forecast_path, "w") as f:
                json.dump(payload, f, indent=2)
            processed += 1
            if update.last_mjd is not None:
                max_seen_mjd = update.last_mjd if max_seen_mjd is None else max(max_seen_mjd, update.last_mjd)
            click.echo(f"[forecast] {update.object_id} p(TDE)={result.prob_tde:.4f} -> {forecast_path}")

        state.update(
            {
                "broker": broker,
                "survey": survey,
                "checkpoint": str(checkpoint),
                "last_mjd": max_seen_mjd,
                "last_poll_index": poll_idx,
                "last_processed_count": processed,
                "last_skipped_count": skipped,
            }
        )
        _save_stream_state(state_path, state)
        click.echo(f"[poll {poll_idx}] processed={processed} skipped={skipped} watermark={max_seen_mjd}")
        if max_polls != 0 and poll_idx >= max_polls:
            break
        time.sleep(poll_interval)


@main.command("forecast-gui")
@click.option("--forecast-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8080, show_default=True)
def forecast_gui(
    forecast_dir: Path,
    host: str,
    port: int,
):
    run_forecast_viewer(forecast_dir, host=host, port=port)


if __name__ == '__main__':
    main()
