from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
import torch
from torch.utils.data import DataLoader

from .alerts import AlerceBrokerClient, ForecastResult, JointAlertForecaster
from .batch import NPBatch
from .config import AttentiveNPJointConfig, ConvGNPJointConfig, JointLossConfig
from .data import MallornCollateConfig, MallornDataset, apply_ebv_correction, build_mallorn_datasets_from_ids, compute_morphology_norm_stats, compute_z_norm_stats, load_all_data, make_mallorn_collate, prepare_mallorn_datasets, preprocess_mallorn_training_tables
from .gui import run_forecast_viewer
from .model import build_joint_config, build_joint_model, load_joint_model_checkpoint
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
    model_type: str,
    epoch: int,
    phase: str,
    model,
    model_cfg,
    loss_cfg: JointLossConfig,
    z_min: float,
    z_max: float,
    morph_mean,
    morph_std,
    flux_center_by_band,
    flux_scale_by_band,
    metrics: dict[str, float],
    composite: float,
    history: list[dict],
) -> dict:
    return {
        "model_type": model_type,
        "epoch": epoch,
        "phase": phase,
        "model_state": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "loss_cfg": asdict(loss_cfg),
        "z_min": z_min,
        "z_max": z_max,
        "morph_mean": morph_mean.tolist(),
        "morph_std": morph_std.tolist(),
        "flux_center_by_band": flux_center_by_band.tolist(),
        "flux_scale_by_band": flux_scale_by_band.tolist(),
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


def _load_joint_checkpoint(checkpoint: Path, device: str):
    return load_joint_model_checkpoint(checkpoint, device)


@torch.no_grad()
def _collect_validation_predictions(model, loader, device: str | torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, float | str | int]] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = torch.sigmoid(out.class_logits).detach().cpu().numpy() if out.class_logits is not None else np.full(batch.context_x.shape[0], np.nan)
        labels = batch.labels.detach().cpu().numpy().astype(int) if batch.labels is not None else np.full(batch.context_x.shape[0], -1)
        oids = batch.metadata.get("oid") if batch.metadata else [f"obj_{i}" for i in range(batch.context_x.shape[0])]
        for oid, label, prob in zip(oids, labels, probs):
            rows.append({"object_id": oid, "label": int(label), "prob_tde": float(prob)})
    return pd.DataFrame(rows)


def _find_best_binary_threshold(df: pd.DataFrame) -> dict[str, float]:
    y = df["label"].to_numpy(dtype=int)
    p = df["prob_tde"].to_numpy(dtype=float)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 197):
        pred = (p >= thr).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    ap = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {"best_threshold": best_thr, "best_f1": best_f1, "ap": ap}


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
        "flux_center_by_band": result.flux_center_by_band,
        "flux_scale_by_band": result.flux_scale_by_band,
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


def _load_mallorn_test_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dirs = sorted(data_dir.glob("split_*"))
    lc_parts = [
        pd.read_csv(sd / "test_full_lightcurves.csv")
        for sd in split_dirs
        if (sd / "test_full_lightcurves.csv").exists()
    ]
    log_f = data_dir / "test_log.csv"
    if not lc_parts or not log_f.exists():
        raise click.ClickException(f"Could not find Mallorn test data under {data_dir}")
    lc = pd.concat(lc_parts, ignore_index=True)
    log = pd.read_csv(log_f)
    lc = lc.rename(columns={"Time (MJD)": "mjd", "Flux": "flux", "Flux_err": "flux_err", "Filter": "band"})
    lc = lc.dropna(subset=["flux", "flux_err"])
    lc = lc[lc["flux_err"] > 0].copy()
    keep = [c for c in ["object_id", "Z", "Z_err", "EBV", "SpecType", "split"] if c in log.columns]
    log = log[keep].copy()
    return lc, log


def _build_full_context_batch(items: list[dict], z_min: float, z_max: float) -> NPBatch:
    def _pad(vals: list[torch.Tensor], pad_value: float = 0.0, dtype=None):
        max_len = max((len(v) for v in vals), default=1)
        dtype = vals[0].dtype if vals else (torch.float32 if dtype is None else dtype)
        out = torch.full((len(vals), max_len), pad_value, dtype=dtype)
        for i, v in enumerate(vals):
            if len(v) > 0:
                out[i, :len(v)] = v
        return out

    context_x = [torch.as_tensor(item["t_norm"], dtype=torch.float32) for item in items]
    context_y = [torch.as_tensor(item["flux_norm"], dtype=torch.float32) for item in items]
    context_yerr = [torch.as_tensor(item["ferr_norm"], dtype=torch.float32) for item in items]
    context_band = [torch.as_tensor(item["band_idx"], dtype=torch.long) for item in items]
    context_mask = [torch.ones(len(item["t_norm"]), dtype=torch.float32) for item in items]
    target_x = [torch.as_tensor(item["t_norm"], dtype=torch.float32) for item in items]
    target_y = [torch.zeros(len(item["t_norm"]), dtype=torch.float32) for item in items]
    target_yerr = [torch.ones(len(item["t_norm"]), dtype=torch.float32) for item in items]
    target_band = [torch.as_tensor(item["band_idx"], dtype=torch.long) for item in items]
    target_mask = [torch.ones(len(item["t_norm"]), dtype=torch.float32) for item in items]
    redshifts = [float(item.get("z", np.nan)) for item in items]
    return NPBatch(
        context_x=_pad(context_x),
        context_y=_pad(context_y),
        context_yerr=_pad(context_yerr, pad_value=1.0),
        context_band=_pad(context_band, pad_value=0, dtype=torch.long).long(),
        context_mask=_pad(context_mask),
        target_x=_pad(target_x),
        target_y=_pad(target_y),
        target_yerr=_pad(target_yerr, pad_value=1.0),
        target_band=_pad(target_band, pad_value=0, dtype=torch.long).long(),
        target_mask=_pad(target_mask),
        labels=None,
        redshift=compute_redshift_batch(redshifts, z_min, z_max),
        metadata={"oid": [item["oid"] for item in items]},
    )


def compute_redshift_batch(redshifts: list[float], z_min: float, z_max: float) -> torch.Tensor:
    z_raw = torch.as_tensor(redshifts, dtype=torch.float32)
    span = max(z_max - z_min, 1e-6)
    z_norm = (z_raw.clone() - z_min) / span
    z_norm = z_norm.clamp(0.0, 1.0)
    z_norm[~torch.isfinite(z_norm)] = 0.5
    return z_norm


@torch.no_grad()
def _predict_dataset_full_context(model, dataset: MallornDataset, batch_size: int, z_min: float, z_max: float, device: str | torch.device) -> pd.DataFrame:
    model.eval()
    object_ids = list(dataset.object_ids)
    rows: list[dict[str, float | str | int]] = []
    for start in range(0, len(object_ids), batch_size):
        batch_ids = object_ids[start : start + batch_size]
        items = []
        for oid in batch_ids:
            item = dataset.data[oid].copy()
            item["oid"] = oid
            items.append(item)
        batch = _build_full_context_batch(items, z_min, z_max).to(device)
        out = model(batch)
        probs = torch.sigmoid(out.class_logits).detach().cpu().numpy() if out.class_logits is not None else np.full(len(batch_ids), np.nan)
        for oid, prob in zip(batch_ids, probs):
            rows.append(
                {
                    "object_id": oid,
                    "label": int(dataset.data[oid].get("target", 0)),
                    "prob_tde": float(prob),
                }
            )
    return pd.DataFrame(rows)


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
@click.option("--context-strategy", type=click.Choice(["prefix", "random"]), default="random", show_default=True)
@click.option("--min-context-points-total", type=int, default=3, show_default=True)
@click.option("--min-target-points-total", type=int, default=1, show_default=True)
@click.option("--prefix-anchor-snr-threshold", type=float, default=5.0, show_default=True)
@click.option("--prefix-anchor-cluster-min-points", type=int, default=2, show_default=True)
@click.option("--prefix-anchor-cluster-window-days", type=float, default=7.0, show_default=True)
@click.option("--prefix-anchor-require-positive-flux/--no-prefix-anchor-require-positive-flux", default=True, show_default=True)
@click.option("--prefix-pre-anchor-days", type=float, default=30.0, show_default=True)
@click.option("--prefix-context-days-min", type=float, default=7.0, show_default=True)
@click.option("--prefix-context-days-max", type=float, default=30.0, show_default=True)
@click.option("--prefix-target-horizon-days", type=float, default=90.0, show_default=True)
@click.option("--prefix-context-frac-min", type=float, default=0.15, show_default=True)
@click.option("--prefix-context-frac-max", type=float, default=0.60, show_default=True)
@click.option("--mask-prob", type=float, default=0.50, show_default=True)
@click.option("--block-mask-prob", type=float, default=0.50, show_default=True)
@click.option("--block-mask-frac", type=float, default=0.35, show_default=True)
@click.option("--min-ctx-per-band", type=int, default=2, show_default=True)
@click.option("--grid-size", type=int, default=256, show_default=True)
@click.option("--grid-feat-dim", type=int, default=128, show_default=True)
@click.option("--conv-layers", type=int, default=6, show_default=True)
@click.option("--model-type", type=click.Choice(["convgnp", "attnnp"]), default="convgnp", show_default=True)
@click.option("--latent-dim", type=int, default=16, show_default=True)
@click.option("--attn-hidden-dim", type=int, default=128, show_default=True)
@click.option("--attn-heads", type=int, default=4, show_default=True)
@click.option("--attn-layers", type=int, default=2, show_default=True)
@click.option("--attn-dropout", type=float, default=0.1, show_default=True)
@click.option("--beta-kl", type=float, default=1e-3, show_default=True)
@click.option("--lambda-recon", type=float, default=1.0, show_default=True)
@click.option("--lambda-cls", type=float, default=1.0, show_default=True)
@click.option("--lambda-interesting", type=float, default=0.0, show_default=True)
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
    context_strategy: str,
    min_context_points_total: int,
    min_target_points_total: int,
    prefix_anchor_snr_threshold: float,
    prefix_anchor_cluster_min_points: int,
    prefix_anchor_cluster_window_days: float,
    prefix_anchor_require_positive_flux: bool,
    prefix_pre_anchor_days: float,
    prefix_context_days_min: float,
    prefix_context_days_max: float,
    prefix_target_horizon_days: float,
    prefix_context_frac_min: float,
    prefix_context_frac_max: float,
    mask_prob: float,
    block_mask_prob: float,
    block_mask_frac: float,
    min_ctx_per_band: int,
    grid_size: int,
    grid_feat_dim: int,
    conv_layers: int,
    model_type: str,
    latent_dim: int,
    attn_hidden_dim: int,
    attn_heads: int,
    attn_layers: int,
    attn_dropout: float,
    beta_kl: float,
    lambda_recon: float,
    lambda_cls: float,
    lambda_interesting: float,
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
    flux_center_by_band = train_ds.flux_center_by_band
    flux_scale_by_band = train_ds.flux_scale_by_band
    base_collate_cfg = MallornCollateConfig(
        seed=seed,
        context_strategy=context_strategy,
        min_ctx_points_total=min_context_points_total,
        min_target_points_total=min_target_points_total,
        prefix_anchor_snr_threshold=prefix_anchor_snr_threshold,
        prefix_anchor_cluster_min_points=prefix_anchor_cluster_min_points,
        prefix_anchor_cluster_window_days=prefix_anchor_cluster_window_days,
        prefix_anchor_require_positive_flux=prefix_anchor_require_positive_flux,
        prefix_pre_anchor_days=prefix_pre_anchor_days,
        prefix_context_days_min=prefix_context_days_min,
        prefix_context_days_max=prefix_context_days_max,
        prefix_target_horizon_days=prefix_target_horizon_days,
        prefix_context_frac_min=prefix_context_frac_min,
        prefix_context_frac_max=prefix_context_frac_max,
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

    if model_type == "convgnp":
        model_cfg = ConvGNPJointConfig(grid_size=grid_size, grid_feat_dim=grid_feat_dim, conv_layers=conv_layers, latent_dim=latent_dim)
    else:
        model_cfg = AttentiveNPJointConfig(
            latent_dim=latent_dim,
            attn_hidden_dim=attn_hidden_dim,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            attn_dropout=attn_dropout,
        )
    model = build_joint_model(model_type, model_cfg).to(device)
    pos_count = float((log[log["object_id"].isin(train_ids)]["target"] == 1).sum())
    neg_count = float((log[log["object_id"].isin(train_ids)]["target"] == 0).sum())
    pos_weight = neg_count / max(pos_count, 1.0)
    interesting_pos = 0.0
    for oid in train_ids:
        item = train_ds.data[oid]
        if np.any((item["obs_snr"] >= base_collate_cfg.interesting_snr_threshold) & (item["flux_raw"] > 0.0)):
            interesting_pos += 1.0
    interesting_neg = max(float(len(train_ids)) - interesting_pos, 0.0)
    interesting_pos_weight = interesting_neg / max(interesting_pos, 1.0)
    loss_cfg = JointLossConfig(
        lambda_recon=lambda_recon,
        lambda_cls=lambda_cls,
        lambda_interesting=lambda_interesting,
        lambda_morph=lambda_morph,
        beta_kl=beta_kl,
        focal_gamma=focal_gamma,
        pos_weight=pos_weight,
        interesting_pos_weight=interesting_pos_weight,
        peak_weight_snr_boost=peak_weight_snr_boost,
        peak_weight_flux_boost=peak_weight_flux_boost,
        peak_weight_snr_threshold=peak_weight_snr_threshold,
        peak_weight_flux_quantile=peak_weight_flux_quantile,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=2, eta_min=1.5e-6)

    click.echo(f"[data] train={len(train_ds)} val={len(val_ds)} z_norm=[{z_min:.3f}, {z_max:.3f}] pos_weight={pos_weight:.3f}")
    click.echo(
        f"[context] strategy={context_strategy} min_ctx_total={min_context_points_total} "
        f"min_tgt_total={min_target_points_total} anchor_snr={prefix_anchor_snr_threshold:.1f} "
        f"anchor_cluster={prefix_anchor_cluster_min_points}@{prefix_anchor_cluster_window_days:.1f}d "
        f"anchor_positive={str(prefix_anchor_require_positive_flux).lower()} "
        f"pre_anchor_days={prefix_pre_anchor_days:.1f} context_days=[{prefix_context_days_min:.1f}, {prefix_context_days_max:.1f}] "
        f"target_horizon_days={prefix_target_horizon_days:.1f} "
        f"fallback_frac=[{prefix_context_frac_min:.2f}, {prefix_context_frac_max:.2f}]"
    )
    click.echo(f"[interesting] lambda_interesting={lambda_interesting:.3f} pos_weight={interesting_pos_weight:.3f}")
    click.echo(f"[morph] lambda_morph={lambda_morph:.3f} mean={morph_mean.tolist()} std={morph_std.tolist()}")
    click.echo(f"[loss] peak_weight_snr_boost={peak_weight_snr_boost:.3f} peak_weight_flux_boost={peak_weight_flux_boost:.3f} threshold={peak_weight_snr_threshold:.2f} flux_q={peak_weight_flux_quantile:.2f}")
    click.echo(f"[selection] primary_checkpoint={checkpoint_metric} stage2_from={stage2_from_metric} stage2_epochs={stage2_epochs}")
    click.echo(f"[model] type={model_type} parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,} device={device}")
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
                    model_type=model_type,
                    phase="stage1",
                    model=model,
                    model_cfg=model_cfg,
                    loss_cfg=loss_cfg,
                    z_min=z_min,
                    z_max=z_max,
                    morph_mean=morph_mean,
                    morph_std=morph_std,
                    flux_center_by_band=flux_center_by_band,
                    flux_scale_by_band=flux_scale_by_band,
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
                    model_type=model_type,
                    phase="stage1",
                    model=model,
                    model_cfg=model_cfg,
                    loss_cfg=loss_cfg,
                    z_min=z_min,
                    z_max=z_max,
                    morph_mean=morph_mean,
                    morph_std=morph_std,
                    flux_center_by_band=flux_center_by_band,
                    flux_scale_by_band=flux_scale_by_band,
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
                    model_type=model_type,
                    phase="stage1",
                    model=model,
                    model_cfg=model_cfg,
                    loss_cfg=loss_cfg,
                    z_min=z_min,
                    z_max=z_max,
                    morph_mean=morph_mean,
                    morph_std=morph_std,
                    flux_center_by_band=flux_center_by_band,
                    flux_scale_by_band=flux_scale_by_band,
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
                        model_type=model_type,
                        phase="stage2",
                        model=model,
                        model_cfg=model_cfg,
                        loss_cfg=stage2_loss_cfg,
                        z_min=z_min,
                        z_max=z_max,
                        morph_mean=morph_mean,
                        morph_std=morph_std,
                        flux_center_by_band=flux_center_by_band,
                        flux_scale_by_band=flux_scale_by_band,
                        metrics=val_metrics,
                        composite=comp,
                        history=history,
                    ),
                    out_dir / "best_stage2_f1_checkpoint.pt",
                )
                torch.save(
                    _checkpoint_payload(
                        epoch=phase_epoch,
                        model_type=model_type,
                        phase="stage2",
                        model=model,
                        model_cfg=model_cfg,
                        loss_cfg=stage2_loss_cfg,
                        z_min=z_min,
                        z_max=z_max,
                        morph_mean=morph_mean,
                        morph_std=morph_std,
                        flux_center_by_band=flux_center_by_band,
                        flux_scale_by_band=flux_scale_by_band,
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
                        model_type=model_type,
                        phase="stage2",
                        model=model,
                        model_cfg=model_cfg,
                        loss_cfg=stage2_loss_cfg,
                        z_min=z_min,
                        z_max=z_max,
                        morph_mean=morph_mean,
                        morph_std=morph_std,
                        flux_center_by_band=flux_center_by_band,
                        flux_scale_by_band=flux_scale_by_band,
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


@main.command("crossval-mallorn")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--folds", type=str, default="all", show_default=True, help="Comma-separated Mallorn split names to use as validation folds, or 'all'.")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--epochs", type=int, default=300, show_default=True)
@click.option("--patience", type=int, default=60, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--context-strategy", type=click.Choice(["random", "prefix"]), default="random", show_default=True)
@click.option("--model-type", type=click.Choice(["convgnp", "attnnp"]), default="convgnp", show_default=True)
@click.option("--grid-size", type=int, default=256, show_default=True)
@click.option("--grid-feat-dim", type=int, default=128, show_default=True)
@click.option("--conv-layers", type=int, default=6, show_default=True)
@click.option("--latent-dim", type=int, default=16, show_default=True)
@click.option("--attn-hidden-dim", type=int, default=128, show_default=True)
@click.option("--attn-heads", type=int, default=4, show_default=True)
@click.option("--attn-layers", type=int, default=2, show_default=True)
@click.option("--attn-dropout", type=float, default=0.1, show_default=True)
@click.option("--lambda-recon", type=float, default=1.0, show_default=True)
@click.option("--lambda-cls", type=float, default=1.0, show_default=True)
@click.option("--lambda-morph", type=float, default=0.15, show_default=True)
@click.option("--beta-kl", type=float, default=1e-3, show_default=True)
@click.option("--focal-gamma", type=float, default=None)
@click.option("--max-obs", type=int, default=200, show_default=True)
@click.option("--keep-all-snr-gt", type=float, default=5.0, show_default=True)
@click.option("--device", type=str, default=None)
def crossval_mallorn(
    data_dir: Path,
    out_dir: Path,
    folds: str,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
    num_workers: int,
    context_strategy: str,
    model_type: str,
    grid_size: int,
    grid_feat_dim: int,
    conv_layers: int,
    latent_dim: int,
    attn_hidden_dim: int,
    attn_heads: int,
    attn_layers: int,
    attn_dropout: float,
    lambda_recon: float,
    lambda_cls: float,
    lambda_morph: float,
    beta_kl: float,
    focal_gamma: float | None,
    max_obs: int,
    keep_all_snr_gt: float,
    device: str | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lc_raw, log_raw = load_all_data(data_dir)
    lc, log = preprocess_mallorn_training_tables(lc_raw, log_raw, max_obs=max_obs, keep_all_snr_gt=keep_all_snr_gt)
    available_folds = sorted(log["split"].dropna().unique().tolist()) if "split" in log.columns else []
    selected_folds = available_folds if folds == "all" else [f.strip() for f in folds.split(",") if f.strip()]
    selected_folds = [f for f in selected_folds if f in available_folds]
    if not selected_folds:
        raise click.ClickException(f"No valid folds selected. Available folds: {available_folds}")

    click.echo(f"[cv] model_type={model_type} folds={','.join(selected_folds)} device={device}")
    oof_parts: list[pd.DataFrame] = []
    fold_summaries: list[dict] = []
    for fold_idx, fold_name in enumerate(selected_folds, start=1):
        fold_dir = out_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        val_ids = log.loc[log["split"] == fold_name, "object_id"].tolist()
        train_ids = log.loc[log["split"] != fold_name, "object_id"].tolist()
        _, _, train_ds, val_ds, train_ids, val_ids = build_mallorn_datasets_from_ids(lc, log, train_ids, val_ids)
        z_min, z_max = compute_z_norm_stats(train_ds)
        morph_mean, morph_std = compute_morphology_norm_stats(train_ds)
        flux_center_by_band = train_ds.flux_center_by_band
        flux_scale_by_band = train_ds.flux_scale_by_band
        base_collate_cfg = MallornCollateConfig(
            seed=seed + fold_idx,
            context_strategy=context_strategy,
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
        if model_type == "convgnp":
            model_cfg = ConvGNPJointConfig(grid_size=grid_size, grid_feat_dim=grid_feat_dim, conv_layers=conv_layers, latent_dim=latent_dim)
        else:
            model_cfg = AttentiveNPJointConfig(
                latent_dim=latent_dim,
                attn_hidden_dim=attn_hidden_dim,
                attn_heads=attn_heads,
                attn_layers=attn_layers,
                attn_dropout=attn_dropout,
            )
        model = build_joint_model(model_type, model_cfg).to(device)
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
            peak_weight_snr_boost=1.0,
            peak_weight_flux_boost=0.5,
            peak_weight_snr_threshold=5.0,
            peak_weight_flux_quantile=0.90,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=2, eta_min=1.5e-6)
        best_f1 = float("-inf")
        best_metrics: dict[str, float] | None = None
        best_ckpt_path = fold_dir / "best_f1_checkpoint.pt"
        patience_count = 0
        history: list[dict] = []
        click.echo(f"[cv:{fold_name}] train={len(train_ds)} val={len(val_ds)} pos_weight={pos_weight:.3f}")
        for epoch in range(1, epochs + 1):
            train_loader = _build_train_loader(train_ds, batch_size, num_workers, base_collate_cfg, epoch)
            train_metrics = fit_epoch(model, train_loader, optimizer, loss_cfg, device)
            val_metrics = evaluate_mallorn_epoch(model, val_loader, loss_cfg, device)
            scheduler.step()
            row = {"epoch": epoch, **train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            history.append(row)
            current_f1 = float(val_metrics.get("best_f1", float("-inf")))
            if epoch == 1 or epoch % 20 == 0 or current_f1 > best_f1:
                click.echo(f"[cv:{fold_name}] ep={epoch} train_total={train_metrics['total']:.4f} val_f1={current_f1:.4f} ap={val_metrics.get('ap', float('nan')):.4f}")
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_metrics = val_metrics
                patience_count = 0
                torch.save(
                    _checkpoint_payload(
                        model_type=model_type,
                        epoch=epoch,
                        phase=f"cv:{fold_name}",
                        model=model,
                        model_cfg=model_cfg,
                        loss_cfg=loss_cfg,
                        z_min=z_min,
                        z_max=z_max,
                        morph_mean=morph_mean,
                        morph_std=morph_std,
                        flux_center_by_band=flux_center_by_band,
                        flux_scale_by_band=flux_scale_by_band,
                        metrics=val_metrics,
                        composite=float("nan"),
                        history=history,
                    ),
                    best_ckpt_path,
                )
            else:
                patience_count += 1
                if patience_count >= patience:
                    break
        best_model, best_ckpt = _load_joint_checkpoint(best_ckpt_path, device)
        fold_oof = _predict_dataset_full_context(best_model, val_ds, batch_size, z_min, z_max, device)
        fold_oof["fold"] = fold_name
        oof_parts.append(fold_oof)
        fold_thr = _find_best_binary_threshold(fold_oof)
        fold_summaries.append(
            {
                "fold": fold_name,
                "checkpoint": str(best_ckpt_path),
                "val_best_f1": None if best_metrics is None else best_metrics.get("best_f1"),
                "val_ap": None if best_metrics is None else best_metrics.get("ap"),
                "full_context_best_threshold": fold_thr["best_threshold"],
                "full_context_best_f1": fold_thr["best_f1"],
                "full_context_ap": fold_thr["ap"],
            }
        )
        pd.DataFrame(history).to_json(fold_dir / "training_log.json", orient="records", indent=2)
        fold_oof.to_csv(fold_dir / "oof_predictions.csv", index=False)

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(columns=["object_id", "label", "prob_tde", "fold"])
    overall = _find_best_binary_threshold(oof) if len(oof) else {"best_threshold": 0.5, "best_f1": float("nan"), "ap": float("nan")}
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    summary = {
        "model_type": model_type,
        "folds": selected_folds,
        "oof_best_threshold": overall["best_threshold"],
        "oof_best_f1": overall["best_f1"],
        "oof_ap": overall["ap"],
        "fold_summaries": fold_summaries,
    }
    with open(out_dir / "oof_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    click.echo(
        f"[cv done] folds={len(selected_folds)} oof_best_threshold={overall['best_threshold']:.4f} "
        f"oof_best_f1={overall['best_f1']:.4f} oof_ap={overall['ap']:.4f}"
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


@main.command("evaluate-mallorn-checkpoint")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--val-frac", type=float, default=0.15, show_default=True)
@click.option("--max-obs", type=int, default=200, show_default=True)
@click.option("--keep-all-snr-gt", type=float, default=5.0, show_default=True)
@click.option("--eval-context-strategy", type=click.Choice(["prefix", "random"]), default="prefix", show_default=True)
@click.option("--min-context-points-total", type=int, default=3, show_default=True)
@click.option("--min-target-points-total", type=int, default=1, show_default=True)
@click.option("--prefix-anchor-snr-threshold", type=float, default=5.0, show_default=True)
@click.option("--prefix-anchor-cluster-min-points", type=int, default=2, show_default=True)
@click.option("--prefix-anchor-cluster-window-days", type=float, default=7.0, show_default=True)
@click.option("--prefix-anchor-require-positive-flux/--no-prefix-anchor-require-positive-flux", default=True, show_default=True)
@click.option("--prefix-pre-anchor-days", type=float, default=30.0, show_default=True)
@click.option("--prefix-context-days-min", type=float, default=7.0, show_default=True)
@click.option("--prefix-context-days-max", type=float, default=30.0, show_default=True)
@click.option("--prefix-target-horizon-days", type=float, default=90.0, show_default=True)
@click.option("--out-json", type=click.Path(dir_okay=False, path_type=Path), default=None)
def evaluate_mallorn_checkpoint(
    data_dir: Path,
    checkpoint: Path,
    batch_size: int,
    num_workers: int,
    device: str | None,
    seed: int,
    val_frac: float,
    max_obs: int,
    keep_all_snr_gt: float,
    eval_context_strategy: str,
    min_context_points_total: int,
    min_target_points_total: int,
    prefix_anchor_snr_threshold: float,
    prefix_anchor_cluster_min_points: int,
    prefix_anchor_cluster_window_days: float,
    prefix_anchor_require_positive_flux: bool,
    prefix_pre_anchor_days: float,
    prefix_context_days_min: float,
    prefix_context_days_max: float,
    prefix_target_horizon_days: float,
    out_json: Path | None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, _, train_ds, val_ds, _, _ = prepare_mallorn_datasets(
        data_dir=data_dir,
        seed=seed,
        val_frac=val_frac,
        max_obs=max_obs,
        keep_all_snr_gt=keep_all_snr_gt,
    )
    model, ckpt = _load_joint_checkpoint(checkpoint, device)
    ckpt_loss_cfg = JointLossConfig(**ckpt["loss_cfg"]) if "loss_cfg" in ckpt else JointLossConfig()
    z_min = float(ckpt.get("z_min", compute_z_norm_stats(train_ds)[0]))
    z_max = float(ckpt.get("z_max", compute_z_norm_stats(train_ds)[1]))
    morph_mean = ckpt.get("morph_mean")
    morph_std = ckpt.get("morph_std")
    eval_collate_cfg = MallornCollateConfig(
        seed=seed,
        context_strategy=eval_context_strategy,
        min_ctx_points_total=min_context_points_total,
        min_target_points_total=min_target_points_total,
        prefix_anchor_snr_threshold=prefix_anchor_snr_threshold,
        prefix_anchor_cluster_min_points=prefix_anchor_cluster_min_points,
        prefix_anchor_cluster_window_days=prefix_anchor_cluster_window_days,
        prefix_anchor_require_positive_flux=prefix_anchor_require_positive_flux,
        prefix_pre_anchor_days=prefix_pre_anchor_days,
        prefix_context_days_min=prefix_context_days_min,
        prefix_context_days_max=prefix_context_days_max,
        prefix_target_horizon_days=prefix_target_horizon_days,
        z_min=z_min,
        z_max=z_max,
        morph_mean=None if morph_mean is None else torch.as_tensor(morph_mean).cpu().numpy(),
        morph_std=None if morph_std is None else torch.as_tensor(morph_std).cpu().numpy(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=make_mallorn_collate(training=False, cfg=eval_collate_cfg),
    )
    metrics = evaluate_mallorn_epoch(model, val_loader, ckpt_loss_cfg, device)
    click.echo(
        f"[eval] checkpoint={checkpoint} strategy={eval_context_strategy} "
        f"best_f1={_fmt_metric(metrics.get('best_f1'))} ap={_fmt_metric(metrics.get('ap'))} "
        f"val_nll={_fmt_metric(metrics.get('val_nll'))} rmse_pk={_fmt_metric(metrics.get('rmse_tde_peak'))}"
    )
    if "interesting_best_f1" in metrics or "interesting_ap" in metrics:
        click.echo(
            f"  interesting: best_f1={_fmt_metric(metrics.get('interesting_best_f1'))} "
            f"ap={_fmt_metric(metrics.get('interesting_ap'))} "
            f"thr={_fmt_metric(metrics.get('interesting_best_threshold'))}"
        )
    for limit in (3, 5, 10, 20, 40, 80):
        f1_key = f"best_f1_ctx_le_{limit}"
        ap_key = f"ap_ctx_le_{limit}"
        n_key = f"n_ctx_le_{limit}"
        if f1_key in metrics or ap_key in metrics or n_key in metrics:
            click.echo(
                f"  ctx<={limit:>3}: n={_fmt_metric(metrics.get(n_key))} "
                f"best_f1={_fmt_metric(metrics.get(f1_key))} ap={_fmt_metric(metrics.get(ap_key))}"
            )
    for limit in (7, 30, 60, 90, 120, 180):
        f1_key = f"best_f1_days_le_{limit}"
        ap_key = f"ap_days_le_{limit}"
        n_key = f"n_days_le_{limit}"
        if f1_key in metrics or ap_key in metrics or n_key in metrics:
            click.echo(
                f"  days<={limit:>3}: n={_fmt_metric(metrics.get(n_key))} "
                f"best_f1={_fmt_metric(metrics.get(f1_key))} ap={_fmt_metric(metrics.get(ap_key))}"
            )
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        click.echo(f"[done] metrics -> {out_json}")


@main.command("predict-mallorn-test")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-csv", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--threshold", type=float, default=None, help="Probability threshold used to convert p(TDE) into binary predictions. Defaults to the checkpoint's saved best threshold, or 0.5 if unavailable.")
@click.option("--write-probabilities/--write-binary", default=False, show_default=True, help="Write raw probabilities instead of the required binary submission labels.")
def predict_mallorn_test(
    data_dir: Path,
    checkpoint: Path,
    out_csv: Path,
    batch_size: int,
    device: str | None,
    threshold: float,
    write_probabilities: bool,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_joint_checkpoint(checkpoint, device)
    train_lc, train_log = load_all_data(data_dir)
    train_lc = apply_ebv_correction(train_lc, train_log)
    train_ds = MallornDataset(train_log["object_id"].tolist(), train_lc, train_log)
    z_min = float(ckpt.get("z_min", compute_z_norm_stats(train_ds)[0]))
    z_max = float(ckpt.get("z_max", compute_z_norm_stats(train_ds)[1]))
    flux_center_by_band = np.asarray(ckpt.get("flux_center_by_band", getattr(train_ds, "flux_center_by_band", np.zeros(6))), dtype=np.float32)
    flux_scale_by_band = np.asarray(ckpt.get("flux_scale_by_band", getattr(train_ds, "flux_scale_by_band", np.ones(6))), dtype=np.float32)

    test_lc, test_log = _load_mallorn_test_data(data_dir)
    test_lc = apply_ebv_correction(test_lc, test_log)
    test_ds = MallornDataset(
        test_log["object_id"].tolist(),
        test_lc,
        test_log,
        flux_center_by_band=flux_center_by_band,
        flux_scale_by_band=flux_scale_by_band,
    )
    object_ids = list(test_ds.object_ids)
    rows: list[dict[str, float | str]] = []
    if threshold is None:
        threshold = float(ckpt.get("metrics", {}).get("best_threshold", 0.5))
    click.echo(
        f"[predict] model={checkpoint} test_objects={len(object_ids)} device={device} "
        f"mode={'probabilities' if write_probabilities else 'binary'} threshold={threshold:.3f}"
    )
    for start in range(0, len(object_ids), batch_size):
        batch_ids = object_ids[start : start + batch_size]
        items = []
        for oid in batch_ids:
            item = test_ds.data[oid].copy()
            item["oid"] = oid
            items.append(item)
        batch = _build_full_context_batch(items, z_min, z_max).to(device)
        out = model(batch)
        probs = torch.sigmoid(out.class_logits).detach().cpu().numpy() if out.class_logits is not None else np.full(len(batch_ids), np.nan)
        for oid, prob in zip(batch_ids, probs):
            pred = float(prob) if write_probabilities else int(float(prob) >= threshold)
            rows.append({"object_id": oid, "prediction": pred})
        if start == 0 or ((start // batch_size) + 1) % 20 == 0:
            click.echo(f"[predict] processed={min(start + len(batch_ids), len(object_ids))}/{len(object_ids)}")

    submission = pd.DataFrame(rows)
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        submission = sample[["object_id"]].merge(submission, on="object_id", how="left")
        submission["prediction"] = submission["prediction"].fillna(0.0)
        if not write_probabilities:
            submission["prediction"] = submission["prediction"].astype(int)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_csv, index=False)
    click.echo(f"[done] submission -> {out_csv}")


@main.command("predict-mallorn-test-ensemble")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--checkpoint", "checkpoints", type=click.Path(exists=True, dir_okay=False, path_type=Path), multiple=True, required=True)
@click.option("--out-csv", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--threshold", type=float, default=None)
@click.option("--threshold-json", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Optional OOF summary JSON providing `oof_best_threshold`.")
@click.option("--write-probabilities/--write-binary", default=False, show_default=True)
def predict_mallorn_test_ensemble(
    data_dir: Path,
    checkpoints: tuple[Path, ...],
    out_csv: Path,
    batch_size: int,
    device: str | None,
    threshold: float | None,
    threshold_json: Path | None,
    write_probabilities: bool,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if threshold is None and threshold_json is not None:
        with open(threshold_json) as f:
            threshold = float(json.load(f).get("oof_best_threshold", 0.5))
    if threshold is None:
        threshold = 0.5

    test_lc, test_log = _load_mallorn_test_data(data_dir)
    test_lc = apply_ebv_correction(test_lc, test_log)
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        object_ids = pd.read_csv(sample_path)["object_id"].tolist()
    else:
        object_ids = test_log["object_id"].tolist()

    prob_cols = []
    for checkpoint in checkpoints:
        model, ckpt = _load_joint_checkpoint(checkpoint, device)
        flux_center_by_band = np.asarray(ckpt.get("flux_center_by_band", np.zeros(6)), dtype=np.float32)
        flux_scale_by_band = np.asarray(ckpt.get("flux_scale_by_band", np.ones(6)), dtype=np.float32)
        test_ds = MallornDataset(
            test_log["object_id"].tolist(),
            test_lc,
            test_log,
            flux_center_by_band=flux_center_by_band,
            flux_scale_by_band=flux_scale_by_band,
        )
        z_min = float(ckpt.get("z_min", 0.0))
        z_max = float(ckpt.get("z_max", 1.0))
        pred_df = _predict_dataset_full_context(model, test_ds, batch_size, z_min, z_max, device)[["object_id", "prob_tde"]]
        pred_df = pred_df.rename(columns={"prob_tde": checkpoint.stem})
        prob_cols.append(pred_df)
        click.echo(f"[ensemble] checkpoint={checkpoint} done")

    merged = pd.DataFrame({"object_id": object_ids})
    for df in prob_cols:
        merged = merged.merge(df, on="object_id", how="left")
    prob_matrix = merged.drop(columns=["object_id"]).to_numpy(dtype=float)
    mean_prob = np.nanmean(prob_matrix, axis=1)
    merged["prediction"] = mean_prob if write_probabilities else (mean_prob >= threshold).astype(int)
    submission = merged[["object_id", "prediction"]].copy()
    if not write_probabilities:
        submission["prediction"] = submission["prediction"].astype(int)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_csv, index=False)
    click.echo(
        f"[done] ensemble_submission -> {out_csv} checkpoints={len(checkpoints)} "
        f"mode={'probabilities' if write_probabilities else 'binary'} threshold={threshold:.4f}"
    )


if __name__ == '__main__':
    main()
