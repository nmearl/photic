#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import click

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from photic.batch import NPBatch
from photic.config import JointLossConfig
from photic.data import (
    BANDS,
    BAND2IDX,
    MallornCollateConfig,
    compute_z_norm_stats,
    compute_morphology_norm_stats,
    make_mallorn_collate,
    prepare_mallorn_datasets,
)
from photic.model import load_joint_model_checkpoint
from photic.train import evaluate_binary_predictions, evaluate_mallorn_epoch

BAND_COLORS = {
    "u": "tab:purple",
    "g": "tab:green",
    "r": "tab:red",
    "i": "tab:orange",
    "z": "tab:brown",
    "y": "tab:gray",
}


def _to_device(batch: NPBatch, device: str | torch.device) -> NPBatch:
    return batch.to(device)


def _load_checkpoint(checkpoint_path: Path, device: str | torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, ckpt = load_joint_model_checkpoint(checkpoint_path, device=device)
    return model, ckpt


def _loss_cfg_from_ckpt(ckpt: dict[str, Any]) -> JointLossConfig:
    if "loss_cfg" in ckpt:
        return JointLossConfig(**ckpt["loss_cfg"])
    return JointLossConfig()


@torch.no_grad()
def collect_validation_outputs(model: torch.nn.Module, loader: DataLoader, device: str | torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_logits, all_labels = [], []
    for batch in loader:
        batch_dev = _to_device(batch, device)
        out = model(batch_dev)
        probs = torch.sigmoid(out.class_logits).detach().cpu().numpy() if out.class_logits is not None else np.full(batch.context_x.shape[0], np.nan)
        labels = batch.labels.detach().cpu().numpy().astype(int) if batch.labels is not None else np.full(batch.context_x.shape[0], -1)
        if out.class_logits is not None and batch.labels is not None:
            all_logits.append(out.class_logits.detach().cpu())
            all_labels.append(batch.labels.detach().cpu())
        pred_sigma = torch.sqrt(out.pred_var.clamp_min(1e-8)).detach()
        obs_snr = batch.metadata.get("obs_snr") if batch.metadata else None
        oids = batch.metadata.get("oid") if batch.metadata else [f"obj_{i}" for i in range(batch.target_x.shape[0])]
        for i in range(batch.target_x.shape[0]):
            mask = batch.target_mask[i].bool()
            if mask.sum().item() == 0:
                rmse = rmse_peak = mean_sigma = float("nan")
                n_targets = n_peak = 0
            else:
                y_true = batch.target_y[i][mask].detach().cpu().numpy()
                y_pred = out.pred_mean[i][mask].detach().cpu().numpy()
                sigma = pred_sigma[i][mask].detach().cpu().numpy()
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                mean_sigma = float(np.mean(sigma))
                snr = obs_snr[i][mask].detach().cpu().numpy() if obs_snr is not None else np.zeros_like(y_true)
                peak_mask = snr > 5.0
                n_targets = int(mask.sum().item())
                n_peak = int(peak_mask.sum())
                rmse_peak = float(np.sqrt(np.mean((y_true[peak_mask] - y_pred[peak_mask]) ** 2))) if peak_mask.sum() > 0 else float("nan")
            rows.append({
                "object_id": oids[i], "label": int(labels[i]), "prob_tde": float(probs[i]), "pred_tde_05": int(probs[i] >= 0.5) if np.isfinite(probs[i]) else -1,
                "n_context": int(batch.context_mask[i].sum().item()), "n_targets": n_targets, "n_peak_targets": n_peak,
                "rmse": rmse, "rmse_peak": rmse_peak, "mean_pred_sigma": mean_sigma,
            })
    df = pd.DataFrame(rows)
    if all_logits and all_labels:
        cls_metrics = evaluate_binary_predictions(torch.cat(all_logits), torch.cat(all_labels))
        for k, v in cls_metrics.items():
            df.attrs[k] = v
    return df


@torch.no_grad()
def make_full_reconstruction_batch(item: dict[str, Any], z_min: float, z_max: float, grid_points_per_band: int = 256) -> tuple[NPBatch, dict[str, np.ndarray]]:
    context_x = torch.as_tensor(item["t_norm"], dtype=torch.float32).unsqueeze(0)
    context_y = torch.as_tensor(item["flux_norm"], dtype=torch.float32).unsqueeze(0)
    context_yerr = torch.as_tensor(item["ferr_norm"], dtype=torch.float32).unsqueeze(0)
    context_band = torch.as_tensor(item["band_idx"], dtype=torch.long).unsqueeze(0)
    context_mask = torch.ones_like(context_x)
    tq_list, tb_list = [], []
    for bi in range(len(BANDS)):
        tq = np.linspace(0.0, 1.0, grid_points_per_band, dtype=np.float32)
        tq_list.append(tq)
        tb_list.append(np.full(grid_points_per_band, bi, dtype=np.int64))
    target_x = torch.as_tensor(np.concatenate(tq_list), dtype=torch.float32).unsqueeze(0)
    target_band = torch.as_tensor(np.concatenate(tb_list), dtype=torch.long).unsqueeze(0)
    target_y = torch.zeros_like(target_x)
    target_yerr = torch.ones_like(target_x)
    target_mask = torch.ones_like(target_x)
    z = np.array([item.get("z", np.nan)], dtype=np.float32)
    z_norm = (z - z_min) / max(z_max - z_min, 1e-6)
    z_norm = np.clip(z_norm, 0.0, 1.0)
    z_norm[~np.isfinite(z_norm)] = 0.5
    batch = NPBatch(
        context_x=context_x, context_y=context_y, context_yerr=context_yerr, context_band=context_band, context_mask=context_mask,
        target_x=target_x, target_y=target_y, target_yerr=target_yerr, target_band=target_band, target_mask=target_mask,
        labels=torch.tensor([float(item["target"])], dtype=torch.float32), redshift=torch.as_tensor(z_norm, dtype=torch.float32), metadata={"oid": [item.get("oid", "unknown")]},
    )
    return batch, {"query_t_norm": target_x.squeeze(0).cpu().numpy(), "query_band_idx": target_band.squeeze(0).cpu().numpy()}


@torch.no_grad()
def reconstruct_object(model: torch.nn.Module, item: dict[str, Any], device: str | torch.device, z_min: float, z_max: float, grid_points_per_band: int = 256, latent_samples: int = 0) -> dict[str, Any]:
    batch, aux = make_full_reconstruction_batch(item, z_min=z_min, z_max=z_max, grid_points_per_band=grid_points_per_band)
    batch = batch.to(device)
    out = model(batch)
    prob = float(torch.sigmoid(out.class_logits)[0].detach().cpu().item()) if out.class_logits is not None else float("nan")
    q_t = aux["query_t_norm"]
    q_b = aux["query_band_idx"]
    pred_mean = out.pred_mean[0].detach().cpu().numpy()
    pred_sigma = torch.sqrt(out.pred_var[0].clamp_min(1e-8)).detach().cpu().numpy()
    query_centers = item["flux_center_by_band"][q_b]
    query_scales = item["flux_scale_by_band"][q_b]
    sample_means_raw = None
    if latent_samples and getattr(model.cfg, 'use_latent', False):
        means_s, _ = model.sample_predictions(batch, num_samples=latent_samples)
        sample_means_raw = means_s[:, 0].detach().cpu().numpy() * query_scales[None, :] + query_centers[None, :]
    pred_mean_raw = pred_mean * query_scales + query_centers
    pred_sigma_raw = pred_sigma * query_scales
    pred_time_raw = q_t * item["t_span"] + item["t_min"]
    obs_centers = item["flux_center_by_band"][item["band_idx"]]
    obs_scales = item["flux_scale_by_band"][item["band_idx"]]
    return {
        "prob_tde": prob, "pred_time_raw": pred_time_raw, "pred_band_idx": q_b,
        "pred_mean_raw": pred_mean_raw, "pred_sigma_raw": pred_sigma_raw, "latent_sample_means_raw": sample_means_raw,
        "obs_time_raw": item["t_raw"], "obs_flux_raw": item["flux_norm"] * obs_scales + obs_centers, "obs_fluxerr_raw": item["ferr_norm"] * obs_scales, "obs_band_idx": item["band_idx"],
        "label": item["target"], "z": item.get("z", np.nan), "oid": item.get("oid", "unknown"),
    }


@torch.no_grad()
def reconstruct_from_context(
    model: torch.nn.Module,
    batch: NPBatch,
    item: dict[str, Any],
    device: str | torch.device,
    z_min: float,
    z_max: float,
    grid_points_per_band: int = 256,
    latent_samples: int = 0,
) -> dict[str, Any]:
    cx = batch.context_x[0][batch.context_mask[0] > 0].detach().cpu().numpy()
    cy = batch.context_y[0][batch.context_mask[0] > 0].detach().cpu().numpy()
    ce = batch.context_yerr[0][batch.context_mask[0] > 0].detach().cpu().numpy()
    cb = batch.context_band[0][batch.context_mask[0] > 0].detach().cpu().numpy()
    tx_obs = batch.target_x[0][batch.target_mask[0] > 0].detach().cpu().numpy()
    ty_obs = batch.target_y[0][batch.target_mask[0] > 0].detach().cpu().numpy()
    te_obs = batch.target_yerr[0][batch.target_mask[0] > 0].detach().cpu().numpy()
    tb_obs = batch.target_band[0][batch.target_mask[0] > 0].detach().cpu().numpy()

    tq_list, tb_list = [], []
    for bi in range(len(BANDS)):
        tq = np.linspace(0.0, 1.0, grid_points_per_band, dtype=np.float32)
        tq_list.append(tq)
        tb_list.append(np.full(grid_points_per_band, bi, dtype=np.int64))
    query_t = np.concatenate(tq_list)
    query_b = np.concatenate(tb_list)
    z = np.array([item.get("z", np.nan)], dtype=np.float32)
    z_norm = (z - z_min) / max(z_max - z_min, 1e-6)
    z_norm = np.clip(z_norm, 0.0, 1.0)
    z_norm[~np.isfinite(z_norm)] = 0.5
    dense = NPBatch(
        context_x=torch.as_tensor(cx, dtype=torch.float32).unsqueeze(0),
        context_y=torch.as_tensor(cy, dtype=torch.float32).unsqueeze(0),
        context_yerr=torch.as_tensor(ce, dtype=torch.float32).unsqueeze(0),
        context_band=torch.as_tensor(cb, dtype=torch.long).unsqueeze(0),
        context_mask=torch.ones((1, len(cx)), dtype=torch.float32),
        target_x=torch.as_tensor(query_t, dtype=torch.float32).unsqueeze(0),
        target_y=torch.zeros((1, len(query_t)), dtype=torch.float32),
        target_yerr=torch.ones((1, len(query_t)), dtype=torch.float32),
        target_band=torch.as_tensor(query_b, dtype=torch.long).unsqueeze(0),
        target_mask=torch.ones((1, len(query_t)), dtype=torch.float32),
        labels=torch.tensor([float(item["target"])], dtype=torch.float32),
        redshift=torch.as_tensor(z_norm, dtype=torch.float32),
        metadata={"oid": [item.get("oid", "unknown")]},
    ).to(device)
    out = model(dense)
    prob_tde = float(torch.sigmoid(out.class_logits)[0].detach().cpu().item()) if out.class_logits is not None else float("nan")
    prob_interesting = float(torch.sigmoid(out.interesting_logits)[0].detach().cpu().item()) if getattr(out, "interesting_logits", None) is not None else float("nan")
    pred_mean = out.pred_mean[0].detach().cpu().numpy()
    pred_sigma = torch.sqrt(out.pred_var[0].clamp_min(1e-8)).detach().cpu().numpy()
    centers_q = item["flux_center_by_band"][query_b]
    scales_q = item["flux_scale_by_band"][query_b]
    pred_mean_raw = pred_mean * scales_q + centers_q
    pred_sigma_raw = pred_sigma * scales_q
    pred_time_raw = query_t * item["t_span"] + item["t_min"]

    centers_ctx = item["flux_center_by_band"][cb]
    scales_ctx = item["flux_scale_by_band"][cb]
    centers_tgt = item["flux_center_by_band"][tb_obs]
    scales_tgt = item["flux_scale_by_band"][tb_obs]
    return {
        "oid": item["oid"],
        "label": item["target"],
        "z": item.get("z", np.nan),
        "prob_tde": prob_tde,
        "prob_interesting": prob_interesting,
        "context_time_raw": cx * item["t_span"] + item["t_min"],
        "context_flux_raw": cy * scales_ctx + centers_ctx,
        "context_fluxerr_raw": ce * scales_ctx,
        "context_band_idx": cb,
        "target_time_raw": tx_obs * item["t_span"] + item["t_min"],
        "target_flux_raw": ty_obs * scales_tgt + centers_tgt,
        "target_fluxerr_raw": te_obs * scales_tgt,
        "target_band_idx": tb_obs,
        "pred_time_raw": pred_time_raw,
        "pred_mean_raw": pred_mean_raw,
        "pred_sigma_raw": pred_sigma_raw,
        "pred_band_idx": query_b,
    }


def choose_tdes(df: pd.DataFrame, n: int, mode: str, seed: int) -> list[str]:
    tdes = df[df["label"] == 1].copy()
    if len(tdes) == 0:
        return []
    if n >= len(tdes):
        return tdes["object_id"].tolist()
    rng = np.random.default_rng(seed)
    if mode == "random":
        return tdes.sample(n=n, random_state=seed)["object_id"].tolist()
    if mode == "hard":
        return tdes.sort_values(["prob_tde", "rmse_peak"], ascending=[True, False]).head(n)["object_id"].tolist()
    if mode == "easy":
        return tdes.sort_values(["prob_tde", "rmse_peak"], ascending=[False, True]).head(n)["object_id"].tolist()
    tdes = tdes.sort_values("prob_tde").reset_index(drop=True)
    idx = np.linspace(0, len(tdes) - 1, n).round().astype(int)
    idx = np.unique(idx)
    chosen = tdes.iloc[idx]["object_id"].tolist()
    if len(chosen) < n:
        remaining = [oid for oid in tdes["object_id"].tolist() if oid not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: n - len(chosen)])
    return chosen[:n]


def plot_reconstructions(model: torch.nn.Module, dataset, object_ids: list[str], device: str | torch.device, z_min: float, z_max: float, output_path: Path, grid_points_per_band: int = 256, cols: int = 2, latent_samples: int = 0):
    n = len(object_ids)
    cols = min(cols, max(n, 1))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.axis("off")
    for ax, oid in zip(axes_flat, object_ids):
        item = dataset.data[oid].copy(); item["oid"] = oid
        recon = reconstruct_object(model, item, device=device, z_min=z_min, z_max=z_max, grid_points_per_band=grid_points_per_band, latent_samples=latent_samples)
        for band in BANDS:
            bi = BAND2IDX[band]
            obs_mask = recon["obs_band_idx"] == bi
            if np.any(obs_mask):
                ax.errorbar(recon["obs_time_raw"][obs_mask], recon["obs_flux_raw"][obs_mask], yerr=recon["obs_fluxerr_raw"][obs_mask], fmt="o", ms=3, alpha=0.75, color=BAND_COLORS[band], label=f"{band} obs")
            pred_mask = recon["pred_band_idx"] == bi
            if np.any(pred_mask):
                order = np.argsort(recon["pred_time_raw"][pred_mask])
                tx = recon["pred_time_raw"][pred_mask][order]
                mu = recon["pred_mean_raw"][pred_mask][order]
                sg = recon["pred_sigma_raw"][pred_mask][order]
                if recon["latent_sample_means_raw"] is not None:
                    for s in recon["latent_sample_means_raw"]:
                        ax.plot(tx, s[pred_mask][order], color=BAND_COLORS[band], lw=0.9, alpha=0.12)
                ax.plot(tx, mu, color=BAND_COLORS[band], lw=1.7, alpha=0.95)
                ax.fill_between(tx, mu - sg, mu + sg, color=BAND_COLORS[band], alpha=0.12)
        title = f"{oid}\nlabel={recon['label']}  p(TDE)={recon['prob_tde']:.3f}"
        if np.isfinite(recon['z']):
            title += f"  z={recon['z']:.3f}"
        ax.set_title(title)
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux (μJy, EBV-corrected)")
        ax.grid(alpha=0.2)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    if dedup:
        fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=min(len(dedup), 6), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prefix_reconstructions(
    model: torch.nn.Module,
    dataset,
    object_ids: list[str],
    collate_cfg: MallornCollateConfig,
    device: str | torch.device,
    z_min: float,
    z_max: float,
    output_path: Path,
    grid_points_per_band: int = 256,
    cols: int = 2,
):
    n = len(object_ids)
    cols = min(cols, max(n, 1))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.axis("off")
    collate = make_mallorn_collate(training=False, cfg=collate_cfg)
    for ax, oid in zip(axes_flat, object_ids):
        item = dataset[dataset.object_ids.index(oid)]
        batch = collate([item])
        recon = reconstruct_from_context(model, batch, item, device=device, z_min=z_min, z_max=z_max, grid_points_per_band=grid_points_per_band)
        for band in BANDS:
            bi = BAND2IDX[band]
            ctx_mask = recon["context_band_idx"] == bi
            tgt_mask = recon["target_band_idx"] == bi
            pred_mask = recon["pred_band_idx"] == bi
            if np.any(ctx_mask):
                ax.errorbar(recon["context_time_raw"][ctx_mask], recon["context_flux_raw"][ctx_mask], yerr=recon["context_fluxerr_raw"][ctx_mask], fmt="o", ms=4, alpha=0.9, color=BAND_COLORS[band], label=f"{band} context")
            if np.any(tgt_mask):
                ax.errorbar(recon["target_time_raw"][tgt_mask], recon["target_flux_raw"][tgt_mask], yerr=recon["target_fluxerr_raw"][tgt_mask], fmt="x", ms=4, alpha=0.55, color=BAND_COLORS[band], label=f"{band} heldout")
            if np.any(pred_mask):
                order = np.argsort(recon["pred_time_raw"][pred_mask])
                tx = recon["pred_time_raw"][pred_mask][order]
                mu = recon["pred_mean_raw"][pred_mask][order]
                sg = recon["pred_sigma_raw"][pred_mask][order]
                ax.plot(tx, mu, color=BAND_COLORS[band], lw=1.7, alpha=0.95)
                ax.fill_between(tx, mu - sg, mu + sg, color=BAND_COLORS[band], alpha=0.10)
        title = f"{oid}\nlabel={recon['label']}  p(TDE)={recon['prob_tde']:.3f}"
        if np.isfinite(recon["prob_interesting"]):
            title += f"  p(int)={recon['prob_interesting']:.3f}"
        ax.set_title(title)
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux (μJy, EBV-corrected)")
        ax.grid(alpha=0.2)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    if dedup:
        fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=min(len(dedup), 6), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@click.command()
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--val-frac", type=float, default=0.15, show_default=True)
@click.option("--max-obs", type=int, default=200, show_default=True)
@click.option("--keep-all-snr-gt", type=float, default=5.0, show_default=True)
@click.option("--num-tdes", type=int, default=6, show_default=True)
@click.option("--selection", type=click.Choice(["spread", "hard", "easy", "random"]), default="spread", show_default=True)
@click.option("--grid-points-per-band", type=int, default=256, show_default=True)
@click.option("--plot-cols", type=int, default=2, show_default=True)
@click.option("--latent-samples", type=int, default=0, show_default=True, help="Overlay latent reconstruction samples on the plot.")
@click.option("--context-strategy", type=click.Choice(["random", "prefix"]), default="random", show_default=True)
def main(data_dir: Path, checkpoint: Path, out_dir: Path, batch_size: int, num_workers: int, device: str | None, seed: int, val_frac: float, max_obs: int, keep_all_snr_gt: float, num_tdes: int, selection: str, grid_points_per_band: int, plot_cols: int, latent_samples: int, context_strategy: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_checkpoint(checkpoint, device=device)
    loss_cfg = _loss_cfg_from_ckpt(ckpt)
    _, log, train_ds, val_ds, train_ids, val_ids = prepare_mallorn_datasets(data_dir=data_dir, seed=seed, val_frac=val_frac, max_obs=max_obs, keep_all_snr_gt=keep_all_snr_gt)
    z_min = float(ckpt.get("z_min", compute_z_norm_stats(train_ds)[0]))
    z_max = float(ckpt.get("z_max", compute_z_norm_stats(train_ds)[1]))
    morph_mean = np.asarray(ckpt.get("morph_mean"), dtype=np.float32) if ckpt.get("morph_mean") is not None else None
    morph_std = np.asarray(ckpt.get("morph_std"), dtype=np.float32) if ckpt.get("morph_std") is not None else None
    if morph_mean is None or morph_std is None:
        morph_mean, morph_std = compute_morphology_norm_stats(train_ds)
    collate_cfg = MallornCollateConfig(seed=seed, context_strategy=context_strategy, z_min=z_min, z_max=z_max, morph_mean=morph_mean, morph_std=morph_std)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=make_mallorn_collate(training=False, cfg=collate_cfg), pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    metrics = evaluate_mallorn_epoch(model, val_loader, loss_cfg, device)
    per_object = collect_validation_outputs(model, val_loader, device)
    per_object = per_object.sort_values(["label", "prob_tde", "rmse_peak"], ascending=[False, False, False]).reset_index(drop=True)
    selected_ids = choose_tdes(per_object, n=num_tdes, mode=selection, seed=seed)
    fig_path = out_dir / "tde_reconstructions.png"
    if selected_ids:
        plot_reconstructions(model, val_ds, selected_ids, device=device, z_min=z_min, z_max=z_max, output_path=fig_path, grid_points_per_band=grid_points_per_band, cols=plot_cols, latent_samples=latent_samples)
    prefix_fig_path = out_dir / "tde_prefix_reconstructions.png"
    if selected_ids and context_strategy == "prefix":
        plot_prefix_reconstructions(model, val_ds, selected_ids, collate_cfg=collate_cfg, device=device, z_min=z_min, z_max=z_max, output_path=prefix_fig_path, grid_points_per_band=grid_points_per_band, cols=plot_cols)
    summary = {
        "checkpoint": str(checkpoint), "checkpoint_epoch": ckpt.get("epoch"), "checkpoint_metrics": ckpt.get("metrics", {}), "checkpoint_composite": ckpt.get("composite"), "model_cfg": ckpt.get("model_cfg", {}), "loss_cfg": ckpt.get("loss_cfg", {}),
        "data": {"data_dir": str(data_dir), "n_train": len(train_ds), "n_val": len(val_ds), "n_val_tde": int((log[log["object_id"].isin(val_ids)]["target"] == 1).sum()), "z_min": z_min, "z_max": z_max, "seed": seed},
        "validation_metrics": metrics, "selected_tdes": selected_ids,
    }
    (out_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2))
    per_object.to_csv(out_dir / "val_object_metrics.csv", index=False)
    per_object[per_object["label"] == 0].sort_values("prob_tde", ascending=False).head(25).to_csv(out_dir / "top_false_positive_candidates.csv", index=False)
    per_object[per_object["label"] == 1].sort_values("prob_tde", ascending=True).head(25).to_csv(out_dir / "top_false_negative_candidates.csv", index=False)
    click.echo(f"[done] summary -> {out_dir / 'diagnostics_summary.json'}")
    click.echo(f"[done] TDE reconstruction figure -> {fig_path if selected_ids else 'no TDEs selected'}")
    if context_strategy == "prefix":
        click.echo(f"[done] prefix reconstruction figure -> {prefix_fig_path if selected_ids else 'no TDEs selected'}")
    click.echo(f"[done] selected TDEs: {selected_ids}")
    for k, v in metrics.items():
        if isinstance(v, float):
            click.echo(f"  {k:20s}: {v:.6f}" if math.isfinite(v) else f"  {k:20s}: nan")
        else:
            click.echo(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
