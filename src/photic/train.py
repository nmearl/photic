from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .batch import NPBatch
from .config import JointLossConfig
from .losses import JointLosses, joint_loss
from .model import ConvGNPJointModel


@torch.no_grad()
def evaluate_binary_predictions(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy().astype(int)
    pred05 = (probs >= 0.5).astype(int)
    out = {
        "f1@0.5": float(f1_score(y, pred05, zero_division=0)),
        "precision@0.5": float(precision_score(y, pred05, zero_division=0)),
        "recall@0.5": float(recall_score(y, pred05, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y, probs))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["ap"] = float(average_precision_score(y, probs))
    except Exception:
        out["ap"] = float("nan")

    best_f1, best_thr, best_prec, best_rec = -1.0, 0.5, 0.0, 0.0
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (probs >= thr).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
            best_prec = float(precision_score(y, pred, zero_division=0))
            best_rec = float(recall_score(y, pred, zero_division=0))
    out["best_f1"] = best_f1
    out["best_threshold"] = best_thr
    out["precision@best"] = best_prec
    out["recall@best"] = best_rec
    return out


def train_step(model: ConvGNPJointModel, batch: NPBatch, optimizer: torch.optim.Optimizer, loss_cfg: JointLossConfig) -> JointLosses:
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    losses = joint_loss(out, batch, loss_cfg)
    losses.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return losses


@torch.no_grad()
def eval_step(model: ConvGNPJointModel, batch: NPBatch, loss_cfg: JointLossConfig) -> tuple[JointLosses, dict[str, float]]:
    out = model(batch)
    losses = joint_loss(out, batch, loss_cfg)
    metrics = {}
    if out.class_logits is not None and batch.labels is not None:
        metrics.update(evaluate_binary_predictions(out.class_logits, batch.labels))
    return losses, metrics


def _mean_dict(list_of_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not list_of_dicts:
        return {}
    keys = list_of_dicts[0].keys()
    return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}


def fit_epoch(model: ConvGNPJointModel, loader, optimizer, loss_cfg: JointLossConfig, device: str | torch.device) -> dict[str, float]:
    model.train()
    metrics = []
    for batch in loader:
        batch = batch.to(device)
        losses = train_step(model, batch, optimizer, loss_cfg)
        metrics.append({
            "total": float(losses.total.item()),
            "recon": float(losses.recon.item()),
            "cls": float(losses.cls.item()),
            "morph": float(losses.morph.item()),
            "kl": float(losses.kl.item()),
        })
    return _mean_dict(metrics)


@torch.no_grad()
def evaluate_epoch(model: ConvGNPJointModel, loader, loss_cfg: JointLossConfig, device: str | torch.device) -> dict[str, float]:
    model.eval()
    loss_rows, cls_rows = [], []
    for batch in loader:
        batch = batch.to(device)
        losses, metrics = eval_step(model, batch, loss_cfg)
        loss_rows.append({
            "total": float(losses.total.item()),
            "recon": float(losses.recon.item()),
            "cls": float(losses.cls.item()),
            "morph": float(losses.morph.item()),
            "kl": float(losses.kl.item()),
        })
        if metrics:
            cls_rows.append(metrics)
    out = _mean_dict(loss_rows)
    out.update(_mean_dict(cls_rows))
    return out


@torch.no_grad()
def evaluate_mallorn_epoch(model: ConvGNPJointModel, loader, loss_cfg: JointLossConfig, device: str | torch.device) -> dict[str, float]:
    model.eval()
    loss_rows = []
    records = {"tde": [], "nontde": []}
    all_logits, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        losses = joint_loss(out, batch, loss_cfg)
        loss_rows.append({
            "total": float(losses.total.item()),
            "recon": float(losses.recon.item()),
            "cls": float(losses.cls.item()),
            "morph": float(losses.morph.item()),
            "kl": float(losses.kl.item()),
        })

        if out.class_logits is not None and batch.labels is not None:
            all_logits.append(out.class_logits.detach().cpu())
            all_labels.append(batch.labels.detach().cpu())

        obs_snr = batch.metadata["obs_snr"] if batch.metadata is not None and "obs_snr" in batch.metadata else None
        for i in range(batch.target_x.shape[0]):
            m = batch.target_mask[i] > 0
            fo = batch.target_y[i][m].detach().cpu().numpy()
            fp = out.pred_mean[i][m].detach().cpu().numpy()
            sig = torch.sqrt(out.pred_var[i][m]).detach().cpu().numpy()
            snr = obs_snr[i][m].detach().cpu().numpy() if obs_snr is not None else np.zeros_like(fo)
            key = "tde" if int(batch.labels[i].item()) == 1 else "nontde"
            records[key].append({"fo": fo, "fp": fp, "sig": sig, "snr": snr})

    metrics = _mean_dict(loss_rows)
    if all_logits:
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        metrics.update(evaluate_binary_predictions(logits, labels))

    for key in ["tde", "nontde"]:
        recs = records[key]
        if not recs:
            metrics[f"rmse_{key}"] = float("nan")
            metrics[f"rmse_{key}_peak"] = float("nan")
            metrics[f"sigma_sat_{key}"] = float("nan")
            continue
        fo = np.concatenate([r["fo"] for r in recs]) if recs else np.array([])
        fp = np.concatenate([r["fp"] for r in recs]) if recs else np.array([])
        sig = np.concatenate([r["sig"] for r in recs]) if recs else np.array([])
        snr = np.concatenate([r["snr"] for r in recs]) if recs else np.array([])
        if len(fo) == 0:
            metrics[f"rmse_{key}"] = float("nan")
            metrics[f"rmse_{key}_peak"] = float("nan")
            metrics[f"sigma_sat_{key}"] = float("nan")
        else:
            res = fo - fp
            metrics[f"rmse_{key}"] = float(np.sqrt(np.mean(res ** 2)))
            pk = snr > 5.0
            if pk.sum() > 0:
                metrics[f"rmse_{key}_peak"] = float(np.sqrt(np.mean(res[pk] ** 2)))
                metrics[f"sigma_sat_{key}"] = float(np.mean(sig[pk] > 1.0))
            else:
                metrics[f"rmse_{key}_peak"] = float("nan")
                metrics[f"sigma_sat_{key}"] = float("nan")

    metrics["val_nll"] = metrics.get("recon", float("nan"))
    return metrics
