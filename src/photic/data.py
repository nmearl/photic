from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .batch import NPBatch

BANDS = ["u", "g", "r", "i", "z", "y"]
BAND2IDX = {b: i for i, b in enumerate(BANDS)}
EBV_COEFFS = {
    "u": 4.757, "g": 3.661, "r": 2.701,
    "i": 2.054, "z": 1.590, "y": 1.308,
}
MORPH_NAMES = ["occupancy", "contrast", "peak_spread", "decline_smoothness"]


def stable_hash(text: str) -> int:
    return int(hashlib.blake2b(text.encode(), digest_size=8).hexdigest(), 16) % (2**63 - 1)


def det_rng(seed: int, oid: str, role: str) -> np.random.Generator:
    return np.random.default_rng(seed ^ stable_hash(f"{oid}::{role}"))


def train_rng(seed: int, oid: str, band: int, epoch: int, worker_id: int = 0) -> np.random.Generator:
    key = f"{seed}::{oid}::b{band}::ep{epoch}::w{worker_id}"
    h = stable_hash(key)
    return np.random.default_rng(h)


def load_all_data(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    split_dirs = sorted(data_dir.glob("split_*"))
    lc_parts = [
        pd.read_csv(sd / "train_full_lightcurves.csv")
        for sd in split_dirs
        if (sd / "train_full_lightcurves.csv").exists()
    ]
    log_f = data_dir / "train_log.csv"

    if lc_parts and log_f.exists():
        lc = pd.concat(lc_parts, ignore_index=True)
    else:
        lc_f = data_dir / "train_full_lightcurves.csv"
        if not lc_f.exists() or not log_f.exists():
            raise FileNotFoundError(f"Cannot find training data under {data_dir!s}")
        lc = pd.read_csv(lc_f)

    log = pd.read_csv(log_f)
    lc = lc.rename(columns={"Time (MJD)": "mjd", "Flux": "flux", "Flux_err": "flux_err", "Filter": "band"})
    keep = [c for c in ["object_id", "Z", "EBV", "target", "SpecType"] if c in log.columns]
    log = log[keep].copy()
    log["target"] = log["target"].fillna(0).astype(int)
    lc = lc[lc["band"].isin(BANDS)].dropna(subset=["flux", "flux_err"])
    lc = lc[lc["flux_err"] > 0].copy()
    return lc, log


def apply_ebv_correction(lc: pd.DataFrame, log: pd.DataFrame, ebv_coeffs: Dict[str, float] | None = None) -> pd.DataFrame:
    ebv_coeffs = EBV_COEFFS if ebv_coeffs is None else ebv_coeffs
    ebv_map = dict(zip(log["object_id"], log["EBV"].fillna(0.0)))
    lc = lc.copy()
    lc["_e"] = lc["object_id"].map(ebv_map).fillna(0.0)
    lc["_c"] = lc["band"].map(ebv_coeffs).fillna(0.0)
    f = 10.0 ** (0.4 * lc["_e"] * lc["_c"])
    lc["flux"] *= f
    lc["flux_err"] *= f
    lc.drop(columns=["_e", "_c"], inplace=True)
    return lc


def cap_observations(obj: pd.DataFrame, max_obs: int, snr_thr: float) -> pd.DataFrame:
    if len(obj) <= max_obs:
        return obj.sort_values("mjd").reset_index(drop=True)
    obj = obj.sort_values("mjd").reset_index(drop=True).copy()
    snr = np.abs(obj["flux"].values) / (obj["flux_err"].values + 1e-6)
    hi = np.where(snr > snr_thr)[0]
    lo = np.setdiff1d(np.arange(len(obj)), hi)
    budget = max_obs - len(hi)
    if budget <= 0:
        idx = np.linspace(0, len(hi) - 1, max_obs).round().astype(int)
        chosen = np.unique(hi[idx])
    elif len(lo) <= budget:
        chosen = np.sort(np.concatenate([hi, lo]))
    else:
        idx = np.linspace(0, len(lo) - 1, budget).round().astype(int)
        chosen = np.sort(np.concatenate([hi, np.unique(lo[idx])]))
    return obj.iloc[chosen].sort_values("mjd").reset_index(drop=True)


def _compute_morphology_targets(t_norm: np.ndarray, flux_norm: np.ndarray, obs_snr: np.ndarray, band_idx: np.ndarray) -> np.ndarray:
    valid = np.isfinite(t_norm) & np.isfinite(flux_norm)
    if valid.sum() < 3:
        return np.zeros(4, dtype=np.float32)
    t = t_norm[valid]
    y = flux_norm[valid]
    s = obs_snr[valid]
    b = band_idx[valid]

    high_state = (y > 0.5) & (s > 2.0)
    occupancy = float(high_state.mean())

    baseline = float(np.median(y))
    contrast = float(np.quantile(y, 0.95) - baseline)

    peak_times = []
    for bi in np.unique(b):
        bm = b == bi
        if bm.sum() < 2:
            continue
        peak_times.append(float(t[bm][np.argmax(y[bm])]))
    peak_spread = float(np.std(peak_times)) if len(peak_times) >= 2 else 0.0

    peak_idx = int(np.argmax(y))
    post = y[peak_idx:]
    if len(post) >= 3:
        diffs = np.diff(post)
        decline_smoothness = float((diffs <= 0).mean())
    else:
        decline_smoothness = 0.5

    return np.array([occupancy, contrast, peak_spread, decline_smoothness], dtype=np.float32)


def valid_object_ids(lc: pd.DataFrame, min_obs: int = 3) -> List[str]:
    counts = lc.groupby("object_id").size()
    return counts[counts >= min_obs].index.tolist()


class MallornDataset(Dataset):
    def __init__(self, object_ids: List[str], lc: pd.DataFrame, log: pd.DataFrame):
        target_map = dict(zip(log["object_id"], log["target"]))
        z_map = dict(zip(
            log["object_id"],
            log["Z"].fillna(np.nan).astype(float) if "Z" in log.columns else [np.nan] * len(log),
        ))
        spectype_map = {
            oid: (log.loc[log["object_id"] == oid, "SpecType"].iloc[0] if "SpecType" in log.columns else "unknown")
            for oid in object_ids
        }
        self.data: Dict[str, dict] = {}
        lc_grp = lc.groupby("object_id")
        for oid in object_ids:
            if oid not in lc_grp.groups:
                continue
            obj = lc_grp.get_group(oid).sort_values("mjd").reset_index(drop=True)
            t_raw = obj["mjd"].values.astype(np.float32)
            flux_raw = obj["flux"].values.astype(np.float32)
            ferr_raw = obj["flux_err"].values.astype(np.float32)
            band_idx = obj["band"].map(BAND2IDX).values.astype(np.int64)
            t_min = t_raw.min()
            t_span = max(float(t_raw.max() - t_raw.min()), 1.0)
            t_norm = (t_raw - t_min) / t_span
            fmean = float(flux_raw.mean())
            fstd = max(float(flux_raw.std()), 1e-6)
            fn = (flux_raw - fmean) / fstd
            fen = ferr_raw / fstd
            obs_snr = (np.abs(flux_raw) / (ferr_raw + 1e-6)).astype(np.float32)
            if len(t_norm) < 3:
                continue
            morph_raw = _compute_morphology_targets(t_norm, fn, obs_snr, band_idx)
            self.data[oid] = dict(
                t_norm=t_norm,
                flux_norm=fn,
                ferr_norm=fen,
                obs_snr=obs_snr,
                band_idx=band_idx,
                t_raw=t_raw,
                flux_mean=fmean,
                flux_std=fstd,
                t_span=t_span,
                t_min=t_min,
                target=int(target_map.get(oid, 0)),
                spectype=spectype_map.get(oid, "unknown"),
                z=float(z_map.get(oid, np.nan)),
                morph_raw=morph_raw,
            )
        self.object_ids = [o for o in object_ids if o in self.data]

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> dict:
        oid = self.object_ids[idx]
        d = self.data[oid]
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        out["oid"] = oid
        return out


def compute_z_norm_stats(dataset: MallornDataset, p1: float = 0.01, p99: float = 0.99) -> Tuple[float, float]:
    z_vals = [d["z"] for d in dataset.data.values() if np.isfinite(d.get("z", np.nan))]
    if not z_vals:
        return 0.0, 1.0
    z_arr = np.array(z_vals)
    return float(np.quantile(z_arr, p1)), float(np.quantile(z_arr, p99))


def compute_morphology_norm_stats(dataset: MallornDataset) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.stack([d["morph_raw"] for d in dataset.data.values()], axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_redshift(z_raw: torch.Tensor, z_min: float, z_max: float) -> torch.Tensor:
    span = max(z_max - z_min, 1e-6)
    z_norm = (z_raw.clone() - z_min) / span
    z_norm = z_norm.clamp(0.0, 1.0)
    z_norm[~torch.isfinite(z_norm)] = 0.5
    return z_norm


def _split_band_random(t_b, f_b, fe_b, snr_b, n_ctx: int, rng: np.random.Generator):
    n = len(t_b)
    if n == 0:
        empty = np.array([], dtype=np.float32)
        empty_i = np.array([], dtype=np.int64)
        return empty, empty, empty, empty, empty, empty, empty_i, empty_i
    n_ctx = min(n_ctx, n)
    ctx_idx = np.sort(rng.choice(n, size=n_ctx, replace=False))
    tgt_idx = np.setdiff1d(np.arange(n), ctx_idx)
    return t_b[ctx_idx], f_b[ctx_idx], t_b[tgt_idx], f_b[tgt_idx], fe_b[tgt_idx], snr_b[tgt_idx], ctx_idx, tgt_idx


def _split_band_block(t_b, f_b, fe_b, snr_b, block_frac: float, min_ctx: int, rng: np.random.Generator):
    n = len(t_b)
    if n == 0:
        empty = np.array([], dtype=np.float32)
        empty_i = np.array([], dtype=np.int64)
        return empty, empty, empty, empty, empty, empty, empty_i, empty_i
    block_len = max(1, int(round(block_frac * n)))
    block_len = min(block_len, max(0, n - min_ctx))
    if block_len == 0:
        return _split_band_random(t_b, f_b, fe_b, snr_b, max(min_ctx, n - 1), rng)
    start = int(rng.integers(0, n - block_len + 1))
    tgt_idx = np.arange(start, start + block_len)
    ctx_idx = np.setdiff1d(np.arange(n), tgt_idx)
    return t_b[ctx_idx], f_b[ctx_idx], t_b[tgt_idx], f_b[tgt_idx], fe_b[tgt_idx], snr_b[tgt_idx], ctx_idx, tgt_idx


def _split_band(t_b, f_b, fe_b, snr_b, n_ctx: int, rng: np.random.Generator, block_mask: bool, block_frac: float, min_ctx: int):
    if block_mask:
        return _split_band_block(t_b, f_b, fe_b, snr_b, block_frac, min_ctx, rng)
    return _split_band_random(t_b, f_b, fe_b, snr_b, n_ctx, rng)


@dataclass(slots=True)
class MallornCollateConfig:
    seed: int = 42
    mask_prob: float = 0.50
    block_mask_prob: float = 0.50
    block_mask_frac: float = 0.35
    min_ctx_per_band: int = 2
    deterministic_val_fraction: float = 0.50
    z_min: float = 0.0
    z_max: float = 1.0
    morph_mean: np.ndarray | None = None
    morph_std: np.ndarray | None = None
    epoch: int = 0


@dataclass(slots=True)
class MallornCollate:
    training: bool
    cfg: MallornCollateConfig

    def __call__(self, batch: List[dict]) -> NPBatch:
        return build_mallorn_batch(batch, training=self.training, cfg=self.cfg)


def build_mallorn_batch(items: List[dict], training: bool, cfg: MallornCollateConfig) -> NPBatch:
    context_x, context_y, context_yerr, context_band, context_mask = [], [], [], [], []
    target_x, target_y, target_yerr, target_band, target_mask = [], [], [], [], []
    labels, redshifts = [], []
    morph_targets = []
    metadata = {"oid": [], "obs_snr": [], "target_is_peak": [], "morph_names": MORPH_NAMES}

    for item in items:
        ctx_t_parts, ctx_f_parts, ctx_fe_parts, ctx_b_parts = [], [], [], []
        tgt_t_parts, tgt_f_parts, tgt_fe_parts, tgt_b_parts = [], [], [], []
        tgt_snr_parts = []

        for bi in range(len(BANDS)):
            bm = item["band_idx"] == bi
            t_b = item["t_norm"][bm]
            f_b = item["flux_norm"][bm]
            fe_b = item["ferr_norm"][bm]
            snr_b = item["obs_snr"][bm]
            n = len(t_b)
            if training:
                worker_info = torch.utils.data.get_worker_info()
                worker_id = 0 if worker_info is None else worker_info.id
                rng = train_rng(cfg.seed, item["oid"], bi, cfg.epoch, worker_id)
                n_ctx = max(cfg.min_ctx_per_band, int(round((1.0 - cfg.mask_prob) * n)))
                use_block = rng.random() < cfg.block_mask_prob
            else:
                rng = det_rng(cfg.seed, item["oid"], f"val_b{bi}")
                n_ctx = max(cfg.min_ctx_per_band, int(round(cfg.deterministic_val_fraction * n)))
                use_block = False
            t_ctx, f_ctx, t_tgt, f_tgt, fe_tgt, snr_tgt, ctx_idx, tgt_idx = _split_band(
                t_b, f_b, fe_b, snr_b, n_ctx, rng,
                block_mask=use_block, block_frac=cfg.block_mask_frac, min_ctx=cfg.min_ctx_per_band,
            )
            if len(t_ctx) > 0:
                ctx_t_parts.append(t_ctx)
                ctx_f_parts.append(f_ctx)
                ctx_fe_parts.append(fe_b[ctx_idx])
                ctx_b_parts.append(np.full(len(t_ctx), bi, dtype=np.int64))
            if len(t_tgt) > 0:
                tgt_t_parts.append(t_tgt)
                tgt_f_parts.append(f_tgt)
                tgt_fe_parts.append(fe_tgt)
                tgt_snr_parts.append(snr_tgt)
                tgt_b_parts.append(np.full(len(t_tgt), bi, dtype=np.int64))

        if ctx_t_parts:
            cx = np.concatenate(ctx_t_parts)
            cy = np.concatenate(ctx_f_parts)
            ce = np.concatenate(ctx_fe_parts)
            cb = np.concatenate(ctx_b_parts)
            order = np.argsort(cx)
            cx, cy, ce, cb = cx[order], cy[order], ce[order], cb[order]
        else:
            cx = cy = ce = np.array([], dtype=np.float32)
            cb = np.array([], dtype=np.int64)

        if tgt_t_parts:
            tx = np.concatenate(tgt_t_parts)
            ty = np.concatenate(tgt_f_parts)
            te = np.concatenate(tgt_fe_parts)
            ts = np.concatenate(tgt_snr_parts)
            tb = np.concatenate(tgt_b_parts)
            order = np.argsort(tx)
            tx, ty, te, ts, tb = tx[order], ty[order], te[order], ts[order], tb[order]
        else:
            tx = ty = te = ts = np.array([], dtype=np.float32)
            tb = np.array([], dtype=np.int64)

        context_x.append(torch.as_tensor(cx, dtype=torch.float32))
        context_y.append(torch.as_tensor(cy, dtype=torch.float32))
        context_yerr.append(torch.as_tensor(ce if len(ce) else np.array([1.0], dtype=np.float32), dtype=torch.float32)[:len(cx)] if len(cx) else torch.zeros(0, dtype=torch.float32))
        context_band.append(torch.as_tensor(cb, dtype=torch.long))
        context_mask.append(torch.ones(len(cx), dtype=torch.float32))
        target_x.append(torch.as_tensor(tx, dtype=torch.float32))
        target_y.append(torch.as_tensor(ty, dtype=torch.float32))
        target_yerr.append(torch.as_tensor(te if len(te) else np.array([1.0], dtype=np.float32), dtype=torch.float32)[:len(tx)] if len(tx) else torch.zeros(0, dtype=torch.float32))
        target_band.append(torch.as_tensor(tb, dtype=torch.long))
        target_mask.append(torch.ones(len(tx), dtype=torch.float32))
        labels.append(float(item["target"]))
        redshifts.append(float(item.get("z", np.nan)))
        raw_morph = item.get("morph_raw", np.zeros(4, dtype=np.float32)).astype(np.float32)
        if cfg.morph_mean is not None and cfg.morph_std is not None:
            raw_morph = (raw_morph - cfg.morph_mean) / cfg.morph_std
        morph_targets.append(torch.as_tensor(raw_morph, dtype=torch.float32))
        metadata["oid"].append(item["oid"])
        metadata["obs_snr"].append(torch.as_tensor(ts, dtype=torch.float32))
        metadata["target_is_peak"].append(torch.as_tensor((ts > 5.0).astype(np.float32), dtype=torch.float32))

    def _pad(vals: List[torch.Tensor], pad_value: float = 0.0, dtype=None):
        max_len = max((len(v) for v in vals), default=1)
        dtype = vals[0].dtype if vals else (torch.float32 if dtype is None else dtype)
        out = torch.full((len(vals), max_len), pad_value, dtype=dtype)
        for i, v in enumerate(vals):
            if len(v) > 0:
                out[i, :len(v)] = v
        return out

    batch = NPBatch(
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
        labels=torch.as_tensor(labels, dtype=torch.float32),
        redshift=normalize_redshift(torch.as_tensor(redshifts, dtype=torch.float32), cfg.z_min, cfg.z_max),
        morph_targets=torch.stack(morph_targets, dim=0),
        metadata={
            "oid": metadata["oid"],
            "obs_snr": _pad(metadata["obs_snr"]),
            "target_is_peak": _pad(metadata["target_is_peak"]),
            "morph_names": metadata["morph_names"],
        },
    )
    return batch


def make_mallorn_collate(training: bool, cfg: MallornCollateConfig) -> Callable[[List[dict]], NPBatch]:
    return MallornCollate(training=training, cfg=cfg)


def prepare_mallorn_datasets(
    data_dir: str | Path,
    seed: int = 42,
    val_frac: float = 0.15,
    max_obs: int = 200,
    keep_all_snr_gt: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, MallornDataset, MallornDataset, List[str], List[str]]:
    lc, log = load_all_data(data_dir)
    lc = apply_ebv_correction(lc, log)
    lc_ids = set(lc["object_id"].unique())
    log = log[log["object_id"].isin(lc_ids)].reset_index(drop=True)
    long_obj = lc.groupby("object_id").size()
    long_obj = long_obj[long_obj > max_obs].index.tolist()
    if long_obj:
        parts = [lc[~lc["object_id"].isin(long_obj)]]
        for oid in long_obj:
            parts.append(cap_observations(lc[lc["object_id"] == oid].copy(), max_obs, keep_all_snr_gt))
        lc = pd.concat(parts, ignore_index=True)

    usable_ids = valid_object_ids(lc, min_obs=3)
    lc = lc[lc["object_id"].isin(usable_ids)].reset_index(drop=True)
    log = log[log["object_id"].isin(usable_ids)].reset_index(drop=True)

    all_ids = log["object_id"].tolist()
    all_tgts = log["target"].tolist()
    train_ids, val_ids = train_test_split(all_ids, test_size=val_frac, stratify=all_tgts, random_state=seed)
    train_ds = MallornDataset(train_ids, lc, log)
    val_ds = MallornDataset(val_ids, lc, log)
    train_ids = train_ds.object_ids
    val_ids = val_ds.object_ids
    return lc, log, train_ds, val_ds, train_ids, val_ids
