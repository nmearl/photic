from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .batch import NPBatch
from .config import ConvGNPJointConfig
from .data import BANDS, BAND2IDX, normalize_redshift
from .model import ConvGNPJointModel


@dataclass(slots=True)
class AlertPhotometryPoint:
    mjd: float
    band: str
    flux: float
    flux_err: float


@dataclass(slots=True)
class AlertObject:
    object_id: str
    photometry: list[AlertPhotometryPoint]
    redshift: float | None = None


class AlertBrokerClient(Protocol):
    def fetch_alert_photometry(self, object_id: str) -> AlertObject:
        ...


@dataclass(slots=True)
class BrokerObjectUpdate:
    object_id: str
    first_mjd: float | None = None
    last_mjd: float | None = None
    ndet: int | None = None
    raw: dict | None = None


@dataclass(slots=True)
class ForecastBandCurve:
    band: str
    mjd: np.ndarray
    flux_mean: np.ndarray
    flux_sigma: np.ndarray
    latent_flux_samples: np.ndarray | None = None


@dataclass(slots=True)
class ForecastResult:
    object_id: str
    prob_tde: float
    class_logit: float
    redshift: float | None
    context_points: int
    flux_mean: float
    flux_std: float
    t_min: float
    t_span: float
    bands: dict[str, ForecastBandCurve]


def _normalize_alert_photometry(points: list[AlertPhotometryPoint]) -> dict[str, np.ndarray]:
    if not points:
        raise ValueError("At least one photometry point is required.")

    rows = [
        (float(p.mjd), p.band, float(p.flux), max(float(p.flux_err), 1e-6))
        for p in points
        if p.band in BAND2IDX
    ]
    if not rows:
        raise ValueError(f"No photometry points with supported bands: {BANDS}")

    rows.sort(key=lambda x: x[0])
    t_raw = np.asarray([r[0] for r in rows], dtype=np.float32)
    flux_raw = np.asarray([r[2] for r in rows], dtype=np.float32)
    ferr_raw = np.asarray([r[3] for r in rows], dtype=np.float32)
    band_idx = np.asarray([BAND2IDX[r[1]] for r in rows], dtype=np.int64)

    t_min = float(t_raw.min())
    t_span = max(float(t_raw.max() - t_raw.min()), 1.0)
    t_norm = (t_raw - t_min) / t_span
    flux_mean = float(flux_raw.mean())
    flux_std = max(float(flux_raw.std()), 1e-6)
    flux_norm = (flux_raw - flux_mean) / flux_std
    ferr_norm = ferr_raw / flux_std

    return {
        "t_raw": t_raw,
        "flux_raw": flux_raw,
        "ferr_raw": ferr_raw,
        "band_idx": band_idx,
        "t_min": t_min,
        "t_span": t_span,
        "t_norm": t_norm.astype(np.float32),
        "flux_mean": flux_mean,
        "flux_std": flux_std,
        "flux_norm": flux_norm.astype(np.float32),
        "ferr_norm": ferr_norm.astype(np.float32),
    }


def _future_query_grid(forecast_days: float, grid_points_per_band: int) -> np.ndarray:
    if grid_points_per_band < 2:
        raise ValueError("grid_points_per_band must be at least 2.")
    if forecast_days <= 0:
        raise ValueError("forecast_days must be positive.")
    return np.linspace(0.0, forecast_days, grid_points_per_band, dtype=np.float32)


class AlerceBrokerClient:
    def __init__(
        self,
        *,
        survey: str = "lsst",
        client=None,
    ):
        self.survey = survey
        if client is None:
            try:
                from alerce.core import Alerce
            except ImportError as exc:
                raise ImportError(
                    "ALeRCE support requires the optional 'alerce' package. "
                    "Install it with `pip install alerce`."
                ) from exc
            client = Alerce()
        self.client = client

    def fetch_alert_photometry(self, object_id: str) -> AlertObject:
        points = self._fetch_best_photometry(object_id)
        redshift = self._extract_redshift(object_id)
        return AlertObject(object_id=object_id, photometry=points, redshift=redshift)

    def fetch_alert_object(self, object_id: str) -> AlertObject:
        return self.fetch_alert_photometry(object_id)

    def poll_recent_objects(
        self,
        *,
        since_mjd: float | None = None,
        page: int = 1,
        page_size: int = 100,
        order_by: str = "lastmjd",
        order_mode: str = "DESC",
        **filters,
    ) -> list[BrokerObjectUpdate]:
        query_filters = dict(filters)
        if since_mjd is not None:
            query_filters.setdefault("lastmjd", since_mjd)
        rows = self.client.query_objects(
            survey=self.survey,
            format="json",
            page=page,
            page_size=page_size,
            order_by=order_by,
            order_mode=order_mode,
            **query_filters,
        )
        if isinstance(rows, dict):
            rows = rows.get("items", rows.get("results", []))
        updates = []
        for row in rows or []:
            oid = row.get("oid") or row.get("objectId") or row.get("object_id")
            if oid is None:
                continue
            updates.append(
                BrokerObjectUpdate(
                    object_id=str(oid),
                    first_mjd=_safe_float(row.get("firstmjd") or row.get("mjdstarthist")),
                    last_mjd=_safe_float(row.get("lastmjd") or row.get("mjdendhist")),
                    ndet=_safe_int(row.get("ndet") or row.get("ndethist")),
                    raw=row,
                )
            )
        return updates

    def _extract_redshift(self, object_id: str) -> float | None:
        try:
            obj = self.client.query_object(object_id, survey=self.survey, format="json")
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        for key in ("redshift", "z", "host_redshift", "best_redshift"):
            val = _safe_float(obj.get(key))
            if val is not None:
                return val
        return None

    def _fetch_best_photometry(self, object_id: str) -> list[AlertPhotometryPoint]:
        points: list[AlertPhotometryPoint] = []

        try:
            forced = self.client.query_forced_photometry(
                object_id,
                survey=self.survey,
                format="json",
            )
            points = self._photometry_rows_to_points(forced)
        except Exception:
            points = []

        if len(points) >= 2:
            return points

        try:
            lightcurve = self.client.query_lightcurve(
                object_id,
                survey=self.survey,
                format="json",
            )
            lightcurve_points = self._lightcurve_to_points(lightcurve)
            points = self._merge_points(points, lightcurve_points)
        except Exception:
            pass

        if len(points) >= 2:
            return points

        detections = self.client.query_detections(
            object_id,
            survey=self.survey,
            format="json",
        )
        detection_points = self._detections_to_points(detections)
        return self._merge_points(points, detection_points)

    @staticmethod
    def _detections_to_points(detections) -> list[AlertPhotometryPoint]:
        return AlerceBrokerClient._photometry_rows_to_points(detections)

    @staticmethod
    def _lightcurve_to_points(lightcurve) -> list[AlertPhotometryPoint]:
        if isinstance(lightcurve, dict):
            rows = []
            for key in ("forced_photometry", "forcedphotometry", "detections", "lightcurve", "items"):
                value = lightcurve.get(key)
                if isinstance(value, list):
                    rows.extend(value)
            if rows:
                return AlerceBrokerClient._photometry_rows_to_points(rows)
        return AlerceBrokerClient._photometry_rows_to_points(lightcurve)

    @staticmethod
    def _photometry_rows_to_points(rows) -> list[AlertPhotometryPoint]:
        if isinstance(rows, dict):
            rows = rows.get("items", rows.get("detections", rows))
        points: list[AlertPhotometryPoint] = []
        for row in rows or []:
            band = row.get("band_name") or row.get("fid") or row.get("band") or row.get("filter")
            band_name = _normalize_alerce_band(band)
            mjd = _safe_float(row.get("mjd") or row.get("jd") or row.get("midpointmjd"))
            flux = _safe_float(
                row.get("psfFlux")
                or row.get("psf_flux")
                or row.get("difference_flux")
                or row.get("flux")
                or row.get("psf_flux")
            )
            flux_err = _safe_float(
                row.get("psfFluxErr")
                or row.get("psf_flux_err")
                or row.get("difference_flux_error")
                or row.get("fluxerr")
                or row.get("e_flux")
            )
            if band_name is None or mjd is None or flux is None or flux_err is None:
                continue
            points.append(
                AlertPhotometryPoint(
                    mjd=mjd,
                    band=band_name,
                    flux=flux,
                    flux_err=max(flux_err, 1e-6),
                )
            )
        if not points:
            raise ValueError("No usable photometry points returned from ALeRCE.")
        points.sort(key=lambda p: p.mjd)
        return points

    @staticmethod
    def _merge_points(
        left: list[AlertPhotometryPoint],
        right: list[AlertPhotometryPoint],
    ) -> list[AlertPhotometryPoint]:
        seen = set()
        merged: list[AlertPhotometryPoint] = []
        for point in [*left, *right]:
            key = (round(point.mjd, 7), point.band, round(point.flux, 7), round(point.flux_err, 7))
            if key in seen:
                continue
            seen.add(key)
            merged.append(point)
        merged.sort(key=lambda p: p.mjd)
        return merged


def _normalize_alerce_band(band) -> str | None:
    if band is None:
        return None
    if isinstance(band, str):
        candidate = band.strip().lower()
        if candidate in BAND2IDX:
            return candidate
        if candidate.startswith("lsst"):
            candidate = candidate[-1]
            return candidate if candidate in BAND2IDX else None
        return candidate if candidate in BAND2IDX else None
    try:
        fid = int(band)
    except (TypeError, ValueError):
        return None
    # Prefer Rubin/LSST band indexing when numeric bands are returned directly.
    if 0 <= fid < len(BANDS):
        return BANDS[fid]
    # ZTF uses fid 1/2/3 for g/r/i.
    fid_map = {1: "g", 2: "r", 3: "i"}
    return fid_map.get(fid)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class JointAlertForecaster:
    def __init__(
        self,
        model: ConvGNPJointModel,
        *,
        z_min: float,
        z_max: float,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> "JointAlertForecaster":
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_cfg = ConvGNPJointConfig(**ckpt["model_cfg"])
        model = ConvGNPJointModel(model_cfg)
        model.load_state_dict(ckpt["model_state"])
        return cls(
            model,
            z_min=float(ckpt.get("z_min", 0.0)),
            z_max=float(ckpt.get("z_max", 1.0)),
            device=device,
        )

    def build_batch(
        self,
        alert: AlertObject,
        *,
        forecast_days: float = 365.0,
        grid_points_per_band: int = 256,
    ) -> tuple[NPBatch, dict[str, float | np.ndarray]]:
        norm = _normalize_alert_photometry(alert.photometry)
        future_days = _future_query_grid(forecast_days=forecast_days, grid_points_per_band=grid_points_per_band)
        query_t_raw_parts = []
        query_t_norm_parts = []
        query_band_parts = []
        for band in BANDS:
            query_t_raw = norm["t_raw"][-1] + future_days
            query_t_norm = (query_t_raw - norm["t_min"]) / norm["t_span"]
            query_t_raw_parts.append(query_t_raw.astype(np.float32))
            query_t_norm_parts.append(query_t_norm.astype(np.float32))
            query_band_parts.append(np.full(grid_points_per_band, BAND2IDX[band], dtype=np.int64))

        z_raw = np.array([np.nan if alert.redshift is None else float(alert.redshift)], dtype=np.float32)
        z_norm = normalize_redshift(torch.as_tensor(z_raw, dtype=torch.float32), self.z_min, self.z_max)

        batch = NPBatch(
            context_x=torch.as_tensor(norm["t_norm"], dtype=torch.float32).unsqueeze(0),
            context_y=torch.as_tensor(norm["flux_norm"], dtype=torch.float32).unsqueeze(0),
            context_yerr=torch.as_tensor(norm["ferr_norm"], dtype=torch.float32).unsqueeze(0),
            context_band=torch.as_tensor(norm["band_idx"], dtype=torch.long).unsqueeze(0),
            context_mask=torch.ones((1, len(norm["t_norm"])), dtype=torch.float32),
            target_x=torch.as_tensor(np.concatenate(query_t_norm_parts), dtype=torch.float32).unsqueeze(0),
            target_y=torch.zeros((1, len(BANDS) * grid_points_per_band), dtype=torch.float32),
            target_yerr=torch.ones((1, len(BANDS) * grid_points_per_band), dtype=torch.float32),
            target_band=torch.as_tensor(np.concatenate(query_band_parts), dtype=torch.long).unsqueeze(0),
            target_mask=torch.ones((1, len(BANDS) * grid_points_per_band), dtype=torch.float32),
            labels=None,
            redshift=z_norm,
            metadata={"oid": [alert.object_id]},
        )
        aux = {
            "query_t_raw": np.concatenate(query_t_raw_parts),
            "query_band_idx": np.concatenate(query_band_parts),
            "flux_mean": norm["flux_mean"],
            "flux_std": norm["flux_std"],
            "t_min": norm["t_min"],
            "t_span": norm["t_span"],
            "n_context": len(norm["t_norm"]),
        }
        return batch, aux

    @torch.no_grad()
    def forecast(
        self,
        alert: AlertObject,
        *,
        forecast_days: float = 365.0,
        grid_points_per_band: int = 256,
        latent_samples: int = 0,
    ) -> ForecastResult:
        batch, aux = self.build_batch(
            alert,
            forecast_days=forecast_days,
            grid_points_per_band=grid_points_per_band,
        )
        batch = batch.to(self.device)
        out = self.model(batch)

        class_logit = float(out.class_logits[0].detach().cpu().item()) if out.class_logits is not None else float("nan")
        prob_tde = float(torch.sigmoid(out.class_logits)[0].detach().cpu().item()) if out.class_logits is not None else float("nan")
        pred_mean = out.pred_mean[0].detach().cpu().numpy() * float(aux["flux_std"]) + float(aux["flux_mean"])
        pred_sigma = torch.sqrt(out.pred_var[0].clamp_min(1e-8)).detach().cpu().numpy() * float(aux["flux_std"])

        sample_means = None
        if latent_samples > 0 and getattr(self.model.cfg, "use_latent", False):
            means_s, _ = self.model.sample_predictions(batch, num_samples=latent_samples)
            sample_means = means_s[:, 0].detach().cpu().numpy() * float(aux["flux_std"]) + float(aux["flux_mean"])

        bands: dict[str, ForecastBandCurve] = {}
        query_t_raw = np.asarray(aux["query_t_raw"], dtype=np.float32)
        query_band_idx = np.asarray(aux["query_band_idx"], dtype=np.int64)
        for band in BANDS:
            bi = BAND2IDX[band]
            mask = query_band_idx == bi
            band_samples = None if sample_means is None else sample_means[:, mask]
            bands[band] = ForecastBandCurve(
                band=band,
                mjd=query_t_raw[mask],
                flux_mean=pred_mean[mask],
                flux_sigma=pred_sigma[mask],
                latent_flux_samples=band_samples,
            )

        return ForecastResult(
            object_id=alert.object_id,
            prob_tde=prob_tde,
            class_logit=class_logit,
            redshift=alert.redshift,
            context_points=int(aux["n_context"]),
            flux_mean=float(aux["flux_mean"]),
            flux_std=float(aux["flux_std"]),
            t_min=float(aux["t_min"]),
            t_span=float(aux["t_span"]),
            bands=bands,
        )

    def forecast_from_broker(
        self,
        broker: AlertBrokerClient,
        object_id: str,
        *,
        forecast_days: float = 365.0,
        grid_points_per_band: int = 256,
        latent_samples: int = 0,
    ) -> ForecastResult:
        alert = broker.fetch_alert_photometry(object_id)
        return self.forecast(
            alert,
            forecast_days=forecast_days,
            grid_points_per_band=grid_points_per_band,
            latent_samples=latent_samples,
        )
