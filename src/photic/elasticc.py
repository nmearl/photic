from __future__ import annotations

from pathlib import Path

from astropy.io import fits
import numpy as np

from .data import BAND2IDX, BANDS

FOCUS_RELEASES = [
    "ELASTICC_TRAIN_TDE",
    "ELASTICC_TRAIN_AGN",
    "ELASTICC_TRAIN_SLSN-I+host",
    "ELASTICC_TRAIN_SLSN-I_no_host",
    "ELASTICC_TRAIN_PISN",
    "ELASTICC_TRAIN_SNIa-SALT2",
    "ELASTICC_TRAIN_SNIa-91bg",
    "ELASTICC_TRAIN_SNIax",
    "ELASTICC_TRAIN_SNII+HostXT_V19",
    "ELASTICC_TRAIN_SNII-NMF",
    "ELASTICC_TRAIN_SNII-Templates",
    "ELASTICC_TRAIN_SNIIn+HostXT_V19",
    "ELASTICC_TRAIN_SNIIn-MOSFIT",
    "ELASTICC_TRAIN_SNIb+HostXT_V19",
    "ELASTICC_TRAIN_SNIb-Templates",
    "ELASTICC_TRAIN_SNIc+HostXT_V19",
    "ELASTICC_TRAIN_SNIc-Templates",
    "ELASTICC_TRAIN_SNIcBL+HostXT_V19",
    "ELASTICC_TRAIN_SNIIb+HostXT_V19",
]

# Physical class taxonomy: template/model variants collapsed into 7 groups
CLASS_NAMES: list[str] = ["TDE", "AGN", "SLSN", "PISN", "SN-Ia", "SN-II", "SN-Ibc"]
NUM_ELASTICC_CLASSES: int = len(CLASS_NAMES)

RELEASE_TO_CLASS: dict[str, int] = {
    "ELASTICC_TRAIN_TDE": 0,
    "ELASTICC_TRAIN_AGN": 1,
    "ELASTICC_TRAIN_SLSN-I+host": 2,
    "ELASTICC_TRAIN_SLSN-I_no_host": 2,
    "ELASTICC_TRAIN_PISN": 3,
    "ELASTICC_TRAIN_SNIa-SALT2": 4,
    "ELASTICC_TRAIN_SNIa-91bg": 4,
    "ELASTICC_TRAIN_SNIax": 4,
    "ELASTICC_TRAIN_SNII+HostXT_V19": 5,
    "ELASTICC_TRAIN_SNII-NMF": 5,
    "ELASTICC_TRAIN_SNII-Templates": 5,
    "ELASTICC_TRAIN_SNIIn+HostXT_V19": 5,
    "ELASTICC_TRAIN_SNIIn-MOSFIT": 5,
    "ELASTICC_TRAIN_SNIIb+HostXT_V19": 6,
    "ELASTICC_TRAIN_SNIb+HostXT_V19": 6,
    "ELASTICC_TRAIN_SNIb-Templates": 6,
    "ELASTICC_TRAIN_SNIc+HostXT_V19": 6,
    "ELASTICC_TRAIN_SNIc-Templates": 6,
    "ELASTICC_TRAIN_SNIcBL+HostXT_V19": 6,
}

META_FIELDS = [
    "MWEBV",
    "REDSHIFT_FINAL",
    "HOSTGAL_PHOTOZ",
    "HOSTGAL_SNSEP",
    "HOSTGAL_DDLR",
    "HOSTGAL_LOGMASS",
    "HOSTGAL_LOGSFR",
    "HOSTGAL_LOGsSFR",
    "HOSTGAL_COLOR",
    "HOSTGAL_ELLIPTICITY",
    "HOSTGAL_MAG_g",
    "HOSTGAL_MAG_r",
    "HOSTGAL_MAG_i",
    "HOSTGAL_MAG_z",
    "HOSTGAL_MAG_Y",
]


def _decode(value):
    if isinstance(value, bytes):
        return value.decode().strip()
    if isinstance(value, np.bytes_):
        return value.decode().strip()
    return value


def _table_to_native(path: str | Path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        cols = {}
        for name in data.names:
            arr = np.asarray(data[name])
            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
            cols[name] = arr
    return cols


def _meta_from_head(cols: dict[str, np.ndarray], idx: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(META_FIELDS), dtype=np.float32)
    mask = np.zeros(len(META_FIELDS), dtype=np.float32)
    for j, field in enumerate(META_FIELDS):
        if field not in cols:
            continue
        value = cols[field][idx]
        value = float(value) if np.isfinite(value) else np.nan
        if np.isfinite(value):
            values[j] = value
            mask[j] = 1.0
    return values, mask


def load_elasticc_focus_records(
    data_dir: str | Path,
    *,
    max_release_dirs: int | None = None,
    max_shards_per_release: int | None = None,
    max_objects_per_release: int | None = None,
) -> list[dict]:
    data_dir = Path(data_dir)
    release_names = [name for name in FOCUS_RELEASES if (data_dir / name).exists()]
    if max_release_dirs is not None:
        release_names = release_names[:max_release_dirs]

    records: list[dict] = []
    for release_name in release_names:
        release_dir = data_dir / release_name
        head_paths = sorted(release_dir.glob("*_HEAD.FITS.gz"))
        if max_shards_per_release is not None:
            head_paths = head_paths[:max_shards_per_release]
        release_count = 0
        target = RELEASE_TO_CLASS.get(release_name, -1)

        for head_path in head_paths:
            phot_path = Path(str(head_path).replace("_HEAD.FITS.gz", "_PHOT.FITS.gz"))
            if not phot_path.exists():
                continue
            head = _table_to_native(head_path)
            phot = _table_to_native(phot_path)
            n_rows = len(head["SNID"])
            for i in range(n_rows):
                start = int(head["PTROBS_MIN"][i]) - 1
                end = int(head["PTROBS_MAX"][i])
                bands = np.asarray([str(_decode(v)).strip().lower() for v in phot["BAND"][start:end]])
                keep = np.array([b in BAND2IDX for b in bands], dtype=bool)
                if not np.any(keep):
                    continue
                flux = np.asarray(phot["FLUXCAL"][start:end], dtype=np.float32)[keep]
                ferr = np.asarray(phot["FLUXCALERR"][start:end], dtype=np.float32)[keep]
                mjd = np.asarray(phot["MJD"][start:end], dtype=np.float32)[keep]
                bands = bands[keep]
                pos_err = np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
                if pos_err.sum() < 4:
                    continue
                flux = flux[pos_err]
                ferr = ferr[pos_err]
                mjd = mjd[pos_err]
                bands = bands[pos_err]
                band_idx = np.array([BAND2IDX[b] for b in bands], dtype=np.int64)
                meta_values, meta_mask = _meta_from_head(head, i)
                z = float(head["REDSHIFT_FINAL"][i]) if np.isfinite(head["REDSHIFT_FINAL"][i]) else np.nan
                oid = f"{release_name}:{_decode(head['SNID'][i])}"
                records.append(
                    {
                        "oid": oid,
                        "t_raw": mjd,
                        "flux_raw": flux,
                        "ferr_raw": ferr,
                        "band_idx": band_idx,
                        "target": target,
                        "z": z,
                        "meta_values": meta_values,
                        "meta_mask": meta_mask,
                        "release_name": release_name,
                    }
                )
                release_count += 1
                if max_objects_per_release is not None and release_count >= max_objects_per_release:
                    break
            if max_objects_per_release is not None and release_count >= max_objects_per_release:
                break
    return records
