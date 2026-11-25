"""Read-only CT loading and training-only clinical preprocessing for NLST.

Prepared volumes need explicit intensity mode: ``unit`` means [0, 1], and
``hu`` means Hounsfield units. DICOM is converted from stored values to HU and
windowed to [-1000, 400]. No missing imaging modality or outcome is invented.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CLINICAL_FEATURES = (
    "age", "sex_male", "ever_smoker", "pack_years", "quit_years", "family_history"
)
HORIZON_YEARS = (1, 3, 5)
WINDOW_HU = (-1000.0, 400.0)


def read_manifest(source) -> list[dict[str, Any]]:
    """Read CSV records without coercing patient identifiers to integers."""
    if isinstance(source, (str, Path)):
        with Path(source).open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if hasattr(source, "to_dict"):
        return list(source.to_dict(orient="records"))
    return [dict(row) for row in source]


def _validated_rows(manifest):
    rows = read_manifest(manifest)
    patient_splits = {}
    for row in rows:
        pid = str(row.get("patient_id", "")).strip()
        if pid.lower() in ("", "none", "nan", "<na>", "nat"):
            raise ValueError("Every manifest row needs a non-missing patient_id")
        split = row.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError("Every manifest split must be train, validation, or test")
        previous = patient_splits.setdefault(pid, split)
        if previous != split:
            raise ValueError("Patient appears in multiple dataset splits")
    return rows


def _number(value, default: float = math.nan) -> float:
    if value is None or str(value).strip() in ("", "M", ".M", "N", ".N", "NA", "nan", "None"):
        return default
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Expected a numeric value, received {value!r}") from None
    return result if math.isfinite(result) else default


class ClinicalPreprocessor:
    """Six standardized baseline features, then six missing indicators.

    Fit selects only rows with split ``train``. Means and population standard
    deviations use median-imputed training values. An all-missing feature gets
    median zero and standardized value zero, even if future rows contain it:
    there is no training distribution from which to estimate a scale.
    """
    features = CLINICAL_FEATURES

    def fit(self, manifest) -> "ClinicalPreprocessor":
        rows = [r for r in _validated_rows(manifest) if r["split"] == "train"]
        if not rows:
            raise ValueError("Clinical preprocessing requires at least one training row")
        values = np.asarray([[_number(r.get(k)) for k in self.features] for r in rows], dtype=np.float64)
        self.all_missing = np.isnan(values).all(axis=0)
        self.medians = np.asarray([
            0.0 if absent else np.nanmedian(values[:, j])
            for j, absent in enumerate(self.all_missing)
        ])
        imputed = np.where(np.isnan(values), self.medians, values)
        self.means = imputed.mean(axis=0)
        self.stds = imputed.std(axis=0)
        self.stds[self.stds < 1e-8] = 1.0
        self.training_rows = len(rows)
        return self

    def transform(self, row: Mapping[str, Any]) -> np.ndarray:
        if not hasattr(self, "medians"):
            raise RuntimeError("Fit or load clinical preprocessing before transforming")
        values = np.asarray([_number(row.get(k)) for k in self.features], dtype=np.float64)
        missing = np.isnan(values)
        standardized = (np.where(missing, self.medians, values) - self.means) / self.stds
        standardized[self.all_missing] = 0.0
        return np.concatenate((standardized, missing.astype(np.float64))).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        if not hasattr(self, "medians"):
            raise RuntimeError("Clinical preprocessing has not been fitted")
        return {
            "format_version": 1,
            "features": list(self.features),
            "output_features": list(self.features) + [f"{k}_missing" for k in self.features],
            "fit_split": "train", "training_rows": self.training_rows,
            "medians": self.medians.tolist(), "means": self.means.tolist(),
            "stds": self.stds.tolist(), "all_missing": self.all_missing.tolist(),
            "all_missing_policy": "training median 0; standardized value always 0",
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ClinicalPreprocessor":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("format_version") != 1 or tuple(state.get("features", ())) != cls.features:
            raise ValueError("Incompatible clinical preprocessing schema")
        if state.get("fit_split") != "train" or int(state.get("training_rows", 0)) < 1:
            raise ValueError("Preprocessing metadata must describe a training-only fit")
        obj = cls()
        for key in ("medians", "means", "stds", "all_missing"):
            arr = np.asarray(state[key], dtype=bool if key == "all_missing" else np.float64)
            if arr.shape != (len(cls.features),) or not np.isfinite(arr).all():
                raise ValueError(f"Invalid clinical preprocessing {key}")
            setattr(obj, key, arr)
        if np.any(obj.stds <= 0):
            raise ValueError("Clinical standard deviations must be positive")
        obj.training_rows = int(state["training_rows"])
        return obj


def sample_slice_indices(slice_count: int, n_slices: int) -> np.ndarray:
    """Cover a whole stack deterministically, repeating slices if short."""
    if slice_count < 1 or n_slices < 1:
        raise ValueError("slice_count and n_slices must be positive")
    return np.rint(np.linspace(0, slice_count - 1, n_slices)).astype(np.int64)


def _finish_slices(slices: torch.Tensor, image_size: int, mode: str) -> torch.Tensor:
    slices = slices.to(dtype=torch.float32)
    if not torch.isfinite(slices).all():
        raise ValueError("CT contains non-finite intensities")
    if mode == "hu":
        low, high = WINDOW_HU
        slices = (slices.clamp(low, high) - low) / (high - low)
    elif mode == "unit":
        if torch.any(slices < -1e-5) or torch.any(slices > 1.0 + 1e-5):
            raise ValueError("Prepared CT declared unit intensities but values are outside [0,1]")
        slices = slices.clamp(0, 1)
    else:
        raise ValueError("Prepared CT requires explicit cached_intensity_mode='unit' or 'hu'")
    return F.interpolate(slices, size=(image_size, image_size), mode="bilinear", align_corners=False).contiguous()


def _prepared_ct(path: Path, series_uid: str, n_slices: int, image_size: int, intensity_mode: str | None):
    if path.suffix.lower() == ".npy":
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        metadata = {}
    elif path.suffix.lower() == ".pt":
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        metadata = loaded if isinstance(loaded, dict) else {}
        volume = metadata.get("volume") if metadata else loaded
    else:
        raise ValueError("Prepared volume must be a .pt or .npy file")
    declared_uid = metadata.get("selected_series_uid", metadata.get("series_instance_uid"))
    if declared_uid is not None and str(declared_uid) != series_uid:
        raise ValueError("Prepared CT SeriesInstanceUID disagrees with manifest")
    declared_mode = metadata.get("intensity_mode")
    if intensity_mode is not None and declared_mode is not None and declared_mode != intensity_mode:
        raise ValueError("Prepared CT intensity metadata disagrees with requested intensity mode")
    mode = intensity_mode if intensity_mode is not None else declared_mode
    if mode not in ("hu", "unit"):
        raise ValueError("Prepared CT needs explicit cached_intensity_mode='unit' or 'hu'; preprocess_id alone is insufficient")
    if not isinstance(volume, (np.ndarray, torch.Tensor)):
        raise ValueError("Prepared CT requires a tensor or dictionary containing 'volume'")
    # Established cache format: [1,depth,height,width]. Also accept
    # [depth,height,width] and unambiguous [depth,1,height,width].
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]
    elif volume.ndim == 4 and volume.shape[1] == 1:
        volume = volume[:, 0]
    if volume.ndim != 3 or any(dim < 1 for dim in volume.shape):
        raise ValueError("Prepared CT must have shape [depth,height,width] or [1,depth,height,width]")
    indices = sample_slice_indices(volume.shape[0], n_slices)
    selected = volume[indices]
    if isinstance(selected, np.ndarray):
        selected = torch.from_numpy(np.array(selected, copy=True))
    return _finish_slices(selected.unsqueeze(1), image_size, mode)


def _dicom_headers(path: Path, series_uid: str):
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("DICOM loading requires pydicom") from exc
    candidates = []
    for source in sorted(p for p in path.rglob("*") if p.is_file()):
        try:
            header = pydicom.dcmread(source, stop_before_pixels=True)
        except (OSError, ValueError, pydicom.errors.InvalidDicomError):
            continue
        if str(getattr(header, "SeriesInstanceUID", "")) != series_uid:
            continue
        raw_image_type = getattr(header, "ImageType", [])
        # pydicom represents VM=1 ImageType as a str rather than MultiValue.
        # Iterating that string would hide LOCALIZER/SCOUT among its letters.
        image_type = (raw_image_type if isinstance(raw_image_type, str)
                      else " ".join(map(str, raw_image_type))).upper()
        if str(getattr(header, "Modality", "")) != "CT" or any(x in image_type for x in ("LOCALIZER", "SCOUT")):
            continue
        try:
            if int(getattr(header, "NumberOfFrames", 1)) != 1:
                continue
            slope, intercept = float(header.RescaleSlope), float(header.RescaleIntercept)
            if not math.isfinite(slope) or not math.isfinite(intercept) or slope == 0:
                continue
            orientation = np.asarray(header.ImageOrientationPatient, dtype=np.float64)
            position = np.asarray(header.ImagePositionPatient, dtype=np.float64)
            shape = (int(header.Rows), int(header.Columns))
            if orientation.shape != (6,) or position.shape != (3,) or min(shape) < 1:
                continue
            if not np.isfinite(orientation).all() or not np.isfinite(position).all():
                continue
            normal = np.cross(orientation[:3], orientation[3:])
            if np.linalg.norm(normal) < 0.9:
                continue
        except (AttributeError, ValueError, TypeError):
            continue
        candidates.append((source, header, orientation, position, shape))
    if not candidates:
        raise ValueError("No usable single-frame CT slices matching manifest SeriesInstanceUID")
    reference = candidates[0]
    normal = np.cross(reference[2][:3], reference[2][3:])
    normal /= np.linalg.norm(normal)
    for _, _, orientation, _, shape in candidates:
        if shape != reference[4] or not np.allclose(orientation, reference[2], atol=1e-4):
            raise ValueError("DICOM series has inconsistent slice orientation or matrix size")
    candidates.sort(key=lambda item: (float(np.dot(item[3], normal)), str(item[0])))
    # Duplicate reconstructions at one physical location must not give that
    # location extra weight when sampling an evenly spaced stack.
    unique = []
    for item in candidates:
        if not unique or abs(float(np.dot(item[3] - unique[-1][3], normal))) > 1e-3:
            unique.append(item)
    return unique


def _dicom_ct(path: Path, series_uid: str, n_slices: int, image_size: int):
    import pydicom
    headers = _dicom_headers(path, series_uid)
    indices = sample_slice_indices(len(headers), n_slices)
    decoded = {}
    for index in np.unique(indices):
        source, header, *_ = headers[int(index)]
        try:
            ds = pydicom.dcmread(source)
            stored = np.asarray(ds.pixel_array, dtype=np.float32)
            if stored.ndim != 2:
                raise ValueError("Expected a two-dimensional single-frame CT slice")
            hu = stored * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        except (RuntimeError, NotImplementedError) as exc:
            # SimpleITK/GDCM handles compressed CT if pydicom lacks a codec.
            # SimpleITK already applies DICOM rescale slope/intercept.
            try:
                import SimpleITK as sitk
                decoded_image = sitk.GetArrayFromImage(sitk.ReadImage(str(source)))
                if decoded_image.shape[0] != 1:
                    raise ValueError("Expected a single CT frame")
                hu = np.asarray(decoded_image[0], dtype=np.float32)
            except ImportError:
                raise RuntimeError("Compressed CT needs a pydicom codec or SimpleITK") from exc
        if tuple(hu.shape) != (int(header.Rows), int(header.Columns)):
            raise ValueError("Decoded CT dimensions disagree with DICOM header")
        decoded[int(index)] = _finish_slices(torch.from_numpy(hu)[None, None], image_size, "hu")[0]
    return torch.stack([decoded[int(index)] for index in indices])


def _labels(row: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    labels = {}
    for name, count in (("subtype", 2), ("stage", 3)):
        mask = _number(row.get(f"{name}_mask"), 0.0)
        value = _number(row.get(name), -100.0)
        if mask not in (0.0, 1.0) or (mask and (value != int(value) or not 0 <= value < count)):
            raise ValueError(f"Invalid {name} target or mask")
        labels[name] = torch.tensor(int(value) if mask else -100, dtype=torch.long)
        labels[f"{name}_mask"] = torch.tensor(bool(mask))
    for outcome in ("os", "dfs"):
        mask = _number(row.get(f"{outcome}_mask"), 0.0)
        time = _number(row.get(f"{outcome}_time"))
        event = _number(row.get(f"{outcome}_event"))
        if mask not in (0.0, 1.0) or (mask and (not math.isfinite(time) or time < 0 or event not in (0.0, 1.0))):
            raise ValueError(f"Invalid {outcome} time, event, or mask")
        labels[f"{outcome}_time"] = torch.tensor(time if mask else 0.0, dtype=torch.float32)
        labels[f"{outcome}_event"] = torch.tensor(event if mask else 0.0, dtype=torch.float32)
        labels[f"{outcome}_mask"] = torch.tensor(bool(mask))
        targets, masks = [], []
        for year in HORIZON_YEARS:
            target = _number(row.get(f"{outcome}_y{year}"))
            target_mask = _number(row.get(f"{outcome}_m{year}"), 0.0)
            if target_mask not in (0.0, 1.0) or (target_mask and (not mask or target not in (0.0, 1.0))):
                raise ValueError(f"Invalid {outcome} horizon target or mask")
            targets.append(target if target_mask else 0.0)
            masks.append(bool(target_mask))
        labels[f"{outcome}_targets"] = torch.tensor(targets, dtype=torch.float32)
        labels[f"{outcome}_target_mask"] = torch.tensor(masks, dtype=torch.bool)
    return labels


class NLSTDataset(Dataset):
    """One whole-CT series and clinical vector per manifest row.

    Raw image directories are read only. Optional ``cache_dir`` stores resized,
    windowed tensors outside the image directory. Keys include source file sizes
    and modification times and preprocessing configuration. Filenames and
    patient IDs are not copied into processed cache payloads.
    """
    def __init__(self, manifest, split: str, clinical_preprocessor: ClinicalPreprocessor,
                 n_slices: int = 30, image_size: int = 224,
                 cached_intensity_mode: str | None = None, cache_dir: str | Path | None = None):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if n_slices < 1 or image_size < 1:
            raise ValueError("n_slices and image_size must be positive")
        if cached_intensity_mode not in (None, "hu", "unit"):
            raise ValueError("cached_intensity_mode must be None, 'hu', or 'unit'")
        all_rows = _validated_rows(manifest)
        self.rows = [r for r in all_rows if r.get("split") == split]
        self.preprocessor = clinical_preprocessor
        self.preprocessor.to_dict()
        self.n_slices, self.image_size = int(n_slices), int(image_size)
        self.cached_intensity_mode = cached_intensity_mode
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        for row in self.rows:
            if not str(row.get("image_path", "")).strip() or not str(row.get("series_instance_uid", "")).strip():
                raise ValueError("Every selected row needs image_path and series_instance_uid")
            _labels(row)

    def __len__(self):
        return len(self.rows)

    def _load_ct(self, row):
        source = Path(str(row["image_path"])).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        cache_path = None
        if self.cache_dir is not None:
            if source.is_dir() and (self.cache_dir == source or source in self.cache_dir.parents):
                raise ValueError("cache_dir must be outside the raw DICOM directory")
            files = sorted(p for p in source.rglob("*") if p.is_file()) if source.is_dir() else [source]
            identity = {
                "version": 1, "path": str(source), "series_uid": str(row["series_instance_uid"]),
                "n_slices": self.n_slices, "image_size": self.image_size,
                "cached_intensity_mode": self.cached_intensity_mode,
                "files": [(str(p.relative_to(source) if source.is_dir() else p.name), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
            }
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"{key}.pt"
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                ct = cached.get("ct") if isinstance(cached, dict) else None
                if (isinstance(ct, torch.Tensor) and tuple(ct.shape) == (self.n_slices, 1, self.image_size, self.image_size)
                        and torch.isfinite(ct).all() and not torch.any((ct < 0) | (ct > 1))):
                    return ct.float()
                raise ValueError("Invalid processed CT cache; remove the corrupt cache entry")
        if source.is_dir():
            ct = _dicom_ct(source, str(row["series_instance_uid"]), self.n_slices, self.image_size)
        else:
            ct = _prepared_ct(source, str(row["series_instance_uid"]), self.n_slices, self.image_size, self.cached_intensity_mode)
        if cache_path is not None:
            temporary = cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                torch.save({"ct": ct, "intensity_mode": "unit", "format_version": 1}, temporary)
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        return ct

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "ct": self._load_ct(row),
            "clinical": torch.from_numpy(self.preprocessor.transform(row)),
            "patient_id": str(row["patient_id"]),
            "labels": _labels(row),
        }


NLSTPrognosisDataset = NLSTDataset


