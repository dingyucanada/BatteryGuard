"""Adapter for common MATR/Severson MATLAB battery dataset layouts.

The public Severson batches exist both as MATLAB v7 files readable by SciPy and
as MATLAB v7.3/HDF5 files.  The latter path is supported when ``h5py`` is
present; it remains an optional dependency so the offline synthetic demo does
not require it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from batteryguard.ingestion.standard import CanonicalDataset, DataIngestionError, empty_timeseries
from batteryguard.schemas.data import CellRecord, CycleRecord, TimeSeriesPoint


def _normalized(name: object) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items() if not str(key).startswith("__")}
    if hasattr(value, "_fieldnames"):
        return {str(name): getattr(value, name) for name in value._fieldnames}
    if isinstance(value, np.void) and value.dtype.names:
        return {str(name): value[name] for name in value.dtype.names}
    raise DataIngestionError(f"expected MATLAB struct, got {type(value).__name__}")


def _lookup(record: Mapping[str, Any], aliases: Iterable[str], default: Any = None) -> Any:
    # Check exact spellings first: Severson uses both ``t`` (time) and ``T``
    # (temperature), which intentionally collide under case normalization.
    for alias in aliases:
        if alias in record:
            return record[alias]
    indexed = {_normalized(key): value for key, value in record.items()}
    for alias in aliases:
        key = _normalized(alias)
        if key in indexed:
            return indexed[key]
    return default


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    array = np.asarray(value, dtype=object).reshape(-1)
    if array.size == 0:
        return default
    item = array[0]
    if isinstance(item, np.generic):
        item = item.item()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return item


def _numeric_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return array


def _infer_batch_size(batch: Mapping[str, Any]) -> int:
    for aliases in (
        ("cell_life", "cycle_life", "life"),
        ("barcode", "cell_id", "cellid"),
        ("policy_readable", "protocol_id", "policy"),
    ):
        value = _lookup(batch, aliases)
        if value is None or isinstance(value, str | bytes):
            continue
        array = np.asarray(value, dtype=object)
        if array.size > 1:
            return int(array.size)
    return 1


def _select_cell(value: Any, index: int, count: int) -> Any:
    if count == 1:
        return value
    if isinstance(value, Mapping):
        return {str(key): _select_cell(item, index, count) for key, item in value.items()}
    if isinstance(value, str | bytes):
        return value
    if isinstance(value, Sequence) and not isinstance(value, np.ndarray):
        return value[index] if len(value) == count else value
    if isinstance(value, np.ndarray):
        squeezed = np.squeeze(value)
        if squeezed.ndim == 0:
            return squeezed.item()
        if squeezed.dtype == object and squeezed.size == count:
            return squeezed.reshape(-1)[index]
        if squeezed.shape[-1] == count:
            return np.take(squeezed, index, axis=-1)
        if squeezed.shape[0] == count:
            return np.take(squeezed, index, axis=0)
        if squeezed.size == count:
            return squeezed.reshape(-1)[index]
    return value


def _batch_records(batch: Any) -> list[dict[str, Any]]:
    if isinstance(batch, np.ndarray) and batch.dtype.names:
        return [_mapping(item) for item in batch.reshape(-1)]
    if isinstance(batch, list | tuple):
        return [_mapping(item) for item in batch]
    if isinstance(batch, np.ndarray) and batch.dtype == object:
        return [_mapping(item) for item in batch.reshape(-1)]
    record = _mapping(batch)
    count = _infer_batch_size(record)
    return [
        {key: _select_cell(value, index, count) for key, value in record.items()}
        for index in range(count)
    ]


def _cycle_structs(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [_mapping(item) for item in value]
    if isinstance(value, np.ndarray) and (value.dtype == object or value.dtype.names):
        return [_mapping(item) for item in value.reshape(-1)]
    record = _mapping(value)
    # HDF5 decoding can yield a field-wise object array, one object per cycle.
    object_lengths = [
        np.asarray(item, dtype=object).size
        for item in record.values()
        if isinstance(item, np.ndarray) and item.dtype == object
    ]
    if object_lengths and max(object_lengths) > 1:
        count = max(object_lengths)
        return [
            {key: _select_cell(item, index, count) for key, item in record.items()}
            for index in range(count)
        ]
    return [record]


def _summary_vectors(summary: Any) -> dict[str, np.ndarray]:
    if summary is None:
        return {}
    if isinstance(summary, pd.DataFrame):
        return {str(column): summary[column].to_numpy() for column in summary.columns}
    if isinstance(summary, list | tuple) or (
        isinstance(summary, np.ndarray) and (summary.dtype == object or summary.dtype.names)
    ):
        records = [_mapping(item) for item in np.asarray(summary, dtype=object).reshape(-1)]
        if not records:
            return {}
        keys = sorted({key for record in records for key in record})
        return {
            key: np.asarray([_scalar(record.get(key), np.nan) for record in records])
            for key in keys
        }
    record = _mapping(summary)
    result: dict[str, np.ndarray] = {}
    for key, value in record.items():
        if isinstance(value, str | bytes):
            continue
        try:
            array = np.asarray(value).squeeze()
        except (TypeError, ValueError):
            continue
        if array.ndim <= 1:
            result[str(key)] = array.reshape(-1)
    return result


def _summary_value(
    vectors: Mapping[str, np.ndarray], aliases: Iterable[str], index: int
) -> float | None:
    indexed = {_normalized(key): value for key, value in vectors.items()}
    for alias in aliases:
        values = indexed.get(_normalized(alias))
        if values is None or values.size == 0 or index >= values.size:
            continue
        try:
            value = float(values.reshape(-1)[index])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _time_multiplier(unit: str, values: np.ndarray) -> float:
    normalized = unit.lower()
    if normalized in {"minute", "minutes", "min"}:
        return 60.0
    if normalized in {"second", "seconds", "s"}:
        return 1.0
    if normalized == "auto":
        finite = values[np.isfinite(values)]
        # Severson cycle time is in minutes; a complete cycle below five hours
        # in raw units is overwhelmingly likely to be minutes.
        return 60.0 if finite.size and float(np.max(finite)) < 300.0 else 1.0
    raise DataIngestionError("source_time_unit must be 'minutes', 'seconds', or 'auto'")


def _duration_from_progress(time_s: np.ndarray, progress: np.ndarray) -> float | None:
    if time_s.size < 2 or progress.size != time_s.size:
        return None
    increments = np.diff(progress, prepend=progress[0])
    durations = np.diff(time_s, prepend=time_s[0])
    selected = durations[increments > 1e-10]
    total = float(np.sum(selected))
    return total if np.isfinite(total) and total > 0 else None


def _raw_cycle(
    cell_id: str,
    cycle_index: int,
    raw: Mapping[str, Any],
    *,
    source_time_unit: str,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    current = _numeric_vector(_lookup(raw, ("I", "current", "current_a")))
    voltage = _numeric_vector(_lookup(raw, ("V", "voltage", "voltage_v")))
    charge = _numeric_vector(_lookup(raw, ("Qc", "q_charge", "charge_capacity_ah")))
    discharge = _numeric_vector(_lookup(raw, ("Qd", "q_discharge", "discharge_capacity_ah")))
    time = _numeric_vector(_lookup(raw, ("t", "time", "time_s")))
    temperature = _numeric_vector(_lookup(raw, ("T", "temperature", "temperature_c")))

    required = {"current": current, "voltage": voltage, "time": time}
    available_lengths = [array.size for array in required.values() if array.size]
    if not available_lengths:
        return [], {
            "charge_capacity": None,
            "discharge_capacity": None,
            "charge_time": None,
            "discharge_time": None,
            "avg_temp": None,
            "max_temp": None,
        }
    length = available_lengths[0]
    mismatched = {name: array.size for name, array in required.items() if array.size != length}
    if mismatched:
        raise DataIngestionError(
            f"{cell_id} cycle {cycle_index} has mismatched time-series lengths: {mismatched}"
        )
    if charge.size not in {0, length} or discharge.size not in {0, length}:
        raise DataIngestionError(
            f"{cell_id} cycle {cycle_index} capacity arrays do not align with time-series"
        )
    if charge.size == 0:
        charge = np.zeros(length, dtype=float)
    if discharge.size == 0:
        discharge = np.zeros(length, dtype=float)
    if temperature.size not in {0, length}:
        raise DataIngestionError(
            f"{cell_id} cycle {cycle_index} temperature array does not align with time-series"
        )
    multiplier = _time_multiplier(source_time_unit, time)
    time_s = time * multiplier
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) < 0):
        raise DataIngestionError(
            f"{cell_id} cycle {cycle_index} time must be finite and non-decreasing"
        )

    rows: list[dict[str, Any]] = []
    for sample_index in range(length):
        current_value = float(current[sample_index])
        if current_value > 1e-8:
            step = "charge"
        elif current_value < -1e-8:
            step = "discharge"
        else:
            step = "rest"
        temperature_value = (
            float(temperature[sample_index])
            if temperature.size and np.isfinite(temperature[sample_index])
            else None
        )
        rows.append(
            {
                "cell_id": cell_id,
                "cycle_index": cycle_index,
                "sample_index": sample_index,
                "time_s": float(time_s[sample_index]),
                "current_a": current_value,
                "voltage_v": float(voltage[sample_index]),
                "charge_capacity_ah": float(max(charge[sample_index], 0.0)),
                "discharge_capacity_ah": float(max(discharge[sample_index], 0.0)),
                "temperature_c": temperature_value,
                "step_type": step,
            }
        )
    finite_temperature = temperature[np.isfinite(temperature)]
    return rows, {
        "charge_capacity": float(np.nanmax(charge)) if charge.size else None,
        "discharge_capacity": float(np.nanmax(discharge)) if discharge.size else None,
        "charge_time": _duration_from_progress(time_s, charge),
        "discharge_time": _duration_from_progress(time_s, discharge),
        "avg_temp": float(np.mean(finite_temperature)) if finite_temperature.size else None,
        "max_temp": float(np.max(finite_temperature)) if finite_temperature.size else None,
    }


def _positive(value: float | None) -> float | None:
    return value if value is not None and np.isfinite(value) and value > 0 else None


class MATRAdapter:
    """Convert common Severson batch structs into canonical tables."""

    def __init__(
        self,
        *,
        chemistry: str = "LFP_graphite",
        nominal_capacity_ah: float = 1.10,
        eol_threshold: float = 0.80,
        source_time_unit: str = "minutes",
    ) -> None:
        if nominal_capacity_ah <= 0:
            raise ValueError("nominal_capacity_ah must be positive")
        if not 0 < eol_threshold < 1:
            raise ValueError("eol_threshold must be in (0, 1)")
        _time_multiplier(source_time_unit, np.asarray([], dtype=float))
        self.chemistry = chemistry
        self.nominal_capacity_ah = float(nominal_capacity_ah)
        self.eol_threshold = float(eol_threshold)
        self.source_time_unit = source_time_unit

    def load(self, source: str | Path, *, dataset_id: str | None = None) -> CanonicalDataset:
        path = Path(source)
        if not path.is_file():
            raise DataIngestionError(f"MATR source does not exist: {path}")
        if path.suffix.lower() != ".mat":
            raise DataIngestionError(f"MATR adapter expects a .mat file, got: {path.name}")
        root = self._load_mat(path)
        batch = root.get("batch")
        if batch is None:
            available = sorted(key for key in root if not key.startswith("__"))
            raise DataIngestionError(
                f"{path.name} has no 'batch' variable; available top-level keys={available}"
            )
        records = _batch_records(batch)
        if not records:
            raise DataIngestionError(f"{path.name} contains an empty batch")

        cell_rows: list[dict[str, Any]] = []
        cycle_rows: list[dict[str, Any]] = []
        timeseries_rows: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            default_id = f"{path.stem}-cell-{index + 1:04d}"
            raw_id = _scalar(_lookup(record, ("cell_id", "cellid", "barcode", "name")))
            cell_id = str(raw_id).strip() if raw_id is not None else default_id
            if not cell_id:
                cell_id = default_id
            protocol_raw = _scalar(
                _lookup(record, ("protocol_id", "policy_readable", "policy", "charge_policy"))
            )
            protocol_id = str(protocol_raw).strip() if protocol_raw is not None else "unknown"
            if not protocol_id:
                protocol_id = "unknown"

            life_raw = _scalar(_lookup(record, ("cycle_life", "cell_life", "life")))
            censored_raw = _scalar(_lookup(record, ("censored",)), False)
            censored = bool(censored_raw)
            cycle_life: int | None
            try:
                numeric_life = float(life_raw) if life_raw is not None else float("nan")
                cycle_life = int(round(numeric_life)) if np.isfinite(numeric_life) else None
            except (TypeError, ValueError):
                cycle_life = None
            if cycle_life is not None and cycle_life <= 0:
                cycle_life = None
            if cycle_life is None:
                censored = True
            if censored:
                cycle_life = None

            batch_raw = _scalar(_lookup(record, ("batch_id", "batch")))
            chemistry = str(
                _scalar(_lookup(record, ("chemistry",)), self.chemistry)
            ).strip()
            nominal_raw = _scalar(
                _lookup(record, ("nominal_capacity_ah", "nominalcapacity", "capacity")),
                self.nominal_capacity_ah,
            )
            try:
                nominal_capacity = float(nominal_raw)
            except (TypeError, ValueError) as exc:
                raise DataIngestionError(f"{cell_id} has invalid nominal capacity") from exc
            eol_raw = _scalar(_lookup(record, ("eol_threshold",)), self.eol_threshold)
            cell_rows.append(
                {
                    "cell_id": cell_id,
                    "chemistry": chemistry,
                    "nominal_capacity_ah": nominal_capacity,
                    "batch_id": str(batch_raw) if batch_raw is not None else path.stem,
                    "protocol_id": protocol_id,
                    "cycle_life": cycle_life,
                    "eol_threshold": float(eol_raw),
                    "censored": censored,
                }
            )

            summary = _summary_vectors(_lookup(record, ("summary", "cycle_summary")))
            raw_cycles = _cycle_structs(_lookup(record, ("cycles", "cycle_data")))
            summary_lengths = [values.size for values in summary.values()]
            cycle_count = max([len(raw_cycles), *summary_lengths], default=0)
            if cycle_count == 0:
                raise DataIngestionError(f"{cell_id} has neither cycle summaries nor cycle curves")

            cycle_numbers = next(
                (
                    values
                    for key, values in summary.items()
                    if _normalized(key) in {"cycle", "cycleindex", "cycleidx"}
                ),
                np.asarray([], dtype=float),
            )
            for cycle_offset in range(cycle_count):
                if cycle_offset < cycle_numbers.size:
                    try:
                        cycle_index = int(round(float(cycle_numbers[cycle_offset])))
                    except (TypeError, ValueError):
                        cycle_index = cycle_offset + 1
                else:
                    cycle_index = cycle_offset + 1
                if cycle_index < 1:
                    cycle_index = cycle_offset + 1

                raw_rows: list[dict[str, Any]] = []
                derived: dict[str, float | None] = {}
                if cycle_offset < len(raw_cycles):
                    raw_rows, derived = _raw_cycle(
                        cell_id,
                        cycle_index,
                        raw_cycles[cycle_offset],
                        source_time_unit=self.source_time_unit,
                    )
                    timeseries_rows.extend(raw_rows)

                q_charge = _positive(
                    _summary_value(
                        summary,
                        ("QCharge", "charge_capacity_ah", "charge_capacity"),
                        cycle_offset,
                    )
                ) or _positive(derived.get("charge_capacity"))
                q_discharge = _positive(
                    _summary_value(
                        summary,
                        ("QDischarge", "discharge_capacity_ah", "discharge_capacity"),
                        cycle_offset,
                    )
                ) or _positive(derived.get("discharge_capacity"))
                if q_charge is None or q_discharge is None:
                    raise DataIngestionError(
                        f"{cell_id} cycle {cycle_index} lacks positive charge/discharge capacity"
                    )
                ce = _summary_value(
                    summary,
                    ("CE", "coulombic_efficiency", "coulombicefficiency"),
                    cycle_offset,
                )
                if ce is None:
                    ce = q_discharge / q_charge

                charge_time = _positive(
                    _summary_value(
                        summary,
                        ("charge_time_s", "chargetime", "charge_time"),
                        cycle_offset,
                    )
                )
                if charge_time is not None:
                    charge_time *= _time_multiplier(
                        self.source_time_unit, np.asarray([charge_time], dtype=float)
                    )
                charge_time = charge_time or _positive(derived.get("charge_time"))
                discharge_time = _positive(
                    _summary_value(
                        summary,
                        ("discharge_time_s", "dischargetime", "discharge_time"),
                        cycle_offset,
                    )
                )
                if discharge_time is not None:
                    discharge_time *= _time_multiplier(
                        self.source_time_unit, np.asarray([discharge_time], dtype=float)
                    )
                discharge_time = discharge_time or _positive(derived.get("discharge_time"))
                if charge_time is None or discharge_time is None:
                    raise DataIngestionError(
                        f"{cell_id} cycle {cycle_index} lacks charge/discharge duration; "
                        "provide cycle curves or explicit time fields"
                    )

                avg_temp = _summary_value(
                    summary, ("Tavg", "avg_temp_c", "temperature_avg"), cycle_offset
                )
                max_temp = _summary_value(
                    summary, ("Tmax", "max_temp_c", "temperature_max"), cycle_offset
                )
                avg_temp = avg_temp if avg_temp is not None else derived.get("avg_temp")
                max_temp = max_temp if max_temp is not None else derived.get("max_temp")
                dcir = _positive(
                    _summary_value(summary, ("IR", "dcir_ohm", "dcir"), cycle_offset)
                )
                cycle_rows.append(
                    {
                        "cell_id": cell_id,
                        "cycle_index": cycle_index,
                        "charge_capacity_ah": q_charge,
                        "discharge_capacity_ah": q_discharge,
                        "coulombic_efficiency": float(ce),
                        "charge_time_s": charge_time,
                        "discharge_time_s": discharge_time,
                        "dcir_ohm": dcir,
                        "avg_temp_c": avg_temp,
                        "max_temp_c": max_temp,
                        "charge_energy_wh": None,
                        "discharge_energy_wh": None,
                    }
                )

        timeseries = (
            pd.DataFrame(timeseries_rows, columns=list(TimeSeriesPoint.model_fields))
            if timeseries_rows
            else empty_timeseries()
        )
        return CanonicalDataset(
            cells=pd.DataFrame(cell_rows, columns=list(CellRecord.model_fields)),
            cycles=pd.DataFrame(cycle_rows, columns=list(CycleRecord.model_fields)),
            timeseries=timeseries,
            dataset_id=dataset_id or path.stem,
        ).validate()

    @staticmethod
    def _load_mat(path: Path) -> dict[str, Any]:
        try:
            loaded = loadmat(path, simplify_cells=True)
            return {str(key): value for key, value in loaded.items()}
        except NotImplementedError:
            return MATRAdapter._load_hdf5_mat(path)
        except ValueError as exc:
            if "7.3" in str(exc) or "HDF" in str(exc).upper():
                return MATRAdapter._load_hdf5_mat(path)
            raise DataIngestionError(f"failed to parse MATLAB file {path}: {exc}") from exc
        except OSError as exc:
            raise DataIngestionError(f"failed to read MATLAB file {path}: {exc}") from exc

    @staticmethod
    def _load_hdf5_mat(path: Path) -> dict[str, Any]:
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DataIngestionError(
                "MATLAB v7.3/HDF5 input requires the optional 'h5py' package; "
                "install h5py or convert the batch to standardized Parquet"
            ) from exc

        def decode(node: Any, handle: Any) -> Any:
            if isinstance(node, h5py.Group):
                return {str(key): decode(value, handle) for key, value in node.items()}
            data = node[()]
            matlab_class = node.attrs.get("MATLAB_class", b"")
            if isinstance(matlab_class, np.ndarray) and matlab_class.size:
                matlab_class = matlab_class.reshape(-1)[0]
            if isinstance(matlab_class, bytes):
                matlab_class = matlab_class.decode("ascii", errors="ignore")
            if matlab_class == "char":
                return "".join(chr(int(value)) for value in np.asarray(data).reshape(-1) if value)
            if h5py.check_dtype(ref=np.asarray(data).dtype) is not None:
                decoded = [decode(handle[reference], handle) for reference in np.asarray(data).reshape(-1)]
                return decoded[0] if len(decoded) == 1 else decoded
            return np.asarray(data).squeeze()

        try:
            with h5py.File(path, "r") as handle:
                return {
                    str(key): decode(value, handle)
                    for key, value in handle.items()
                    if not str(key).startswith("#")
                }
        except Exception as exc:
            raise DataIngestionError(f"failed to decode MATLAB v7.3 file {path}: {exc}") from exc


def load_matr(source: str | Path, **kwargs: Any) -> CanonicalDataset:
    """Functional entry point accepted by the ingestion registry."""

    dataset_id = kwargs.pop("dataset_id", None)
    adapter = MATRAdapter(**kwargs)
    return adapter.load(source, dataset_id=dataset_id)
