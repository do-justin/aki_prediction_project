"""Native packet loading and signal quality for the isolated EEG study."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from struct import error as StructError
from time import perf_counter
from typing import Any
import zlib

import numpy as np
import pycatch22
import scipy.signal
import vitaldb


EEG_WAVEFORM = "BIS/EEG1_WAV"
EEG_FALLBACK = "BIS/EEG2_WAV"
SQI_TRACK = "BIS/SQI"
EEG_TRACKS = (EEG_WAVEFORM, EEG_FALLBACK, SQI_TRACK)
PROTOCOL = "eeg-qc-v1"
NATIVE_RATE = 128
WINDOW_SAMPLES = 10 * NATIVE_RATE
STRIDE_SAMPLES = 5 * NATIVE_RATE
SQI_MAX_AGE_SECONDS = 1.0
SQI_MINIMUM = 50
MAX_AMPLITUDE_UV = 100
MIN_SD_UV = 1e-6
FILTER_ORDER = 4
FILTER_BAND_HZ = (0.5, 50.0)
UNIT_TO_UV = {"uv": 1.0, "mv": 1_000.0, "v": 1_000_000.0}
AGGREGATES = ("mean", "std", "min", "max")


@dataclass
class EEGCleanResult:
    """Operative-grid arrays and QC. Source arrays are empty if packets could not be mapped."""

    times: np.ndarray
    observed_times: np.ndarray
    raw: np.ndarray
    sqi: np.ndarray
    clean: np.ndarray | None
    masks: dict[str, np.ndarray]
    qc: dict[str, Any]


def protocol_fingerprint() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "reader": "vitaldb.VitalFile local native packet records",
        "versions": {name: version(name) for name in ("vitaldb", "numpy", "scipy", "pycatch22")},
        "channels": list(EEG_TRACKS),
        "fallback": "absent_empty_unloadable_preferred_track_only",
        "native_rate_hz": NATIVE_RATE,
        "whole_rate_hz": NATIVE_RATE,
        "window_rate_hz": NATIVE_RATE,
        "operative_grid": "int(opstart*rate):int(opend*rate) on case-relative clock",
        "packet_mapping": "nearest_slot_half_ties_later_contradictory_collisions_missing",
        "conversion_to_uv": UNIT_TO_UV.copy(),
        "integer_waveform_scaling": "value_times_gain_plus_offset",
        "sqi_alignment": "bounded_previous_observation_actual_eeg_time",
        "sqi_duplicates": "collapse_agreement_contradictions_unknown",
        "sqi_max_age_seconds": SQI_MAX_AGE_SECONDS,
        "sqi_valid_range": [0, 100],
        "sqi_minimum_inclusive": SQI_MINIMUM,
        "raw_max_amplitude_uv_inclusive": MAX_AMPLITUDE_UV,
        "unusable_policy": "union_over_entire_expected_grid_max_inclusive_5_percent",
        "interpolation": "linear_interior_nearest_finite_endpoints_once",
        "filter": {
            "order": FILTER_ORDER, "band_hz": list(FILTER_BAND_HZ),
            "type": "butterworth_sos_bandpass_forward_backward",
            "padtype": "odd", "padlen": None,
        },
        "minimum_sd_uv": MIN_SD_UV,
        "features": "pycatch22.catch22_all(catch24=True)",
        "window_samples": WINDOW_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "window_aggregates": list(AGGREGATES),
        "descriptor_missingness": "nonfinite_to_nan_propagate_across_included_windows",
        "window_sd_ddof": 1,
    }


def raw_recording_path(raw_dir: str | Path, caseid: int) -> Path:
    return Path(raw_dir) / "vital_files" / f"{int(caseid):04d}.vital"


def _track_status(recording, name: str) -> str:
    track = recording.trks.get(name)
    if track is None:
        return "absent"
    if not any(np.size(record.get("val")) for record in track.recs):
        return "empty"
    return "present"


def _read_recording(path: Path, names: tuple[str, ...]):
    recording = vitaldb.VitalFile(str(path), track_names=list(names))
    if not np.isfinite(recording.dtstart) or recording.dtstart <= 0:
        raise OSError(f"VitalDB recording has no verified case clock: {path.name}")
    return recording


def _select_source(path: Path):
    """Read both tracks once in the normal case, then support track-local decode recovery."""
    if not path.is_file():
        raise FileNotFoundError(path)
    decode_errors = (OSError, EOFError, StructError, KeyError, ValueError)
    try:
        recording = _read_recording(path, EEG_TRACKS)
    except decode_errors:
        track_status = {EEG_WAVEFORM: "unreadable", EEG_FALLBACK: "not_checked"}
        read_errors = []
        for name in (EEG_WAVEFORM, EEG_FALLBACK):
            try:
                recording = _read_recording(path, (name, SQI_TRACK))
            except decode_errors as exc:
                track_status[name] = "unreadable"
                read_errors.append(exc)
                continue
            track_status[name] = _track_status(recording, name)
            if track_status[name] == "present":
                return recording, name, track_status
        if read_errors:
            raise OSError(f"Unable to decode an EEG track from {path.name}") from read_errors[-1]
        return recording, None, track_status

    track_status = {name: _track_status(recording, name) for name in (EEG_WAVEFORM, EEG_FALLBACK)}
    selected = next((name for name, status in track_status.items() if status == "present"), None)
    return recording, selected, track_status


def _scale_to_uv(track) -> tuple[float, float, float]:
    unit = str(track.unit).strip().lower().replace("µ", "u").replace("μ", "u")
    if unit not in UNIT_TO_UV:
        raise ValueError("unexpected_waveform_unit")
    # The pinned VitalDB reader stores floating packets in physical units.
    gain, offset = (float(track.gain), float(track.offset)) if track.fmt > 2 else (1.0, 0.0)
    if not np.isfinite([gain, offset]).all() or gain == 0:
        raise ValueError("invalid_waveform_gain_or_offset")
    return gain, offset, UNIT_TO_UV[unit]


def _different(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a != b) & ~(np.isnan(a) & np.isnan(b))


def _map_waveform(
    records: list[dict], origin: float, start: int, stop: int,
    gain: float, offset: float, scale: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Place each native sample on its nearest slot without stretching the waveform."""
    count = len(records)
    packet_times = np.fromiter((rec["dt"] - origin for rec in records), dtype=float, count=count)
    lengths = np.fromiter((np.size(rec["val"]) for rec in records), dtype=np.int64, count=count)
    finite = np.isfinite(packet_times)
    packet_slots = np.zeros(count, dtype=np.int64)
    packet_slots[finite] = np.floor(packet_times[finite] * NATIVE_RATE + 0.5).astype(np.int64)
    selected = np.flatnonzero(finite & (packet_slots < stop) & (packet_slots + lengths > start))
    raw = np.full(stop - start, np.nan)
    observed_times = np.full_like(raw, np.nan)
    audit = {
        "waveform_packets": count,
        "operative_waveform_packets": int(selected.size),
        "nonfinite_waveform_packet_timestamps": int(np.count_nonzero(~finite)),
        "nonmonotone_waveform_packet_timestamps": int(np.count_nonzero(np.diff(packet_times[finite]) < 0)),
        "observed_packet_samples": 0,
        "waveform_collision_slots": 0,
        "waveform_conflict_slots": 0,
        "max_abs_packet_grid_offset_seconds": None,
    }
    if not selected.size:
        return raw, observed_times, audit

    selected_lengths = lengths[selected]
    packet_offsets = np.cumsum(selected_lengths) - selected_lengths
    within_packet = np.arange(selected_lengths.sum()) - np.repeat(packet_offsets, selected_lengths)
    slots = np.repeat(packet_slots[selected] - start, selected_lengths) + within_packet
    times = np.repeat(packet_times[selected], selected_lengths) + within_packet / NATIVE_RATE
    values = np.concatenate([records[i]["val"] for i in selected]).astype(float)
    inside = (slots >= 0) & (slots < raw.size)
    slots, times, values = slots[inside], times[inside], values[inside]
    with np.errstate(over="ignore", invalid="ignore"):
        values = (values * gain + offset) * scale
    audit["observed_packet_samples"] = int(slots.size)
    audit["max_abs_packet_grid_offset_seconds"] = float(
        np.max(np.abs(packet_times[selected] - packet_slots[selected] / NATIVE_RATE))
    )
    if np.any(slots[1:] <= slots[:-1]):
        order = np.argsort(slots, kind="stable")
        slots, times, values = slots[order], times[order], values[order]
        duplicate = slots[1:] == slots[:-1]
        differs = _different(values[1:], values[:-1]) | (times[1:] != times[:-1])
        collided = np.unique(slots[1:][duplicate])
        conflicts = np.unique(slots[1:][duplicate & differs])
        audit["waveform_collision_slots"] = int(collided.size)
        audit["waveform_conflict_slots"] = int(conflicts.size)
        keep = np.concatenate(([True], ~duplicate)) if slots.size else np.array([], dtype=bool)
        slots, times, values = slots[keep], times[keep], values[keep]
    else:
        conflicts = np.array([], dtype=np.int64)

    raw[slots] = values
    observed_times[slots] = times
    raw[conflicts] = np.nan
    observed_times[conflicts] = np.nan
    return raw, observed_times, audit


def _align_sqi(track, origin: float, timestamps: np.ndarray) -> tuple[np.ndarray, dict]:
    sqi = np.full(timestamps.shape, np.nan)
    entries = [] if track is None else track.recs
    audit = {
        "sqi_status": "available",
        "sqi_observations": len(entries),
        "sqi_nonfinite_timestamps": 0,
        "sqi_nonmonotone_timestamps": 0,
        "sqi_duplicate_timestamps": 0,
        "sqi_conflicting_timestamps": 0,
        "sqi_invalid_values": 0,
        "sqi_inter_observation_median_seconds": None,
    }
    if not entries:
        audit["sqi_status"] = "absent" if track is None else "empty"
        return sqi, audit
    if track.type != 2 or str(track.unit).strip() != "%":
        audit["sqi_status"] = "invalid_numeric_track_metadata"
        return sqi, audit

    times = np.fromiter((rec["dt"] - origin for rec in entries), dtype=float, count=len(entries))
    values = np.fromiter((rec["val"] for rec in entries), dtype=float, count=len(entries))
    audit["sqi_invalid_values"] = int(np.count_nonzero(~np.isfinite(values) | (values < 0) | (values > 100)))
    finite = np.isfinite(times)
    audit["sqi_nonfinite_timestamps"] = int(np.count_nonzero(~finite))
    times, values = times[finite], values[finite]
    if not times.size:
        return sqi, audit
    audit["sqi_nonmonotone_timestamps"] = int(np.count_nonzero(np.diff(times) < 0))
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    duplicate = times[1:] == times[:-1]
    conflicts = np.unique(times[1:][duplicate & _different(values[1:], values[:-1])])
    audit["sqi_duplicate_timestamps"] = int(np.count_nonzero(duplicate))
    audit["sqi_conflicting_timestamps"] = int(conflicts.size)
    keep = np.concatenate(([True], ~duplicate))
    times, values = times[keep], values[keep]
    values[np.searchsorted(times, conflicts)] = np.nan
    if times.size > 1:
        audit["sqi_inter_observation_median_seconds"] = float(np.median(np.diff(times)))

    latest = np.searchsorted(times, timestamps, side="right") - 1
    index = np.maximum(latest, 0)
    valid = (
        (latest >= 0) & (timestamps - times[index] <= SQI_MAX_AGE_SECONDS)
        & np.isfinite(values[index]) & (values[index] >= 0) & (values[index] <= 100)
    )
    sqi[valid] = values[index[valid]]
    return sqi, audit


def _gap_audit(unusable: np.ndarray) -> dict[str, int | float]:
    starts_and_stops = np.flatnonzero(np.diff(unusable, prepend=False, append=False))
    starts, stops = starts_and_stops[::2], starts_and_stops[1::2]
    lengths = stops - starts
    interior = lengths[(starts > 0) & (stops < unusable.size)]
    return {
        "unusable_gap_count": int(starts.size),
        "interior_gap_count": int(interior.size),
        "max_interior_gap_seconds": float(interior.max(initial=0) / NATIVE_RATE),
        "leading_gap_seconds": float(lengths[0] / NATIVE_RATE) if starts.size and starts[0] == 0 else 0.0,
        "trailing_gap_seconds": float(lengths[-1] / NATIVE_RATE) if stops.size and stops[-1] == unusable.size else 0.0,
    }


def _mask_audit(raw: np.ndarray, observed_times: np.ndarray, sqi: np.ndarray):
    source_missing = ~np.isfinite(observed_times)
    source_nonfinite = ~source_missing & ~np.isfinite(raw)
    sqi_unknown = ~np.isfinite(sqi)
    low_sqi = sqi < SQI_MINIMUM
    high_amplitude = np.isfinite(raw) & (np.abs(raw) > MAX_AMPLITUDE_UV)
    unusable = source_missing | source_nonfinite | sqi_unknown | low_sqi | high_amplitude
    masks = {
        "source_missing": source_missing,
        "source_nonfinite": source_nonfinite,
        "sqi_unknown": sqi_unknown,
        "sqi_low": low_sqi,
        "raw_amplitude_high": high_amplitude,
        "unusable_union": unusable,
    }
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    audit = {
        "observed_samples": int(raw.size - counts["source_missing"]),
        "observed_duration_seconds": (raw.size - counts["source_missing"]) / NATIVE_RATE,
        "valid_sqi_samples": int(raw.size - counts["sqi_unknown"]),
        "mask_counts": counts,
        "mask_fractions": {name: count / raw.size for name, count in counts.items()},
        **_gap_audit(unusable),
    }
    return masks, audit


def _clean_signal(raw: np.ndarray, unusable: np.ndarray, qc: dict) -> tuple[np.ndarray | None, str | None]:
    # Integer comparison preserves the inclusive 5% boundary for any grid length.
    if qc["mask_counts"]["unusable_union"] * 20 > raw.size:
        return None, "unusable_fraction_above_5_pct"
    good = np.flatnonzero(~unusable)
    if good.size < 2:
        return None, "insufficient_usable_samples"

    sos = scipy.signal.butter(FILTER_ORDER, FILTER_BAND_HZ, btype="bandpass", fs=NATIVE_RATE, output="sos")
    zeros_at_origin = min(np.count_nonzero(sos[:, 2] == 0), np.count_nonzero(sos[:, 5] == 0))
    pad_length = int(3 * (2 * len(sos) + 1 - zeros_at_origin))
    qc["filter_default_pad_samples"] = pad_length
    if raw.size <= pad_length:
        return None, "insufficient_filter_length"

    signal = raw.copy()
    if unusable.any():
        signal[unusable] = np.interp(np.flatnonzero(unusable), good, signal[good])
    qc["interpolated_samples"] = qc["mask_counts"]["unusable_union"]
    qc["interpolated_fraction"] = qc["interpolated_samples"] / raw.size
    qc["leading_filled_seconds"] = qc["leading_gap_seconds"]
    qc["trailing_filled_seconds"] = qc["trailing_gap_seconds"]
    clean = scipy.signal.sosfiltfilt(sos, signal, padtype="odd", padlen=None)
    if not np.isfinite(clean).all():
        return None, "nonfinite_filtered_signal"
    qc["clean_mean_uv"] = float(np.mean(clean))
    qc["clean_sd_uv"] = float(np.std(clean))
    if qc["clean_sd_uv"] < MIN_SD_UV:
        return None, "below_numerical_sd"
    return clean, None


def load_clean_eeg(caseid: int, opstart: float, opend: float, raw_dir: str | Path) -> EEGCleanResult:
    """Load the chosen raw EEG and clean the single common operative signal."""
    qc: dict[str, Any] = {
        "caseid": int(caseid),
        "protocol": PROTOCOL,
        "selected_source": None,
        "fallback_reason": None,
        "expected_samples": 0,
        "observed_samples": 0,
        "interpolated_samples": 0,
    }
    empty = np.array([], dtype=float)
    if not np.isfinite([opstart, opend]).all() or opstart < 0 or opend <= opstart:
        qc.update(status="failed", failure_reason="invalid_operation_bounds")
        return EEGCleanResult(empty, empty, empty, empty, None, {}, qc)
    start, stop = int(opstart * NATIVE_RATE), int(opend * NATIVE_RATE)
    if stop <= start:
        qc.update(status="failed", failure_reason="empty_operation_grid")
        return EEGCleanResult(empty, empty, empty, empty, None, {}, qc)

    recording_path = raw_recording_path(raw_dir, caseid)
    recording, selected, track_status = _select_source(recording_path)
    qc.update(
        selected_source=selected, track_status=track_status,
        fallback_reason=None if selected == EEG_WAVEFORM else track_status[EEG_WAVEFORM],
        source_recording=recording_path.name, case_clock_origin=float(recording.dtstart),
        expected_samples=stop - start, expected_duration_seconds=(stop - start) / NATIVE_RATE,
    )
    times = np.arange(start, stop) / NATIVE_RATE
    sqi_track = recording.trks.get(SQI_TRACK)
    if selected is None:
        _, sqi_audit = _align_sqi(sqi_track, recording.dtstart, empty)
        qc.update(sqi_audit, status="failed", failure_reason="no_eeg_source")
        return EEGCleanResult(times, empty, empty, empty, None, {}, qc)

    track = recording.trks[selected]
    rate = float(track.srate)
    qc.update(native_rate_hz=rate if np.isfinite(rate) else None, source_unit=str(track.unit),
              source_format=int(track.fmt))
    if track.type != 1 or rate != NATIVE_RATE:
        qc.update(status="failed", failure_reason="unexpected_waveform_type_or_rate")
        return EEGCleanResult(times, empty, empty, empty, None, {}, qc)
    try:
        gain, offset, scale = _scale_to_uv(track)
    except ValueError as exc:
        qc.update(status="failed", failure_reason=str(exc))
        return EEGCleanResult(times, empty, empty, empty, None, {}, qc)
    qc.update(applied_gain=gain, applied_offset=offset, physical_unit="µV", unit_scale_to_uv=scale)

    raw, observed_times, timing_audit = _map_waveform(track.recs, recording.dtstart, start, stop, gain, offset, scale)
    sqi, sqi_audit = _align_sqi(
        sqi_track, recording.dtstart, np.where(np.isfinite(observed_times), observed_times, times)
    )
    masks, mask_audit = _mask_audit(raw, observed_times, sqi)
    qc.update(timing_audit)
    qc.update(sqi_audit)
    qc.update(mask_audit)
    clean, reason = _clean_signal(raw, masks["unusable_union"], qc)
    qc.update(status="passed" if reason is None else "failed", failure_reason=reason)
    return EEGCleanResult(times, observed_times, raw, sqi, clean, masks, qc)


def _features(signal: np.ndarray) -> tuple[list[str], np.ndarray]:
    result = pycatch22.catch22_all(signal, catch24=True)
    values = np.asarray(result["values"], dtype=float)
    values[~np.isfinite(values)] = np.nan
    return result["names"], values


def _encoded_masks(masks: dict[str, np.ndarray]) -> dict:
    return {
        "encoding": "np.packbits(bitorder=little), zlib, base64; truncate to expected_samples",
        "data": {
            name: base64.b64encode(zlib.compress(np.packbits(mask, bitorder="little"), level=1)).decode("ascii")
            for name, mask in masks.items()
        },
    }


def extract_eeg_case(caseid: int, opstart: float, opend: float, raw_dir: str | Path):
    """Return both Step 02 records, timings and JSON-serializable QC for one operation."""
    start = perf_counter()
    loaded = load_clean_eeg(caseid, opstart, opend, raw_dir)
    timings = {"eeg_load_grid_qc": perf_counter() - start}
    identity = {"caseid": int(caseid), "waveform": EEG_WAVEFORM}
    qc = loaded.qc
    qc["masks"] = _encoded_masks(loaded.masks)
    qc.update(total_windows=0, retained_windows=0, failed_windows=0)
    if loaded.clean is None:
        reason = qc["failure_reason"]
        qc["branch_status"] = {"full": reason, "windowed": reason}
        return ({name: {**identity, "error": reason} for name in ("full", "windowed")}, timings, qc)

    signal = loaded.clean
    del loaded
    start = perf_counter()
    names, values = _features(signal)
    full = {**identity, **dict(zip(names, values))}
    timings["process_full"] = perf_counter() - start
    start = perf_counter()
    feature_rows = []
    starts = range(0, signal.size - WINDOW_SAMPLES + 1, STRIDE_SAMPLES)
    for offset in starts:
        window = signal[offset:offset + WINDOW_SAMPLES]
        if np.std(window) >= MIN_SD_UV:
            _, values = _features(window)
            feature_rows.append(values)
    total, retained = len(starts), len(feature_rows)
    qc.update(total_windows=total, retained_windows=retained, failed_windows=total - retained)
    if not feature_rows:
        reason = "no_valid_windows" if total else "insufficient_window_length"
        windowed = {**identity, "error": reason}
    else:
        reason = None
        matrix = np.asarray(feature_rows, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            sample_sd = matrix.std(axis=0, ddof=1) if retained > 1 else np.full(len(names), np.nan)
            aggregates = (matrix.mean(axis=0), sample_sd, matrix.min(axis=0), matrix.max(axis=0))
        windowed = {
            **identity,
            **{f"{name}_{kind}": value if np.isfinite(value) else np.nan
               for kind, row in zip(AGGREGATES, aggregates) for name, value in zip(names, row)},
        }
        qc["undefined_window_descriptor_counts"] = dict(zip(names, (~np.isfinite(matrix)).sum(axis=0).tolist()))
        qc["undefined_window_aggregates"] = [
            name for name, value in windowed.items() if name not in identity and not np.isfinite(value)
        ]
    timings["process_windowed"] = perf_counter() - start
    qc["undefined_full_descriptors"] = [name for name in names if not np.isfinite(full[name])]
    qc["branch_status"] = {"full": "success", "windowed": reason or "success"}
    return {"full": full, "windowed": windowed}, timings, qc
