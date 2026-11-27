from __future__ import annotations

import gzip
import hashlib
import json
from multiprocessing import BoundedSemaphore, Pool, TimeoutError as WorkerTimeout
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from vitaldb import dataset

from data_preparation.artifact_metadata import (
    ArtifactCompatibilityError, STEP_01_COHORT_ARTIFACT, file_sha256, metadata_path_for, read_versioned_csv,
)
from data_preparation.catch24 import descriptor_names
from data_preparation.eeg import (
    AMPLITUDE_PROTOCOL, EEG_WAVEFORM, extract_eeg_case, protocol_fingerprint, qc_protocol, raw_recording_path,
)
from data_preparation.eeg_study import EEG_PROFILE, QC_NAME, TRACK_CATALOG_NAME, track_catalog, write_csv, write_json
from data_preparation.inputs import (
    CATCH_22_ERROR_FILE, CATCH_22_FILE, CATCH_22_WINDOWED_FILE, COHORT_FILE, GENERATE_FULL_FEATURES,
    GENERATE_WINDOWED_FEATURES, MANDATORY_WAVEFORMS, PROCESSED_DIR, RAW_DIR,
)
from data_preparation.step_02_catch_22 import COHORT_REQUIRED_COLUMNS, _process_case, build_extraction_fingerprint
from runtime_resources import available_cpu_count

EEG_EXTRACTION_PROTOCOL_VERSION = 4
MODES = ("full", "windowed")
WINDOW_STATS = ("mean", "std", "min", "max")
CHANNELS = (*MANDATORY_WAVEFORMS, EEG_WAVEFORM)
# Reader that produced the frozen strict run's original-four checkpoints.
STRICT_TRACK_READER_SHA256 = "d3f3ed2c526db55f400a60ef58af08d500d3272acc419b169dfec52bc080c014"
_cache_dir: Path
_download_slots = None
_download_failure: str | None
_track_sources: dict[str, str]
_w4_reference: Path | None = None
_w4_checkpoint_hashes: dict[int, str]


def _raw_release() -> dict[str, str]:
    return {
        name: digest for digest, name in (
            line.split(maxsplit=1) for line in (RAW_DIR / "SHA256SUMS.txt").read_text().splitlines() if line.strip()
        )
    }


def _recording_identities(cohort: pd.DataFrame) -> str:
    release = _raw_release()
    digest = hashlib.sha256()
    for caseid in sorted(cohort["caseid"]):
        recording = raw_recording_path(RAW_DIR, caseid)
        source = recording.relative_to(RAW_DIR).as_posix()
        stat = recording.stat()
        expected = release.get(source)
        if expected is None:
            raise ArtifactCompatibilityError(f"Local VitalDB recording is missing from the release catalog: {source}")
        digest.update(f"{source}:{expected}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _branch_output_paths(processed: Path | None = None) -> dict[str, tuple[Path, Path]]:
    paths = {
        "full": (CATCH_22_FILE, CATCH_22_ERROR_FILE),
        "windowed": (CATCH_22_WINDOWED_FILE,
                     CATCH_22_ERROR_FILE.with_name(CATCH_22_ERROR_FILE.stem + "_windowed.csv")),
    }
    if processed is None:
        return paths
    return {mode: (processed / feature.name, processed / error.name)
            for mode, (feature, error) in paths.items()}


def _reference_file(path: Path) -> Path:
    if not path.is_file() or path.resolve() != path:
        raise ArtifactCompatibilityError(f"Missing or aliased strict EEG reference file: {path}")
    return path


def _legacy_reference() -> tuple[Path | None, dict]:
    configured = os.getenv("AKI_EEG_W4_REFERENCE")
    if not configured:
        return None, {}
    if qc_protocol() != AMPLITUDE_PROTOCOL:
        raise ValueError("AKI_EEG_W4_REFERENCE requires the amplitude-only EEG protocol.")
    root = Path(os.environ["AKI_ARTIFACT_ROOT"]).resolve()
    reference_root = Path(configured).expanduser().resolve(strict=True)
    if reference_root.is_relative_to(root) or root.is_relative_to(reference_root):
        raise ValueError("The strict EEG reference and the writable amplitude study must be disjoint.")
    processed = reference_root / "data/processed"
    paths = [processed / "eeg_extraction_checkpoints/fingerprint.json",
             processed / "eeg_track_source_hashes.json",
             *(metadata_path_for(features) for features, _ in _branch_output_paths(processed).values())]
    return processed, {
        "study_root": str(reference_root),
        "files": {str(path.relative_to(processed)): file_sha256(_reference_file(path)) for path in paths},
    }


def eeg_fingerprints(cohort: pd.DataFrame, *, recording_sha256: str | None = None) -> dict[str, dict]:
    recordings = recording_sha256 or _recording_identities(cohort)
    protocol_hashes = {
        name: file_sha256(Path(__file__).with_name(name)) for name in ("eeg.py", "eeg_study.py", "eeg_extraction.py")
    }
    fingerprints = {
        mode: {
            **build_extraction_fingerprint(cohort, mode),
            "extraction_protocol_version": EEG_EXTRACTION_PROTOCOL_VERSION,
            "study_profile": EEG_PROFILE,
            "waveforms": list(CHANNELS),
            "eeg_protocol": protocol_fingerprint(),
            "eeg_code_sha256": protocol_hashes,
            "track_catalog_sha256": file_sha256(PROCESSED_DIR / TRACK_CATALOG_NAME),
            "waveform_release_sha256": file_sha256(RAW_DIR / "SHA256SUMS.txt"),
            "recording_identity_sha256": recordings,
            "original_four_source": "frozen VitalDB public track CSV cache, pinned loader value/grid semantics",
        }
        for mode in MODES
    }
    reference, attestation = _legacy_reference()
    if reference is not None:
        for fingerprint in fingerprints.values():
            fingerprint["original_four_reference"] = attestation
    return fingerprints


def _cached_track(tid: str, interval: float = 1) -> np.ndarray:
    """Decode the original four channels with the pinned dataset.load_trk grid rules."""
    global _download_failure
    tid = str(tid)
    if not tid.isalnum():
        raise ValueError(f"Invalid VitalDB track identifier: {tid}")
    path = _cache_dir / f"{tid}.csv.gz"
    if not path.is_file():
        last_error = None
        for attempt in range(3):
            try:
                with _download_slots:
                    response = requests.get(f"https://api.vitaldb.net/{tid}", timeout=(10, 90))
                    response.raise_for_status()
                    content = response.content
                if not content:
                    raise ValueError("empty HTTP response")
                temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                temporary.write_bytes(gzip.compress(content, compresslevel=1, mtime=0))
                temporary.replace(path)
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        else:
            _download_failure = f"VitalDB track {tid} could not be downloaded: {last_error}"
            raise RuntimeError(_download_failure)
    try:
        _track_sources[tid] = file_sha256(path)
        values = pd.read_csv(path, na_values="-nan(ind)").values
        if len(values) == 0:
            return np.empty(0)
        values[:, 0] /= interval
        sample_count = int(np.nanmax(values[:, 0])) + 1
        if np.isnan(values[:, 0]).any():
            if sample_count == len(values):
                return values[:, 1]
            return np.take(values[:, 1], np.linspace(0, len(values) - 1, sample_count).astype(np.int64))
        result = np.full(sample_count, np.nan)
        for position, value in values:
            result[int(position)] = value
        return result
    except (OSError, ValueError, IndexError) as exc:
        _download_failure = f"Cannot verify or decode cached VitalDB track {tid}: {exc}"
        raise RuntimeError(_download_failure) from exc


def _init_worker(processed: Path, download_slots, reference: Path | None = None,
                 checkpoint_hashes: dict[int, str] | None = None) -> None:
    global _cache_dir, _download_slots, _w4_reference, _w4_checkpoint_hashes
    _cache_dir = processed / "eeg_legacy_track_cache"
    _download_slots = download_slots
    _w4_reference = reference
    _w4_checkpoint_hashes = checkpoint_hashes or {}
    if reference is None:
        dataset.dftrks = track_catalog(processed)
        dataset.load_trk = _cached_track


def _extract_operation(task: tuple[int, float, float, str]) -> dict:
    global _download_failure, _track_sources
    caseid, start, end, expected_sha = task
    _track_sources = {}
    _download_failure = None
    recording = raw_recording_path(RAW_DIR, caseid)
    if file_sha256(recording) != expected_sha:
        raise ArtifactCompatibilityError(f"VitalDB recording checksum differs from release: {recording.name}")
    eeg, timings, qc = extract_eeg_case(caseid, start, end, RAW_DIR)
    results = {EEG_WAVEFORM: eeg}
    if _w4_reference is None:
        for channel in MANDATORY_WAVEFORMS:
            legacy, legacy_timings = _process_case((caseid, start, end, channel))
            if _download_failure:
                raise RuntimeError(_download_failure)
            if "error" in legacy:
                legacy = dict.fromkeys(MODES, legacy["error"])
            results[channel] = legacy
            timings[channel] = legacy_timings
    else:
        source = recording.relative_to(RAW_DIR).as_posix()
        saved = _read_reference_checkpoint(_w4_reference, caseid, {source: expected_sha}, _w4_checkpoint_hashes[caseid])
        for channel in MANDATORY_WAVEFORMS:
            results[channel] = saved["results"][channel]
        _track_sources = saved["sources"]
    for channel, branch_records in results.items():
        if set(branch_records) != set(MODES):
            raise RuntimeError(f"Unexpected branch results for {caseid}/{channel}")
    for channel in CHANNELS:
        results[channel] = {
            mode: {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
                   for key, value in row.items()}
            for mode, row in results[channel].items()
        }
    return {"caseid": caseid, "eeg_recording_sha256": expected_sha, "sources": _track_sources,
            "results": results, "timings": timings, "qc": qc}


def _checked_checkpoint(path: Path, caseid: int, release: dict[str, str], *, sha256: str | None = None) -> dict:
    data = path.read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ArtifactCompatibilityError(f"Strict EEG checkpoint changed after reference verification: {path}")
    payload = json.loads(data)
    raw_name = raw_recording_path(RAW_DIR, caseid).relative_to(RAW_DIR).as_posix()
    if payload.get("caseid") != caseid or payload.get("eeg_recording_sha256") != release.get(raw_name):
        raise ArtifactCompatibilityError(f"EEG checkpoint operation or recording identity differs: {path}")
    results = payload.get("results", {})
    if set(results) != set(CHANNELS) or payload.get("qc", {}).get("caseid") != caseid:
        raise ArtifactCompatibilityError(f"EEG checkpoint is incomplete: {path}")
    for channel, branches in results.items():
        if set(branches) != set(MODES) or any(
            row.get("caseid") != caseid or row.get("waveform") != channel for row in branches.values()
        ):
            raise ArtifactCompatibilityError(f"EEG checkpoint branch identity differs: {path}")
    return payload


def _read_reference_checkpoint(reference: Path, caseid: int, release: dict[str, str], digest: str) -> dict:
    checkpoint = _reference_file(reference / "eeg_extraction_checkpoints" / f"{caseid}.json")
    return _checked_checkpoint(checkpoint, caseid, release, sha256=digest)


def _flatten_qc(payload: dict) -> dict:
    qc = {"caseid": payload["caseid"]}
    for key, value in payload["qc"].items():
        if key == "masks":
            continue
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                qc[f"{key}_{subkey}"] = subvalue
        if isinstance(value, (str, int, float, bool)) or value is None:
            qc[key] = value
    for channel, records in payload["results"].items():
        for mode in MODES:
            qc[f"{mode}_{channel.replace('/', '_')}_failure"] = records[mode].get("error")
    return qc


def _branch_frames(mode: str, features: list[dict], errors: list[dict], names: list[str]):
    descriptors = names if mode == "full" else [f"{name}_{stat}" for stat in WINDOW_STATS for name in names]
    return (pd.DataFrame(features, columns=["caseid", "waveform", *descriptors]),
            pd.DataFrame(errors, columns=["caseid", "waveform", "error"]))


def _verify_legacy_reference(reference: Path, caseids: list[int], fingerprints: dict[str, dict],
                             release: dict[str, str]) -> dict[int, str]:
    """Reconstruct strict published outputs from the exact checkpoint bytes to be reused."""
    checkpoints = reference / "eeg_extraction_checkpoints"
    recorded = fingerprints[MODES[0]]["original_four_reference"]["files"]
    for name, expected in recorded.items():
        if file_sha256(_reference_file(reference / name)) != expected:
            raise ArtifactCompatibilityError(f"Strict EEG reference changed during validation: {name}")
    saved_fingerprints = json.loads((checkpoints / "fingerprint.json").read_text())
    if set(saved_fingerprints) != set(MODES):
        raise ArtifactCompatibilityError("The strict EEG reference must contain both extraction branches.")
    excluded = {"eeg_protocol", "eeg_code_sha256", "original_four_reference"}
    for mode in MODES:
        prior = saved_fingerprints[mode]
        if (prior["eeg_protocol"]["protocol"] != "eeg-qc-v1"
                or prior["eeg_code_sha256"]["eeg_extraction.py"] != STRICT_TRACK_READER_SHA256
                or prior.keys() != fingerprints[mode].keys() - {"original_four_reference"}
                or any(prior[key] != fingerprints[mode][key] for key in prior.keys() - excluded)):
            raise ArtifactCompatibilityError(f"Incompatible original-four reference extraction parameters: {mode}")

    features = {mode: [] for mode in MODES}
    errors = {mode: [] for mode in MODES}
    track_hashes = json.loads((reference / "eeg_track_source_hashes.json").read_text())
    seen_tracks = set()
    hashes = {}
    for caseid in caseids:
        checkpoint = _reference_file(checkpoints / f"{caseid}.json")
        hashes[caseid] = file_sha256(checkpoint)
        payload = _checked_checkpoint(checkpoint, caseid, release, sha256=hashes[caseid])
        for tid, digest in payload["sources"].items():
            if not tid.isalnum() or track_hashes.get(tid) != digest:
                raise ArtifactCompatibilityError(f"Strict EEG checkpoint track identity differs: {checkpoint}")
            seen_tracks.add(tid)
        for channel in CHANNELS:
            for mode, row in payload["results"][channel].items():
                (errors if "error" in row else features)[mode].append(row)
    if seen_tracks != track_hashes.keys():
        raise ArtifactCompatibilityError("Strict EEG reference track roster differs from its extraction checkpoints.")

    names = descriptor_names()
    for mode, (feature_path, error_path) in _branch_output_paths(reference).items():
        published = json.loads(metadata_path_for(feature_path).read_text())
        if (published["requested_caseids"] != caseids
                or published["extraction_fingerprint"] != saved_fingerprints[mode]
                or published["source_track_hashes_sha256"] != recorded["eeg_track_source_hashes.json"]
                or published["data_sha256"] != file_sha256(_reference_file(feature_path))):
            raise ArtifactCompatibilityError(f"Incompatible strict EEG assembled feature metadata: {feature_path}")
        for frame, path in zip(_branch_frames(mode, features[mode], errors[mode], names), (feature_path, error_path)):
            actual = hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
            if actual != file_sha256(_reference_file(path)):
                raise ArtifactCompatibilityError(f"Strict EEG checkpoints do not reconstruct published output: {path}")
    return hashes


def _assemble(checkpoints: Path, caseids: list[int], fingerprints: dict[str, dict], release: dict[str, str],
              *, reference: Path | None = None, reference_hashes: dict[int, str] | None = None) -> None:
    outputs = _branch_output_paths()
    source_dir = (reference if reference is not None else PROCESSED_DIR) / "eeg_legacy_track_cache"
    features = {mode: [] for mode in MODES}
    errors = {mode: [] for mode in MODES}
    audits = []
    source_hashes = {}
    for caseid in caseids:
        path = checkpoints / f"{caseid}.json"
        payload = _checked_checkpoint(path, caseid, release)
        if reference is not None:
            prior = _read_reference_checkpoint(reference, caseid, release, reference_hashes[caseid])
            if (payload["sources"] != prior["sources"] or any(
                payload["results"][channel] != prior["results"][channel] for channel in MANDATORY_WAVEFORMS
            )):
                raise ArtifactCompatibilityError(f"Reused original-four data differs from the strict reference: {path}")
        audits.append(_flatten_qc(payload))
        for tid, expected in payload["sources"].items():
            if not tid.isalnum() or (tid in source_hashes and source_hashes[tid] != expected):
                raise ArtifactCompatibilityError(f"Conflicting cached VitalDB track identity: {tid}")
            if tid not in source_hashes:
                source = source_dir / f"{tid}.csv.gz"
                if reference is not None:
                    _reference_file(source)
                if not source.is_file() or file_sha256(source) != expected:
                    raise ArtifactCompatibilityError(f"Cached VitalDB source changed: {source}")
                source_hashes[tid] = expected
        for channel in CHANNELS:
            for mode, row in payload["results"][channel].items():
                (errors if "error" in row else features)[mode].append(row)
    names = descriptor_names()
    qc = pd.DataFrame(audits) if audits else pd.DataFrame(columns=["caseid"])
    write_csv(PROCESSED_DIR / QC_NAME, qc)
    write_json(PROCESSED_DIR / "eeg_track_source_hashes.json", source_hashes)
    for mode, (feature_path, error_path) in outputs.items():
        feature_frame, error_frame = _branch_frames(mode, features[mode], errors[mode], names)
        write_csv(feature_path, feature_frame)
        write_csv(error_path, error_frame)
        write_json(metadata_path_for(feature_path), {
            "extraction_protocol_version": EEG_EXTRACTION_PROTOCOL_VERSION,
            "extraction_fingerprint": fingerprints[mode], "error_file": str(error_path),
            "requested_caseids": caseids, "successful_pairs": len(features[mode]),
            "source_track_hashes_sha256": file_sha256(PROCESSED_DIR / "eeg_track_source_hashes.json"),
            "qc_sha256": file_sha256(PROCESSED_DIR / QC_NAME), "data_sha256": file_sha256(feature_path),
        })


def main() -> None:
    if os.getenv("AKI_STUDY_PROFILE") != EEG_PROFILE:
        raise ValueError("Use run_catch22 experiments --study-profile eeg for EEG extraction.")
    from experimental.eeg.runtime import _require_isolation
    study_root = Path(os.environ["AKI_ARTIFACT_ROOT"])
    _require_isolation(study_root, {"processed": PROCESSED_DIR}, RAW_DIR)
    if not (GENERATE_FULL_FEATURES and GENERATE_WINDOWED_FEATURES):
        raise ValueError("The EEG study requires both extraction branches.")
    cohort, _ = read_versioned_csv(COHORT_FILE, artifact_role=STEP_01_COHORT_ARTIFACT,
                                  required_columns=COHORT_REQUIRED_COLUMNS)
    if cohort["caseid"].duplicated().any():
        raise ValueError("Duplicate EEG extraction case IDs.")
    fingerprints = eeg_fingerprints(cohort)
    reference, _ = _legacy_reference()
    release = _raw_release()
    caseids = sorted(int(caseid) for caseid in cohort["caseid"])
    reference_hashes = {}
    if reference is not None:
        print(f"Checking published strict reference checkpoints for {len(caseids)} operations", flush=True)
        reference_hashes = _verify_legacy_reference(reference, caseids, fingerprints, release)
    checkpoints = PROCESSED_DIR / "eeg_extraction_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if reference is None:
        (PROCESSED_DIR / "eeg_legacy_track_cache").mkdir(exist_ok=True)
    identity = checkpoints / "fingerprint.json"
    if identity.exists():
        if json.loads(identity.read_text()) != fingerprints:
            raise ArtifactCompatibilityError("Incompatible EEG checkpoint. Select a fresh processed directory.")
    else:
        if any(checkpoints.iterdir()):
            raise ArtifactCompatibilityError("EEG checkpoint has no fingerprint.")
        write_json(identity, fingerprints)

    pending = []
    for caseid, start, end in cohort[COHORT_REQUIRED_COLUMNS].sort_values("caseid").itertuples(index=False, name=None):
        caseid = int(caseid)
        if not (checkpoints / f"{caseid}.json").exists():
            name = raw_recording_path(RAW_DIR, caseid).relative_to(RAW_DIR).as_posix()
            pending.append((caseid, float(start), float(end), release[name]))
    # Start the largest recordings early to keep them from becoming a serial tail.
    pending.sort(key=lambda task: (-(task[2] - task[1]), task[0]))
    available_cpus = available_cpu_count()
    workers = min(available_cpus, int(os.getenv("AKI_EXTRACTION_WORKERS", str(available_cpus))))
    timeout = int(os.getenv("AKI_EEG_EXTRACTION_IDLE_TIMEOUT", "14400"))
    downloads = int(os.getenv("AKI_EEG_DOWNLOAD_WORKERS", "4"))
    if min(workers, timeout, downloads) < 1:
        raise ValueError("EEG extraction worker count, download limit, and idle timeout must be positive.")
    if pending:
        workers = min(workers, len(pending))
        source_status = (f"up to {downloads} concurrent VitalDB downloads" if reference is None
                         else "verified original-four waveform checkpoints")
        print(f"Extracting {len(pending)} remaining EEG-study operations with {workers} workers "
              f"and {source_status}", flush=True)
        download_slots = BoundedSemaphore(downloads) if reference is None else None
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(PROCESSED_DIR, download_slots, reference, reference_hashes)) as pool:
            results = pool.imap_unordered(_extract_operation, pending, chunksize=1)
            for completed in range(1, len(pending) + 1):
                try:
                    payload = results.next(timeout=timeout)
                except WorkerTimeout as exc:
                    raise RuntimeError(f"No EEG extraction progress for {timeout}s. Completed checkpoints can resume.") from exc
                write_json(checkpoints / f"{payload['caseid']}.json", payload)
                if completed % 25 == 0 or completed == len(pending):
                    print(f"Committed {completed}/{len(pending)} EEG-study operations", flush=True)
    _assemble(checkpoints, caseids, fingerprints, release, reference=reference, reference_hashes=reference_hashes)


if __name__ == "__main__":
    main()
