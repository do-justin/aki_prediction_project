from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import shutil
from time import perf_counter

from artifact_paths import DEFAULT_ARTIFACT_ROOT, DEFAULT_MEDIA_ROOT, path_is_within

BASELINE_REPO = Path("/home/exouser/projects/aki_prediction_project")
PRESERVATION_ROOT = Path("/home/exouser/codex-handoffs/joms-manuscript-pre-eeg-2026-11-25")
ORIGINAL_EEG_ROOT = DEFAULT_ARTIFACT_ROOT / "studies/eeg-feasibility-v1"
DEFAULT_STUDY_ROOT = DEFAULT_ARTIFACT_ROOT / "studies/eeg-feasibility-pycatch22-1.0.2-v1"
AMPLITUDE_STUDY_ROOT = DEFAULT_ARTIFACT_ROOT / "studies/eeg-feasibility-pycatch22-1.0.2-amplitude-v1"
DEFAULT_RAW_ROOT = DEFAULT_MEDIA_ROOT / "projects/aki_prediction_project/data/raw"
LAYOUT_OVERRIDES = {
    "DATA_DIR": "data", "PROCESSED_DIR": "data/processed",
    "RESULTS_DIR": "results/catch22/experiments", "PAPER_DIR": "results/catch22/paper",
    "SMOKE_ROOT": "diagnostic", "LOG_FILE": "logs/experiment_log.txt",
}


def _require_isolation(root: Path, paths: dict[str, Path], raw: Path) -> None:
    from data_preparation.eeg import AMPLITUDE_PROTOCOL, qc_protocol

    amplitude = qc_protocol() == AMPLITUDE_PROTOCOL
    protected = [raw, DEFAULT_RAW_ROOT, BASELINE_REPO, PRESERVATION_ROOT, ORIGINAL_EEG_ROOT,
                 DEFAULT_STUDY_ROOT if amplitude else AMPLITUDE_STUDY_ROOT,
                 *DEFAULT_STUDY_ROOT.parent.glob("manuscript-*")]
    if amplitude and (reference := os.getenv("AKI_EEG_W4_REFERENCE")):
        protected.append(Path(reference).expanduser().resolve())
    for name, path in {"study_root": root, **paths}.items():
        if not path_is_within(path, root):
            raise ValueError(f"EEG writable path {name}={path} must resolve inside {root}.")
        for original in protected:
            if path_is_within(path, original) or path_is_within(original, path):
                raise ValueError(f"EEG writable path {name} overlaps a protected path: {original}.")
    if root.exists():
        for parent, directories, files in os.walk(root):
            for name in [*directories, *files]:
                child = Path(parent) / name
                if child.is_symlink() or (child.is_file() and child.stat().st_nlink > 1):
                    raise ValueError(f"EEG study contains a writable path alias: {child}.")


def configure_environment(*, dry_run: bool = False, root_override: Path | None = None) -> Path:
    """Establish a single safe write boundary before importing profile consumers."""
    from importlib.metadata import version
    from data_preparation.eeg import AMPLITUDE_PROTOCOL, qc_protocol

    if version("pycatch22") != "1.0.2":
        raise ValueError("This EEG upgrade worktree requires the pinned pycatch22 1.0.2 candidate environment.")
    default = AMPLITUDE_STUDY_ROOT if qc_protocol() == AMPLITUDE_PROTOCOL else DEFAULT_STUDY_ROOT
    root = (root_override or Path(os.getenv("AKI_ARTIFACT_ROOT", default))).expanduser()
    if root.is_symlink():
        raise ValueError(f"EEG study root may not be a symbolic link: {root}")
    root = root.resolve()
    raw = Path(os.getenv("RAW_DIR", DEFAULT_RAW_ROOT)).expanduser().resolve()
    paths = {
        name: Path(os.environ[name]).expanduser().resolve() if root_override is None and name in os.environ
        else root / relative
        for name, relative in LAYOUT_OVERRIDES.items()
    }
    for name in ("LOG_FILE",):
        if paths[name] != root / LAYOUT_OVERRIDES[name]:
            raise ValueError(f"EEG {name} must use the study layout path {root / LAYOUT_OVERRIDES[name]}.")
    _require_isolation(root, paths, raw)
    scratch = root / "tmp"
    os.environ.update({name: str(path) for name, path in paths.items()})
    os.environ.update(
        AKI_ARTIFACT_ROOT=str(root), RAW_DIR=str(raw), INPUT_FILE=str(raw / "clinical_data.csv"),
        LAB_DATA_FILE=str(raw / "lab_data.csv"), AKI_STUDY_PROFILE="eeg", AKI_LR_CLASS_WEIGHT="balanced",
        TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch), JOBLIB_TEMP_FOLDER=str(scratch / "joblib"),
        MPLCONFIGDIR=str(scratch / "matplotlib"), XDG_CACHE_HOME=str(scratch / "cache"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    reference = PRESERVATION_ROOT / "studies/manuscript-core-v1/data/processed/aki_features_master_wide.csv"
    if reference.is_file():
        os.environ.setdefault("AKI_EEG_MANUSCRIPT_REFERENCE", str(reference))
    if not dry_run:
        from artifact_paths import enforce_storage_policy
        enforce_storage_policy({"eeg_study": root, **paths})
        scratch.mkdir(parents=True, exist_ok=True)
    return root


def _pilot_plot(caseid, loaded, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not loaded.raw.size:
        return
    width = min(128 * 20, loaded.raw.size)
    centers = [loaded.raw.size // 2]
    mask = loaded.masks.get("unusable_union")
    if mask is not None and mask.any():
        centers.append(int(np.flatnonzero(mask)[len(np.flatnonzero(mask)) // 2]))
    figure, axes = plt.subplots(len(centers), 1, figsize=(10, 3 * len(centers)), squeeze=False)
    for axis, center in zip(axes.ravel(), centers):
        start = min(max(0, center - width // 2), loaded.raw.size - width)
        span = slice(start, start + width)
        axis.plot(loaded.times[span], loaded.raw[span], linewidth=0.6, label="Observed EEG")
        if loaded.clean is not None:
            axis.plot(loaded.times[span], loaded.clean[span], linewidth=0.7, label="Cleaned EEG")
        if mask is not None:
            axis.fill_between(loaded.times[span], 0, 1, where=mask[span], color="#a33e36",
                              alpha=0.17, step="mid", transform=axis.get_xaxis_transform(), label="Masked")
        axis.set(xlabel="Seconds from case start", ylabel="EEG (µV)")
        axis.legend(loc="upper right")
    figure.suptitle(f"Blinded pilot operation {caseid}")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _benchmark_case(task):
    from data_preparation.eeg import extract_eeg_case

    caseid, start, end, raw = task
    started = perf_counter()
    _, timings, qc = extract_eeg_case(caseid, start, end, raw)
    return {"caseid": caseid, "source": qc.get("selected_source"), "qc_status": qc["status"],
            "failure_reason": qc.get("failure_reason"), "full_status": qc["branch_status"]["full"],
            "windowed_status": qc["branch_status"]["windowed"],
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_seconds": perf_counter() - started, **timings}, qc


def run_pilot(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    root = configure_environment()
    from data_preparation.artifact_metadata import file_sha256
    from data_preparation.eeg import EEG_FALLBACK, EEG_WAVEFORM, load_clean_eeg, protocol_fingerprint
    from data_preparation.eeg_study import available_cases, track_catalog, valid_operation_times, write_csv, write_json
    from data_preparation.inputs import MANDATORY_WAVEFORMS, RAW_DIR, WAVEFORM_SUBSTITUTIONS
    import numpy as np
    import pandas as pd

    pilot = root / "pilot"
    pilot.mkdir(exist_ok=True)
    catalog = track_catalog(pilot, create=True)
    clinical = pd.read_csv(RAW_DIR / "clinical_data.csv", usecols=["caseid", "opstart", "opend", "ane_type", "asa"])
    general = clinical["ane_type"].astype("string").str.strip().str.casefold().eq("general")
    allowed_asa = ~clinical["asa"].astype("string").str.strip().str.upper().isin({"5", "5.0", "6", "6.0", "V", "VI"})
    clinical = clinical[general & allowed_asa & valid_operation_times(clinical)].copy()
    original_four = set.intersection(*(available_cases(
        catalog, (channel, *WAVEFORM_SUBSTITUTIONS.get(channel, ())), suffix=True
    ) for channel in MANDATORY_WAVEFORMS))
    eeg1 = available_cases(catalog, (EEG_WAVEFORM,))
    eeg2 = available_cases(catalog, (EEG_FALLBACK,))
    clinical = clinical[clinical["caseid"].isin(original_four)]
    clinical["duration_seconds"] = clinical["opend"] - clinical["opstart"]
    eligible = clinical[clinical["caseid"].isin(eeg1 | eeg2)].sort_values(["duration_seconds", "caseid"])
    if eligible.empty:
        raise ValueError("No EEG-bearing operations in the blinded source inventory.")
    rng = np.random.default_rng(42)
    candidates = []
    for label, positions in zip(("short", "median", "long"), np.array_split(np.arange(len(eligible)), 3)):
        n = min(args.cases_per_stratum, len(positions))
        for index in sorted(rng.choice(positions, size=n, replace=False).tolist()):
            candidates.append((label, eligible.iloc[index]))
    for label, ids in (("catalog_fallback", eeg2 - eeg1), ("catalog_absent", original_four - (eeg1 | eeg2))):
        eligible_extra = clinical[clinical["caseid"].isin(ids)]
        if not eligible_extra.empty:
            candidates.append((label, eligible_extra.sort_values("caseid").iloc[0]))
    rows = {}
    qualified = {}
    used = set()
    for stratum, row in candidates:
        caseid = int(row["caseid"])
        if caseid in used:
            continue
        used.add(caseid)
        loaded = load_clean_eeg(caseid, row["opstart"], row["opend"], RAW_DIR)
        _pilot_plot(caseid, loaded, pilot / f"{caseid}_raw_clean.png")
        qc_path = pilot / f"{caseid}_qc.json"
        if not qc_path.is_file():
            write_json(qc_path, loaded.qc)
        rows[caseid] = {"caseid": caseid, "stratum": stratum, "duration_seconds": float(row["duration_seconds"]),
                        "source": loaded.qc.get("selected_source"), "qc_status": loaded.qc["status"],
                        "failure_reason": loaded.qc.get("failure_reason"),
                        "full_status": "not_benchmarked", "windowed_status": "not_benchmarked"}
        if loaded.clean is not None and stratum in ("short", "median", "long"):
            qualified.setdefault(stratum, []).append((caseid, float(row["opstart"]), float(row["opend"]), RAW_DIR))
        print(f"Pilot QC {stratum} {caseid}: {loaded.qc['status']}", flush=True)
    tasks = []
    for stratum in ("short", "median", "long"):
        options = qualified.get(stratum, [])
        if options:
            ordered = sorted(options, key=lambda item: (item[2] - item[1], item[0]))
            tasks.append(ordered[len(ordered) // 2])
    worker_count = max(1, len(tasks))
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {}
        for task in tasks:
            caseid = task[0]
            path = pilot / f"{caseid}_benchmark.json"
            if path.is_file():
                cached = json.loads(path.read_text())
                if cached.get("protocol") == protocol_fingerprint() and cached.get("interval") == list(task[1:3]):
                    rows[caseid].update(cached["summary"])
                    continue
            futures[pool.submit(_benchmark_case, task)] = task
        for future in as_completed(futures):
            caseid, start, end, _ = futures[future]
            summary, qc = future.result()
            rows[caseid].update(summary)
            write_json(pilot / f"{caseid}_qc.json", qc)
            write_json(pilot / f"{caseid}_benchmark.json", {"protocol": protocol_fingerprint(),
                                                            "interval": [start, end], "summary": summary})
            print(f"Pilot benchmark {rows[caseid]['stratum']} {caseid}: {summary['full_status']} / "
                  f"{summary['windowed_status']} ({summary['elapsed_seconds']:.1f}s, "
                  f"{summary['peak_rss_kib'] / 1024:.0f} MiB peak)", flush=True)
    write_csv(pilot / "pilot_cases.csv", pd.DataFrame(rows.values()))
    successful_strata = {row["stratum"] for row in rows.values()
                         if row["full_status"] == row["windowed_status"] == "success"}
    missing = sorted({"short", "median", "long"} - successful_strata)
    status = "benchmarked" if not missing else "incomplete"
    write_json(pilot / "protocol_freeze.json", {"status": status, "protocol": protocol_fingerprint(),
                                               "cases": len(rows), "workers": worker_count, "missing_benchmark_strata": missing,
                                               "track_catalog_sha256": file_sha256(pilot / "eeg_source_tracks.csv")})
    if missing:
        raise ValueError(f"Pilot had no qualifying extraction in {', '.join(missing)}. Increase the blinded sample.")
    return 0


def require_amplitude_cohort_confirmation(root: Path) -> None:
    from data_preparation.eeg import AMPLITUDE_PROTOCOL, qc_protocol

    if qc_protocol() != AMPLITUDE_PROTOCOL:
        return
    from data_preparation.artifact_metadata import file_sha256
    from experimental.eeg.verify_amplitude_cohort import CONFIRMATION_NAME

    confirmation = root / "control" / CONFIRMATION_NAME
    if not confirmation.is_file():
        raise ValueError("Confirm the amplitude EEG cohort against the saved-mask replay before fitting.")
    expected = json.loads(confirmation.read_text())
    processed = Path(os.environ["PROCESSED_DIR"])
    sources = {"split_preflight_sha256": root / "control/eeg_split_preflight.json",
               "case_audit_sha256": processed / "eeg_operation_attrition.csv",
               "fingerprint_sha256": processed / "eeg_extraction_checkpoints/fingerprint.json"}
    if ((expected.get("status"), expected.get("protocol")) != ("confirmed", AMPLITUDE_PROTOCOL)
            or any(not path.is_file() or file_sha256(path) != expected.get(key) for key, path in sources.items())):
        raise ValueError(f"Amplitude EEG cohort confirmation is stale or incomplete: {confirmation}")


def prepare_diagnostic_root(root: Path) -> Path:
    from data_preparation.artifact_metadata import file_sha256, metadata_path_for
    from data_preparation.eeg_study import TRACK_CATALOG_NAME

    gate = root / "control/eeg_split_preflight.json"
    if not gate.is_file():
        raise ValueError("EEG diagnostic fitting requires a validated all-scope split preflight.")
    preflight = json.loads(gate.read_text())
    if preflight.get("status") != "validated":
        raise ValueError("EEG diagnostic fitting is held by the definitive split preflight.")
    require_amplitude_cohort_confirmation(root)
    source = Path(os.environ["PROCESSED_DIR"])
    basenames = {
        "non_windowed": "aki_features_master_wide.csv",
        "windowed": "aki_features_master_wide_windowed.csv",
    }
    for branch, name in basenames.items():
        path = source / name
        if (file_sha256(path) != preflight["dataset_sha256"][branch]
                or file_sha256(metadata_path_for(path)) != preflight["input_provenance_sha256"][branch]):
            raise ValueError("EEG diagnostic source differs from the validated preflight.")
    diagnostic = Path(os.environ["SMOKE_ROOT"])
    _require_isolation(root, {"diagnostic": diagnostic}, Path(os.environ["RAW_DIR"]))
    configure_environment(root_override=diagnostic)
    target = Path(os.environ["PROCESSED_DIR"])
    target.mkdir(parents=True, exist_ok=True)
    for name in (*basenames.values(), "aki_pleth_ecg_co2_awp.csv", "aki_preop_processed.csv"):
        for item in (source / name, metadata_path_for(source / name)):
            shutil.copy2(item, target / item.name)
    shutil.copy2(source / TRACK_CATALOG_NAME, target / TRACK_CATALOG_NAME)
    return diagnostic
