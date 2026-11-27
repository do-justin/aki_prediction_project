from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import requests

from data_preparation.artifact_metadata import file_sha256
from data_preparation.cohort_scopes import SCOPE_OUTCOMES, select_cohort
from data_preparation.outcome_registry import get_outcome_spec

EEG_PROFILE = "eeg"
EEG_WAVEFORM = "BIS/EEG1_WAV"
EEG_FALLBACK = "BIS/EEG2_WAV"
EEG_SQI = "BIS/SQI"
EEG_SCOPE_IDENTITIES = {
    "main_shared": "eeg_main_shared_v1",
    "icu_waveform_sensitivity": "eeg_icu_waveform_sensitivity_v1",
}
TRACK_CATALOG_NAME = "eeg_source_tracks.csv"
SOURCE_COHORT_NAME = "eeg_source_cohort_audit.csv"
QC_NAME = "eeg_case_qc.csv"


def eeg_profile() -> bool:
    return os.getenv("AKI_STUDY_PROFILE") == EEG_PROFILE


def eeg_scope_identities() -> dict[str, str]:
    from data_preparation.eeg import AMPLITUDE_PROTOCOL, qc_protocol

    if qc_protocol() == AMPLITUDE_PROTOCOL:
        return {scope: f"eeg_{scope}_amplitude_v1" for scope in EEG_SCOPE_IDENTITIES}
    return EEG_SCOPE_IDENTITIES


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def track_catalog(processed: Path, *, create: bool = False) -> pd.DataFrame:
    """Freeze the VitalDB track-to-case map used by both source selection and W4 loading."""
    path = processed / TRACK_CATALOG_NAME
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    frozen = os.getenv("AKI_EEG_TRACK_CATALOG") if create else None
    if frozen:
        frozen_path = Path(frozen)
        _read_track_catalog(frozen_path)
        if not path.is_file():
            temporary = path.with_suffix(".tmp")
            shutil.copyfile(frozen_path, temporary)
            temporary.replace(path)
        if file_sha256(path) != file_sha256(frozen_path):
            raise ValueError("The definitive track catalog differs from the blinded pilot catalog.")
    if create and not path.is_file():
        response = requests.get("https://api.vitaldb.net/trks", timeout=(10, 90))
        response.raise_for_status()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(response.content)
        try:
            _read_track_catalog(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return _read_track_catalog(path)


def _read_track_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path, usecols=["caseid", "tname", "tid"], dtype={"tname": str, "tid": str})
    if catalog.empty or catalog.isna().any().any():
        raise ValueError(f"Invalid VitalDB track catalog at {path}")
    return catalog


def available_cases(catalog: pd.DataFrame, names: tuple[str, ...], *, suffix: bool = False) -> set[int]:
    match = catalog["tname"].str.endswith(names) if suffix else catalog["tname"].isin(names)
    return set(catalog.loc[match, "caseid"])


def valid_operation_times(frame: pd.DataFrame) -> pd.Series:
    start = pd.to_numeric(frame["opstart"], errors="coerce")
    end = pd.to_numeric(frame["opend"], errors="coerce")
    return np.isfinite(start) & np.isfinite(end) & start.ge(0) & end.gt(start)


def write_source_cohort(
    processed: Path, source: pd.DataFrame, clinical: pd.DataFrame, four: pd.DataFrame,
    catalog: pd.DataFrame,
) -> set[int]:
    columns = ["caseid", "subjectid", "eligible_any_aki", "aki_label", "eligible_icu_admission", "y_icu_admit"]
    available_descriptives = [name for name in ("age", "sex", "asa", "emop") if name in source]
    ledger = source[[*columns, *available_descriptives]].copy()
    ledger["eligible_clinical"] = ledger["caseid"].isin(clinical["caseid"])
    ledger["has_four_waveform_tracks"] = ledger["caseid"].isin(four["caseid"])
    for field, track in (("has_eeg1", EEG_WAVEFORM), ("has_eeg2", EEG_FALLBACK), ("has_sqi", EEG_SQI)):
        ledger[field] = ledger["caseid"].isin(available_cases(catalog, (track,)))
    ledger["eeg_extraction_candidate"] = ledger["has_four_waveform_tracks"] & (
        ledger["has_eeg1"] | ledger["has_eeg2"]
    )
    write_csv(processed / SOURCE_COHORT_NAME, ledger.sort_values("caseid"))
    return set(ledger.loc[ledger["eeg_extraction_candidate"], "caseid"])


def _outcome_counts(frame: pd.DataFrame, outcome: str) -> dict:
    spec = get_outcome_spec(outcome)
    eligible = frame[spec.eligibility_col].eq(1) & frame[spec.target_col].isin((0, 1))
    positives = eligible & frame[spec.target_col].eq(1)
    n_eligible = int(eligible.sum())
    n_positive = int(positives.sum())
    return {
        "eligible_operations": n_eligible, "eligible_patients": int(frame.loc[eligible, "subjectid"].nunique()),
        "positive_operations": n_positive, "positive_patients": int(frame.loc[positives, "subjectid"].nunique()),
        "prevalence": n_positive / n_eligible if n_eligible else None,
    }


def _counts(frame: pd.DataFrame, outcomes: tuple[str, ...] = ("any_aki", "icu_admission")) -> dict:
    return {
        "operations": len(frame), "patients": int(frame["subjectid"].nunique()),
        "outcomes": {outcome: _outcome_counts(frame, outcome) for outcome in outcomes},
    }


def _descriptive_selection(frame: pd.DataFrame) -> dict:
    values = {}
    if "age" in frame:
        valid = pd.to_numeric(frame["age"], errors="coerce").dropna()
        values["age"] = {"median": float(valid.median()) if not valid.empty else None,
                         "observed": len(valid), "missing": len(frame) - len(valid)}
    for name in ("sex", "asa", "emop"):
        if name in frame:
            values[name] = {str(level): int(n) for level, n in frame[name].fillna("Missing").value_counts().items()}
    return values


def write_attrition(processed: Path, complete: set[int], *, branch_complete: dict[str, set[int]]) -> None:
    audit = pd.read_csv(processed / SOURCE_COHORT_NAME)
    qc = pd.read_csv(processed / QC_NAME)
    audit = audit.merge(qc, on="caseid", how="left", validate="one_to_one")
    audit["both_branches"] = audit["caseid"].isin(complete)
    for branch, caseids in branch_complete.items():
        audit[f"complete_{branch}"] = audit["caseid"].isin(caseids)
    cohort = audit[audit["both_branches"]]
    candidates = audit["eeg_extraction_candidate"]
    clean_eeg = audit["status"].eq("passed") if "status" in audit else candidates & False
    eeg_features = candidates.copy()
    for mode in ("full", "windowed"):
        field = f"branch_status_{mode}"
        eeg_features &= audit[field].eq("success") if field in audit else False
    stages = [
        ("source", "All source operations", pd.Series(True, index=audit.index), None),
        ("clinical", "Valid operative interval, general anesthesia and ASA eligibility", audit["eligible_clinical"], "clinical"),
        ("four_waveform_tracks", "Original four waveform tracks, allowing ECG substitution", audit["has_four_waveform_tracks"], "original_waveform_track"),
        ("eeg_track", "EEG1 or EEG2 track listed", candidates, "eeg_track"),
        ("eeg_qc", "Selected EEG passes source and signal QC", candidates & clean_eeg, "eeg_source_or_qc"),
        ("eeg_features", "EEG features available in both branches", eeg_features, "eeg_feature_extraction"),
        ("both_branches", "All five modalities extracted in both branches", audit["both_branches"], "original_modality_extraction"),
    ]
    previous = pd.Series(False, index=audit.index)
    audit["first_exclusion"] = pd.Series(pd.NA, index=audit.index, dtype="string")
    stage_records = []
    for key, title, mask, reason in stages:
        excluded = previous & ~mask
        if reason:
            audit.loc[excluded, "first_exclusion"] = reason
        stage_records.append({"key": key, "title": title, "first_exclusion": reason, **_counts(audit[mask]),
                              "excluded_at_stage": _counts(audit[excluded])})
        previous = mask
    scopes = {}
    identities = eeg_scope_identities()
    for scope, identity in identities.items():
        selected = select_cohort(cohort, scope)
        audit[f"in_{scope}"] = audit["caseid"].isin(selected["caseid"])
        scopes[scope] = {"cohort_manifest_identity": identity, **_counts(selected, SCOPE_OUTCOMES[scope])}
    write_csv(processed / "eeg_operation_attrition.csv", audit)
    selections = {"retained_both_branches": cohort,
                  "excluded_after_four_tracks": audit[audit["has_four_waveform_tracks"] & ~audit["both_branches"]]}
    reference = os.getenv("AKI_EEG_MANUSCRIPT_REFERENCE")
    reference_info = None
    if reference:
        reference_path = Path(reference)
        reference_data = pd.read_csv(reference_path, usecols=["caseid", "subjectid", "eligible_any_aki", "aki_label",
                                                             "eligible_icu_admission", "y_icu_admit"])
        reference_info = {"path": str(reference_path), "sha256": file_sha256(reference_path)}
        for scope in identities:
            roster = select_cohort(reference_data, scope)
            selections[f"manuscript_{scope}"] = audit[audit["caseid"].isin(roster["caseid"])]
    overlapping = {}
    for field in audit.columns:
        if field.startswith("mask_counts_"):
            overlapping[field] = _counts(audit[pd.to_numeric(audit[field], errors="coerce").gt(0)])
        elif field.endswith("_failure") or field == "failure_reason":
            overlapping[field] = {str(reason): _counts(audit[audit[field].eq(reason)])
                                  for reason in sorted(audit[field].dropna().unique())}
    write_json(processed / "eeg_attrition.json", {
        "study_profile": EEG_PROFILE, "stages": stage_records, "scopes": scopes,
        "selection_counts": {key: _counts(frame) for key, frame in selections.items()},
        "selection_descriptives": {key: _descriptive_selection(frame) for key, frame in selections.items()},
        "manuscript_reference": reference_info, "overlapping_reasons": overlapping,
        "case_audit_sha256": file_sha256(processed / "eeg_operation_attrition.csv"),
    })
