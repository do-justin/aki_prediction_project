"""Confirm the amplitude-only EEG cohort against the frozen strict-run masks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

import pandas as pd

from data_preparation.artifact_metadata import file_sha256
from data_preparation.cohort_scopes import SCOPE_OUTCOMES, select_cohort, selected_case_hash
from data_preparation.eeg import AMPLITUDE_PROTOCOL, PROTOCOL
from data_preparation.eeg_study import TRACK_CATALOG_NAME, _counts, write_json
from model_creation.split_plan import eeg_input_fingerprints, merged_branch_path

CONFIRMATION_NAME = "eeg_amplitude_cohort_confirmation.json"
SCOPES = ("main_shared", "icu_waveform_sensitivity")
BRANCHES = ("non_windowed", "windowed")
ORIGINAL_FAILURES = [f"{mode}_{channel}_failure" for mode in ("full", "windowed") for channel in (
    "SNUADC_PLETH", "SNUADC_ECG_II", "Primus_CO2", "Primus_AWP",
)]
AMPLITUDE_MASKS = ("source_missing", "source_nonfinite", "raw_amplitude_high")
EXTRACTION_SOURCES = ("cohort_roster_sha256", "track_catalog_sha256", "waveform_release_sha256",
                      "recording_identity_sha256")
IDENTITY_COLUMNS = ("caseid", "subjectid", "eligible_any_aki", "aki_label", "eligible_icu_admission", "y_icu_admit")


def _audits(processed: Path) -> tuple[pd.DataFrame, dict, str]:
    path = processed / "eeg_operation_attrition.csv"
    summary = json.loads((processed / "eeg_attrition.json").read_text())
    digest = file_sha256(path)
    if summary["case_audit_sha256"] != digest or summary["study_profile"] != "eeg":
        raise ValueError(f"Could not verify EEG operation audit: {path}")
    return pd.read_csv(path).set_index("caseid", verify_integrity=True, drop=False), summary, digest


def _source_identity(processed: Path, protocol: str) -> tuple[dict, str]:
    path = processed / "eeg_extraction_checkpoints/fingerprint.json"
    fingerprint = json.loads(path.read_text())
    sources = {key: fingerprint["full"][key] for key in EXTRACTION_SOURCES}
    for mode in ("full", "windowed"):
        entry = fingerprint[mode]
        if (entry["eeg_protocol"]["protocol"] != protocol
                or {key: entry[key] for key in EXTRACTION_SOURCES} != sources):
            raise ValueError(f"Unexpected EEG source or protocol in {path} ({mode}).")
    if file_sha256(processed / TRACK_CATALOG_NAME) != sources["track_catalog_sha256"]:
        raise ValueError(f"EEG track catalog hash differs for {processed}.")
    return sources, file_sha256(path)


def _packed_union_count(qc: dict) -> int:
    samples = qc["expected_samples"]
    masks = qc["masks"]
    if samples <= 0 or masks["encoding"] != "np.packbits(bitorder=little), zlib, base64; truncate to expected_samples":
        raise ValueError("Unexpected strict EEG mask encoding or operative grid size.")
    union = 0
    for name in AMPLITUDE_MASKS:
        packed = zlib.decompress(base64.b64decode(masks["data"][name], validate=True))
        flags = int.from_bytes(packed, "little")
        if len(packed) != (samples + 7) // 8 or flags.bit_length() > samples or flags.bit_count() != qc["mask_counts"][name]:
            raise ValueError(f"Archived EEG mask {name} differs from its recorded count or operative grid.")
        union |= flags
    return union.bit_count()


def _replay(strict: pd.DataFrame, checkpoints: Path) -> tuple[pd.Series, str]:
    counts = {}
    digest = hashlib.sha256()
    candidates = strict.loc[strict["eeg_extraction_candidate"].eq(True)].sort_index()
    for case in candidates.itertuples(index=False):
        path = checkpoints / f"{case.caseid}.json"
        raw = path.read_bytes()
        digest.update(hashlib.sha256(raw).digest())
        saved = json.loads(raw)
        qc = saved["qc"]
        if (saved["caseid"] != case.caseid or qc["caseid"] != case.caseid or qc["protocol"] != PROTOCOL
                or qc.get("selected_source") != (None if pd.isna(case.selected_source) else case.selected_source)
                or qc["expected_samples"] != case.expected_samples):
            raise ValueError(f"Strict EEG checkpoint differs from its operation audit: {path}")
        if qc["selected_source"] is not None:
            if any(qc["mask_counts"][name] != getattr(case, f"mask_counts_{name}") for name in AMPLITUDE_MASKS):
                raise ValueError(f"Strict EEG mask counts differ from the operation audit: {path}")
            counts[case.caseid] = _packed_union_count(qc)
    return pd.Series(counts, dtype="int64"), digest.hexdigest()


def _require_roster(label: str, expected: pd.DataFrame, observed: pd.DataFrame, audit: pd.DataFrame) -> None:
    if selected_case_hash(expected) != selected_case_hash(observed):
        missing = sorted(set(expected["caseid"]) - set(observed["caseid"]))
        extra = sorted(set(observed["caseid"]) - set(expected["caseid"]))
        reasons = audit.loc[missing[:12], "failure_reason"].fillna("no QC failure recorded").to_dict()
        raise ValueError(f"{label} differs: {len(expected):,} expected, {len(observed):,} observed. "
                         f"Missing cases and recorded QC reasons: {reasons}. Extra cases: {extra[:12]}. "
                         "The operation/patient mapping must match exactly.")


def _describe(scopes: dict[str, pd.DataFrame], strict: pd.DataFrame) -> dict:
    output = {}
    for scope, frame in scopes.items():
        outcomes = SCOPE_OUTCOMES[scope]
        prior = strict.loc[strict[f"in_{scope}"].eq(True)]
        output[scope] = {
            **_counts(frame, outcomes), "selected_case_hash": selected_case_hash(frame),
            "strict_overlap_operations": int(frame["caseid"].isin(prior["caseid"]).sum()),
            "strict_overlap_patients": int(frame.loc[frame["subjectid"].isin(prior["subjectid"]), "subjectid"].nunique()),
        }
    return output


def confirm_cohort(strict_root: Path, study_root: Path | None = None) -> dict:
    strict_dir = strict_root / "data/processed"
    strict, _, strict_audit_hash = _audits(strict_dir)
    sources, strict_fingerprint_hash = _source_identity(strict_dir, PROTOCOL)
    union, checkpoint_hash = _replay(strict, strict_dir / "eeg_extraction_checkpoints")
    eeg_qc = strict.loc[union.index[union.mul(20).le(strict.loc[union.index, "expected_samples"])]]
    both = eeg_qc.loc[eeg_qc[ORIGINAL_FAILURES].isna().all(axis=1)]
    expected = {scope: select_cohort(both, scope) for scope in SCOPES}
    scopes = _describe(expected, strict)
    counts = (len(eeg_qc), len(both), scopes["main_shared"]["operations"], scopes["main_shared"]["patients"],
              scopes["main_shared"]["outcomes"]["any_aki"]["positive_operations"],
              scopes["main_shared"]["outcomes"]["icu_admission"]["positive_operations"],
              scopes["icu_waveform_sensitivity"]["operations"], scopes["icu_waveform_sensitivity"]["patients"],
              scopes["icu_waveform_sensitivity"]["outcomes"]["icu_admission"]["positive_operations"])
    if counts != (5088, 5010, 3153, 3048, 173, 756, 5010, 4851, 936):
        raise ValueError(f"Archived mask replay differs from the projected cohorts: {counts}.")
    report = {
        "status": "projected", "protocol": AMPLITUDE_PROTOCOL, "strict_root": str(strict_root),
        "strict_case_audit_sha256": strict_audit_hash, "strict_fingerprint_sha256": strict_fingerprint_hash,
        "strict_checkpoint_inventory_sha256": checkpoint_hash, "source_fingerprints": sources,
        "eeg_qc_operations": len(eeg_qc), "both_branches_operations": len(both), "scopes": scopes,
    }
    if study_root is None:
        return report

    target = study_root / "data/processed"
    observed, attrition, audit_hash = _audits(target)
    actual_sources, fingerprint_hash = _source_identity(target, AMPLITUDE_PROTOCOL)
    if actual_sources != sources or not observed[list(IDENTITY_COLUMNS)].equals(strict[list(IDENTITY_COLUMNS)]):
        raise ValueError("Amplitude extraction source, outcome, or operation/patient identities differ from the frozen study.")
    candidates = observed["eeg_extraction_candidate"].eq(True)
    recorded_union = observed.loc[union.index, "mask_counts_unusable_union"]
    unchanged = ["eeg_extraction_candidate", "selected_source", "expected_samples", *ORIGINAL_FAILURES]
    if not observed[unchanged].equals(strict[unchanged]):
        raise ValueError("The EEG candidate roster, raw source, operative grid, or original waveform eligibility changed.")
    differs = recorded_union.ne(union)
    if differs.any():
        raise ValueError(f"Amplitude unusable sample counts differ from replay in {differs.sum():,} operations. "
                         f"First cases: {differs.index[differs].tolist()[:12]}.")

    stages = {"eeg_qc": (eeg_qc, observed.loc[candidates & observed["status"].eq("passed")]),
              "eeg_features": (eeg_qc, observed.loc[candidates & observed["branch_status_full"].eq("success")
                                                       & observed["branch_status_windowed"].eq("success")]),
              "both_branches": (both, observed.loc[observed["both_branches"].eq(True)])}
    stage_counts = {row["key"]: row for row in attrition["stages"]}
    for stage, (wanted, actual) in stages.items():
        _require_roster(stage, wanted, actual, observed)
        if any(stage_counts[stage][key] != val for key, val in _counts(actual).items()):
            raise ValueError(f"Recorded EEG attrition counts differ at {stage}.")

    gate_path = study_root / "control/eeg_split_preflight.json"
    gate = json.loads(gate_path.read_text())
    plan = gate["split_plan"]
    if (gate["status"] != "validated" or plan["status"] != "validated"
            or eeg_input_fingerprints(target) != {key: gate[key] for key in ("dataset_sha256", "input_provenance_sha256")}):
        raise ValueError("Amplitude split preflight or prepared input fingerprints have not been validated.")
    if {tuple(item) for item in plan["scope_outcome_branches"]} != {
        (scope, outcome, branch) for scope in SCOPES for outcome in SCOPE_OUTCOMES[scope] for branch in BRANCHES
    }:
        raise ValueError("Amplitude split preflight is missing an outcome or extraction branch.")
    for scope, frame in expected.items():
        _require_roster(scope, frame, observed.loc[observed[f"in_{scope}"].eq(True)], observed)
        if any(attrition["scopes"][scope][key] != val for key, val in _counts(frame, SCOPE_OUTCOMES[scope]).items()):
            raise ValueError(f"Recorded EEG attrition counts differ for {scope}.")
        for outcome in SCOPE_OUTCOMES[scope]:
            item = plan["scope_outcome_plans"][f"{scope}/{outcome}"]
            folds = [caseid for fold in item["outer_test_caseids"] for caseid in fold]
            summary = scopes[scope]["outcomes"][outcome]
            values = {"operations": len(frame), "patients": frame["subjectid"].nunique(),
                      **{key: summary[key] for key in ("positive_operations", "positive_patients")}}
            if sorted(folds) != sorted(frame["caseid"]) or any(item[key] != val for key, val in values.items()):
                raise ValueError(f"Amplitude outer-fold operation IDs or outcome counts differ for {scope}/{outcome}.")
    for branch in BRANCHES:
        matrix = pd.read_csv(merged_branch_path(target, branch), usecols=list(IDENTITY_COLUMNS))
        for scope, frame in expected.items():
            roster = select_cohort(matrix, scope)
            _require_roster(f"{branch}/{scope}", frame, roster, observed)
            if _counts(roster, SCOPE_OUTCOMES[scope]) != _counts(frame, SCOPE_OUTCOMES[scope]):
                raise ValueError(f"Amplitude prepared outcome counts differ for {branch}/{scope}.")
    return {**report, "status": "confirmed", "case_audit_sha256": audit_hash,
            "fingerprint_sha256": fingerprint_hash, "split_preflight_sha256": file_sha256(gate_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-root", required=True, type=Path)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    if args.replay_only == (args.study_root is not None):
        parser.error("Choose one of --study-root or --replay-only.")
    strict = args.strict_root.resolve()
    study = args.study_root.resolve() if args.study_root else None
    if study is not None:
        output = study / "control" / CONFIRMATION_NAME
        if (study.is_relative_to(strict) or strict.is_relative_to(study)
                or not output.resolve().is_relative_to(study)):
            parser.error("The amplitude run and its confirmation must use a study root separate from the strict witness.")
    report = confirm_cohort(strict, study)
    if study is not None:
        write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
