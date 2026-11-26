"""EEG grid identity, feature leakage, and paired prefit regression checks."""

import numpy as np
import pandas as pd
import pytest

from data_preparation.artifact_metadata import STEP_05_MERGED_ARTIFACT, file_sha256, write_artifact_metadata
from model_creation import split_plan, utils
from model_creation.experiment_queue import _eeg_expected_keys
from model_creation.preprocessing import FoldPreprocessor


def _waveform_columns(branch):
    descriptors = utils.catch24_descriptor_names()
    suffixes = descriptors if branch == "non_windowed" else [
        f"{name}_{reducer}" for reducer in ("mean", "std", "min", "max") for name in descriptors
    ]
    return [f"{prefix}_{suffix}" for prefix in utils.EEG_WAVEFORM_PREFIXES.values() for suffix in suffixes]


def test_eeg_exact_feature_grid_keeps_qc_and_outcomes_out(monkeypatch):
    monkeypatch.setenv("AKI_STUDY_PROFILE", "eeg")
    waveform_columns = _waveform_columns("windowed")
    df = pd.DataFrame(np.nan, index=[0], columns=waveform_columns)
    for column in ("age", "preop_cr", "aki_label", "caseid", "subjectid", "cohort_membership",
                   "BIS_SQI", "eeg_interpolated_fraction", "eeg_selected_source", "unapproved_preop"):
        df[column] = 42

    features = utils.get_feature_sets(df, branch="windowed")
    assert tuple(features) == utils.EEG_FEATURE_SETS
    assert len({frozenset(columns) for columns in features.values()}) == 23
    assert features["preop_only"] == ["age", "preop_cr"]
    assert len(features["eeg_only"]) == 96
    assert len(features["original_four_waveforms"]) == 384
    assert len(features["all_five_waveforms"]) == 480
    assert set(features["preop_and_original_four"]) == set(features["preop_and_all_five"]) - set(features["eeg_only"])
    for group in ("pleth", "ecg", "co2", "awp"):
        assert set(features[f"preop_and_five_minus_{group}"]) - set(features[f"preop_and_four_minus_{group}"]) == set(features["eeg_only"])
        assert not set(features[f"preop_and_five_minus_{group}"]) & set(features[f"{group}_only"])
    with pytest.raises(ValueError, match="exact descriptor columns"):
        utils.get_feature_sets(df.assign(BIS_EEG1_WAV_quality=1), branch="windowed")
    with pytest.raises(ValueError, match="exact descriptor columns"):
        utils.get_feature_sets(df, branch="non_windowed")

    monkeypatch.delenv("AKI_STUDY_PROFILE")
    existing = utils.get_feature_sets(df.loc[:, ["age", "preop_cr", *features["original_four_waveforms"]]])
    assert existing["all_waveforms"] == features["original_four_waveforms"]
    assert existing["preop_and_all_waveforms"] == features["preop_and_original_four"]
    assert "all_five_waveforms" not in existing


def test_eeg_descriptor_remains_numeric_with_csv_string_values():
    descriptor = next(name for name in utils.catch24_descriptor_names() if name.casefold() == "dn_mean")
    name = f"BIS_EEG1_WAV_{descriptor}"
    data = pd.DataFrame({name: ["3", "invalid", "4"]})
    preprocessor = FoldPreprocessor().fit(data)
    assert preprocessor.numeric_columns_ == [name]
    assert preprocessor.categorical_columns_ == []
    assert preprocessor.transform(data)[name].tolist()[0] == 3
    assert np.isnan(preprocessor.transform(data)[name].iloc[1])


def _write_eeg_branches(processed, *, mismatch=False):
    patients = np.repeat(np.arange(48), 2)
    identities = pd.DataFrame({
        "caseid": np.arange(len(patients)), "subjectid": patients, "age": 55.0,
        "aki_label": (patients % 3 == 0).astype(float), "eligible_any_aki": (patients < 36).astype(int),
        "y_icu_admit": patients % 2, "eligible_icu_admission": 1,
    })
    identities.loc[identities["eligible_any_aki"].eq(0), "aki_label"] = np.nan
    for column in utils.MERGED_REQUIRED_COLUMNS:
        if column not in identities:
            identities[column] = np.nan
    for branch in ("non_windowed", "windowed"):
        values = pd.DataFrame(1.0, index=identities.index, columns=_waveform_columns(branch))
        df = pd.concat([identities, values], axis=1)
        if branch == "windowed" and mismatch:
            df.loc[0, "y_icu_admit"] = 1
        path = split_plan.merged_branch_path(processed, branch)
        df.to_csv(path, index=False)
        write_artifact_metadata(path, artifact_role=STEP_05_MERGED_ARTIFACT, available_columns=df.columns,
                                extra_metadata={
                                    "study_profile": "eeg", "data_sha256": file_sha256(path),
                                    "extraction_fingerprint": {"protocol": "eeg-qc-v1", "branch": branch},
                                    "cohort_scopes": {
                                        scope: {"cohort_manifest_identity": identity}
                                        for scope, identity in utils.EEG_SCOPE_IDS.items()
                                    },
                                })


def test_eeg_all_scope_split_preflight_shares_branches(tmp_path, monkeypatch):
    monkeypatch.setenv("AKI_STUDY_PROFILE", "eeg")
    _write_eeg_branches(tmp_path)
    preflight = split_plan.prepare_eeg_split_preflight(tmp_path, tmp_path / "cache", workers=2)
    plan = preflight["split_plan"]
    assert plan["unique_outer_keys"] == 3
    assert plan["ebm_bags_checked"]
    assert preflight["split_cache_sha256"]
    assert len(_eeg_expected_keys("main_shared", principal=True)) == 30
    assert len(_eeg_expected_keys("main_shared")) + len(_eeg_expected_keys("icu_waveform_sensitivity")) == 414
    scope_plans = plan["scope_outcome_plans"]
    assert set(scope_plans) == {"main_shared/any_aki", "main_shared/icu_admission", "icu_waveform_sensitivity/icu_admission"}
    for key, scope_plan in scope_plans.items():
        assert plan["fold_assignment_hashes"][f"{key}/non_windowed"] == \
               plan["fold_assignment_hashes"][f"{key}/windowed"] == scope_plan["fold_assignment_hash"]
        assert len(scope_plan["hpo_fold_hashes"]) == len(scope_plan["calibration_fold_hashes"]) == 5
        held_out = scope_plan["outer_test_caseids"]
        assert sorted(case for fold in held_out for case in fold) == list(range(scope_plan["operations"]))
        owners = {case: fold for fold, cases in enumerate(held_out) for case in cases}
        assert all(owners[case] == owners[case + 1] for case in range(0, len(owners), 2))
    assert scope_plans["main_shared/any_aki"]["positive_patients"] == 12
    assert scope_plans["main_shared/icu_admission"]["operations"] == 72
    assert scope_plans["icu_waveform_sensitivity/icu_admission"]["operations"] == 96
    assert len(preflight["feature_membership"]["non_windowed"]["all_five_waveforms"]) == 120


def test_eeg_preflight_rejects_label_drift_before_creating_split_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AKI_STUDY_PROFILE", "eeg")
    _write_eeg_branches(tmp_path, mismatch=True)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="identical ordered operation, patient, and outcome rosters"):
        split_plan.prepare_eeg_split_preflight(tmp_path, cache, workers=1)
    assert not cache.exists()


def test_eeg_queue_infeasible_preflight_never_creates_scheduler(tmp_path, monkeypatch, capsys):
    import run_catch22
    from model_creation import experiment_queue

    monkeypatch.setenv("AKI_STUDY_PROFILE", "eeg")
    monkeypatch.setenv("AKI_LR_CLASS_WEIGHT", "balanced")
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(run_catch22, "_study_identity", lambda: {"test_identity": True})
    monkeypatch.setattr(run_catch22, "_configuration_grid", lambda args: [
        {"model_type": "logistic_regression", "outcome": "icu_admission", "branch": "non_windowed",
         "feature_set": "preop_only", "n_trials": 50},
    ])
    monkeypatch.setattr(experiment_queue, "available_cpu_count", lambda: 8)
    monkeypatch.setattr(experiment_queue, "_load_or_prepare_eeg_preflight",
                        lambda *args: {"status": "infeasible", "reason": "insufficient patient support"})
    monkeypatch.setattr(experiment_queue, "run_configurations", lambda *args, **kwargs: pytest.fail("Admitted a fit"))

    args = run_catch22.build_parser().parse_args(
        ["experiments", "--prep", "skip", "--cpu-budget", "4", "--threads-per-model", "1",
         "--blas-threads", "1", "--max-workers", "1"]
    )
    args.study_profile = "eeg"
    assert experiment_queue.execute(args) == 1
    assert "insufficient patient support" in capsys.readouterr().err
    assert not (tmp_path / "control" / "active_scheduler.json").exists()
    assert not list(tmp_path.glob("results/**/run_manifest.json"))


def test_eeg_resume_revalidates_source_without_overwriting_frozen_gate(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import run_catch22
    from model_creation.experiment_queue import _load_or_prepare_eeg_preflight

    gate = tmp_path / "eeg_split_preflight.json"
    identity = {"code": "test"}
    input_identity = {"non_windowed": "full-data", "windowed": "time-data"}
    saved = {
        "status": "validated", "study_identity": identity, "dataset_sha256": input_identity,
        "split_cache_sha256": {},
    }
    gate.write_text(json.dumps(saved))
    frozen_bytes = gate.read_bytes()
    monkeypatch.setattr(split_plan, "eeg_input_fingerprints", lambda path: {"dataset_sha256": input_identity})
    monkeypatch.setattr(run_catch22, "_base_env", lambda *args, **kwargs: {"profile": "eeg"})
    monkeypatch.setattr(run_catch22, "_validate_processed_artifacts", lambda **kwargs: 1)
    monkeypatch.setattr(run_catch22, "_ensure_data_prepared", lambda *args, **kwargs: pytest.fail("Rebuilt the source"))
    context = {"layout": SimpleNamespace(processed_dir=tmp_path), "log": tmp_path / "validation.log"}
    args = SimpleNamespace(prep="auto", report_only=False)

    result = _load_or_prepare_eeg_preflight(args, context, tmp_path, identity, workers=1)
    assert result["status"] == "blocked"
    assert "source inputs did not pass validation" in result["reason"]
    assert gate.read_bytes() == frozen_bytes
    assert json.loads((tmp_path / "eeg_split_preflight_last_blocked.json").read_text())["status"] == "blocked"

    monkeypatch.setattr(run_catch22, "_validate_processed_artifacts", lambda **kwargs: 0)
    assert _load_or_prepare_eeg_preflight(args, context, tmp_path, identity, workers=1) == saved
    assert gate.read_bytes() == frozen_bytes
