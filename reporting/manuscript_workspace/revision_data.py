"""Read the selected study manifests and validate publication-report inputs."""

import hashlib
import json
from math import isclose
from pathlib import Path

import pandas as pd
from results_recreation.metrics_summary import (
    _evaluation_hash, _load_predictions, _prediction_content_hash, compute_point_metrics,
)


OUTCOMES = {"any_aki": "AKI", "icu_admission": "ICU admission"}
FAMILIES = {
    "logistic_regression": ("LR", "Logistic regression"),
    "xgboost": ("XGBoost", "XGBoost"),
    "ebm": ("EBM", "EBM"),
}
BRANCHES = {"non_windowed": "Whole-operation", "windowed": "Windowed"}
INPUTS = {
    "preop_only": "Preoperative only",
    "all_waveforms": "Waveforms only",
    "preop_and_all_waveforms": "Preoperative + waveforms",
}
CHANNELS = {
    "ecg_only": "ECG", "pleth_only": "PPG", "co2_only": "CO₂",
    "awp_only": "Airway pressure", "all_waveforms": "All four",
}
METRICS = {"auprc": "AP", "auroc": "AUROC", "brier": "Brier score"}
IDENTITY = ["caseid", "subjectid", "y_true", "repeat_id", "outer_fold_id"]
BASELINE_COLUMNS = IDENTITY + ["y_prob_raw", "y_prob_calibrated", "threshold", "y_pred_label"]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class Study:
    def __init__(self, study, scope):
        self.root = Path(study) / "results/catch22/experiments" / scope / "nested_cv"
        self.manifest_path = self.root / "run_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest["status"] != "validated" or self.manifest["diagnostic"]:
            raise ValueError(f"Expected a validated, non-diagnostic study at {self.root}")
        self.configs = {
            tuple(item[field] for field in ("model_type", "outcome", "branch", "feature_set")): item
            for item in self.manifest["configurations"]
        }

    def artifact(self, family, outcome, branch, inputs):
        keys = family, outcome, branch, inputs
        config = self.configs[keys]
        return Path(config.get("artifact_path") or self.root.joinpath("models", *keys))

    def predictions(self, *keys):
        file = self.artifact(*keys) / "predictions/test.csv"
        frame = pd.read_csv(file).sort_values("caseid").reset_index(drop=True)
        if not frame.caseid.is_unique or frame.diagnostic.any() or not frame.is_oof.all():
            raise ValueError(f"Expected one non-diagnostic outer-test row per operation in {file}")
        return frame

    def table(self, name):
        return pd.read_csv(self.root / "tables" / f"{name}.csv")

    def evidence(self):
        return {"path": str(self.manifest_path), "sha256": digest(self.manifest_path),
                "identity": self.manifest["study_identity"], "configurations": len(self.configs),
                "status": self.manifest["status"]}


def metric_row(frame, family, outcome, branch, inputs):
    match = frame[(frame.model_name == FAMILIES[family][1]) & (frame.outcome == outcome)
                  & (frame.branch == branch) & (frame.feature_set == inputs)]
    if len(match) != 1:
        raise ValueError(f"Expected one metric row for {family}, {outcome}, {branch}, {inputs}")
    return match.iloc[0]


def verify_contrast(metrics, family, outcome, branch):
    candidate = metric_row(metrics, family, outcome, branch, "preop_and_all_waveforms")
    reference = metric_row(metrics, family, outcome, branch, "preop_only")
    valid = (candidate.reference_model_name == reference.model_name
             and candidate.reference_feature_set == reference.feature_set
             and candidate.reference_prediction_content_hash == reference.prediction_content_hash
             and candidate.evaluation_case_hash == reference.evaluation_case_hash
             and candidate.bootstrap_draw_hash == reference.bootstrap_draw_hash
             and all(isclose(candidate[f"delta_{k}"], candidate[k] - reference[k],
                             rel_tol=0, abs_tol=1e-12) for k in METRICS))
    if not valid:
        raise ValueError(f"Unmatched paired contrast for {family}, {outcome}, {branch}")
    return candidate


def verify_baselines(studies, outcome, family):
    study = studies[family]
    keys = [(family, outcome, branch, "preop_only") for branch in BRANCHES]
    predictions = [study.predictions(*key) for key in keys]
    paths = [study.artifact(*key) for key in keys]
    folders = [sorted((path / "artifacts/folds").glob("repeat_*/outer_*")) for path in paths]
    if not all(len(folder) == 5 for folder in folders):
        raise ValueError(f"Expected five baseline fold bundles for {family}, {outcome}")
    comparable_metadata = ("inner_split", "calibration_split", "random_state", "n_trials",
                           "best_params", "effective_refit_params", "model_sha256", "preprocessor_sha256")
    differences = []
    for left, right in zip(*folders):
        a, b = [json.loads((p / "metadata.json").read_text()) for p in (left, right)]
        differences.append([field for field in comparable_metadata if a.get(field) != b.get(field)])
    equal = predictions[0][BASELINE_COLUMNS].equals(predictions[1][BASELINE_COLUMNS])
    fit_equal = not any(differences)
    return {"outcome": outcome, "family": family, "operations": len(predictions[0]),
            "predictions_equal": equal, "fitting_metadata_equal": fit_equal,
            "fold_metadata_differences": differences,
            "prediction_sha256": {branch: digest(path / "predictions/test.csv")
                                   for branch, path in zip(BRANCHES, paths)}}


def verify_metric_sources(studies, metrics, *, broader=False):
    outcomes = ("icu_admission",) if broader else OUTCOMES
    features = ("preop_only", "preop_and_all_waveforms") if broader else (
        "preop_only", "preop_and_all_waveforms", *CHANNELS,
    )
    evidence = {}
    for outcome in outcomes:
        for branch in BRANCHES:
            reference = None
            draws = None
            for family, study in studies.items():
                for feature in features:
                    path = study.artifact(family, outcome, branch, feature) / "predictions/test.csv"
                    prediction = _load_predictions(path, strict=True)
                    frame = prediction.df.sort_values("caseid").reset_index(drop=True)
                    row = metric_row(metrics, family, outcome, branch, feature)
                    replay = compute_point_metrics(prediction)
                    if reference is None:
                        reference, draws = frame[IDENTITY], row.bootstrap_draw_hash
                    if not (frame[IDENTITY].equals(reference)
                            and row.evaluation_case_hash == _evaluation_hash(frame)
                            and row.prediction_content_hash == _prediction_content_hash(prediction)
                            and row.bootstrap_draw_hash == draws
                            and row.bootstrap_replicates == 5000
                            and row.bootstrap_seed == 0
                            and not row.bootstrap_stratified
                            and all(isclose(row[key], replay[key], rel_tol=0, abs_tol=1e-12) for key in METRICS)
                            and row.analysis_status == "definitive"):
                        raise ValueError(f"Saved metrics or bootstrap provenance do not match {path}")
                    evidence[str(path)] = {"sha256": digest(path), "prediction_hash": row.prediction_content_hash,
                                           "evaluation_case_hash": row.evaluation_case_hash, "draw_hash": draws}
    return evidence
