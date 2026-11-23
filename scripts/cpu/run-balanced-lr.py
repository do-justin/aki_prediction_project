"""Run the display-scoped balanced LR revision against sealed manuscript inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.artifact_metadata import file_sha256
from model_creation.experiment_queue import _copy_verified_split_caches
from model_creation.run_inventory import SOURCE_NAMES
from model_creation.split_plan import prepare_split_plan
from runtime_resources import available_cpu_count
from run_catch22 import _study_identity, _write_manifest

STUDIES = Path("/media/volume/catch22/data/aki_prediction_project/studies")
PROTECTED_NAMES = (
    "manuscript-core-v1", "manuscript-optimized-v1",
    "manuscript-supplementary-v1", "manuscript-icu-supplementary-v1",
)
RESULTS = Path("results/catch22/experiments")
CORE_FEATURES = ("preop_only", "all_waveforms", "preop_and_all_waveforms")
INDIVIDUAL_CHANNELS = ("ecg_only", "pleth_only", "co2_only", "awp_only")
SCOPES = {
    "main_shared": (("any_aki", "icu_admission"), CORE_FEATURES + INDIVIDUAL_CHANNELS),
    "icu_waveform_sensitivity": (("icu_admission",), ("preop_only", "preop_and_all_waveforms")),
}
REFERENCE_STUDIES = {
    "main_shared": "manuscript-supplementary-v1",
    "icu_waveform_sensitivity": "manuscript-icu-supplementary-v1",
}


def build_plan(studies: Path) -> dict:
    """Validate the inputs and resolve every required reference to its original fit."""
    processed = studies / "manuscript-core-v1/data/processed"
    dataset_hashes = {branch: file_sha256(processed / name) for branch, name in SOURCE_NAMES.items()}
    source_manifests, jobs, folds = {}, [], {}
    identity = _study_identity()
    for scope, (outcomes, features) in SCOPES.items():
        results = studies / REFERENCE_STUDIES[scope] / RESULTS / scope / "nested_cv"
        manifest_path = results / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "validated" or manifest.get("failures"):
            raise ValueError(f"Invalid frozen study status: {manifest_path}")
        source_manifests[str(manifest_path)] = file_sha256(manifest_path)
        configs = {
            (row["outcome"], row["branch"], row["feature_set"]): row
            for row in manifest["configurations"] if row["model_type"] == "logistic_regression"
        }
        for outcome in outcomes:
            for branch, dataset_hash in dataset_hashes.items():
                roster = manifest["branch_rosters"][branch]
                for feature in features:
                    source = configs[outcome, branch, feature]
                    directory = Path(source.get("artifact_path") or
                                     results / "models/logistic_regression" / outcome / branch / feature)
                    validation_path = directory / "artifacts/validation.json"
                    fp = json.loads(validation_path.read_text())["validation_fingerprint"]
                    expected = {
                        "cohort_scope": scope, "model_type": "logistic_regression", "outcome": outcome,
                        "branch": branch, "feature_set": feature, "dataset_hash": dataset_hash,
                        "selected_case_hash": roster["selected_case_hash"], "n_trials": 20,
                        "class_weighting": "none", "outer_folds": 5, "inner_folds": 3,
                        "calibration_folds": 5, "repeats": 1, "base_random_state": 42, "diagnostic": False,
                        "preprocessing_protocol": identity["preprocessing_protocol"],
                        "environment": identity["environment"], "preserve_nan": True,
                    }
                    if roster["dataset_sha256"] != dataset_hash or any(fp.get(k) != v for k, v in expected.items()):
                        raise ValueError(f"Frozen model uses different inputs or validation: {validation_path}")
                    key = "/".join((scope, outcome, branch))
                    if folds.setdefault(key, fp["fold_assignment_hash"]) != fp["fold_assignment_hash"]:
                        raise ValueError(f"Frozen fits disagree about folds for {key}")
                    jobs.append({
                        "cohort_scope": scope, "model_type": "logistic_regression", "outcome": outcome,
                        "branch": branch, "feature_set": feature, "n_trials": 50,
                        "class_weighting": "fold_training_class_weight_balanced",
                        "source_validation": str(validation_path), "source_validation_sha256": file_sha256(validation_path),
                    })
    return {
        "study_identity": identity, "processed_dir": str(processed), "dataset_sha256": dataset_hashes,
        "source_manifests_sha256": source_manifests, "fold_assignment_hashes": folds, "configurations": jobs,
    }


def run_scope(scope: str, budget: int, root: Path, plan: dict, raw: Path) -> None:
    _, features = SCOPES[scope]
    study = root / scope
    env = dict(os.environ)
    for name in ("DATA_DIR", "RESULTS_DIR", "PAPER_DIR", "LOG_FILE", "TABLES_DIR", "FIGURES_DIR",
                 "REPORTS_DIR", "SMOKE_ROOT", "XAI_DIR"):
        env.pop(name, None)
    env.update(AKI_ARTIFACT_ROOT=str(study), RAW_DIR=str(raw), PROCESSED_DIR=plan["processed_dir"],
               AKI_LR_CLASS_WEIGHT="balanced", MPLBACKEND="Agg", PYTHONNOUSERSITE="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    command = [
        sys.executable, "-m", "run_catch22", "experiments", "--prep", "skip",
        "--model-types", "logistic_regression", "--cohort-scope", scope,
        "--n-trials", "50", "--threads-per-model", "12", "--blas-threads", "4",
        "--cpu-budget", str(budget), "--max-workers", str(min(4, budget // 4)),
        "--feature-sets", *features, "--resume", "--bootstrap", "5000",
        "--bootstrap-seed", "0", "--no-bootstrap-stratified",
    ]
    # The canonical storage policy remains enforced when using its research volume.
    log = study / "controller.log"
    with log.open("a") as stream:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
    manifest_path = study / RESULTS / scope / "nested_cv/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {key: value for key, value in plan["fold_assignment_hashes"].items() if key.startswith(scope + "/")}
    if manifest["status"] != "validated" or manifest["split_plan"]["fold_assignment_hashes"] != expected:
        raise ValueError(f"Completed run is missing report verification or uses unexpected folds: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--reference-studies", type=Path, default=STUDIES)
    parser.add_argument("--raw-dir", type=Path, default=Path("/media/volume/catch22/projects/aki_prediction_project/data/raw"))
    parser.add_argument("--cpu-budget", type=int, default=min(56, available_cpu_count() - 4))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root, studies = args.study_root.resolve(), args.reference_studies.resolve()
    if root == studies or any(root.is_relative_to(studies / name) or (studies / name).is_relative_to(root)
                              for name in PROTECTED_NAMES):
        parser.error("Choose an output root separate from the protected manuscript studies.")
    if any(not (root / scope / RESULTS).resolve().is_relative_to(root) for scope in SCOPES):
        parser.error("Scoped output paths must stay under the challenger root.")
    if not (12 <= args.cpu_budget <= available_cpu_count()):
        parser.error("CPU budget must be between 12 and the available CPU count.")
    plan = build_plan(studies)
    plan["cpu_budget"] = args.cpu_budget
    plan["raw_dir"] = str(args.raw_dir.resolve())
    plan["source_sha256"] = {
        name: file_sha256(PROJECT_ROOT / name) for name in
        ("run_catch22.py", "model_creation/experiment_queue.py", "scripts/cpu/run-balanced-lr.py")
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "challenger-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("The recorded source inputs, code or run plan changed. Use a fresh study root.")
    _write_manifest(plan_path, plan)
    for scope in SCOPES:
        study = root / scope
        cache = study / "control/split_cache"
        reference = studies / REFERENCE_STUDIES[scope]
        _copy_verified_split_caches([{
            "source_results_dir": str(reference / RESULTS / scope / "nested_cv"),
            "source_split_cache_dir": str(reference / "control/split_cache"),
        }], cache)
        jobs = [job for job in plan["configurations"] if job["cohort_scope"] == scope]
        splits = prepare_split_plan(Path(plan["processed_dir"]), jobs, cache, workers=min(8, args.cpu_budget))
        expected = {key: value for key, value in plan["fold_assignment_hashes"].items() if key.startswith(scope + "/")}
        if splits["fold_assignment_hashes"] != expected:
            raise ValueError(f"Challenger fold assignments differ from the saved {scope} reference.")
        _write_manifest(study / "verified-split-plan.json", splits)
    main_budget = max(4, args.cpu_budget * 3 // 16 * 4)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_scope, "main_shared", main_budget, root, plan, args.raw_dir),
            executor.submit(run_scope, "icu_waveform_sensitivity", args.cpu_budget - main_budget, root, plan, args.raw_dir),
        ]
        for future in futures:
            future.result()
    for path, digest in plan["source_manifests_sha256"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Protected source manifest changed during execution: {path}")


if __name__ == "__main__":
    main()
