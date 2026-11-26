"""Warm exact-input grouped partition caches before configuration admission."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path


class SplitFeasibilityError(ValueError):
    """Validated inputs cannot support the prespecified patient partitions."""


def _partitions(task):
    import numpy as np
    import pandas as pd
    from model_creation.validation import build_cv_splits, make_ebm_bags
    y_values, group_values, folds, seed, cache_dir, check_bags = task
    os.environ["AKI_SPLIT_CACHE_DIR"] = str(cache_dir)
    y, groups = pd.Series(y_values), pd.Series(group_values)
    try:
        splits, actual = build_cv_splits(pd.DataFrame(index=np.arange(len(y))), y, groups, folds, seed)
        if actual != folds:
            raise ValueError("Requested grouped fold count is infeasible; refusing silent reduction")
        for train, test in splits:
            if y.iloc[train].nunique() != 2 or y.iloc[test].nunique() != 2:
                raise ValueError("Every requested partition must contain both outcome classes")
        if check_bags:
            # HPO's single bag is the first production bag for the same seed.
            for indices in [np.arange(len(y)), *(train for train, _ in splits)]:
                for fraction in (0.1, 0.15, 0.2):
                    make_ebm_bags(y.iloc[indices], groups.iloc[indices], n_bags=14,
                                  validation_size=fraction, random_state=seed)
    except ValueError as error:
        raise SplitFeasibilityError(str(error)) from error
    return [(train.tolist(), test.tolist()) for train, test in splits]


def prepare_split_plan(processed_dir: Path, configurations: list[dict], cache_dir: Path,
                       *, outer_folds=5, inner_folds=3, calibration_folds=5,
                       workers=8, matched_branches=False, frames=None) -> dict:
    """Preserve selection/tie ordering; parallelize only independent cache keys."""
    import pandas as pd
    from model_creation.validation import select_modeling_dataset
    arrays, case_ids = {}, {}
    for config in configurations:
        key = (config["cohort_scope"], config["outcome"], config["branch"])
        if key not in arrays:
            frame = frames[key[2]] if frames is not None else pd.read_csv(merged_branch_path(processed_dir, key[2]))
            _, _, y, cases, groups = select_modeling_dataset(
                frame, key[1], "preop_only", cohort_scope=key[0])
            arrays[key] = (y.tolist(), groups.tolist())
            case_ids[key] = cases.tolist()
    scope_outcome_plans = {}
    if matched_branches:
        for scope, outcome in dict.fromkeys(key[:2] for key in arrays):
            key = (scope, outcome, "non_windowed")
            other = (scope, outcome, "windowed")
            if key not in arrays or other not in arrays or arrays[key] != arrays[other] or case_ids[key] != case_ids[other]:
                raise ValueError(f"EEG branches must have identical ordered operation, patient, and outcome rosters for {scope}/{outcome}.")
            y, groups = arrays[key]
            if set(y) != {0, 1}:
                raise SplitFeasibilityError(f"EEG scope/outcome {scope}/{outcome} lacks two eligible outcome classes.")
            rows = list(zip(case_ids[key], groups, y))
            scope_outcome_plans[f"{scope}/{outcome}"] = {
                "ordered_roster_sha256": _digest(rows),
                "operations": len(rows), "patients": len(set(groups)),
                "positive_operations": sum(y),
                "positive_patients": len({group for group, label in zip(groups, y) if label == 1}),
            }

    check_bags = any(c["model_type"] == "ebm" for c in configurations)
    unique = {}
    for key, (y, groups) in arrays.items():
        digest = _digest([y, groups])
        unique.setdefault(digest, (y, groups))
    # Spawn imports numerical libraries with one thread, avoiding nested BLAS pools.
    names = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            outer_tasks = [(y, g, outer_folds, 42, str(cache_dir), False) for y, g in unique.values()]
            outer_results = list(pool.map(_partitions, outer_tasks))
            by_input = dict(zip(unique, outer_results))
            fold_hashes = {}
            for key, (y, groups) in arrays.items():
                digest = _digest([y, groups])
                cases = case_ids[key]
                roster = [{"train": [cases[i] for i in train], "test": [cases[i] for i in test]}
                          for train, test in by_input[digest]]
                fold_hashes["/".join(key)] = _digest(roster)
                if matched_branches:
                    scope_plan = scope_outcome_plans["/".join(key[:2])]
                    scope_plan["fold_assignment_hash"] = fold_hashes["/".join(key)]
                    scope_plan["outer_test_caseids"] = [row["test"] for row in roster]
            nested_tasks = []
            nested_keys = []
            for digest, (y, groups) in unique.items():
                for fold, (train, _) in enumerate(by_input[digest]):
                    for count in dict.fromkeys((inner_folds, calibration_folds)):
                        nested_tasks.append(([y[i] for i in train], [groups[i] for i in train],
                                             count, 42 + fold, str(cache_dir), check_bags))
                        nested_keys.append((digest, fold, count))
            nested_results = list(pool.map(_partitions, nested_tasks))
            if matched_branches:
                nested_by_input = dict(zip(nested_keys, nested_results))
                for key, (y, groups) in arrays.items():
                    if key[2] != "non_windowed":
                        continue
                    digest = _digest([y, groups])
                    cases = case_ids[key]
                    scope_plan = scope_outcome_plans["/".join(key[:2])]
                    for role, count in (("hpo", inner_folds), ("calibration", calibration_folds)):
                        scope_plan[f"{role}_fold_hashes"] = [
                            _digest([[cases[train[i]] for i in test]
                                     for _, test in nested_by_input[(digest, fold, count)]])
                            for fold, (train, _) in enumerate(by_input[digest])
                        ]
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return {"status": "validated", "scope_outcome_branches": [list(k) for k in arrays],
            "unique_outer_keys": len(unique), "nested_requests": len(nested_tasks),
            "outer_folds": outer_folds, "inner_folds": inner_folds,
            "calibration_folds": calibration_folds, "ebm_bags_checked": check_bags,
            "fold_assignment_hashes": fold_hashes,
            "cache_dir": str(cache_dir),
            **({"scope_outcome_plans": scope_outcome_plans} if matched_branches else {})}


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def merged_branch_path(processed_dir: Path, branch: str) -> Path:
    suffix = {"non_windowed": "", "windowed": "_windowed"}[branch]
    return processed_dir / f"aki_features_master_wide{suffix}.csv"


def eeg_input_fingerprints(processed_dir: Path) -> dict:
    """Bind both matrices and the exact preprocessing provenance for each."""
    from data_preparation.artifact_metadata import file_sha256, load_artifact_metadata, metadata_path_for
    from model_creation.utils import EEG_SCOPE_IDS

    identity = {"dataset_sha256": {}, "input_provenance_sha256": {}}
    for branch in ("non_windowed", "windowed"):
        path = merged_branch_path(processed_dir, branch)
        metadata = load_artifact_metadata(path)
        if (metadata.get("study_profile") != "eeg" or not metadata.get("extraction_fingerprint")
                or any(metadata.get("cohort_scopes", {}).get(scope, {}).get("cohort_manifest_identity") != value
                       for scope, value in EEG_SCOPE_IDS.items())):
            raise ValueError(f"Missing EEG protocol or cohort identity in {path}.")
        dataset_hash = file_sha256(path)
        if metadata.get("data_sha256") != dataset_hash:
            raise ValueError(f"EEG modeling matrix hash violates its provenance sidecar: {path}.")
        identity["dataset_sha256"][branch] = dataset_hash
        identity["input_provenance_sha256"][branch] = file_sha256(metadata_path_for(path))
    return identity


def prepare_eeg_split_preflight(processed_dir: Path, cache_dir: Path, *, workers: int) -> dict:
    """Validate the full experiment's matched model inputs before admitting any fit."""
    from data_preparation.artifact_metadata import STEP_05_MERGED_ARTIFACT, file_sha256, read_versioned_csv
    from data_preparation.cohort_scopes import SCOPE_OUTCOMES
    from model_creation.utils import EEG_SCOPE_IDS, MERGED_REQUIRED_COLUMNS, get_eeg_feature_sets

    fingerprints = eeg_input_fingerprints(processed_dir)
    configurations, frames, memberships = [], {}, {}
    for branch in ("non_windowed", "windowed"):
        frames[branch], _ = read_versioned_csv(
            merged_branch_path(processed_dir, branch), artifact_role=STEP_05_MERGED_ARTIFACT,
            required_columns=MERGED_REQUIRED_COLUMNS,
        )
        memberships[branch] = get_eeg_feature_sets(frames[branch], branch=branch)
        configurations.extend(
            {"cohort_scope": scope, "outcome": outcome, "branch": branch, "model_type": "ebm"}
            for scope in EEG_SCOPE_IDS for outcome in SCOPE_OUTCOMES[scope]
        )
    preop_columns = ["caseid", *memberships["non_windowed"]["preop_only"]]
    if (memberships["non_windowed"]["preop_only"] != memberships["windowed"]["preop_only"]
            or not frames["non_windowed"][preop_columns].equals(frames["windowed"][preop_columns])):
        raise ValueError("EEG branches must have identical ordered preoperative predictor data.")
    plan = prepare_split_plan(processed_dir, configurations, cache_dir, workers=workers,
                              matched_branches=True, frames=frames)
    if eeg_input_fingerprints(processed_dir) != fingerprints:
        raise ValueError("EEG modeling data or provenance changed during split preflight.")
    return {**fingerprints, "split_plan": plan, "feature_membership": memberships,
            "split_cache_sha256": {path.name: file_sha256(path) for path in sorted(cache_dir.glob("*.json"))}}
