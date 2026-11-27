"""Read verified experimental fold bundles and reconstruct EBM displays."""
from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from data_preparation.artifact_metadata import file_sha256
from model_creation.postprocessing import LogisticRecalibrationModel, apply_logistic_recalibration
from model_creation.validation import is_outer_fold_checkpoint_complete
from reporting import publication_style as style
from reporting.display_dictionary import load_display_dictionary
from reporting.manuscript_workspace.revision_data import OUTCOMES

CHANNELS = {"SNUADC_PLETH_": "PPG", "SNUADC_ECG_II_": "ECG", "Primus_CO2_": "CO₂",
            "Primus_AWP_": "Airway pressure", "BIS_EEG1_WAV_": "EEG"}
NUMERIC_FOLD_FILES = {
    "logistic_regression": ("coefficients.csv", "explanation_metadata.json"),
    "xgboost": ("heldout_shap_values.csv.gz", "shap_importance.csv", "explanation_metadata.json"),
    "ebm": ("term_importance.csv", "heldout_term_contributions.csv.gz", "native_explanation.json",
            "special_bin_scores.csv", "explanation_metadata.json"),
}


def _fold_dir(predictions, repeat, fold):
    return predictions.path.parent.parent / "artifacts/folds" / f"repeat_{int(repeat):02d}" / f"outer_{int(fold):02d}"


def _fold_inputs(study, raw, predictions, repeat, fold):
    path = _fold_dir(predictions, repeat, fold)
    if not all(item.resolve().is_relative_to(study["root"]) for item in path.iterdir()):
        raise ValueError(f"EEG explanation fold contains an external artifact: {path}")
    if not is_outer_fold_checkpoint_complete(path):
        raise ValueError(f"EEG explanation requires a complete saved fold: {path}")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["validation_fingerprint"]["dataset_hash"] != raw.attrs["source_dataset_sha256"]:
        raise ValueError(f"EEG explanation dataset does not match fitted fold: {path}")
    preprocessor = joblib.load(path / "preprocessor.pkl")
    model = joblib.load(path / "model.pkl")
    training = pd.read_csv(path / "train_oof.csv", usecols=["caseid", "subjectid"])
    test = predictions.df.loc[predictions.df.repeat_id.eq(repeat) & predictions.df.outer_fold_id.eq(fold)]
    if (set(training.caseid) & set(test.caseid) or set(training.subjectid) & set(test.subjectid)
            or training.caseid.duplicated().any()):
        raise ValueError(f"EEG explanation training and test rosters overlap: {path}")
    provenance = {"fold": fold, "repeat": repeat, "model_sha256": metadata["model_sha256"],
                  "preprocessor_sha256": metadata["preprocessor_sha256"], "metadata_sha256": file_sha256(path / "metadata.json"),
                  "model_bundle": str(path), "prediction_csv": str(predictions.path)}
    return preprocessor, model, training, test, provenance


def _read_modeling_matrix(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.caseid.duplicated().any():
        raise ValueError(f"Duplicate EEG modeling operations: {path}")
    raw.attrs["source_dataset_sha256"] = file_sha256(path)
    return raw


@lru_cache(maxsize=2)
def _fold_export_matrix(matrix: Path, expected_hash: str) -> pd.DataFrame:
    raw = _read_modeling_matrix(matrix)
    if raw.attrs["source_dataset_sha256"] != expected_hash:
        raise ValueError(f"EEG fold export modeling dataset hash mismatch: {matrix}")
    return raw


def _restore_fold_export(fold, expected_hash, destination, cached):
    source, hashes = cached
    try:
        metadata = json.loads((fold / "metadata.json").read_text())
        saved = json.loads((source / "explanation_metadata.json").read_text())
        if (saved["source_fold"] != str(fold)
                or metadata["validation_fingerprint"]["dataset_hash"] != expected_hash
                or any(key not in saved or saved[key] != value for key, value in metadata.items())):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not is_outer_fold_checkpoint_complete(fold):
        raise ValueError(f"EEG report fold checkpoint failed verification: {fold}")

    destination.mkdir(parents=True, exist_ok=True)
    try:
        for name, digest in hashes.items():
            target = destination / name
            shutil.copyfile(source / name, target)
            if file_sha256(target) != digest:
                break
        else:
            return True
    except OSError:
        pass
    for name in hashes:
        (destination / name).unlink(missing_ok=True)
    return False


def _export_fold_source(fold, matrix, expected_hash, destination, cached=None) -> bool:
    from reporting.fold_explanations import export_fold_bundle

    if cached is not None and _restore_fold_export(fold, expected_hash, destination, cached):
        return True
    export_fold_bundle(fold, _fold_export_matrix(matrix, expected_hash), destination, numeric_only=True)
    return False


def _inputs(preprocessor):
    return [*preprocessor.numeric_columns_, *preprocessor.categorical_columns_]


def _fold_responses(loaded, train_inputs, names, *, quantiles=None, limit=2):
    responses, support = [], []
    quantiles = quantiles or {}
    for feature in names:
        observed = []
        for training in train_inputs:
            values = training[feature].to_numpy(dtype=float)
            observed.append(values[np.isfinite(values)])
        if not all(len(values) >= 2 and np.ptp(values) > 0 for values in observed):
            continue
        lower, upper = quantiles.get(feature, (.05, .95))
        lo, reference, hi = np.median([np.quantile(values, [lower, .5, upper]) for values in observed], axis=0)
        if not lo < reference < hi:
            continue
        for bundle, values in zip(loaded, observed, strict=True):
            pre, model, training, _, provenance = bundle
            term = model.term_names_.index(feature)
            index, = model.term_features_[term]
            cuts = np.asarray(model.bins_[index][0], dtype=float)
            x = np.unique(np.r_[lo, cuts[(cuts > lo) & (cuts < hi)], reference, hi])
            probes = np.zeros((len(x), model.n_features_in_))
            probes[:, index] = x
            native = model.eval_terms(probes)[:, term]
            scores = native - native[x == reference][0]
            context = {**provenance, "feature": feature, "reference": reference, "training_n": len(training),
                       "finite_training_n": len(values), "missing_training_n": len(training) - len(values)}
            responses.append(pd.DataFrame({**context, "value": x, "raw_log_odds_minus_reference": scores}))
            counts, edges = np.histogram(values, bins=np.linspace(lo, hi, 26))
            support.append(pd.DataFrame({**context, "bin_lower": edges[:-1], "bin_upper": edges[1:], "count": counts}))
        if len(responses) >= limit * len(loaded):
            break
    return responses, support


def _chosen_cases(frame):
    risk = frame[["caseid", "y_prob_calibrated"]].copy()
    for quantile in (.1, .9):
        target = float(risk.y_prob_calibrated.quantile(quantile))
        closest = risk.assign(distance=(risk.y_prob_calibrated - target).abs()).sort_values(["distance", "caseid"]).iloc[0]
        yield quantile, target, int(closest.caseid)


def _groups(preprocessor, transformed, model):
    categories = sorted(preprocessor.categorical_columns_, key=len, reverse=True)
    for indices in model.term_features_:
        index, = indices
        name = transformed.columns[index]
        channel = next((label for prefix, label in CHANNELS.items() if name.startswith(prefix)), None)
        yield channel or next((field for field in categories if name.startswith(f"{field}_")), name)


def _case_explanation(bundle, case_id, lookup):
    preprocessor, model, _, test, provenance = bundle
    fold = Path(provenance["model_bundle"])
    test = test.set_index("caseid").loc[case_id]
    data = preprocessor.transform(lookup.loc[[case_id], _inputs(preprocessor)])
    raw_probability = float(model.predict_proba(data)[0, 1])
    raw_intercept = float(np.ravel(model.intercept_)[0])
    terms = np.asarray(model.eval_terms(data))[0]
    raw_logit = raw_intercept + terms.sum()
    calibration_path = fold / "calibration.json"
    calibration = LogisticRecalibrationModel(**json.loads(calibration_path.read_text()))
    locked_risk = float(apply_logistic_recalibration(np.array([raw_probability]), calibration)[0])
    clipped_logit = logit(np.clip(raw_probability, calibration.eps, 1 - calibration.eps))
    correction = calibration.slope * (clipped_logit - raw_logit)
    base = calibration.intercept + calibration.slope * raw_intercept
    calibrated = calibration.slope * terms
    reconstructed = expit(base + calibrated.sum() + correction)
    if not (np.isclose(raw_probability, test.y_prob_raw, atol=1e-12, rtol=0)
            and np.isclose(expit(raw_logit), test.y_prob_raw, atol=1e-12, rtol=0)
            and np.isclose(locked_risk, test.y_prob_calibrated, atol=1e-12, rtol=0)
            and np.isclose(reconstructed, test.y_prob_calibrated, atol=1e-12, rtol=0)):
        raise ValueError(f"Unable to reproduce the saved held-out EEG prediction for {case_id}")
    term_table = pd.DataFrame({"feature": model.term_names_, "group": list(_groups(preprocessor, data, model)),
                               "raw_term": terms, "calibrated_log_odds_term": calibrated})
    grouped = term_table.groupby("group").calibrated_log_odds_term.sum().to_dict()
    channels = {label: grouped.pop(label, 0.0) for label in CHANNELS.values()}
    largest = sorted(grouped, key=lambda key: (-abs(grouped[key]), key))[:3]
    plotted = {key: grouped[key] for key in largest}
    plotted["Other clinical terms"] = sum(value for key, value in grouped.items() if key not in largest)
    plotted.update(channels)
    if abs(correction) > 1e-12:
        plotted["Probability clipping correction"] = correction
    common = {**provenance, "caseid": case_id, "raw_probability": raw_probability,
              "calibrated_probability": float(test.y_prob_calibrated), "observed_outcome": int(test.y_true),
              "calibration_sha256": file_sha256(calibration_path), "raw_intercept": raw_intercept,
              "calibrated_intercept": base, "calibration_clipping_correction": correction}
    return common, term_table, plotted


def explanation_sources(study, branch, *, manuscript_ranges=None):
    raw = _read_modeling_matrix(study["matrix_paths"][branch])
    lookup = raw.set_index("caseid")
    outcomes = [outcome for outcome in OUTCOMES if ("ebm", outcome, branch, "preop_and_all_five") in study["predictions"]]
    importance, responses, supports, ledger, terms, contributions = [], [], [], [], [], []
    manuscript_responses, manuscript_support = [], []
    for outcome in outcomes:
        ps = study["predictions"][("ebm", outcome, branch, "preop_and_all_five")]
        keys = sorted(set(zip(ps.df.repeat_id, ps.df.outer_fold_id)))
        loaded = [_fold_inputs(study, raw, ps, repeat, fold) for repeat, fold in keys]
        metadata = {"cohort_scope": ps.cohort_scope, "branch": branch, "outcome": outcome,
                    "model_type": "ebm", "feature_set": ps.feature_set}
        frames = [pd.DataFrame({**metadata, **provenance, "feature": model.term_names_,
                                "importance": model.term_importances()})
                  for _, model, _, _, provenance in loaded]
        feature_names = sorted(set().union(*(set(frame.feature) for frame in frames)))
        score_frames = []
        for frame, (_, _, _, _, provenance) in zip(frames, loaded, strict=True):
            values = frame.set_index("feature").importance.reindex(feature_names)
            score_frames.append(pd.DataFrame({**metadata, **provenance, "feature": feature_names,
                                              "importance": values.fillna(0).tolist(),
                                              "term_present": values.notna().tolist()}))
        scores = pd.concat(score_frames, ignore_index=True)
        means = scores.groupby("feature", as_index=False).importance.mean().sort_values(
            ["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
        means["rank"] = means.index + 1
        scores = scores.merge(means.rename(columns={"importance": "mean_importance"}), on="feature", validate="many_to_one")
        importance.append(scores)
        common_numerics = set.intersection(*(set(pre.numeric_columns_) for pre, *_ in loaded))
        names = [name for name in means.feature if name in common_numerics and all(name in model.term_names_ for _, model, *_ in loaded)]
        train_inputs = [pre.transform(lookup.loc[training.caseid, _inputs(pre)])
                        for pre, _, training, _, _ in loaded]
        curves, support = _fold_responses(loaded, train_inputs, names)
        responses.extend(frame.assign(**metadata) for frame in curves)
        supports.extend(frame.assign(**metadata) for frame in support)
        ranges = manuscript_ranges.get(outcome, {}) if manuscript_ranges is not None else {}
        if ranges:
            if not set(ranges).issubset(names):
                raise ValueError(f"Missing fitted numeric terms for {outcome}: {sorted(set(ranges).difference(names))}")
            curves, support = _fold_responses(loaded, train_inputs, ranges, quantiles=ranges, limit=len(ranges))
            if len(curves) != len(ranges) * len(keys):
                raise ValueError(f"Incomplete outer-training support for {outcome} manuscript responses")
            manuscript_responses.extend(frame.assign(**metadata) for frame in curves)
            manuscript_support.extend(frame.assign(**metadata) for frame in support)
        baseline = study["predictions"][("ebm", outcome, branch, "preop_only")].df.set_index("caseid")
        predictions = ps.df.set_index("caseid")
        for quantile, target, caseid in _chosen_cases(ps.df):
            row = predictions.loc[caseid]
            bundle = loaded[keys.index((row.repeat_id, row.outer_fold_id))]
            common, table, plotted = _case_explanation(bundle, caseid, lookup)
            common.update(**metadata, quantile=quantile, target_probability=target,
                          clinical_probability=float(baseline.loc[caseid, "y_prob_calibrated"]))
            ledger.append(common)
            terms.append(table.assign(**common))
            position = common["calibrated_intercept"]
            for group, value in plotted.items():
                record = {**common, "group": group, "contribution": value, "start": position, "end": position + value}
                if manuscript_ranges is not None:
                    record["raw_group_value"] = lookup.at[caseid, group] if group in lookup.columns else None
                contributions.append(record)
                position += value
    sources = {"importance": pd.concat(importance, ignore_index=True),
               "responses": pd.concat(responses, ignore_index=True) if responses else pd.DataFrame(),
               "support": pd.concat(supports, ignore_index=True) if supports else pd.DataFrame(),
               "case_ledger": pd.DataFrame(ledger), "case_terms": pd.concat(terms, ignore_index=True),
               "case_contributions": pd.DataFrame(contributions)}
    if manuscript_ranges is not None:
        sources["manuscript_responses"] = pd.concat(manuscript_responses, ignore_index=True)
        sources["manuscript_support"] = pd.concat(manuscript_support, ignore_index=True)
    return sources


def _label(feature):
    if feature.startswith("BIS_EEG1_WAV_"):
        descriptor = feature.removeprefix("BIS_EEG1_WAV_")
        label = load_display_dictionary().feature_label(feature, use_short=True, include_unit=True)
        return f"EEG · {descriptor}" if label == feature else label
    return load_display_dictionary().feature_label(feature, use_short=True, include_unit=True)


def importance_figure(sources):
    selected = sources["importance"][sources["importance"]["rank"].le(10)].copy()
    outcomes = selected.outcome.unique()
    fig, axes = plt.subplots(1, len(outcomes), figsize=(7.8, 4.6), squeeze=False, layout="constrained")
    for ax, outcome in zip(axes[0], outcomes, strict=True):
        rows = selected[selected.outcome.eq(outcome)].sort_values(["rank", "fold"])
        terms = rows.drop_duplicates("feature")
        ax.barh(terms["rank"], terms.mean_importance, color=style.color("mean"), height=.68)
        for row in rows.itertuples(index=False):
            ax.plot(row.importance, row.rank + (row.fold - 2) * .09, marker=style.FOLD_MARKERS[row.fold],
                    color=style.FOLD_COLORS[row.fold], ms=3,
                    markerfacecolor=style.FOLD_COLORS[row.fold] if row.term_present else "none")
        ax.set(yticks=terms["rank"], yticklabels=[_label(name) for name in terms.feature],
               title=OUTCOMES[outcome], xlabel="Native mean absolute term score (log-odds)")
        ax.invert_yaxis()
    return fig, selected


def response_figure(sources):
    responses, support = sources["responses"], sources["support"]
    outcomes = sources["importance"].outcome.unique()
    fig, axes = plt.subplots(len(outcomes), 2, figsize=(7.8, 3.0 * len(outcomes)), squeeze=False, layout="constrained")
    for i, outcome in enumerate(outcomes):
        selected = responses[responses.outcome.eq(outcome)] if len(responses) else pd.DataFrame()
        features = list(selected.feature.unique()) if len(selected) else []
        for j in range(2):
            ax = axes[i, j]
            if j >= len(features):
                ax.text(.5, .5, "No eligible term with continuous\nouter-training support", ha="center", va="center")
                ax.set_axis_off()
                continue
            feature = features[j]
            curves = selected[selected.feature.eq(feature)]
            for fold, rows in curves.groupby("fold"):
                ax.step(rows.value, rows.raw_log_odds_minus_reference, where="post", color=style.FOLD_COLORS[fold],
                        linestyle=style.FOLD_STYLES[fold], label=f"Fold {fold+1}")
            ax.axvline(curves.reference.iloc[0], color=style.color("reference"), ls=":", lw=.7)
            ax.axhline(0, color=style.color("reference"), lw=.6)
            h = support[support.outcome.eq(outcome) & support.feature.eq(feature) & support.fold.eq(0)]
            ax2 = ax.twinx()
            ax2.stairs(h["count"], np.r_[h.bin_lower, h.bin_upper.iloc[-1]], alpha=.4, color=style.color("support"))
            ax2.set_yticks([])
            ax.set(title=f"{OUTCOMES[outcome]} · {_label(feature)}", xlabel="Observed value",
                   ylabel="Raw log-odds vs reference" if j == 0 else "")
    return fig, responses


def heldout_figure(sources):
    selected = sources["case_contributions"]
    outcomes = selected.outcome.unique()
    fig, axes = plt.subplots(len(outcomes), 2, figsize=(7.8, 3.05 * len(outcomes)), squeeze=False, layout="constrained")
    for i, outcome in enumerate(outcomes):
        for j, quantile in enumerate((.1, .9)):
            ax = axes[i, j]
            rows = selected[selected.outcome.eq(outcome) & selected["quantile"].eq(quantile)]
            for y, row in enumerate(rows.itertuples(index=False)):
                ink = style.color("orange" if row.contribution < 0 else "blue")
                ax.barh(y, row.contribution, left=row.start, color=ink, height=.65)
            ax.set(yticks=range(len(rows)), yticklabels=[_label(g) for g in rows.group],
                   title=f"{OUTCOMES[outcome]} · {quantile:.0%} risk percentile\nCalibrated risk {rows.calibrated_probability.iloc[0]:.2%}",
                   xlabel="Calibrated log-odds")
            ax.invert_yaxis()
    return fig, selected
