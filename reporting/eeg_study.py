"""Rebuild the EEG feasibility displays from the isolated study's saved results."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import textwrap
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

from artifact_paths import enforce_storage_policy, get_artifact_root, get_processed_dir
from data_preparation.artifact_metadata import file_sha256
from data_preparation.cohort_scopes import selected_case_hash
from reporting import publication_style as style
from reporting.cohort_flow import CohortStage, _draw_cohort_flow_fallback
from reporting.display_dictionary import load_display_dictionary
from reporting.figure_rendering import save_report_figure_bundle
from reporting.manuscript_assets import write_pdf_sections
from reporting.manuscript_workspace.revise_publication import add_frame, document, heading, note
from reporting.manuscript_workspace.revision_data import BRANCHES, FAMILIES, IDENTITY, OUTCOMES
from reporting.manuscript_workspace.revision_tables import CHANNEL_SPECS, DESCRIPTORS
from reporting.preop_descriptives import (
    CATEGORICAL_COLS, CONTINUOUS_COLS, TABLE_FOOTNOTE, _build_descriptive_table,
    _restrict_to_roster, _summarize_continuous,
)
from results_recreation.comparisons import EEG_BASELINE_PAIRS, EEG_CORE, EEG_INCREMENT_PAIRS, load_comparison_artifacts
from results_recreation.metrics_summary import (
    _model_id, _prediction_content_hash, compute_point_metrics, crawl_predictions,
)
from results_recreation.results_analysis import PlotConfig, _calibration_bins

SCOPES = {"main_shared": ("Shared AKI/ICU", "eeg_main_shared_v1"),
          "icu_waveform_sensitivity": ("Broader ICU", "eeg_icu_waveform_sensitivity_v1")}
CHANNELS = ("pleth_only", "ecg_only", "co2_only", "awp_only", "eeg_only", *EEG_CORE[1:3])
LABELS = {
    "preop_only": "Preoperative only", "original_four_waveforms": "Original four waveforms",
    "all_five_waveforms": "All five waveforms", "preop_and_original_four": "Preoperative + original four",
    "preop_and_all_five": "Preoperative + all five", "pleth_only": "PPG", "ecg_only": "ECG",
    "co2_only": "CO₂", "awp_only": "Airway pressure", "eeg_only": "EEG",
}
INPUT_STYLES = dict(zip(EEG_CORE, (('blue', '-'), ('orange', ':'), ('orange', '-'),
                                  ('green', '--'), ('green', '-')), strict=True))
METRIC_LABELS = {"auprc": "AP", "auroc": "AUROC", "brier": "Brier score"}
MAIN = ("Figure 1", "Table 1", "Table 2", "Figure 2", "Figure 3", "Figure 4", "Figure 5")
SUPPLEMENT = ("Table S1", "Table S2", *(f"Figure S{i}" for i in range(3, 11)))
TITLES = dict(zip((*MAIN, *SUPPLEMENT), (
    "Study cohort selection", "Shared cohort characteristics", "Performance by input configuration",
    "Incremental discrimination", "Whole-operation ROC curves", "Global EBM term importance",
    "EBM feature-response functions", "Predictors and waveform feature extraction",
    "Shared and broader ICU cohorts", "Whole-operation precision–recall curves",
    "Whole-operation calibration", "Average precision by waveform input",
    "AUROC by waveform input", "Added discrimination for ICU admission by cohort",
    "Decision curves", "Paired net benefit differences", "Held-out EBM prediction explanations",
), strict=True))
CI_NOTE = "95% patient-cluster intervals are conditional on the saved outer-fold fits."
CAPTIONS = {
    "Figure 1": "First exclusions are counted once. Patients may contribute more than one operation.",
    "Figure 2": f"Candidate minus reference discrimination. P, preoperative. W4/W5, four/five waveforms. {CI_NOTE}",
    "Figure 3": "Pooled calibrated outer-test predictions. The reference line indicates chance discrimination.",
    "Figure 4": "Ten leading terms per outcome by mean native importance across five outer folds. A hollow marker indicates a term absent from that fitted fold.",
    "Figure 5": "Native raw log-odds responses relative to a training-derived reference. Distributions and reference values use outer-training operations.",
    "Figure S3": "Pooled calibrated outer-test predictions. The horizontal line shows operation-level outcome prevalence.",
    "Figure S4": "Binned held-out reliability and the distribution of calibrated outer-test risk. A point summarizes a nonempty risk bin.",
    "Figure S5": f"Waveform-only models. AP denotes average precision. {CI_NOTE}",
    "Figure S6": f"Waveform-only models. {CI_NOTE}",
    "Figure S7": f"Within-cohort contrasts. P, preoperative. W4/W5, four/five waveforms. {CI_NOTE}",
    "Figure S8": "Exploratory decision curves from calibrated outer-test risk at prespecified thresholds.",
    "Figure S9": "Candidate minus reference net benefit. Pointwise 95% patient-cluster intervals conditional on saved fits.",
    "Figure S10": "Cases nearest the empirical 10th and 90th percentiles of combined-model outer-test risk within outcome. Case ID resolves ties. Contributions are on the calibrated log-odds scale.",
}


def _within(path: Path, root: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"EEG report path is outside its study: {path}")
    return path


def _input_style(feature):
    color_name, line = INPUT_STYLES[feature]
    return {"color": style.color(color_name), "linestyle": line, "label": LABELS[feature]}


def _scope_inputs(artifact_root, processed, scope):
    from model_creation.utils import EEG_FEATURE_SETS

    root = _within(artifact_root / "results/catch22/experiments" / scope / "nested_cv", artifact_root)
    manifest_path = _within(root / "run_manifest.json", root)
    manifest = json.loads(manifest_path.read_text())
    cohort_id = SCOPES[scope][1]
    if (manifest.get("status"), manifest.get("diagnostic"), manifest.get("study_profile"),
            manifest.get("cohort_scope"), manifest.get("manifest_cohort_id")) != ("validated", False, "eeg", scope, cohort_id):
        raise ValueError(f"Expected a validated definitive EEG manifest for {scope}: {manifest_path}")
    outcomes = OUTCOMES if scope == "main_shared" else ("icu_admission",)
    expected = {(model, outcome, branch, feature) for model in FAMILIES for outcome in outcomes
                for branch in BRANCHES for feature in EEG_FEATURE_SETS}
    configs = manifest["configurations"]
    keys = [(c["model_type"], c["outcome"], c["branch"], c["feature_set"]) for c in configs]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"EEG report requires the exact definitive grid for {scope}")
    rosters, matrix_paths = manifest["branch_rosters"], {}
    for branch in BRANCHES:
        roster = rosters[branch]
        path = _within(Path(roster["path"]), root)
        case_ids = pd.read_csv(path)
        if selected_case_hash(case_ids) != roster["selected_case_hash"]:
            raise ValueError(f"EEG roster content differs from the manifest: {path}")
        suffix = "_windowed" if branch == "windowed" else ""
        matrix = _within(processed / f"aki_features_master_wide{suffix}.csv", processed)
        if file_sha256(matrix) != roster["dataset_sha256"]:
            raise ValueError(f"EEG report modeling dataset hash mismatch: {matrix}")
        matrix_paths[branch] = matrix
    if len({r["selected_case_hash"] for r in rosters.values()}) != 1:
        raise ValueError(f"EEG branches must use the same roster: {scope}")
    for path in root.glob("models/**/predictions/test.csv"):
        _within(path, root)
    for config in configs:
        if config.get("artifact_path"):
            _within(Path(config["artifact_path"]), root)
    sets = crawl_predictions(root, strict=True, run_manifest=manifest_path, require_reference=True)
    tables = root / "tables"
    summary = pd.read_csv(_within(tables / "metrics_summary.csv", root))
    if len(summary) != len(expected):
        raise ValueError(f"EEG summary row count differs from the requested grid: {scope}")
    for name in ("comparison_inventory.json", "paired_comparisons.csv", "decision_curve_summary.csv",
                 "decision_curve_comparisons.csv", "delong_comparisons.csv"):
        _within(tables / name, root)
    artifacts = load_comparison_artifacts(tables, sets, summary, run_manifest=manifest)
    if artifacts["inventory"]["comparison_status"] != "complete" or (
        artifacts["inventory"]["bootstrap_replicates"], artifacts["inventory"]["bootstrap_seed"],
        artifacts["inventory"]["bootstrap_stratified"],
    ) != (5000, 0, False):
        raise ValueError(f"Expected complete seed-0, 5,000-draw EEG comparisons: {scope}")
    summary = summary.assign(model_type=summary.model_name.map(_model_id))
    summary_by_key = summary.set_index(["model_type", "outcome", "branch", "feature_set"], verify_integrity=True)
    predictions, baseline_identity = {}, {}
    for ps in sets:
        key = (_model_id(ps.model_name), ps.outcome, ps.branch, ps.feature_set)
        frame = ps.df.sort_values("caseid").reset_index(drop=True)
        identity = baseline_identity.setdefault(ps.outcome, frame[IDENTITY])
        if not frame[IDENTITY].equals(identity):
            raise ValueError(f"EEG modeling cases, outcomes or folds differ in {ps.path}")
        row = summary_by_key.loc[key]
        if row.prediction_content_hash != _prediction_content_hash(ps) or any(
            not np.isclose(row[metric], value, rtol=0, atol=1e-12, equal_nan=True)
            for metric, value in compute_point_metrics(ps).items()
        ):
            raise ValueError(f"Stale EEG point metrics or predictions at {ps.path}")
        predictions[key] = ps
    return {"root": root, "manifest_path": manifest_path, "manifest": manifest,
            "tables": tables, "metrics": summary, "comparisons": artifacts,
            "predictions": predictions, "matrix_paths": matrix_paths}


def _attrition_sources(path, studies):
    attrition = json.loads(path.read_text())
    audit = _within(path.with_name("eeg_operation_attrition.csv"), path.parent)
    if attrition.get("study_profile") != "eeg" or file_sha256(audit) != attrition["case_audit_sha256"]:
        raise ValueError("EEG attrition evidence changed after roster freeze")
    stages = attrition["stages"]
    previous = None
    for stage in stages:
        n, p = stage["operations"], stage["patients"]
        if not 0 <= p <= n or (previous and (n > previous[0] or p > previous[1])):
            raise ValueError(f"Invalid EEG attrition order: {stage}")
        previous = n, p
    for scope, study in studies.items():
        expected = attrition["scopes"][scope]
        roster = study["manifest"]["branch_rosters"]["non_windowed"]
        if (expected.get("cohort_manifest_identity") != SCOPES[scope][1]
                or any(expected[key] != roster[key] for key in ("operations", "patients"))):
            raise ValueError(f"EEG attrition counts disagree with {scope} roster")
        for outcome in {key[1] for key in study["predictions"]}:
            frame = study["predictions"][("ebm", outcome, "non_windowed", "preop_only")].df
            events = frame[frame.y_true.eq(1)]
            reported = expected["outcomes"][outcome]
            if (len(frame), len(events), events.subjectid.nunique()) != (
                reported["eligible_operations"], reported["positive_operations"], reported["positive_patients"],
            ):
                raise ValueError(f"EEG outcome denominators disagree with saved {scope}/{outcome} predictions")
    return attrition


def _save_figure(fig, out, label, source):
    stem = label.replace(" ", "_")
    if label in CAPTIONS:
        save_report_figure_bundle(fig, out / stem, formats=("png",), close=False)
        fig.suptitle(f"{label}. {TITLES[label]}", fontweight="bold", fontsize=11)
        fig.supxlabel(textwrap.fill(CAPTIONS[label], width=110), fontsize=8, color=style.color("reference"))
        save_report_figure_bundle(fig, out / stem, formats=("pdf",), close=True)
    else:
        if fig.get_label():
            fig.suptitle(fig.get_label(), fontweight="bold", fontsize=11)
        save_report_figure_bundle(fig, out / stem, formats=("png", "pdf"), close=True)
    source.to_csv(out / f"{stem}.plotted.csv", index=False, float_format="%.17g")


def _flow(attrition):
    values, stages = [], []
    previous = None
    reasons = {"clinical": "Clinical eligibility", "original_waveform_track": "A required original track is unavailable",
               "eeg_track": "Neither raw EEG track is listed", "eeg_source_or_qc": "EEG load, coverage or cleaning failed",
               "eeg_feature_extraction": "EEG feature extraction failed",
               "original_modality_extraction": "An original waveform failed one or both branches"}
    for stage in attrition["stages"]:
        count, patients = stage["operations"], stage["patients"]
        stages.append(CohortStage(title=stage["title"], count=count, n_patients=patients,
                                  removed=None if previous is None else previous - count,
                                  removal_reason=reasons.get(stage.get("first_exclusion"))))
        values.append({"stage": stage["key"], "operations": count, "patients": patients})
        previous = count
    fig = _draw_cohort_flow_fallback(stages, None, None, publication=True)
    ax = fig.axes[0]
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height + 1.5)
    ax.set_position([0, 1.5 / (height + 1.5), 1, height / (height + 1.5)])
    for index, (scope, counts) in enumerate(attrition["scopes"].items()):
        if scope not in SCOPES:
            continue
        outcomes = []
        for outcome, record in counts["outcomes"].items():
            outcomes.append(f"{OUTCOMES[outcome]} {record['positive_operations']:,}")
            values.append({"stage": scope, "outcome": outcome, "operations": counts["operations"],
                           "patients": counts["patients"], **record})
        fig.text(.04, (.99 - .32 * index) / (height + 1.5),
                 f"{SCOPES[scope][0]} · {counts['operations']:,} operations, {counts['patients']:,} patients · "
                 + ", ".join(outcomes), fontsize=8)
    return fig, pd.DataFrame(values)


def _descriptives(studies, processed, attrition):
    source = pd.read_csv(_within(processed / "aki_pleth_ecg_co2_awp.csv", processed))
    derived = pd.read_csv(_within(processed / "aki_preop_processed.csv", processed))
    cohorts = {}
    for scope, study in studies.items():
        roster = Path(study["manifest"]["branch_rosters"]["non_windowed"]["path"])
        cohort = _restrict_to_roster(source, roster)
        additions = _restrict_to_roster(derived, roster)
        added = [col for col in (*CONTINUOUS_COLS, "inpatient_preop") if col in additions and col not in cohort]
        cohorts[scope] = cohort.merge(additions[["caseid", *added]], on="caseid", validate="one_to_one", how="left")
    shared, broader = (cohorts[scope] for scope in SCOPES)
    if not set(shared.caseid).issubset(broader.caseid):
        raise ValueError("The broader ICU roster must include the shared EEG roster")
    table = _build_descriptive_table(
        shared, None, continuous_features=[col for col in CONTINUOUS_COLS if col in shared],
        categorical_features=[col for col in (*CATEGORICAL_COLS, "inpatient_preop") if col in shared],
        display_dictionary=load_display_dictionary(), alpha=.05, max_sample=5000, random_state=42,
    ).drop(columns="Type")
    table.columns = ["Characteristic", "Finding", "Missing n (%)"]
    rows = []
    for scope, cohort in cohorts.items():
        data = attrition["scopes"][scope]
        row = {"Cohort": SCOPES[scope][0], "Operations": data["operations"], "Patients": data["patients"]}
        for outcome in OUTCOMES:
            counts = data["outcomes"].get(outcome)
            row[OUTCOMES[outcome] + " positive operations"] = counts["positive_operations"] if counts else "—"
            row[OUTCOMES[outcome] + " positive patients"] = counts["positive_patients"] if counts else "—"
            row[OUTCOMES[outcome] + " prevalence"] = f"{counts['positive_operations'] / data['operations']:.1%}" if counts else "—"
        row.update(_cohort_characteristics(cohort))
        rows.append(row)
    return table, pd.DataFrame(rows).set_index("Cohort").T.rename_axis("Characteristic").reset_index()


def _cohort_characteristics(frame):
    duration = (pd.to_numeric(frame.opend, errors="coerce") - pd.to_numeric(frame.opstart, errors="coerce")) / 60
    numeric = {"Age, years": frame.age, "BMI, kg/m²": frame.bmi, "Operative duration, min": duration,
               "Baseline creatinine, mg/dL": frame.preop_cr}
    categories = {
        "Male sex": frame.sex.eq("M"), "Emergency surgery": frame.emop.eq(1),
        "ASA-PS III–IV": frame.asa.isin([3, 4]), "Open surgical approach": frame.approach.eq("Open"),
        "Colorectal procedure": frame.optype.eq("Colorectal"), "Transplantation": frame.optype.eq("Transplantation"),
        "Hypertension": frame.preop_htn.eq(1), "Diabetes": frame.preop_dm.eq(1),
        "Baseline creatinine absent or invalid": ~frame.aki_has_baseline_cr.eq(1),
        "Baseline creatinine >4 mg/dL": pd.to_numeric(frame.preop_cr, errors="coerce").gt(4),
        "No finite positive postoperative creatinine within 72 h": ~frame.aki_has_postop_cr_72h.eq(1),
        "AKI ascertainment unavailable": ~frame.eligible_any_aki.eq(1),
        "C-reactive protein unavailable": ~np.isfinite(pd.to_numeric(frame.preop_crp, errors="coerce")),
    }
    return {**{key: _summarize_continuous(values)[0] for key, values in numeric.items()},
            **{key: f"{values.sum():,} ({values.sum() / len(frame):.1%})" for key, values in categories.items()}}


def _performance(metrics):
    rows = []
    for outcome in OUTCOMES:
        for family in FAMILIES:
            for branch in BRANCHES:
                for feature in EEG_CORE:
                    rows.extend(metrics.loc[metrics.outcome.eq(outcome) & metrics.model_type.eq(family)
                                           & metrics.branch.eq(branch) & metrics.feature_set.eq(feature)].to_dict("records"))
    source = pd.DataFrame(rows)
    output = pd.DataFrame({"Outcome": source.outcome.map(OUTCOMES),
                           "Model": source.model_type.map({k: v[0] for k, v in FAMILIES.items()}),
                           "Branch": source.branch.map(BRANCHES), "Input": source.feature_set.map(LABELS)})
    for metric, label in METRIC_LABELS.items():
        output[label] = [f"{value:.3f} ({lower:.3f}–{upper:.3f})" for value, lower, upper in
                         source[[metric, f"{metric}_lower", f"{metric}_upper"]].itertuples(index=False, name=None)]
    return output, source


def _descriptor_table(extractor_version: str) -> pd.DataFrame:
    if extractor_version == "0.4.5":
        descriptors = pd.DataFrame(DESCRIPTORS.items(), columns=["Catch24 descriptor", "Description"])
    else:
        import pycatch22

        installed = version("pycatch22")
        if installed != extractor_version or not hasattr(pycatch22, "feature_table"):
            raise ValueError(
                f"EEG descriptor definitions require the frozen pycatch22 {extractor_version} build. "
                f"Installed version is {installed}."
            )
        table = pycatch22.feature_table(catch24=True)
        descriptors = pd.DataFrame(
            ((row["name"], row["hctsa_call"]) for row in table),
            columns=["Catch24 descriptor", "hctsa reference call"],
        )
    if len(descriptors) != 24 or not descriptors["Catch24 descriptor"].is_unique:
        raise ValueError(f"Invalid Catch24 definitions for pycatch22 {extractor_version}.")
    descriptors.attrs["pycatch22_version"] = extractor_version
    return descriptors


def _predictor_tables(study, *, extractor_version: str):
    ps = study["predictions"][("ebm", "any_aki", "non_windowed", "preop_only")]
    import joblib
    fold_dir = ps.path.parent.parent / "artifacts/folds/repeat_00/outer_00"
    pre = joblib.load(_within(fold_dir / "preprocessor.pkl", study["root"]))
    labels = load_display_dictionary()
    clinical = []
    for keys, kind in ((pre.categorical_columns_, "Categorical"), (pre.numeric_columns_, "Numeric")):
        for key in sorted(keys):
            clinical.append({"Clinical predictor": labels.feature_label(key, include_unit=True), "Input": key, "Encoding": kind})
    channels = [(key, *value) for key, value in CHANNEL_SPECS.items()]
    channels.append(("BIS_EEG1_WAV", "Frontal EEG (EEG)", "BIS/EEG1_WAV with BIS/EEG2_WAV load fallback",
                     "128 Hz", "0.5–50 Hz band-pass"))
    waveform = pd.DataFrame(channels, columns=["Prefix", "Waveform", "Source", "Native rate", "Filter"])
    waveform["Whole-operation rate"] = ["10 Hz"] * 4 + ["128 Hz"]
    waveform["Windowed rate"] = ["100 Hz", "100 Hz", "62.5 Hz", "62.5 Hz", "128 Hz"]
    return (pd.DataFrame(clinical), waveform, _descriptor_table(extractor_version))


def _axes(rows, cols=3, *, height=2.15):
    return plt.subplots(rows, cols, squeeze=False, figsize=(7.8, height * rows + .55), layout="constrained")


def _core_legend(fig, ax, columns=2):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=columns, loc="outside lower center", fontsize=7)


def _curve_tables(study):
    metrics = study["metrics"].set_index(["model_type", "outcome", "branch", "feature_set"])
    sources = {key: [] for key in ("roc", "pr", "calibration", "risk_support")}
    for (family, outcome, branch, feature), ps in study["predictions"].items():
        frame = ps.df
        row = metrics.loc[(family, outcome, branch, feature)]
        metadata = {"cohort_scope": ps.cohort_scope, "outcome": outcome,
                    "branch": branch, "model_type": family, "feature_set": feature,
                    "prediction_content_hash": row.prediction_content_hash, "operations": len(frame),
                    "patients": frame.subjectid.nunique()}
        risk, truth = frame.y_prob_calibrated, frame.y_true
        fpr, tpr, roc_thresholds = roc_curve(truth, risk)
        precision, recall, pr_thresholds = precision_recall_curve(truth, risk)
        sources["roc"].append(pd.DataFrame({**metadata, "x": fpr, "y": tpr,
                                            "threshold": roc_thresholds, "score": row.auroc}))
        sources["pr"].append(pd.DataFrame({**metadata, "x": recall[::-1], "y": precision[::-1],
                                           "threshold": np.r_[pr_thresholds, np.nan][::-1], "score": row.auprc,
                                           "prevalence": float(truth.mean())}))
        x, y, n = _calibration_bins(truth, risk, PlotConfig())
        sources["calibration"].append(pd.DataFrame({**metadata, "x": x, "y": y,
                                                    "bin_n": n, "brier": row.brier}))
        counts, edges = np.histogram(risk, bins=np.linspace(0, 1, 41))
        sources["risk_support"].append(pd.DataFrame({**metadata, "bin_lower": edges[:-1],
                                                     "bin_upper": edges[1:], "bin_n": counts}))
    return {key: pd.concat(frames, ignore_index=True) for key, frames in sources.items()}


def _discrimination(curves, branch, kind):
    selected = curves[kind].loc[lambda df: df.branch.eq(branch) & df.feature_set.isin(EEG_CORE)].copy()
    support = curves["risk_support"].loc[lambda df: df.branch.eq(branch) & df.feature_set.isin(EEG_CORE)]
    outcomes = [outcome for outcome in OUTCOMES if outcome in set(selected.outcome)]
    fig, axes = _axes(len(outcomes), height=3.2 if kind == "calibration" else 2.5)
    for i, outcome in enumerate(outcomes):
        for j, (family, (short, _)) in enumerate(FAMILIES.items()):
            ax = axes[i, j]
            frame = selected[selected.outcome.eq(outcome) & selected.model_type.eq(family)]
            for feature in EEG_CORE:
                rows = frame[frame.feature_set.eq(feature)]
                line = _input_style(feature)
                if kind == "calibration":
                    ax.plot(rows.x, rows.y, marker="o", markersize=2.2, **line)
                    histogram = support[support.outcome.eq(outcome) & support.model_type.eq(family)
                                        & support.feature_set.eq(feature)]
                    inset = ax.inset_axes([.56, .06, .42, .28]) if feature == EEG_CORE[0] else inset
                    inset.stairs(histogram.bin_n, np.r_[histogram.bin_lower, histogram.bin_upper.iloc[-1]], **line)
                    inset.set(xticks=[0, 1], yticks=[], title="Risk distribution")
                    inset.tick_params(axis="x", labelsize=6)
                    inset.title.set_fontsize(6)
                else:
                    ax.plot(rows.x, rows.y, **line)
            if kind == "pr":
                ax.axhline(frame.prevalence.iloc[0], color=style.color("reference"), ls="--", lw=.65)
                xlabel, ylabel = "Recall", "Precision"
            elif kind == "roc":
                ax.plot([0, 1], [0, 1], color=style.color("reference"), ls="--", lw=.65)
                xlabel, ylabel = "False positive rate", "True positive rate"
            else:
                ax.plot([0, 1], [0, 1], color=style.color("reference"), ls="--", lw=.65)
                xlabel, ylabel = "Predicted risk", "Observed risk"
            ax.set(title=f"{OUTCOMES[outcome]} · {short}", xlim=(0, 1), ylim=(0, 1),
                   xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel=xlabel, ylabel=ylabel if j == 0 else "")
    _core_legend(fig, axes[0, 0])
    return fig, selected


def _contrast(studies, *, icu_only):
    pairs = (*EEG_BASELINE_PAIRS, *EEG_INCREMENT_PAIRS)
    pairs_label = ("P + W4 − P", "P + W5 − P", "(P + W5) − (P + W4)", "W5 − W4")
    frames = []
    for study in studies.values():
        table = study["comparisons"]["paired"]
        table = table[table.reference_model.eq(table.candidate_model) & table.metric.isin(["auprc", "auroc"])]
        if icu_only:
            table = table[table.outcome.eq("icu_admission")]
        for index, ((reference, candidate), label) in enumerate(zip(pairs, pairs_label, strict=True)):
            frames.append(table[table.reference_feature_set.eq(reference) & table.candidate_feature_set.eq(candidate)]
                          .assign(contrast=label, contrast_order=index))
    data = pd.concat(frames, ignore_index=True)
    row_key = "cohort_scope" if icu_only else "outcome"
    groups = list(data.groupby(row_key, sort=False))
    fig, axes = _axes(len(groups), 2, height=3.1)
    for i, (key, group) in enumerate(groups):
        for j, metric in enumerate(("auprc", "auroc")):
            ax = axes[i, j]
            for row in group[group.metric.eq(metric)].itertuples():
                family_offset = (list(FAMILIES).index(row.candidate_model) - 1) * .2
                branch_offset = -.045 if row.branch == "non_windowed" else .045
                y = row.contrast_order + family_offset + branch_offset
                ink = style.FAMILY_COLORS[row.candidate_model]
                marker = "o" if row.branch == "non_windowed" else "s"
                ax.hlines(y, row.lower, row.upper, color=ink, linewidth=.75)
                ax.plot(row.delta, y, marker=marker, ms=3, color=ink, mfc=ink if marker == "o" else "white")
            ax.axvline(0, color=style.color("reference"), ls="--", lw=.65)
            ax.set(yticks=range(len(pairs)), yticklabels=pairs_label, ylim=(len(pairs) - .6, -.4),
                   title=f"{SCOPES[key][0] if icu_only else OUTCOMES[key]} · Δ{METRIC_LABELS[metric]}")
    proxies = [plt.Line2D([], [], color=style.FAMILY_COLORS[family], label=name[0]) for family, name in FAMILIES.items()]
    proxies.extend([plt.Line2D([], [], color="#222222", ls="none", marker=marker, label=name)
                    for marker, name in (("o", "Whole-operation"), ("s", "Windowed"))])
    fig.legend(handles=proxies, loc="outside lower center", ncol=5, fontsize=7)
    return fig, data


def _channels(studies, metric):
    data = pd.concat([study["metrics"] for study in studies.values()], ignore_index=True)
    selected = data[data.feature_set.isin(CHANNELS)].copy()
    groups = list(selected.groupby(["cohort_scope", "outcome"], sort=False))
    fig, axes = _axes(len(groups), len(BRANCHES), height=2.35)
    for i, ((scope, outcome), frame) in enumerate(groups):
        for j, (branch, branch_name) in enumerate(BRANCHES.items()):
            ax = axes[i, j]
            for row in frame[frame.branch.eq(branch)].itertuples():
                y = CHANNELS.index(row.feature_set) + (list(FAMILIES).index(row.model_type) - 1) * .22
                ink = style.FAMILY_COLORS[row.model_type]
                ax.hlines(y, getattr(row, f"{metric}_lower"), getattr(row, f"{metric}_upper"), color=ink, linewidth=.7)
                ax.plot(getattr(row, metric), y, marker="o", ms=3, color=ink)
            ax.set(yticks=range(len(CHANNELS)), yticklabels=[LABELS[key] for key in CHANNELS],
                   ylim=(len(CHANNELS) - .6, -.4), title=f"{SCOPES[scope][0]} · {OUTCOMES[outcome]}\n{branch_name}")
    fig.legend(handles=[plt.Line2D([], [], color=style.FAMILY_COLORS[family], label=name[0])
                        for family, name in FAMILIES.items()], loc="outside lower center", ncol=3, fontsize=7)
    return fig, selected


def _dca(study, branch, *, paired):
    key = "dca_comparisons" if paired else "dca"
    table = study["comparisons"][key]
    table = table[table.branch.eq(branch)]
    if paired:
        pairs = (EEG_BASELINE_PAIRS[-1], EEG_INCREMENT_PAIRS[0])
        frames = []
        for (ref, candidate), text in zip(pairs, ("P + W5 − P", "(P + W5) − (P + W4)"), strict=True):
            frames.append(table[table.reference_model.eq(table.candidate_model) & table.reference_feature_set.eq(ref)
                                & table.candidate_feature_set.eq(candidate)].assign(contrast=text))
        table = pd.concat(frames, ignore_index=True)
    else:
        table = table[table.feature_set.isin(EEG_CORE)]
    outcomes = [outcome for outcome in OUTCOMES if outcome in set(table.outcome)]
    fig, axes = _axes(len(outcomes), height=2.55)
    for i, outcome in enumerate(outcomes):
        for j, (family, (name, _)) in enumerate(FAMILIES.items()):
            ax = axes[i, j]
            model_key = "candidate_model" if paired else "model_type"
            selected = table[table.outcome.eq(outcome) & table[model_key].eq(family)]
            if paired:
                for k, (contrast, rows) in enumerate(selected.groupby("contrast", sort=False)):
                    ink = style.color(("blue", "green")[k])
                    rows = rows.sort_values("threshold_probability")
                    ax.plot(rows.threshold_probability, rows.delta, color=ink, label=contrast)
                    ax.fill_between(rows.threshold_probability, rows.lower, rows.upper, color=ink, alpha=.16)
                ax.axhline(0, color=style.color("reference"), lw=.75)
            else:
                for feature in EEG_CORE:
                    rows = selected[selected.feature_set.eq(feature)].sort_values("threshold_probability")
                    ax.plot(rows.threshold_probability, rows.model, **_input_style(feature))
                ax.plot(rows.threshold_probability, rows.treat_all, color=style.color("reference"), ls="--", label="Treat all")
                ax.axhline(0, color=style.color("ink"), lw=.75, label="Treat none")
            ax.set(title=f"{OUTCOMES[outcome]} · {name}", xlabel="Threshold probability",
                   ylabel=("Δ net benefit" if paired else "Net benefit") if j == 0 else "")
    _core_legend(fig, axes[0, 0])
    return fig, table


def _ablation(studies):
    data = pd.concat([study["comparisons"]["paired"] for study in studies.values()], ignore_index=True)
    include = data.comparison_type.isin(["ablation", "ablation_parent"]) | (
        data.reference_feature_set.eq(EEG_INCREMENT_PAIRS[0][0]) & data.candidate_feature_set.eq(EEG_INCREMENT_PAIRS[0][1]))
    selected = data[include].copy()
    selected["ablation_parent"] = np.where(selected.candidate_feature_set.str.contains("four")
                                             | selected.reference_feature_set.str.contains("four_minus"), "W4", "W5")
    return selected


def _ablation_figure(ablations, *, scope):
    source = ablations[ablations.cohort_scope.eq(scope) & ablations.metric.eq("auprc")
                       & ablations.comparison_type.isin(["ablation_parent", "eeg_increment"])].copy()
    source["dropped"] = np.where(source.comparison_type.eq("eeg_increment"), "eeg",
                                  source.reference_feature_set.str.rsplit("_", n=1).str[-1])
    groups = list(source.groupby(["outcome", "branch"], sort=False))
    fig, axes = _axes(len(groups), 2, height=2.1)
    channels = ("pleth", "ecg", "co2", "awp", "eeg")
    for i, ((outcome, branch), group) in enumerate(groups):
        for j, parent in enumerate(("W4", "W5")):
            ax = axes[i, j]
            for row in group[group.ablation_parent.eq(parent)].itertuples():
                x = channels.index(row.dropped)
                y = x + (list(FAMILIES).index(row.candidate_model) - 1) * .2
                ink = style.FAMILY_COLORS[row.candidate_model]
                ax.hlines(y, row.lower, row.upper, color=ink, linewidth=.75)
                ax.plot(row.delta, y, marker="o", ms=3, color=ink)
            names = [LABELS[f"{channel}_only"] for channel in (channels[:4] if parent == "W4" else channels)]
            ax.set(yticks=range(len(names)), yticklabels=names, ylim=(len(names) - .6, -.4),
                   title=f"{OUTCOMES[outcome]} · {BRANCHES[branch]} · {parent}", xlabel="Complete minus ablated AP")
            ax.axvline(0, color=style.color("reference"), ls="--", lw=.65)
    fig.legend(handles=[plt.Line2D([], [], color=style.FAMILY_COLORS[family], label=name[0])
                        for family, name in FAMILIES.items()], loc="outside lower center", ncol=3, fontsize=7)
    return fig, source


def _table_sections(label, frames):
    title = f"{label}. {TITLES[label]}"
    headings = {
        "Table S1": ("Clinical inputs", "Waveform inputs", "Catch24 descriptors"),
    }.get(label, (title,) * len(frames))
    notes = {
        "Table 1": TABLE_FOOTNOTE, "Table 2": CI_NOTE,
        "Table S1": "All-five inputs yield 120 whole-operation or 480 windowed waveform features. "
        "EEG masks missing samples, unknown or low SQI, and raw amplitudes outside ±100 µV over the full operative grid. "
        "The union may cover at most 5% of samples. SQI must be at least 50, using a bounded prior observation. "
        "Linear interpolation and nearest endpoint fill precede forward/backward SOS filtering (design N=4). "
        "A 10-second EEG window advances by 5 seconds. "
        "Each descriptor is summarized by mean, sample SD, minimum and maximum.",
        "Table S2": "The cohorts overlap. Positive patients had at least one positive operation.",
    }
    if label == "Table S1" and (extractor_version := frames[-1].attrs.get("pycatch22_version")):
        notes[label] = f"Waveform descriptors use pycatch22 {extractor_version}. {notes[label]}"
    return [{"heading": heading, "dataframe": frame, "notes": notes[label] if i == len(frames) - 1 else None}
            for i, (heading, frame) in enumerate(zip(headings, frames, strict=True))]


def _books(out, tables):
    pdfunite = shutil.which("pdfunite")
    if not pdfunite:
        raise RuntimeError("Poppler pdfunite is required to assemble the EEG vector PDF books")
    widths = {"Table 1": ((3.3, 2.2, 1.55),), "Table 2": ((.6, .6, 1.15, 1.6, 1.05, 1.05, 1.0),),
              "Table S1": ((2.9, 2.7, 1.45), (1.0, 1.1, 1.55, .8, .85, .9, .85), (3.3, 3.75)),
              "Table S2": ((4.05, 1.5, 1.5),)}
    books = {}
    for inventory, stem in ((MAIN, "Main_Figures_and_Tables"), (SUPPLEMENT, "Supplementary_Figures_and_Tables")):
        doc = document(stem.replace("_", " "))
        pdfs, image_labels, table_count = [], [], 0
        for index, label in enumerate(inventory):
            title = f"{label}. {TITLES[label]}"
            heading(doc, title, page_break=index > 0)
            file_stem = label.replace(" ", "_")
            if label in tables:
                sections = _table_sections(label, tables[label])
                for section, column_widths in zip(sections, widths[label], strict=True):
                    if section["heading"] != title:
                        heading(doc, section["heading"])
                    add_frame(doc, section["dataframe"], column_widths)
                    table_count += 1
                    if section.get("notes"):
                        note(doc, section["notes"])
                path = out / f"{file_stem}.pdf"
                write_pdf_sections(path, title=title, sections=sections)
                pdfs.append(path)
            else:
                p = doc.add_paragraph()
                p.paragraph_format.keep_with_next = True
                image_path = out / f"{file_stem}.png"
                with Image.open(image_path) as raster:
                    placed_width = min(6.4, .999 * raster.width / style.PNG_DPI)
                image = p.add_run().add_picture(str(image_path), width=Inches(placed_width))
                if image.height.inches > 8.7:
                    ratio = 8.7 / image.height.inches
                    image.width, image.height = round(image.width * ratio), round(image.height * ratio)
                note(doc, CAPTIONS[label])
                image_labels.append(label)
                pdfs.append(out / f"{file_stem}.pdf")
        doc_path, pdf_path = out / f"{stem}.docx", out / f"{stem}.pdf"
        doc.save(doc_path)
        subprocess.run([pdfunite, *map(str, pdfs), str(pdf_path)], check=True)
        check = Document(doc_path)
        titles = [p.text for p in check.paragraphs if p.text in {f"{label}. {TITLES[label]}" for label in inventory}]
        if titles != [f"{label}. {TITLES[label]}" for label in inventory] or (
            len(check.inline_shapes), len(check.tables)
        ) != (len(image_labels), table_count):
            raise ValueError(f"Incomplete EEG display book: {doc_path}")
        placement = {}
        for label, shape in zip(image_labels, check.inline_shapes, strict=True):
            image = check.part.related_parts[shape._inline.graphic.graphicData.pic.blipFill.blip.embed].image
            dpi = min(image.px_width / shape.width.inches, image.px_height / shape.height.inches)
            if dpi + .001 < style.PNG_DPI:
                raise ValueError(f"EEG Word artwork for {label} is below {style.PNG_DPI} effective dpi")
            placement[label] = round(dpi, 2)
        books[stem] = {"displays": inventory, "placed_dpi": placement, "tables": table_count,
                       "docx": file_sha256(doc_path), "pdf": file_sha256(pdf_path)}
    return books


def _export_fold_sources(study, out):
    from reporting.eeg_explanations import _raw
    from reporting.fold_explanations import export_fold_bundle

    for branch in BRANCHES:
        raw = _raw(study, branch)
        for (family, outcome, selected_branch, feature), ps in study["predictions"].items():
            if branch != selected_branch:
                continue
            fold_dirs = sorted((ps.path.parent.parent / "artifacts/folds").glob("repeat_*/outer_*"))
            if len(fold_dirs) != 5:
                raise ValueError(f"Incomplete verified EEG fold explanations: {ps.path}")
            for fold_dir in fold_dirs:
                _within(fold_dir, study["root"])
                for path in fold_dir.iterdir():
                    _within(path, study["root"])
                export_fold_bundle(fold_dir, raw, out.joinpath(family, outcome, branch, feature, *fold_dir.parts[-2:]),
                                   numeric_only=True)


def _write_scope_assets(study, out, scope, *, curves=None, whole_explanations=None):
    from reporting.eeg_explanations import explanation_sources, heldout_figure, importance_figure, response_figure

    out.mkdir(parents=True)
    study["metrics"].to_csv(out / "all_metrics.csv", index=False)
    for key, name in (("paired", "paired_comparisons"), ("dca", "decision_curves"),
                      ("dca_comparisons", "paired_net_benefit"), ("delong", "delong_eligibility")):
        study["comparisons"][key].to_csv(out / f"{name}.csv", index=False)
    if curves is None:
        curves = _curve_tables(study)
    for key, frame in curves.items():
        frame.to_csv(out / f"{key}_source.csv.gz", index=False, compression="gzip", float_format="%.17g")
    for branch in BRANCHES:
        branch_out = out / branch
        branch_out.mkdir()
        for kind in ("roc", "pr", "calibration"):
            fig, source = _discrimination(curves, branch, kind)
            fig.set_label(f"{SCOPES[scope][0]} · {BRANCHES[branch]} · {kind.upper()}")
            _save_figure(fig, branch_out, kind, source)
        for paired in (False, True):
            key = "paired_net_benefit" if paired else "net_benefit"
            fig, source = _dca(study, branch, paired=paired)
            _save_figure(fig, branch_out, key, source)
        sources = (whole_explanations if branch == "non_windowed" and whole_explanations is not None
                   else explanation_sources(study, branch))
        for name, table in sources.items():
            table.to_csv(branch_out / f"ebm_{name}.csv", index=False, float_format="%.17g")
        for key, build in (("ebm_importance", importance_figure), ("ebm_responses", response_figure),
                           ("ebm_heldout", heldout_figure)):
            fig, source = build(sources)
            _save_figure(fig, branch_out, key, source)
    _export_fold_sources(study, out / "fold_explanations")
    return curves


def build_report(artifact_root: Path, processed: Path) -> Path:
    from reporting.eeg_explanations import explanation_sources, heldout_figure, importance_figure, response_figure

    artifact_root = artifact_root.resolve()
    processed = _within(processed, artifact_root)
    output = _within(artifact_root / "results/catch22/eeg_report", artifact_root)
    enforce_storage_policy({"eeg_report": output})
    studies = {scope: _scope_inputs(artifact_root, processed, scope) for scope in SCOPES}
    attrition_path = _within(processed / "eeg_attrition.json", processed)
    attrition = _attrition_sources(attrition_path, studies)
    shared = studies["main_shared"]
    gate = _within(artifact_root / "control/eeg_split_preflight.json", artifact_root)
    gate_sha = file_sha256(gate)
    if any(study["manifest"].get("eeg_split_preflight_sha256") != gate_sha for study in studies.values()):
        raise ValueError("EEG reporting requires the split and feature plan recorded in both run manifests")
    pilot = _within(artifact_root / "pilot/protocol_freeze.json", artifact_root)
    protocol = json.loads(pilot.read_text())["protocol"]
    table_1, table_s2 = _descriptives(studies, processed, attrition)
    table_2, performance = _performance(shared["metrics"])
    table_s1 = _predictor_tables(shared, extractor_version=protocol["versions"]["pycatch22"])
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=output) as work, style.style_context():
        out = Path(work)
        tables = {"Table 1": (table_1,), "Table 2": (table_2,), "Table S1": table_s1, "Table S2": (table_s2,)}
        for label, frames in tables.items():
            for index, frame in enumerate(frames):
                suffix = f"_{index + 1}" if len(frames) > 1 else ""
                frame.to_csv(out / f"{label.replace(' ', '_')}{suffix}.csv", index=False)
        performance.to_csv(out / "Table_2_source_metrics.csv", index=False)
        shutil.copyfile(attrition_path, out / "cohort_attrition.json")
        shutil.copyfile(attrition_path.with_name("eeg_operation_attrition.csv"), out / "eeg_operation_attrition.csv")
        shutil.copyfile(gate, out / "eeg_split_and_feature_plan.json")
        shutil.copyfile(pilot, out / "eeg_protocol_freeze.json")
        _save_figure(*_figure_args(_flow(attrition), out, "Figure 1"))
        _save_figure(*_figure_args(_contrast({"main_shared": shared}, icu_only=False), out, "Figure 2"))
        _save_figure(*_figure_args(_contrast(studies, icu_only=True), out, "Figure S7"))
        for label, metric in (("Figure S5", "auprc"), ("Figure S6", "auroc")):
            _save_figure(*_figure_args(_channels(studies, metric), out, label))
        curves = _curve_tables(shared)
        for label, kind in (("Figure 3", "roc"), ("Figure S3", "pr"), ("Figure S4", "calibration")):
            _save_figure(*_figure_args(_discrimination(curves, "non_windowed", kind), out, label))
        curves["risk_support"].loc[lambda df: df.branch.eq("non_windowed") & df.feature_set.isin(EEG_CORE)].to_csv(
            out / "Figure_S4.risk_support.csv", index=False,
        )
        for label, paired in (("Figure S8", False), ("Figure S9", True)):
            _save_figure(*_figure_args(_dca(shared, "non_windowed", paired=paired), out, label))
        explanations = explanation_sources(shared, "non_windowed")
        for label, build in (("Figure 4", importance_figure), ("Figure 5", response_figure),
                             ("Figure S10", heldout_figure)):
            _save_figure(*_figure_args(build(explanations), out, label))
        explanations["support"].to_csv(out / "Figure_5.training_support.csv", index=False)
        explanations["case_ledger"].to_csv(out / "Figure_S10.case_ledger.csv", index=False)
        explanations["case_terms"].to_csv(out / "Figure_S10.case_terms.csv", index=False)
        ablations = _ablation(studies)
        ablations.to_csv(out / "ablation_comparisons.csv", index=False)
        for scope, study in studies.items():
            companion = out / "companion" / scope
            _write_scope_assets(study, companion, scope, curves=curves if scope == "main_shared" else None,
                                whole_explanations=explanations if scope == "main_shared" else None)
            _save_figure(*_figure_args(_ablation_figure(ablations, scope=scope), companion, "ablation_ap"))
        books = _books(out, tables)
        ledger = {"study_profile": "eeg", "books": books, "style_sha256": style.STYLE_SHA256,
                  "eeg_protocol": protocol, "split_preflight_sha256": gate_sha,
                  "delta_convention": "Candidate minus reference. Lower Brier scores indicate better performance.",
                  "intervals": "Patient-cluster percentile intervals conditional on saved fits. Ablations are nominal and exploratory.",
                  "descriptive_sources": {str(processed / name): file_sha256(processed / name)
                                          for name in ("aki_pleth_ecg_co2_awp.csv", "aki_preop_processed.csv")},
                  "displays": {label: {"title": TITLES[label], "files": sorted(p.name for p in out.glob(
                      f"{label.replace(' ', '_')}.*" if label.startswith("Figure") else f"{label.replace(' ', '_')}*"))}
                      for label in (*MAIN, *SUPPLEMENT)},
                  "attrition": {"file": str(attrition_path), "sha256": file_sha256(attrition_path)},
                  "sources": {scope: {"run_manifest": str(study["manifest_path"]),
                                      "run_manifest_sha256": file_sha256(study["manifest_path"]),
                                      "study_identity": study["manifest"]["study_identity"],
                                      "comparisons_sha256": file_sha256(study["tables"] / "comparison_inventory.json"),
                                      "predictions": {str(ps.path): _prediction_content_hash(ps)
                                                      for ps in study["predictions"].values()}}
                              for scope, study in studies.items()},
                  "files_sha256": {str(path.relative_to(out)): file_sha256(path) for path in sorted(out.rglob("*")) if path.is_file()}}
        for path in sorted(out.rglob("*")):
            if path.is_file():
                destination = _within(output / path.relative_to(out), output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
        staged = out / "display_manifest.json"
        staged.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
        path = _within(output / staged.name, output)
        staged.replace(path)
    return path


def _figure_args(figure_source, out, label):
    figure, source = figure_source
    return figure, out, label, source


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=get_artifact_root())
    parser.add_argument("--processed-dir", type=Path, default=get_processed_dir(project))
    args = parser.parse_args()
    print(build_report(args.artifact_root, args.processed_dir))


if __name__ == "__main__":
    main()
