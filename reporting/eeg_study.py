"""Rebuild the EEG feasibility displays from the isolated study's saved results."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from joblib import Parallel, delayed, parallel_config
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

from artifact_paths import enforce_storage_policy, get_artifact_root, get_processed_dir, get_raw_dir
from data_preparation.artifact_metadata import file_sha256
from data_preparation.cohort_scopes import selected_case_hash
from data_preparation.eeg_study import eeg_scope_identities
from reporting import publication_style as style
from reporting.cohort_flow import CohortStage
from reporting.display_dictionary import load_display_dictionary
from reporting.eeg_explanations import NUMERIC_FOLD_FILES
from reporting.eeg_manuscript_figures import (
    CALIBRATION_INPUTS, channels as _channels, contrasts as _contrast,
    decision_curves as _dca, discrimination as _discrimination,
)
from reporting.eeg_manuscript_tables import (
    COHORT_SOURCES, PREDICTOR_DICTIONARIES, add_table_display, descriptive_tables,
    performance_table, predictor_tables,
)
from reporting.figure_rendering import save_report_figure_bundle
from reporting.manuscript_workspace.revise_publication import FIGURE_TITLES, add_image, document, heading
from reporting.manuscript_workspace.revision_data import BRANCHES, FAMILIES, IDENTITY, OUTCOMES
from reporting.manuscript_workspace.revision_figures import cohort_flow_figure
from results_recreation.comparisons import EEG_INCREMENT_PAIRS, load_comparison_artifacts
from results_recreation.metrics_summary import (
    _model_id, _prediction_content_hash, compute_point_metrics, crawl_predictions,
)
from results_recreation.results_analysis import PlotConfig, _calibration_bins
from runtime_resources import resource_snapshot

SCOPES = {"main_shared": "Shared AKI/ICU", "icu_waveform_sensitivity": "Broader ICU"}
LABELS = {
    "pleth_only": "PPG", "ecg_only": "ECG", "co2_only": "CO₂", "awp_only": "Airway pressure", "eeg_only": "EEG",
}
MAIN = ("Figure 1", "Table 1", "Table 2", "Figure 2", "Figure 3", "Figure 4", "Figure 5")
SUPPLEMENT = ("Table S1", "Table S2", *(f"Figure S{i}" for i in range(3, 11)))
TITLES = {**FIGURE_TITLES, "Table 1": "Cohort characteristics", "Table 2": "Performance by input configuration",
          "Table S1": "Predictors and feature extraction", "Table S2": "Main and broader ICU cohorts"}
CI_NOTE = "95% patient-cluster intervals are conditional on the saved outer-fold fits."
CAPTIONS = {
    "Figure 1": "First exclusions are counted once. The shared cohort supports both acute kidney injury (AKI) and intensive care unit (ICU) admission analysis.",
    "Figure 2": f"Paired differences within each outcome and model. P denotes preoperative inputs, W4 the original four waveforms, and W5 all five including EEG. {CI_NOTE}",
    "Figure 3": "Pooled held-out predictions. AUROCs follow input legend order across both rows of values in each panel. The diagonal indicates chance.",
    "Figure 4": "Mean absolute contribution to model log-odds, weighted by the training distribution. Bars show means across five folds and symbols show individual folds. Hollow symbols indicate absent terms.",
    "Figure 5": "Combined whole-operation models with EEG. Curves show fold-specific changes in model log-odds relative to the indicated reference values, before recalibration. Histograms show the first outer training fold.",
    "Figure S3": "Pooled held-out predictions. Average precision follows input legend order across both rows of values. Horizontal guides show outcome prevalence.",
    "Figure S4": "Up to ten equal-frequency bins of pooled held-out predictions. Histograms show the distribution of predicted risk and insets show the lower AKI risk range.",
    "Figure S5": f"Waveform-only models in the shared cohort. AP denotes average precision. {CI_NOTE}",
    "Figure S6": f"Waveform-only models in the shared cohort. {CI_NOTE}",
    "Figure S7": f"Within-cohort differences. P denotes preoperative inputs, W4 the original four waveforms, and W5 all five including EEG. {CI_NOTE}",
    "Figure S8": "Whole-operation net benefit per 100 operations. Arrows mark treat-all values below the plotted range.",
    "Figure S9": "Paired differences in whole-operation net benefit per 100 operations. Shading shows pointwise 95% patient-cluster intervals. P denotes preoperative inputs, W4 the original four waveforms, and W5 all five including EEG.",
    "Figure S10": "Cases nearest the indicated percentiles of whole-operation combined-model risk. Contributions add to the intercept on the calibrated log-odds scale. Preoperative-only risks use separate fits.",
}


def _within(path: Path, root: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"EEG report path is outside its study: {path}")
    return path



def _scope_inputs(artifact_root, processed, scope):
    from model_creation.utils import EEG_FEATURE_SETS

    root = _within(artifact_root / "results/catch22/experiments" / scope / "nested_cv", artifact_root)
    manifest_path = _within(root / "run_manifest.json", root)
    manifest = json.loads(manifest_path.read_text())
    cohort_id = eeg_scope_identities()[scope]
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
        if (expected.get("cohort_manifest_identity") != study["manifest"]["manifest_cohort_id"]
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
        fig.savefig(out / f"{stem}.png", dpi=style.PNG_DPI, facecolor="white")
        fig.savefig(out / f"{stem}.pdf", facecolor="white")
        plt.close(fig)
    else:
        if fig.get_label():
            fig.suptitle(fig.get_label(), fontweight="bold", fontsize=11)
        save_report_figure_bundle(fig, out / stem, formats=("png", "pdf"), close=True)
    source.to_csv(out / f"{stem}.plotted.csv", index=False, float_format="%.17g")


def _flow(attrition):
    labels = {"source": "VitalDB operations", "clinical": "Clinical eligibility",
              "four_waveform_tracks": "Original four waveforms available", "eeg_track": "Raw EEG track available",
              "eeg_qc": "Usable EEG signal", "eeg_features": "EEG features in both branches",
              "both_branches": "Broader ICU analysis cohort"}
    previous = None
    values, stages = [], []
    reasons = {"clinical": "Clinical eligibility not met", "original_waveform_track": "Required original track unavailable",
               "eeg_track": "No raw EEG track", "eeg_source_or_qc": "EEG load or quality check failed",
               "eeg_feature_extraction": "EEG feature extraction failed",
               "original_modality_extraction": "Original waveform extraction failed"}
    for stage in attrition["stages"]:
        count, patients = stage["operations"], stage["patients"]
        values.append({"stage": stage["key"], "operations": count, "patients": patients})
        if stage["key"] != "eeg_features" or count != previous:
            stages.append(CohortStage(title=labels[stage["key"]], count=count, n_patients=patients,
                                      removed=None if previous is None else previous - count,
                                      removal_reason=reasons.get(stage.get("first_exclusion"))))
        previous = count
    main = attrition["scopes"]["main_shared"]
    broader = attrition["scopes"]["icu_waveform_sensitivity"]
    stages.append(CohortStage(title="Shared AKI/ICU analysis cohort", count=main["operations"],
                              n_patients=main["patients"], removed=broader["operations"] - main["operations"],
                              removal_reason="AKI ascertainment unavailable"))
    for scope, counts in attrition["scopes"].items():
        if scope not in SCOPES:
            continue
        for outcome, record in counts["outcomes"].items():
            values.append({"stage": scope, "outcome": outcome, "operations": counts["operations"],
                           "patients": counts["patients"], **record})
    def events(scope, outcome):
        record = scope["outcomes"][outcome]
        return f"{OUTCOMES[outcome]}: {record['positive_operations']:,} operations ({record['prevalence']:.1%})"

    fig = cohort_flow_figure(stages, [events(main, outcome) for outcome in OUTCOMES],
                             cohort_note=f"Broader ICU cohort · {events(broader, 'icu_admission')}")
    return fig, pd.DataFrame(values)



def _axes(rows, cols=3, *, height=2.15):
    return plt.subplots(rows, cols, squeeze=False, figsize=(7.8, height * rows + .55), layout="constrained")



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



def _pdf_renderer():
    command = os.environ.get("AKI_EEG_SOFFICE", "soffice")
    if executable := shutil.which(command):
        return executable
    raise RuntimeError("LibreOffice Writer is required for manuscript-layout PDF proofs. Set AKI_EEG_SOFFICE to its executable.")


def _books(out, tables, *, pdf_renderer=None):
    pdf_renderer = pdf_renderer or _pdf_renderer()
    books = {}
    with tempfile.TemporaryDirectory(prefix="eeg-word-proofs-") as temporary:
        staging = Path(temporary)
        proof_sources = []
        for inventory, stem in ((MAIN, "Main_Figures_and_Tables"), (SUPPLEMENT, "Supplementary_Figures_and_Tables")):
            doc = document(stem.replace("_", " "))
            figure_labels, table_count = [], 0
            for index, label in enumerate(inventory):
                title = f"{label}. {TITLES[label]}"
                file_stem = label.replace(" ", "_")
                if label in tables:
                    heading(doc, title, page_break=index > 0)
                    table_count += add_table_display(doc, label, tables[label])
                    table_doc = document(title)
                    heading(table_doc, title)
                    add_table_display(table_doc, label, tables[label])
                    proof_sources.append(staging / f"{file_stem}.docx")
                    table_doc.save(proof_sources[-1])
                else:
                    image_path = out / f"{file_stem}.png"
                    with Image.open(image_path) as raster:
                        width = min(6.35 if label == "Figure S10" else 6.85, raster.width / style.PNG_DPI,
                                    8.65 * raster.width / raster.height)
                    add_image(doc, label, image_path, CAPTIONS[label], page_break=index > 0, width=width)
                    figure_labels.append(label)
            doc_path = out / f"{stem}.docx"
            doc.save(doc_path)
            proof_sources.append(doc_path)
            check = Document(doc_path)
            expected = [f"{label}. {TITLES[label]}" for label in inventory]
            titles = [p.text for p in check.paragraphs if p.text in expected]
            if titles != expected or (len(check.inline_shapes), len(check.tables)) != (len(figure_labels), table_count):
                raise ValueError(f"Incomplete EEG display book: {doc_path}")
            placement = {}
            for label, shape in zip(figure_labels, check.inline_shapes, strict=True):
                image = check.part.related_parts[shape._inline.graphic.graphicData.pic.blipFill.blip.embed].image
                dpi = min(image.px_width / shape.width.inches, image.px_height / shape.height.inches)
                if dpi + .001 < style.PNG_DPI:
                    raise ValueError(f"EEG Word artwork for {label} is below {style.PNG_DPI} effective dpi")
                placement[label] = round(dpi, 2)
            books[stem] = {"displays": inventory, "placed_dpi": placement, "tables": table_count,
                           "docx": file_sha256(doc_path)}

        # The final manuscript PDFs were exported from the editable Word layout.
        export_filter = ('pdf:writer_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},'
                         '"ReduceImageResolution":{"type":"boolean","value":"false"}}')
        proc = subprocess.run([pdf_renderer, f"-env:UserInstallation={(staging / 'profile').as_uri()}", "--headless",
                               "--convert-to", export_filter, "--outdir", str(out), *map(str, proof_sources)],
                              capture_output=True, text=True, timeout=180)
        missing = [path.stem for path in proof_sources if not (out / f"{path.stem}.pdf").is_file()]
        if proc.returncode or missing:
            raise RuntimeError(f"LibreOffice PDF export failed ({proc.returncode}, missing {missing}). "
                               f"{proc.stdout[-1000:]}\n{proc.stderr[-3000:]}")
    for stem, book in books.items():
        book["pdf"] = file_sha256(out / f"{stem}.pdf")
    return books


def _fold_export_fingerprint() -> dict:
    from model_creation import preprocessing
    from reporting import display_dictionary, fold_explanations
    from reporting.eeg_explanations import _read_modeling_matrix

    return {
        "source_sha256": {module.__name__: file_sha256(Path(module.__file__))
                          for module in (preprocessing, display_dictionary, fold_explanations)},
        "matrix_reader_sha256": hashlib.sha256(inspect.getsource(_read_modeling_matrix).encode()).hexdigest(),
        "display_dictionary_sha256": file_sha256(load_display_dictionary().source),
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "xgboost-cpu", "interpret-core", "shap", "pycatch22",
        )},
    }


def _published_fold_cache(output, fingerprint):
    manifest = output / "display_manifest.json"
    try:
        previous = json.loads(_within(manifest, output).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(previous, dict) or previous.get("fold_export_fingerprint") != fingerprint:
        return {}
    hashes = previous.get("files_sha256")
    if not isinstance(hashes, dict):
        return {}
    caches = {}
    for scope in SCOPES:
        prefix = Path("companion", scope, "fold_explanations")
        cache_hashes = {Path(path).relative_to(prefix): digest for path, digest in hashes.items()
                        if Path(path).is_relative_to(prefix) and isinstance(digest, str)}
        if cache_hashes:
            caches[scope] = (_within(output / prefix, output), cache_hashes)
    return caches


def _cached_fold(relative, family, cache):
    if cache is None:
        return None
    root, files = cache
    names = NUMERIC_FOLD_FILES[family]
    if any(relative / name not in files for name in names):
        return None
    try:
        source = _within(root / relative, root)
        for name in names:
            _within(source / name, root)
    except ValueError:
        return None
    return source, {name: files[relative / name] for name in names}


def _export_fold_sources(study, out, *, workers: int, cache=None) -> int:
    from reporting.eeg_explanations import _export_fold_source

    matrices = {}
    for branch, matrix in study["matrix_paths"].items():
        expected_hash = study["manifest"]["branch_rosters"][branch]["dataset_sha256"]
        if file_sha256(matrix) != expected_hash:
            raise ValueError(f"EEG fold export modeling dataset hash mismatch: {matrix}")
        matrices[branch] = (matrix, expected_hash)

    jobs = []
    for (family, outcome, branch, feature), ps in study["predictions"].items():
        matrix, expected_hash = matrices[branch]
        fold_dirs = sorted((ps.path.parent.parent / "artifacts/folds").glob("repeat_*/outer_*"))
        if len(fold_dirs) != 5:
            raise ValueError(f"Incomplete verified EEG fold explanations: {ps.path}")
        for fold in fold_dirs:
            _within(fold, study["root"])
            for path in fold.iterdir():
                _within(path, study["root"])
            relative = Path(family, outcome, branch, feature, *fold.parts[-2:])
            jobs.append((fold, matrix, expected_hash, out / relative, _cached_fold(relative, family, cache)))

    workers = min(workers, max(1, len(jobs)))
    print(f"Preparing {len(jobs):,} fold explanations with {workers} workers", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        reused = sum(Parallel(n_jobs=workers, batch_size=1)(delayed(_export_fold_source)(*job) for job in jobs))
    print(f"Reused {reused:,} verified fold exports and rebuilt {len(jobs) - reused:,}", flush=True)
    return reused


def _write_scope_assets(study, out, scope, *, fold_workers, fold_cache=None, curves=None, whole_explanations=None):
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
            fig.set_label(f"{SCOPES[scope]} · {BRANCHES[branch]} · {kind.upper()}")
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
    _export_fold_sources(study, out / "fold_explanations", workers=fold_workers, cache=fold_cache)
    return curves


def _fold_worker_count(requested: int | None) -> int:
    resources = resource_snapshot()
    available = resources["available_cpus"]
    budget = int(os.getenv("AKI_REPORT_CPU_BUDGET", str(available)))
    if budget < 1 or (requested is not None and requested < 1):
        raise ValueError("The report CPU budget and fold worker count must be positive")
    memory = resources["memory_available_bytes"]
    # Account for each process loading the numerical libraries and both input matrices.
    memory_workers = max(1, memory // (1024 ** 3)) if memory is not None else available
    return min(available, budget, memory_workers, requested if requested is not None else available)


def build_report(artifact_root: Path, processed: Path, *, fold_workers: int | None = None) -> Path:
    from reporting.eeg_manuscript_explanations import (
        heldout_figure, importance_figure, manuscript_explanation_sources, manuscript_response_support, response_figure,
    )

    pdf_renderer = _pdf_renderer()
    fold_workers = _fold_worker_count(fold_workers)
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
    raw = get_raw_dir(Path(__file__).resolve().parents[1])
    table_1, table_s2 = descriptive_tables(studies, processed, attrition)
    table_2, performance = performance_table(shared, artifact_root=artifact_root)
    table_s1 = predictor_tables(shared, extractor_version=protocol["versions"]["pycatch22"], raw=raw,
                                eeg_protocol=protocol["protocol"])
    output.mkdir(parents=True, exist_ok=True)
    fold_fingerprint = _fold_export_fingerprint()
    fold_caches = _published_fold_cache(output, fold_fingerprint)
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
            _save_figure(*_figure_args(_channels(shared, metric), out, label))
        curves = _curve_tables(shared)
        for label, kind in (("Figure 3", "roc"), ("Figure S3", "pr"), ("Figure S4", "calibration")):
            _save_figure(*_figure_args(_discrimination(curves, "non_windowed", kind), out, label))
        curves["risk_support"].loc[lambda df: df.branch.eq("non_windowed") & df.feature_set.isin(CALIBRATION_INPUTS)].to_csv(
            out / "Figure_S4.risk_support.csv", index=False,
        )
        for label, paired in (("Figure S8", False), ("Figure S9", True)):
            _save_figure(*_figure_args(_dca(shared, "non_windowed", paired=paired), out, label))
        explanations = manuscript_explanation_sources(shared, "non_windowed")
        for label, build in (("Figure 4", importance_figure), ("Figure 5", response_figure),
                             ("Figure S10", heldout_figure)):
            _save_figure(*_figure_args(build(explanations), out, label))
        manuscript_response_support(explanations).to_csv(out / "Figure_5.training_support.csv", index=False)
        explanations["case_ledger"].to_csv(out / "Figure_S10.case_ledger.csv", index=False)
        explanations["case_terms"].to_csv(out / "Figure_S10.case_terms.csv", index=False)
        ablations = _ablation(studies)
        ablations.to_csv(out / "ablation_comparisons.csv", index=False)
        for scope, study in studies.items():
            companion = out / "companion" / scope
            _write_scope_assets(study, companion, scope, fold_workers=fold_workers,
                                fold_cache=fold_caches.get(scope), curves=curves if scope == "main_shared" else None,
                                whole_explanations=({key: value for key, value in explanations.items() if not key.startswith("manuscript_")}
                                                    if scope == "main_shared" else None))
            _save_figure(*_figure_args(_ablation_figure(ablations, scope=scope), companion, "ablation_ap"))
            if scope == "icu_waveform_sensitivity":
                for metric in ("auprc", "auroc"):
                    _save_figure(*_figure_args(_channels(study, metric), companion, f"channel_{metric}"))
        books = _books(out, tables, pdf_renderer=pdf_renderer)
        ledger = {"study_profile": "eeg", "books": books, "style_sha256": style.STYLE_SHA256,
                  "display_layout": "manuscript-revision",
                  "eeg_protocol": protocol, "split_preflight_sha256": gate_sha,
                  "fold_export_fingerprint": fold_fingerprint,
                  "delta_convention": "Candidate minus reference. Lower Brier scores indicate better performance.",
                  "intervals": "Patient-cluster percentile intervals conditional on saved fits. Ablations are nominal and exploratory.",
                  "descriptive_sources": {str(processed / name): file_sha256(processed / name)
                                          for name in COHORT_SOURCES},
                  "predictor_dictionary_sources": {str(raw / name): file_sha256(raw / name)
                                                   for name in PREDICTOR_DICTIONARIES},
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
    parser.add_argument("--fold-workers", type=int,
                        help="Maximum fold export processes (default: available report CPU budget)")
    args = parser.parse_args()
    print(build_report(args.artifact_root, args.processed_dir, fold_workers=args.fold_workers))


if __name__ == "__main__":
    main()
