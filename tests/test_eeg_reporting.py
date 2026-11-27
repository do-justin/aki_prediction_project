import json
import pickle
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from model_creation.utils import EEG_FEATURE_SETS
from reporting import eeg_study
from reporting.display_dictionary import load_display_dictionary
from reporting.eeg_explanations import _case_explanation, _chosen_cases, heldout_figure
from reporting.feature_set_display import FeatureSetDisplay
from results_recreation.comparisons import EEG_INCREMENT_PAIRS, FAMILIES, _eeg_inventory
from results_recreation.metrics_summary import PredictionSet


def test_eeg_full_and_diagnostic_contrast_inventories():
    context = ("main_shared", "nested_cv", "any_aki", "non_windowed")
    full = {(family, feature) for family in FAMILIES for feature in EEG_FEATURE_SETS}
    inventory = _eeg_inventory(context, full, full)
    assert len({row["comparison_id"] for row in inventory}) == len(inventory)
    increment = [row for row in inventory if row["comparison_type"] == "eeg_increment"]
    assert {tuple(row[k] for k in ("reference_feature_set", "candidate_feature_set")) for row in increment} == set(EEG_INCREMENT_PAIRS)
    assert len(increment) == 6 and all(row["designation"] == "primary" for row in increment)
    ablations = [row for row in inventory if row["comparison_type"] == "ablation_parent"]
    assert len(ablations) == 24
    assert all(row["reference_feature_set"].split("_minus_")[0] in {"preop_and_four", "preop_and_five"} for row in ablations)
    diagnostic = {("ebm", "preop_only"), ("ebm", "preop_and_all_five")}
    smoke = _eeg_inventory(context, diagnostic, diagnostic)
    assert len(smoke) == 1 and smoke[0]["availability"] == "available"


def test_eeg_plot_sources_keep_population_and_calibration_for_constant_risk():
    frame = pd.DataFrame({"caseid": [1, 2, 3], "subjectid": [10, 10, 20], "y_true": [0, 1, 1],
                          "y_prob_calibrated": [.92, .92, .92]})
    ps = PredictionSet(Path("prediction.csv"), frame, .5, "icu_admission", "windowed", "eeg_only",
                       "EBM", "catch22", "icu_waveform_sensitivity", "nested_cv", "definitive")
    key = ("ebm", "icu_admission", "windowed", "eeg_only")
    metrics = pd.DataFrame([{"model_type": key[0], "outcome": key[1], "branch": key[2], "feature_set": key[3],
                             "prediction_content_hash": "known-source", "auroc": .5, "auprc": 2/3,
                             "brier": (.92 ** 2 + 2 * .08 ** 2) / 3}])
    data = eeg_study._curve_tables({"predictions": {key: ps}, "metrics": metrics})
    cal = data["calibration"].iloc[0]
    assert len(data["calibration"]) == 1
    assert (cal.bin_n, cal.x, cal.y, cal.patients) == (3, .92, 2 / 3, 2)
    assert cal.prediction_content_hash == "known-source"
    assert data["pr"].prevalence.eq(2 / 3).all()
    assert data["risk_support"].bin_n.sum() == 3


def test_eeg_case_percentiles_keep_ties_and_replay_probability_clipping(tmp_path):
    frame = pd.DataFrame({"caseid": [6, 2, 1], "y_prob_calibrated": [.1, .1, .1]})
    assert list(_chosen_cases(frame)) == [(.1, .1, 1), (.9, .1, 1)]
    calibration = {"intercept": -.3, "slope": 1.2, "eps": .01}
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    risk = expit(-.3 + 1.2 * logit(.99))
    clf = SimpleNamespace(intercept_=np.array([10.0]), term_names_=["BIS_EEG1_WAV_DN_Mean"], term_features_=[(0,)],
                          eval_terms=lambda frame: np.array([[10.0]]),
                          predict_proba=lambda frame: np.array([[1-expit(20.0), expit(20.0)]]))
    pre = SimpleNamespace(numeric_columns_=clf.term_names_, categorical_columns_=[], transform=lambda frame: frame)
    test = pd.DataFrame({"caseid": [1], "y_prob_raw": [expit(20)], "y_prob_calibrated": [risk], "y_true": [1]})
    bundle = (pre, clf, None, test, {"model_bundle": str(tmp_path)})
    lookup = pd.DataFrame({clf.term_names_[0]: [2.5]}, index=pd.Index([1], name="caseid"))
    common, terms, plotted = _case_explanation(bundle, 1, lookup)
    assert terms.group.tolist() == ["EEG"]
    assert common["calibrated_probability"] == pytest.approx(risk)
    assert plotted["Probability clipping correction"] < 0
    assert expit(common["calibrated_intercept"] + sum(plotted.values())) == pytest.approx(risk)


def test_eeg_heldout_figure_uses_both_risk_percentiles():
    import matplotlib.pyplot as plt

    contributions = pd.DataFrame({
        "outcome": ["any_aki", "any_aki"],
        "quantile": [.1, .9],
        "group": ["EEG", "EEG"],
        "start": [0.0, 0.0],
        "contribution": [-.2, .4],
        "calibrated_probability": [.05, .4],
    })
    figure, source = heldout_figure({"case_contributions": contributions})
    try:
        pd.testing.assert_frame_equal(source, contributions)
        assert [axis.patches[0].get_width() for axis in figure.axes] == [-.2, .4]
        assert "10% risk percentile" in figure.axes[0].get_title()
        assert "90% risk percentile" in figure.axes[1].get_title()
    finally:
        plt.close(figure)


def test_eeg_labels_have_units_only_for_dimensional_statistics(monkeypatch):
    labels = load_display_dictionary()
    assert "µV" in labels.feature_label("BIS_EEG1_WAV_DN_Mean_std", include_unit=True)
    assert "µV" not in labels.feature_label("BIS_EEG1_WAV_CO_f1ecac", include_unit=True)
    monkeypatch.delenv("AKI_STUDY_PROFILE", raising=False)
    assert "eeg" not in FeatureSetDisplay.from_defaults().config.component_keys
    monkeypatch.setenv("AKI_STUDY_PROFILE", "eeg")
    configs = FeatureSetDisplay.from_defaults().config
    assert "eeg" in configs.feature_set_map["all_five_waveforms"]
    assert "eeg" not in configs.feature_set_map["all_waveforms"]


def test_eeg_report_rejects_paths_outside_study(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    outside = tmp_path / "prior-manuscript.csv"
    outside.write_text("protected values")
    path = root / "predictions.csv"
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="outside its study"):
        eeg_study._within(path, root)


def test_eeg_parallel_fold_exports_match_serial(tmp_path):
    from sklearn.linear_model import LogisticRegression

    from data_preparation.artifact_metadata import file_sha256
    from model_creation.preprocessing import FoldPreprocessor
    from model_creation.validation import CHECKPOINT_BUNDLE_PROTOCOL_VERSION

    raw = pd.DataFrame({"caseid": [1, 2, 3, 4], "age": [40, 60, 50, 70]})
    matrix = tmp_path / "modeling.csv"
    raw.to_csv(matrix, index=False)
    digest = file_sha256(matrix)
    preprocessor = FoldPreprocessor()
    training = preprocessor.fit_transform(raw.loc[:1, ["age"]])
    model = LogisticRegression().fit(training, [0, 1])
    config = tmp_path / "results" / "models" / "preop_only"
    for index in range(5):
        fold = config / "artifacts" / "folds" / "repeat_00" / f"outer_{index:02d}"
        fold.mkdir(parents=True)
        pd.DataFrame({"caseid": [3, 4], "y_true": [0, 1]}).to_csv(fold / "test_predictions.csv", index=False)
        pd.DataFrame({"caseid": [1, 2]}).to_csv(fold / "train_oof.csv", index=False)
        for name in ("calibration.json", "threshold.json", "best_params.json"):
            (fold / name).write_text("{}")
        (fold / "preprocessor.pkl").write_bytes(pickle.dumps(preprocessor))
        (fold / "model.pkl").write_bytes(pickle.dumps(model))
        hashes = {p.name: file_sha256(p) for p in fold.iterdir()}
        (fold / "metadata.json").write_text(json.dumps({
            "status": "completed", "model_type": "logistic_regression", "model_path": "model.pkl",
            "checkpoint_bundle_protocol_version": CHECKPOINT_BUNDLE_PROTOCOL_VERSION,
            "validation_fingerprint": {"dataset_hash": digest,
                                       "checkpoint_bundle_protocol_version": CHECKPOINT_BUNDLE_PROTOCOL_VERSION},
            "model_sha256": hashes["model.pkl"], "preprocessor_sha256": hashes["preprocessor.pkl"],
            "artifact_sha256": hashes,
        }))
    key = ("logistic_regression", "any_aki", "non_windowed", "preop_only")
    study = {
        "root": tmp_path / "results",
        "matrix_paths": {"non_windowed": matrix},
        "manifest": {"branch_rosters": {"non_windowed": {"dataset_sha256": digest}}},
        "predictions": {key: SimpleNamespace(path=config / "predictions" / "test.csv")},
    }
    published = tmp_path / "published"
    serial = published / "companion" / "main_shared" / "fold_explanations"
    parallel = tmp_path / "parallel"
    eeg_study._export_fold_sources(study, serial, workers=1)
    eeg_study._export_fold_sources(study, parallel, workers=2)

    serial_files = {p.relative_to(serial): p.read_bytes() for p in serial.rglob("*") if p.is_file()}
    parallel_files = {p.relative_to(parallel): p.read_bytes() for p in parallel.rglob("*") if p.is_file()}
    assert serial_files == parallel_files
    assert len([p for p in parallel_files if p.name == "coefficients.csv"]) == 5
    output_metadata = parallel.joinpath(*key, "repeat_00", "outer_00", "explanation_metadata.json")
    assert json.loads(output_metadata.read_text())["caseids"] == [3, 4]

    fingerprint = {"exporter": "verified"}
    assert eeg_study._published_fold_cache(published, fingerprint) == {}
    (published / "display_manifest.json").write_text(json.dumps({
        "fold_export_fingerprint": fingerprint,
        "files_sha256": {str((serial / p).relative_to(published)): file_sha256(serial / p) for p in serial_files},
    }))
    assert eeg_study._published_fold_cache(published, {"exporter": "new version"}) == {}
    cache = eeg_study._published_fold_cache(published, fingerprint)["main_shared"]
    bad_csv = next(p for p in serial_files if p.name == "coefficients.csv")
    (serial / bad_csv).write_text("invalid cache")
    rebuilt = tmp_path / "rebuilt"
    assert eeg_study._export_fold_sources(study, rebuilt, workers=2, cache=cache) == 4
    assert serial_files == {p.relative_to(rebuilt): p.read_bytes() for p in rebuilt.rglob("*") if p.is_file()}

    (config / "artifacts/folds/repeat_00/outer_00/model.pkl").write_bytes(b"broken fold")
    with pytest.raises(ValueError, match="checkpoint failed verification"):
        eeg_study._export_fold_sources(study, tmp_path / "invalid", workers=1, cache=cache)


def test_eeg_fold_workers_respect_report_budget_and_available_resources(monkeypatch):
    resources = {"available_cpus": 64, "memory_available_bytes": 128 * 1024 ** 3}
    monkeypatch.setattr(eeg_study, "resource_snapshot", lambda: resources)
    monkeypatch.delenv("AKI_REPORT_CPU_BUDGET", raising=False)
    assert eeg_study._fold_worker_count(None) == 64
    assert eeg_study._fold_worker_count(8) == 8
    assert eeg_study._fold_worker_count(128) == 64
    with pytest.raises(ValueError, match="must be positive"):
        eeg_study._fold_worker_count(0)
    monkeypatch.setenv("AKI_REPORT_CPU_BUDGET", "12")
    assert eeg_study._fold_worker_count(None) == 12
    assert eeg_study._fold_worker_count(32) == 12
    resources["memory_available_bytes"] = 3 * 1024 ** 3
    assert eeg_study._fold_worker_count(32) == 3


def test_eeg_descriptor_table_uses_frozen_extractor_definitions():
    import pycatch22

    legacy = eeg_study._descriptor_table("0.4.5")
    assert legacy.attrs["pycatch22_version"] == "0.4.5"
    assert "DN_Mean" in legacy["Catch24 descriptor"].tolist()

    installed = version("pycatch22")
    with pytest.raises(ValueError, match=r"require the frozen pycatch22 9\.9\.9"):
        eeg_study._descriptor_table("9.9.9")
    if installed == "0.4.5":
        return

    current = eeg_study._descriptor_table(installed)
    definition = pycatch22.feature_table(catch24=True)
    assert current["Catch24 descriptor"].tolist() == [row["name"] for row in definition]
    assert current["hctsa reference call"].tolist() == [row["hctsa_call"] for row in definition]
    assert "SP_Summaries_fft_wq_50" in current["Catch24 descriptor"].tolist()
    sections = eeg_study._table_sections("Table S1", (pd.DataFrame(), pd.DataFrame(), current))
    assert f"pycatch22 {installed}" in sections[-1]["notes"]


def test_eeg_assembles_numbered_word_and_vector_pdf_books(tmp_path):
    import shutil

    import matplotlib.pyplot as plt

    if not shutil.which("pdfunite"):
        pytest.skip("Poppler is needed to merge vector PDF pages")
    fig, ax = plt.subplots(figsize=(7.8, 1.5))
    ax.plot([0, 1], [0, 1])
    eeg_study._save_figure(fig, tmp_path, "Figure 1", pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
    for label in (*eeg_study.MAIN[3:], *eeg_study.SUPPLEMENT[2:]):
        stem = label.replace(" ", "_")
        for suffix in (".pdf", ".png"):
            shutil.copyfile(tmp_path / f"Figure_1{suffix}", tmp_path / f"{stem}{suffix}")
    table = lambda width: pd.DataFrame([list(range(width))], columns=[f"Col {i}" for i in range(width)])
    books = eeg_study._books(tmp_path, {"Table 1": (table(3),), "Table 2": (table(7),),
                                        "Table S1": (table(3), table(7), table(2)), "Table S2": (table(3),)})
    assert len(books["Main_Figures_and_Tables"]["placed_dpi"]) == 5
    assert len(books["Supplementary_Figures_and_Tables"]["placed_dpi"]) == 8
    assert (tmp_path / "Main_Figures_and_Tables.pdf").read_bytes().startswith(b"%PDF-")
