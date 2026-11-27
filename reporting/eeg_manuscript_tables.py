"""Build manuscript-formatted tables from the frozen EEG cohorts and model results."""

from importlib.metadata import version
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
import joblib
import numpy as np
import pandas as pd

from reporting.manuscript_workspace.revise_publication import (
    MODEL_KEY, OUTCOME_KEY, abbreviation_key, add_frame, add_table, heading, note, row,
)
from reporting.manuscript_workspace.revision_data import (
    BRANCHES, FAMILIES, METRICS, OUTCOMES, Study, verify_baselines,
)
from reporting.manuscript_workspace.revision_tables import (
    CATEGORIES, CHANNEL_SPECS, DESCRIPTORS, LABS, clinical_predictor_table,
)
from reporting.preop_descriptives import _ecg_display, _restrict_to_roster
from data_preparation.eeg import AMPLITUDE_PROTOCOL, PROTOCOL


COHORT_SOURCES = ("aki_pleth_ecg_co2_awp.csv", "aki_preop_processed.csv")
PREDICTOR_DICTIONARIES = ("clinical_parameters.csv", "lab_parameters.csv")
INPUT_CONFIGURATIONS = {
    "preop_only": "Preoperative only",
    "original_four_waveforms": "Four waveforms",
    "all_five_waveforms": "Four waveforms + EEG",
    "preop_and_original_four": "Preoperative + four waveforms",
    "preop_and_all_five": "Preoperative + four waveforms + EEG",
}
COHORT_BANDS = (
    "Study population", "Demographics", "Operative characteristics", "Comorbidities",
    "Procedure type", "Preoperative laboratory values",
)
AGE_NOTE = "VitalDB ages recorded as >89 are counted as missing numeric age."
CI_NOTE = "95% patient-cluster intervals are conditional on the saved outer-fold fits."


def _number(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _median(series, precision):
    q1, median, q3 = series.quantile([.25, .5, .75])
    if pd.isna(median):
        return "—"
    return f"{median:.{precision}f} ({q1:.{precision}f}, {q3:.{precision}f})"


def _count(mask):
    return f"{int(mask.sum()):,} ({mask.mean():.1%})"


def _duration(cohort):
    return (_number(cohort.opend) - _number(cohort.opstart)) / 60


def _primary_characteristics(cohort):
    def continuous(label, values, precision):
        return label, _median(values, precision), _count(values.isna())

    def categorical(label, mask, source):
        return label, _count(mask), _count(source.isna())

    age, bmi, asa = (_number(cohort[key]) for key in ("age", "bmi", "asa"))
    records = [
        ("Study population", "", ""),
        ("Operations", f"{len(cohort):,}", ""),
        ("Unique patients", f"{cohort.subjectid.nunique():,}", ""),
        ("AKI within 72 hours", _count(cohort.aki_label.eq(1)), ""),
        (" Stage 2–3 AKI", _count(cohort.aki_stage.ge(2)), ""),
        ("ICU admission", _count(cohort.y_icu_admit.eq(1)), ""),
        ("Demographics", "", ""),
        continuous("Age (years)", age, 0),
        continuous("BMI (kg/m²)", bmi, 1),
        categorical("Male sex", cohort.sex.eq("M"), cohort.sex),
        ("Operative characteristics", "", ""),
        *(categorical(f"ASA-PS {label}", asa.eq(i), asa)
          for i, label in enumerate(("I", "II", "III", "IV"), 1)),
        categorical("Emergency surgery", cohort.emop.eq(1), cohort.emop),
        categorical("Open approach", cohort.approach.eq("Open"), cohort.approach),
        continuous("Operative duration (min)", _duration(cohort), 0),
        ("Comorbidities", "", ""),
        categorical("Hypertension", cohort.preop_htn.eq(1), cohort.preop_htn),
        categorical("Diabetes", cohort.preop_dm.eq(1), cohort.preop_dm),
    ]
    ecg = _ecg_display(cohort.preop_ecg)
    records.append(("Preoperative ECG finding", "", _count(ecg.eq("Missing"))))
    records.extend((f" {label}", _count(ecg.eq(label)), "")
                   for label in ("Normal sinus rhythm", "Abnormal", "Unknown")
                   if label != "Unknown" or ecg.eq(label).any())
    procedures = cohort.optype.astype("string").str.strip().replace("", pd.NA)
    records.append(("Procedure type", "", _count(procedures.isna())))
    records.extend((f" {label}", _count(procedures.eq(label).fillna(False)), "")
                   for label in sorted(procedures.dropna().unique()))
    records.append(("Preoperative laboratory values", "", ""))
    records.extend(continuous(label, _number(cohort[field]), precision) for field, label, precision in (
        ("preop_hb", "Hemoglobin (g/dL)", 1), ("preop_cr", "Creatinine (mg/dL)", 2),
        ("preop_egfr_ckdepi_2021", "eGFR (mL/min/1.73 m²)", 1), ("preop_bun", "BUN (mg/dL)", 0),
        ("preop_alb", "Albumin (g/dL)", 1), ("preop_wbc", "WBC (10³/µL)", 2),
        ("preop_crp", "CRP (mg/dL)", 2),
    ))
    return pd.DataFrame(records, columns=["Characteristic", f"Finding (N = {len(cohort):,})", "Missing n (%)"])


def _comparison_characteristics(cohort, label):
    creatinine = _number(cohort.preop_cr)
    return pd.Series({
        "Operations": f"{len(cohort):,}",
        "Unique patients": f"{cohort.subjectid.nunique():,}",
        "ICU admission": _count(cohort.y_icu_admit.eq(1)),
        "Age, years": _median(_number(cohort.age), 0),
        "Male sex": _count(cohort.sex.eq("M")),
        "BMI, kg/m²": _median(_number(cohort.bmi), 1),
        "Emergency surgery": _count(cohort.emop.eq(1)),
        "ASA-PS III–IV": _count(cohort.asa.isin([3, 4])),
        "Open surgical approach": _count(cohort.approach.eq("Open")),
        "Operative duration, min": _median(_duration(cohort), 0),
        "Colorectal procedure": _count(cohort.optype.eq("Colorectal")),
        "Transplantation": _count(cohort.optype.eq("Transplantation")),
        "Hypertension": _count(cohort.preop_htn.eq(1)),
        "Diabetes": _count(cohort.preop_dm.eq(1)),
        "Preoperative creatinine, mg/dL": _median(creatinine, 2),
        "No finite positive baseline creatinine": _count(~cohort.aki_has_baseline_cr.eq(1)),
        "Baseline creatinine >4 mg/dL": _count(creatinine.gt(4)),
        "No finite positive postoperative creatinine by 72 h": _count(~cohort.aki_has_postop_cr_72h.eq(1)),
        "Ineligible for AKI ascertainment": _count(~cohort.eligible_any_aki.eq(1)),
        "C-reactive protein unavailable": _count(_number(cohort.preop_crp).isna()),
    }, name=label)


def descriptive_tables(studies, processed: Path, attrition):
    source, derived = (pd.read_csv(processed / name) for name in COHORT_SOURCES)
    derived_fields = ("preop_egfr_ckdepi_2021", "preop_wbc", "preop_crp")
    cohorts = {}
    for scope, study in studies.items():
        roster = Path(study["manifest"]["branch_rosters"]["non_windowed"]["path"])
        context = _restrict_to_roster(source, roster)
        additions = _restrict_to_roster(derived, roster)
        cohort = context.merge(additions[["caseid", *derived_fields]], on="caseid", validate="one_to_one")
        expected = attrition["scopes"][scope]
        if (len(cohort), cohort.subjectid.nunique(), int(cohort.y_icu_admit.sum())) != (
            expected["operations"], expected["patients"], expected["outcomes"]["icu_admission"]["positive_operations"],
        ):
            raise ValueError(f"Unexpected cohort characteristics denominator for {scope}")
        cohorts[scope] = cohort
    shared, broader = cohorts["main_shared"], cohorts["icu_waveform_sensitivity"]
    if not set(shared.caseid).issubset(broader.caseid):
        raise ValueError("The broader ICU roster must include the shared EEG roster")
    if int(shared.aki_label.sum()) != attrition["scopes"]["main_shared"]["outcomes"]["any_aki"]["positive_operations"]:
        raise ValueError("Unexpected shared EEG AKI event count")
    comparison = pd.concat([
        _comparison_characteristics(shared, "Main shared cohort"),
        _comparison_characteristics(broader, "Broader ICU cohort"),
    ], axis=1).rename_axis("Characteristic").reset_index()
    return _primary_characteristics(shared), comparison


def performance_table(study, *, artifact_root: Path):
    source = study["metrics"].set_index(["outcome", "model_type", "feature_set", "branch"], verify_integrity=True)
    baseline_study = Study(artifact_root, "main_shared")
    metrics = [key for metric in METRICS for key in (metric, f"{metric}_lower", f"{metric}_upper")]
    displayed, full = [], []
    for outcome, outcome_label in OUTCOMES.items():
        for family, (family_label, _) in FAMILIES.items():
            baseline = verify_baselines({family: baseline_study}, outcome, family)
            matched = baseline["predictions_equal"] and baseline["fitting_metadata_equal"] and source.loc[
                (outcome, family, "preop_only", "non_windowed"), metrics
            ].equals(source.loc[(outcome, family, "preop_only", "windowed"), metrics])
            for features, input_label in INPUT_CONFIGURATIONS.items():
                for branch, branch_label in BRANCHES.items():
                    key = (outcome, family, features, branch)
                    scores = source.loc[key]
                    full.append(dict(zip(source.index.names, key, strict=True), **scores.to_dict()))
                    if features == "preop_only" and matched and branch == "windowed":
                        continue
                    label = input_label if features == "preop_only" and matched else f"{input_label} ({branch_label.lower()})"
                    displayed.append({
                        "Outcome": outcome_label, "Model": family_label, "Input configuration": label,
                        **{name: f"{scores[metric]:.3f}\n({scores[metric+'_lower']:.3f}–{scores[metric+'_upper']:.3f})"
                           for metric, name in METRICS.items()},
                    })
    return pd.DataFrame(displayed), pd.DataFrame(full)


def descriptor_table(extractor_version):
    if extractor_version == "0.4.5":
        table = pd.DataFrame(DESCRIPTORS.items(), columns=["Catch24 descriptor", "Description"])
    else:
        import pycatch22

        installed = version("pycatch22")
        if installed != extractor_version or not hasattr(pycatch22, "feature_table"):
            raise ValueError(f"EEG descriptor definitions require the frozen pycatch22 {extractor_version} build. "
                             f"Installed version is {installed}.")
        definitions = pycatch22.feature_table(catch24=True)
        table = pd.DataFrame(((entry["name"], entry["hctsa_call"]) for entry in definitions),
                             columns=["Catch24 descriptor", "hctsa reference call"])
        table.attrs["legacy_names"] = {entry["name"]: entry["legacy_name"] for entry in definitions}
    if len(table) != 24 or not table["Catch24 descriptor"].is_unique:
        raise ValueError(f"Invalid Catch24 definitions for pycatch22 {extractor_version}.")
    table.attrs["pycatch22_version"] = extractor_version
    return table


def predictor_tables(study, *, extractor_version, raw: Path, eeg_protocol=PROTOCOL):
    baseline = study["predictions"][("ebm", "any_aki", "non_windowed", "preop_only")]
    path = baseline.path.parent.parent / "artifacts/folds/repeat_00/outer_00/preprocessor.pkl"
    if not path.resolve().is_relative_to(study["root"].resolve()):
        raise ValueError(f"EEG predictor source escapes the saved study: {path}")
    fitted = joblib.load(path)
    fields = set(fitted.categorical_columns_) | set(fitted.numeric_columns_)
    if set(fitted.categorical_columns_) != CATEGORIES.keys():
        raise ValueError(f"Unexpected EEG categorical predictor fields: {path}")
    clinical, labs = (pd.read_csv(raw / name).set_index("Parameter", verify_integrity=True)
                      for name in PREDICTOR_DICTIONARIES)
    waveforms = [spec for spec in CHANNEL_SPECS.values()]
    waveforms.append(("Frontal EEG (EEG)", "BIS/EEG1_WAV, loading fallback BIS/EEG2_WAV", "128 Hz", "0.5–50 Hz band-pass"))
    channels = pd.DataFrame(waveforms, columns=["Waveform", "VitalDB track", "Original sampling rate", "Filter"])
    channels.attrs["eeg_protocol"] = eeg_protocol
    channels["Whole-operation rate"] = ["10 Hz"] * 4 + ["128 Hz"]
    channels["Windowed rate"] = ["100 Hz", "100 Hz", "62.5 Hz", "62.5 Hz", "128 Hz"]
    descriptors = descriptor_table(extractor_version)
    if "legacy_names" in descriptors.attrs:
        descriptions = descriptors["Catch24 descriptor"].map(descriptors.attrs["legacy_names"]).map(DESCRIPTORS)
        descriptors.insert(1, "Description", descriptions.fillna(descriptors["hctsa reference call"]))
    return clinical_predictor_table(fields, clinical, labs), channels, descriptors


def _table_1(doc, frame):
    table = add_table(doc, (3.43, 2.16, 1.47))
    row(table, frame.columns, header=True, compact=True)
    for values in frame.itertuples(index=False, name=None):
        row(table, values, band=values[0] in COHORT_BANDS, compact=True)
        for cell in table.rows[-1].cells[1:]:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if values[0] == "Procedure type":
            for run in table.rows[-1].cells[2].paragraphs[0].runs:
                run.bold = False
    note(doc, "Median (first, third quartiles) or n (%) of operations. " + AGE_NOTE + " "
         + abbreviation_key(*OUTCOME_KEY, "ASA-PS", "BMI", "ECG", "eGFR", "BUN", "WBC", "CRP"))
    return 1


def _table_2(doc, frame):
    count = 0
    for _, outcome_rows in frame.groupby("Outcome", sort=False):
        if count:
            heading(doc, "Table 2. Performance by input configuration (continued)", page_break=True)
        table = add_table(doc, (.78, .67, 2.40, 1.07, 1.07, 1.07))
        row(table, frame.columns, header=True, compact=True)
        for _, group in outcome_rows.groupby("Model", sort=False):
            scores = group[list(METRICS.values())].apply(lambda col: col.str.split("\n").str[0].astype(float))
            best = (scores.AP.max(), scores.AUROC.max(), scores["Brier score"].min())
            for i, values in enumerate(group.itertuples(index=False, name=None)):
                shown = values if i == 0 else ("", "", *values[2:])
                bold = tuple(3+j for j, (value, top) in enumerate(zip(scores.iloc[i], best, strict=True)) if value == top)
                row(table, shown, font_size=8.5, compact=True, bold_columns=bold)
        note(doc, CI_NOTE + " Bold marks the highest AP and AUROC and lowest Brier score within each outcome and model. "
             + abbreviation_key(*MODEL_KEY, "AP", "AUROC"))
        count += 1
    return count


def _predictors(doc, clinical, channels, descriptors):
    table = add_table(doc, (1.46, 1.71, 1.81, 2.08))
    row(table, ("Predictor", "Model input", "Unit or coding", "VitalDB clinical-table exceptions"),
        band=True, compact=True)
    categorical = clinical["Type or unit"].str.startswith("Categorical.")
    for label, fields in (("Categorical preoperative predictors", clinical[categorical]),
                          ("Numeric preoperative predictors", clinical[~categorical])):
        row(table, [label], band=True, merge=((0, 3),), compact=True)
        for name, field, unit, source in fields.itertuples(index=False, name=None):
            if source == "VitalDB clinical table":
                source = ""
            elif field in LABS:
                source = f"Lab records ({LABS[field]})"
            row(table, (name.removeprefix("Preoperative "), field,
                        unit.removeprefix("Categorical. ").removeprefix("Continuous, "), source), compact=True)
    row(table, ["Waveform sources and preprocessing"], band=True, merge=((0, 3),), compact=True)
    channels = channels[["Waveform", "VitalDB track", "Original sampling rate", "Filter"]]
    row(table, channels.columns, band=True, compact=True)
    for values in channels.itertuples(index=False, name=None):
        row(table, values, compact=True)
    row(table, ["Crop to the operation. For the original four waveforms, reject segments with more than 5% "
                "nonfinite samples and interpolate qualifying gaps before zero-phase Butterworth filtering "
                "(design N = 4) and polyphase antialiasing resampling."],
        merge=((0, 3),), compact=True)
    if channels.attrs.get("eeg_protocol") == AMPLITUDE_PROTOCOL:
        mask_note = ("For EEG, flag missing or nonfinite samples and amplitudes outside ±100 µV. BIS SQI is descriptive metadata. "
                     "This sample-level mask adapts amplitude limits used for segment or epoch rejection.")
    else:
        mask_note = ("For EEG, mask missing samples, SQI below 50 or unavailable within the preceding second, "
                     "and amplitudes outside ±100 µV.")
    row(table, [f"{mask_note} The combined mask may cover at most 5% of the grid. Interpolate with nearest endpoint fill "
                "before zero-phase Butterworth filtering (design N = 4)."], merge=((0, 3),), compact=True)
    row(table, ["Feature construction"], band=True, merge=((0, 3),), compact=True)
    row(table, ("", "Whole-operation", "Windowed"), band=True, merge=((2, 3),), compact=True)
    for values in (
        ("Sampling rate", "10 Hz original four, 128 Hz EEG", "100 Hz ECG/PPG, 62.5 Hz CO₂/airway pressure, 128 Hz EEG"),
        ("Feature extraction", "Catch24 across the complete operative waveform", "Catch24 on 10-second windows advancing about 5 seconds"),
        ("Aggregation", "None", "Mean, sample SD, minimum and maximum across windows"),
        ("Features per channel", "24", "96"),
        ("Total with EEG", "120 (96 from original four)", "480 (384 from original four)"),
    ):
        row(table, values, merge=((2, 3),), compact=True)
    row(table, ["Window stride is 500 samples at 100 Hz, 312 at 62.5 Hz, and 640 at 128 Hz. "
                "Single-window SD is missing. Undefined descriptors propagate to aggregates."],
        merge=((0, 3),), compact=True)
    row(table, ["Catch24 descriptors"], band=True, merge=((0, 3),), compact=True)
    row(table, ("Canonical descriptor", "Description"), band=True, merge=((0, 1), (2, 3)), compact=True)
    for name, description in descriptors[["Catch24 descriptor", "Description"]].itertuples(index=False, name=None):
        row(table, (name, description), merge=((0, 1), (2, 3)), font_size=8.5, compact=True)
    note(doc, f"The preoperative feature set contained {len(clinical)} raw predictors. The number of encoded inputs varied "
         f"across folds and model families. Waveform descriptors use the frozen pycatch22 "
         f"{descriptors.attrs['pycatch22_version']} build. Catch22 descriptors use internally standardized signals.")
    note(doc, "EEG, electroencephalography. SQI, signal quality index. "
         + abbreviation_key("ALT", "ASA-PS", "AST", "aPTT", "CKD-EPI", "PaCO₂", "PaO₂", "PT", "SD", "SpO₂"))
    return 1


def add_table_display(doc, label, frames):
    """Append one numbered display below the caller's title and return the Word table count."""
    if label == "Table 1":
        return _table_1(doc, *frames)
    if label == "Table 2":
        return _table_2(doc, *frames)
    if label == "Table S1":
        return _predictors(doc, *frames)
    if label == "Table S2":
        add_frame(doc, frames[0], (3.84, 1.61, 1.61))
        note(doc, "Median (first, third quartiles) or n (%) of operations. The cohorts overlap. " + AGE_NOTE
             + " " + abbreviation_key(*OUTCOME_KEY, "BMI", "ASA-PS"))
        return 1
    raise ValueError(f"Unknown EEG manuscript table: {label}")
