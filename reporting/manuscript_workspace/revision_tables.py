"""Build editable publication tables from fitted inputs and frozen cohort rosters."""

from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from reporting.preop_descriptives import _ecg_display
from .revision_data import (
    BRANCHES, CHANNELS, FAMILIES, INPUTS, METRICS, OUTCOMES, digest, metric_row, verify_baselines,
)

CHANNEL_SPECS = {
    "SNUADC_ECG_II": ("Electrocardiogram (ECG)", "SNUADC/ECG_II; loading fallback SNUADC/ECG_V5",
                     "500 Hz", "0.5–40 Hz band-pass"),
    "SNUADC_PLETH": ("Photoplethysmogram (PPG/PLETH)", "SNUADC/PLETH",
                    "500 Hz", "0.1–10 Hz band-pass"),
    "Primus_CO2": ("Capnography (CO₂)", "Primus/CO2", "62.5 Hz", "8 Hz low-pass"),
    "Primus_AWP": ("Airway pressure (AWP)", "Primus/AWP", "62.5 Hz", "12 Hz low-pass"),
}
AGGREGATES = ("mean", "std", "min", "max")
SINGLE_CHANNEL = {
    "ecg_only": "SNUADC_ECG_II", "pleth_only": "SNUADC_PLETH",
    "co2_only": "Primus_CO2", "awp_only": "Primus_AWP",
}
DESCRIPTORS = {
    "DN_HistogramMode_5": "Amplitude mode using five bins",
    "DN_HistogramMode_10": "Amplitude mode using ten bins",
    "CO_f1ecac": "First autocorrelation crossing below 1/e",
    "CO_FirstMin_ac": "Lag of the first autocorrelation minimum",
    "CO_HistogramAMI_even_2_5": "Lag-two auto-mutual information using five bins",
    "CO_trev_1_num": "Mean cubed successive difference (time reversibility)",
    "MD_hrv_classic_pnn40": "Fraction of absolute successive differences above 0.04 standardized units",
    "SB_BinaryStats_mean_longstretch1": "Longest above-mean run statistic in pycatch22 run-count units",
    "SB_TransitionMatrix_3ac_sumdiagcov": "Trace of the three-symbol transition covariance",
    "PD_PeriodicityWang_th0_01": "Periodicity from detrended autocorrelation with a 0.01 threshold",
    "CO_Embed2_Dist_tau_d_expfit_meandiff": "Mean discrepancy between embedding distances and an exponential fit",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi": "First minimum of Gaussian-estimated auto-mutual information",
    "FC_LocalSimple_mean1_tauresrat": "Ratio of residual to original autocorrelation times after one-step prediction",
    "DN_OutlierInclude_p_001_mdrmd": "Timing of large positive excursions across thresholds",
    "DN_OutlierInclude_n_001_mdrmd": "Timing of large negative excursions across thresholds",
    "SP_Summaries_welch_rect_area_5_1": "Power in the lowest fifth of frequencies",
    "SB_BinaryStats_diff_longstretch0": "Longest non-increasing stretch statistic",
    "SB_MotifThree_quantile_hh": "Entropy of successive pairs of three equiprobable symbols",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": "Fraction of scales before the rescaled-range crossover",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": "Fraction of scales before the detrended-fluctuation crossover",
    "SP_Summaries_welch_rect_centroid": "Frequency at half the cumulative spectral power",
    "FC_LocalSimple_mean3_stderr": "Standard deviation of three-sample moving-mean prediction errors",
    "DN_Mean": "Mean in filtered, resampled signal units",
    "DN_Spread_Std": "Standard deviation in filtered, resampled signal units",
}
CATEGORIES = {
    "sex": ("Sex", "Female or male"),
    "emop": ("Emergency surgery", "0/1"),
    "department": ("Surgical department", "Source department; rare levels pooled during training"),
    "approach": ("Surgical approach", "Open, videoscopic or robotic"),
    "asa": ("ASA-PS", "Classes I–IV in included operations"),
    "optype": ("Procedure type", "Source procedure category"),
    "ane_type": ("Anesthesia type", "General anesthesia in both cohorts"),
    "preop_htn": ("Hypertension", "0/1"),
    "preop_dm": ("Diabetes", "0/1"),
    "preop_ecg": ("Preoperative ECG finding", "Source ECG finding text encoded as categories"),
    "preop_pft": ("Pulmonary function test", "Source pulmonary function category"),
}
SPECIAL_CLINICAL = {
    "preop_egfr_ckdepi_2021": ("Estimated glomerular filtration rate", "Continuous, mL/min/1.73 m²",
                              "2021 CKD-EPI creatinine equation using source creatinine, age and sex"),
    "inpatient_preop": ("Admission before case recording", "Binary (0/1)",
                       "Derived from source admission time (adm < 0); unknown time remains missing"),
    "preop_sao2": ("Source preoperative oxygen saturation", "Continuous, %",
                   "VitalDB clinical field, listed as SpO₂ in the source dictionary"),
}
LABS = {"preop_wbc": "wbc", "preop_crp": "crp", "preop_lac": "lac"}
CLINICAL_LABELS = {"preop_alt": "ALT", "preop_ast": "AST"}


def expanded_performance(metrics, studies):
    rows, baselines = [], []
    for outcome, outcome_label in OUTCOMES.items():
        for family, (family_label, _) in FAMILIES.items():
            evidence = verify_baselines(studies, outcome, family)
            baselines.append(evidence)
            common = evidence["predictions_equal"] and evidence["fitting_metadata_equal"]
            for inputs, label in INPUTS.items():
                for branch, branch_label in BRANCHES.items():
                    if inputs == "preop_only" and common and branch == "windowed":
                        continue
                    row = metric_row(metrics, family, outcome, branch, inputs)
                    configuration = label if inputs == "preop_only" and common else f"{label} ({branch_label.lower()})"
                    rows.append({
                        "Outcome": outcome_label, "Model": family_label, "Input configuration": configuration,
                        **{name: f"{row[key]:.3f}\n({row[key+'_lower']:.3f}–{row[key+'_upper']:.3f})"
                           for key, name in METRICS.items()},
                    })
    return pd.DataFrame(rows), baselines


def fitted_predictors(studies):
    clinical = set(CATEGORIES)
    expected = {
        branch: {f"{prefix}_{feature}{suffix}" for prefix in CHANNEL_SPECS for feature in DESCRIPTORS
                 for suffix in (("",) if branch == "non_windowed" else tuple("_" + a for a in AGGREGATES))}
        for branch in BRANCHES
    }
    raw_clinical, dimensions, audited = set(), defaultdict(list), {}
    for scope, selected in studies.items():
        inputs_in_scope = {*INPUTS, *CHANNELS} if scope == "main_shared" else {"preop_only", "preop_and_all_waveforms"}
        for family, study in selected.items():
            for k, outcome, branch, inputs in study.configs:
                if k != family or inputs not in inputs_in_scope:
                    continue
                wave_inputs = expected[branch]
                if inputs == "preop_only":
                    wave_inputs = set()
                elif inputs in SINGLE_CHANNEL:
                    wave_inputs = {col for col in wave_inputs if col.startswith(SINGLE_CHANNEL[inputs] + "_")}
                path = study.artifact(k, outcome, branch, inputs) / "artifacts/folds"
                files = sorted(path.glob("repeat_*/outer_*/preprocessor.pkl"))
                if len(files) != 5:
                    raise ValueError(f"Expected five fitted preprocessors under {path}")
                for file in files:
                    pre = joblib.load(file)
                    wave = {c for c in pre.numeric_columns_ if c.startswith(tuple(p+"_" for p in CHANNEL_SPECS))}
                    raw = (set(pre.numeric_columns_) - wave) | set(pre.categorical_columns_)
                    if wave != wave_inputs:
                        raise ValueError(f"Waveform inventory mismatch in {file}")
                    if inputs not in CHANNELS:
                        if (raw_clinical and raw_clinical != raw) or set(pre.categorical_columns_) != clinical:
                            raise ValueError(f"Clinical inventory mismatch in {file}")
                        raw_clinical = raw
                    elif raw:
                        raise ValueError(f"Unexpected clinical inputs in {file}")
                    dimensions[(scope, family, branch, inputs)].append(len(pre.feature_names_))
                    audited[str(file)] = digest(file)
    return raw_clinical, [{"scope": scope, "family": family, "branch": branch, "inputs": inputs,
                           "min": min(dims), "max": max(dims), "folds": len(dims)}
                          for (scope, family, branch, inputs), dims in sorted(dimensions.items())], audited


def build_predictors(studies, workspace):
    raw, dimensions, audited = fitted_predictors(studies)
    clinical = pd.read_csv(workspace / "04_DATA/raw_source/clinical_parameters.csv").set_index("Parameter")
    labs = pd.read_csv(workspace / "04_DATA/raw_source/lab_parameters.csv").set_index("Parameter")
    observed = []
    for field in sorted(raw, key=lambda k: (k not in CATEGORIES, k)):
        if field in CATEGORIES:
            label, meaning = CATEGORIES[field]
            row = label, field, "Categorical. " + meaning, "VitalDB clinical table"
        elif field in SPECIAL_CLINICAL:
            label, meaning, timing = SPECIAL_CLINICAL[field]
            row = label, field, meaning, timing
        elif field in LABS:
            source = labs.loc[LABS[field]]
            unit = str(source.Unit).replace("×1000/mcL", "10³/µL")
            row = source.Description, field, f"Continuous, {unit}", f"Laboratory record ({LABS[field]}), latest finite result in [−30 days, 0) relative to case recording"
        else:
            source = clinical.loc[field]
            unit = str(source.Unit).replace("kg/m2", "kg/m²").replace("x1000/mcL", "10³/µL")
            label = CLINICAL_LABELS.get(field, source.Description)
            row = label, field, f"Continuous, {unit}" if pd.notna(source.Unit) else "Continuous, unitless", "VitalDB clinical table"
        observed.append(row)
    return {
        "clinical": pd.DataFrame(observed, columns=["Predictor", "Model input", "Type or unit", "Source and timing"]),
        "channels": pd.DataFrame([spec for spec in CHANNEL_SPECS.values()], columns=[
            "Waveform", "VitalDB track", "Original sampling rate", "Filter"]),
        "descriptors": pd.DataFrame(DESCRIPTORS.items(), columns=["Canonical descriptor", "Description"]),
        "dimensions": dimensions, "audited": audited,
    }


def build_cohort(core, broader, workspace):
    main = core.manifest["branch_rosters"]["non_windowed"]["caseids"]
    broad = broader.manifest["branch_rosters"]["non_windowed"]["caseids"]
    if not set(main) < set(broad):
        raise ValueError("The broader ICU cohort must contain the main cohort")
    for study, roster in ((core, main), (broader, broad)):
        if study.manifest["branch_rosters"]["windowed"]["caseids"] != roster:
            raise ValueError("Cohort table requires the same roster in both extraction branches")
    sources = {
        "master": workspace / "04_DATA/processed/aki_features_master_wide.csv",
        "context": workspace / "04_DATA/processed/aki_pleth_ecg_co2_awp.csv",
    }
    master = pd.read_csv(sources["master"]).set_index("caseid").loc[broad]
    context = pd.read_csv(sources["context"]).set_index("caseid").loc[broad]
    if not master.subjectid.equals(context.subjectid) or not master.y_icu_admit.eq(context.y_icu_admit).all():
        raise ValueError("Frozen patient or outcome records disagree")
    ecg = _ecg_display(master.loc[main, "preop_ecg"])
    if not ecg.equals(_ecg_display(context.loc[main, "preop_ecg"])):
        raise ValueError("Frozen main-cohort ECG classifications disagree")
    master["duration"] = (context.opend - context.opstart) / 60
    for field in ("age", "bmi", "preop_cr", "preop_crp"):
        master[field] = pd.to_numeric(master[field], errors="coerce")
    for field in ("aki_has_baseline_cr", "aki_has_postop_cr_72h", "eligible_any_aki"):
        master[field] = context[field]
    outputs = []
    for study, roster, label in ((core, main, "Main shared cohort"), (broader, broad, "Broader ICU cohort")):
        d = master.loc[roster]
        n = len(d)
        r = study.manifest["branch_rosters"]["non_windowed"]
        if (n, d.subjectid.nunique(), int(d.y_icu_admit.sum())) != (
            r["operations"], r["patients"], r["outcomes"]["icu_admission"]["events"]
        ):
            raise ValueError(f"Unexpected denominator or count for {label}")
        def count(mask):
            value = int(mask.sum())
            return f"{value:,} ({100*value/n:.1f}%)"
        def median(field, precision=0):
            q = d[field].quantile([.25, .5, .75]).to_list()
            return f"{q[1]:.{precision}f} ({q[0]:.{precision}f}, {q[2]:.{precision}f})"
        outputs.append(pd.Series({
            "Operations": f"{n:,}",
            "Unique patients": f"{d.subjectid.nunique():,}",
            "ICU admission": count(d.y_icu_admit.eq(1)),
            "Age, years": median("age"),
            "Male sex": count(d.sex.eq("M")),
            "BMI, kg/m²": median("bmi", 1),
            "Emergency surgery": count(d.emop.eq(1)),
            "ASA-PS III–IV": count(d.asa.isin([3, 4])),
            "Open surgical approach": count(d.approach.eq("Open")),
            "Operative duration, min": median("duration"),
            "Colorectal procedure": count(d.optype.eq("Colorectal")),
            "Transplantation": count(d.optype.eq("Transplantation")),
            "Hypertension": count(d.preop_htn.eq(1)),
            "Diabetes": count(d.preop_dm.eq(1)),
            "Preoperative creatinine, mg/dL": median("preop_cr", 2),
            "Source baseline creatinine absent or invalid": count(~d.aki_has_baseline_cr.eq(True)),
            "Baseline creatinine >4 mg/dL": count(d.preop_cr.gt(4)),
            "No finite positive creatinine result by 72 h after operation end": count(~d.aki_has_postop_cr_72h.eq(True)),
            "AKI ascertainment unavailable under study criteria": count(~d.eligible_any_aki.eq(True)),
            "C-reactive protein unavailable": count(~np.isfinite(d.preop_crp)),
        }, name=label))
    cohort = pd.concat(outputs, axis=1).rename_axis("Characteristic").reset_index()
    ecg_counts = ecg.value_counts().reindex(["Normal sinus rhythm", "Abnormal", "Unknown", "Missing"],
                                            fill_value=0).to_dict()
    return cohort, sources, ecg_counts
