"""Use manuscript EBM explanation layouts with the EEG study's saved fits."""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator
from scipy.special import expit

from reporting.display_dictionary import load_display_dictionary
from reporting.eeg_explanations import explanation_sources
from reporting.manuscript_workspace import explanation_figures
from reporting.manuscript_workspace.revision_data import OUTCOMES

RESPONSE_PANELS = (
    ("A", "any_aki", "preop_hb", (.01, .99), "AKI · Hemoglobin", "Hemoglobin (g/dL)", (10, 13, 16)),
    ("C", "icu_admission", "age", (.05, .95), "ICU admission · Age", "Age (years)", (45, 60, 75)),
    ("D", "icu_admission", "preop_egfr_ckdepi_2021", (.05, .95), "ICU admission · eGFR",
     "eGFR (mL/min/1.73 m²)", (65, 80, 95, 110)),
)


def manuscript_explanation_sources(study, branch):
    ranges = {}
    for _, outcome, feature, limits, *_ in RESPONSE_PANELS:
        ranges.setdefault(outcome, {})[feature] = limits
    return explanation_sources(study, branch, manuscript_ranges=ranges)


def _importance_label(feature):
    if feature.startswith("optype_"):
        procedure = feature.removeprefix("optype_")
        return "Colorectal procedure" if procedure == "Colorectal" else procedure
    if feature.startswith("department_"):
        return f"{feature.removeprefix('department_')} department"
    if feature.startswith("asa_"):
        return f"ASA-PS {_asa_class(feature.removeprefix('asa_'))}"
    label = load_display_dictionary().feature_label(feature).replace(" — ", " · ")
    return (label.replace("AWP · ", "Airway pressure · ").replace("ECG II · ", "ECG · ")
            .replace("EtCO₂ · ", "Capnography · ").replace("Detrended fluctuation (50%)", "DFA crossover fraction")
            .replace("DFA fluctuation (50%)", "DFA crossover fraction")
            .replace("Three-state transition covariance trace", "Transition-matrix covariance trace"))


def importance_figure(sources):
    selected = sources["importance"].loc[lambda rows: rows["rank"].le(10)].copy()
    selected["display_label"] = selected.feature.map(_importance_label)
    selected["fold_number"] = selected.fold + 1
    display = selected.rename(columns={"rank": "display_rank", "importance": "native_importance",
                                       "mean_importance": "mean_native_importance"})
    panels = [(outcome, OUTCOMES[outcome], max(baseline, 1.06 * display.loc[display.outcome.eq(outcome),
                                                                         "native_importance"].max()))
              for outcome, baseline in (("any_aki", .72), ("icu_admission", .35))]
    fig = explanation_figures.term_importance_figure(display, panels, show_absent=not selected.term_present.all())
    return fig, selected


def _response_rows(data):
    return pd.concat([data.loc[data.outcome.eq(outcome) & data.feature.eq(feature)].assign(panel=panel)
                      for panel, outcome, feature, *_ in RESPONSE_PANELS], ignore_index=True)


def manuscript_response_support(sources):
    return _response_rows(sources["manuscript_support"].loc[lambda rows: rows.fold.eq(0)])


def response_figure(sources):
    data = _response_rows(sources["manuscript_responses"]).assign(
        raw_contribution_minus_reference=lambda rows: rows.raw_log_odds_minus_reference)
    support = manuscript_response_support(sources)
    panels = [(panel, heading, xlabel, FixedLocator(ticks))
              for panel, _, _, _, heading, xlabel, ticks in RESPONSE_PANELS]
    return explanation_figures.response_figure(data, support, "panel", panels), data


def _asa_class(value):
    value = str(value).removesuffix(".0")
    return {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V"}.get(value, value)


def _case_label(group, value):
    fixed = {"Other clinical terms": "Other\npreoperative", "CO₂": "Capnography",
             "Probability clipping correction": "Calibration\nclipping"}
    if group in fixed:
        return fixed[group]
    if group in ("ECG", "PPG", "Airway pressure", "EEG"):
        return group
    if group in ("optype", "department"):
        prefix = "Procedure" if group == "optype" else "Department"
        return f"{prefix}:\n{'missing' if pd.isna(value) else value}"
    if group == "asa":
        return f"ASA-PS: {'missing' if pd.isna(value) else _asa_class(value)}"
    shorthand = {"preop_wbc": ("WBC", ".2f"), "preop_hb": ("Hb", ".1f"), "preop_bun": ("BUN", ".0f"),
                 "preop_alt": ("ALT", ".0f"), "age": ("Age", ".0f")}
    fallback = load_display_dictionary().feature_label(group, use_short=True)
    label, precision = shorthand.get(group, (fallback, ".3g"))
    if pd.isna(value):
        return f"{label}: missing"
    if isinstance(value, (float, int, np.number)):
        value = format(value, precision)
    if group == "age":
        value = f"{value} years"
    return textwrap.fill(f"{label}: {value}", width=21, break_long_words=False)


def heldout_figure(sources):
    selected = sources["case_contributions"].copy()
    selected["display_label"] = [_case_label(group, value) for group, value in
                                 zip(selected.group, selected.raw_group_value, strict=True)]
    channel_order = {group: order for order, group in enumerate((
        "ECG", "PPG", "CO₂", "Airway pressure", "EEG", "Probability clipping correction",
    ), 1)}
    panels, plotted = [], []
    for (outcome, quantile), rows in selected.groupby(["outcome", "quantile"], sort=False):
        rows = rows.sort_values("group", key=lambda group: group.map(channel_order).fillna(0), kind="stable")
        header = rows.iloc[0]
        ends = header.calibrated_intercept + rows.contribution.cumsum()
        rows = rows.assign(start=ends.shift(fill_value=header.calibrated_intercept), end=ends)
        final = ends.iloc[-1]
        if not np.isclose(expit(final), header.calibrated_probability, atol=1e-12, rtol=0):
            raise ValueError(f"Grouped EBM contributions do not reconstruct the saved risk for {outcome} {quantile:.0%}")
        status = "Yes" if header.observed_outcome else "No"
        panels.append({"title": f"{OUTCOMES[outcome]} · {quantile:.0%}".removesuffix("%") + "th percentile",
                       "summary": f"Preoperative only {header.clinical_probability:.1%}\n"
                                  f"Combined {header.calibrated_probability:.1%}  |  Outcome: {status}",
                       "labels": list(rows.display_label), "starts": rows.start, "ends": rows.end,
                       "intercept": header.calibrated_intercept, "final": final})
        plotted.append(rows)
    fig, _ = explanation_figures.heldout_waterfalls(panels)
    return fig, pd.concat(plotted, ignore_index=True)
