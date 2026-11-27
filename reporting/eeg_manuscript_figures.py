"""Manuscript performance panels populated with matched EEG-study results."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from reporting import publication_style as style
from reporting.manuscript_workspace import revision_figures as manuscript
from reporting.manuscript_workspace.revision_data import BRANCHES, FAMILIES, OUTCOMES
from results_recreation.comparisons import EEG_BASELINE_PAIRS, EEG_CORE, EEG_INCREMENT_PAIRS

INPUTS = {
    "preop_only": "Preoperative only",
    "original_four_waveforms": "Original four",
    "all_five_waveforms": "All five",
    "preop_and_original_four": "Preoperative + four",
    "preop_and_all_five": "Preoperative + five",
}
CHANNELS = {"ecg_only": "ECG", "pleth_only": "PPG", "co2_only": "CO₂", "awp_only": "Airway pressure",
            "eeg_only": "EEG", "original_four_waveforms": "All four", "all_five_waveforms": "All five"}
CALIBRATION_INPUTS = (EEG_CORE[0], *EEG_CORE[-2:])
CONTRASTS = dict(zip((*EEG_BASELINE_PAIRS, *EEG_INCREMENT_PAIRS),
                     ("(P + W4) − P", "(P + W5) − P", "(P + W5) − (P + W4)", "W5 − W4"), strict=True))
METRICS = {"auprc": "AP", "auroc": "AUROC"}


def input_line(key):
    basis = ("preop_only", "all_waveforms", "all_waveforms", "preop_and_all_waveforms", "preop_and_all_waveforms")
    index = EEG_CORE.index(key)
    options = manuscript.input_line(basis[index])
    if index in (2, 4):
        options["ls"] = "-"
    return options


def _reference(label, *, linestyle="--"):
    return Line2D([0], [0], color=style.color("reference"), ls=linestyle, label=label)


def _input_legend(fig, inputs, *references, columns=None):
    handles = [Line2D([0], [0], **input_line(key), label=INPUTS[key]) for key in inputs]
    handles.extend(references)
    columns = columns or len(handles)
    # Matplotlib lays out legend handles by column. Read across rows in score order.
    ordered = [handle for col in range(columns) for handle in handles[col::columns]]
    fig.legend(handles=ordered, loc="outside lower center", ncols=columns)


def _pairs(frame, pairs):
    matched = frame[frame.reference_model.eq(frame.candidate_model)]
    return pd.concat([
        matched[matched.reference_feature_set.eq(ref) & matched.candidate_feature_set.eq(candidate)]
        .assign(contrast=CONTRASTS[ref, candidate]) for ref, candidate in pairs
    ], ignore_index=True)


def discrimination(curves, branch, kind):
    if kind == "calibration":
        return calibration(curves, branch)
    selected = curves[kind].loc[lambda df: df.branch.eq(branch) & df.feature_set.isin(EEG_CORE)].copy()
    outcomes = {key: label for key, label in OUTCOMES.items() if key in set(selected.outcome)}
    fig, axes = manuscript.setup(len(outcomes), 3, 80 if len(outcomes) == 1 else 155,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (outcome, title) in enumerate(outcomes.items()):
        for j, (family, (short, _)) in enumerate(FAMILIES.items()):
            ax = axes[i, j]
            frame = selected[selected.outcome.eq(outcome) & selected.model_type.eq(family)]
            scores = []
            for feature in EEG_CORE:
                rows = frame[frame.feature_set.eq(feature)]
                ax.plot(rows.x, rows.y, **input_line(feature))
                scores.append(f"{rows.score.iloc[0]:.3f}")
            if kind == "pr":
                ax.axhline(frame.prevalence.iloc[0], color=style.color("reference"), ls="--", lw=.7)
                xlabel, ylabel = "Recall", "Precision"
            else:
                ax.plot([0, 1], [0, 1], color=style.color("reference"), ls="--", lw=.7)
                xlabel, ylabel = "False-positive rate", "True-positive rate"
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.text(.03, .05 if kind == "roc" else .09, " / ".join(scores[:3]) + "\n" + " / ".join(scores[3:]),
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1})
            manuscript.panel(ax, i, j, 3, f"{title} · {short}")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(outcomes) - 1:
                ax.set_xlabel(xlabel)
    _input_legend(fig, EEG_CORE, _reference("Chance" if kind == "roc" else "Outcome prevalence"), columns=3)
    return fig, selected


def calibration(curves, branch):
    def inputs_only(frame):
        return frame[frame.branch.eq(branch) & frame.feature_set.isin(CALIBRATION_INPUTS)]

    selected, support = (inputs_only(curves[name]) for name in ("calibration", "risk_support"))
    outcomes = {key: label for key, label in OUTCOMES.items() if key in set(selected.outcome)}
    fig = plt.figure(figsize=(style.WIDTH_MM / 25.4, (175 if len(outcomes) == 2 else 91) / 25.4), layout="constrained")
    grid = fig.add_gridspec(len(outcomes), 3)
    for i, (outcome, title) in enumerate(outcomes.items()):
        histograms = []
        for j, (family, (short, _)) in enumerate(FAMILIES.items()):
            sub = grid[i, j].subgridspec(2, 1, height_ratios=(3.2, 1))
            ax = fig.add_subplot(sub[0])
            hist = fig.add_subplot(sub[1], sharex=ax)
            histograms.append(hist)
            for plot in (ax, hist):
                plot.spines[["top", "right"]].set_visible(False)
            inset = ax.inset_axes([.55, .09, .42, .40]) if outcome == "any_aki" else None
            for feature, marker in zip(CALIBRATION_INPUTS, ("o", "^", "s"), strict=True):
                frame = selected[selected.outcome.eq(outcome) & selected.model_type.eq(family) & selected.feature_set.eq(feature)]
                ax.plot(frame.x, frame.y, marker=marker, ms=3, **input_line(feature))
                if inset is not None:
                    inset.plot(frame.x, frame.y, marker=marker, ms=2, **input_line(feature))
                counts = support[support.outcome.eq(outcome) & support.model_type.eq(family) & support.feature_set.eq(feature)]
                hist.stairs(counts.bin_n, np.r_[counts.bin_lower, counts.bin_upper.iloc[-1]], **input_line(feature))
            if inset is not None:
                inset.plot([0, .35], [0, .35], color=style.color("reference"), ls="--", lw=.7)
                inset.set(xlim=(0, .35), ylim=(0, .35), xticks=[0, .35], yticks=[0, .35], title="AKI detail")
                inset.tick_params(labelsize=8, pad=1, length=1.5)
            ax.plot([0, 1], [0, 1], color=style.color("reference"), ls="--", lw=.7)
            ax.set(xlim=(0, 1), ylim=(0, 1), yticks=[0, .5, 1])
            ax.tick_params(labelbottom=False)
            hist.set(xlim=(0, 1), xticks=[0, .5, 1], xlabel="Predicted risk")
            hist.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            if j == 0:
                ax.set_ylabel("Observed risk")
                hist.set_ylabel("Operations")
            manuscript.panel(ax, i, j, 3, f"{title} · {short}")
        maximum = max(hist.get_ylim()[1] for hist in histograms)
        for hist in histograms:
            hist.set_ylim(0, maximum)
    _input_legend(fig, CALIBRATION_INPUTS, _reference("Perfect calibration"))
    return fig, selected


def contrasts(studies, *, icu_only):
    pairs = (EEG_BASELINE_PAIRS[-1], EEG_INCREMENT_PAIRS[0]) if icu_only else tuple(CONTRASTS)
    data = _pairs(pd.concat([s["comparisons"]["paired"] for s in studies.values()], ignore_index=True), pairs)
    data = data[data.metric.isin(METRICS)].copy()
    if icu_only:
        data = data[data.outcome.eq("icu_admission")].copy()
    for column in ("delta", "lower", "upper"):
        data[f"{column}_pp"] = 100 * data[column]
    cohorts = {"main_shared": ("Shared AKI/ICU", "#385A92"),
               "icu_waveform_sensitivity": ("Broader ICU", "#A34D27")}
    row_labels = BRANCHES if icu_only else OUTCOMES
    colors = [style.color(key) for key in ("orange", "green", "purple", "blue")]
    fig, axes = manuscript.setup(2, 2, 153 if icu_only else 158, sharex="col" if icu_only else False)
    for i, (key, title) in enumerate(row_labels.items()):
        group = data[data["branch" if icu_only else "outcome"].eq(key)]
        for j, (metric, name) in enumerate(METRICS.items()):
            ax = axes[i, j]
            for row in group[group.metric.eq(metric)].itertuples():
                pair = row.reference_feature_set, row.candidate_feature_set
                index = pairs.index(pair)
                y = tuple(FAMILIES).index(row.candidate_model)
                if icu_only:
                    y += (index - .5) * .40 + (-.088 if row.cohort_scope == "main_shared" else .088)
                    ink = cohorts[row.cohort_scope][1]
                    encoding = {"marker": "o" if index == 0 else "D", "filled": index == 0}
                else:
                    y += (index - 1.5) * .18 + (-.039 if row.branch == "non_windowed" else .039)
                    ink = colors[index]
                    encoding = style.BRANCHES[row.branch]
                manuscript.point(ax, row.delta_pp, row.lower_pp, row.upper_pp, y, encoding, color=ink)
            ax.axvline(0, color=style.color("reference"), ls="--", lw=.8)
            ax.set(yticks=range(len(FAMILIES)), yticklabels=[family[0] for family in FAMILIES.values()], ylim=(2.55, -.55))
            if not icu_only or i == 1:
                ax.set_xlabel(f"Δ{name} (percentage points)")
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="x", color=style.color("grid"), lw=.6)
            manuscript.panel(ax, i, j, 2, f"{title} · {name}" if icu_only else title)
    if icu_only:
        handles = [Line2D([0], [0], color=ink, label=label) for label, ink in cohorts.values()]
        handles += [Line2D([0], [0], color=style.color("ink"), marker=marker, mfc="white" if i else style.color("ink"),
                           ls="none", label=CONTRASTS[pair]) for i, (pair, marker) in enumerate(zip(pairs, ("o", "D"), strict=True))]
    else:
        handles = [Line2D([0], [0], color=ink, label=CONTRASTS[pair]) for pair, ink in zip(pairs, colors, strict=True)]
        handles += [Line2D([0], [0], color=style.color("ink"), marker=style.BRANCHES[key]["marker"], ls="none",
                           mfc=style.color("ink") if style.BRANCHES[key]["filled"] else "white", label=title)
                    for key, title in BRANCHES.items()]
    fig.legend(handles=[*handles, _reference("Zero difference")], loc="outside lower center", ncols=3 if icu_only else 4)
    return fig, data


def channels(study, metric):
    selected = study["metrics"].loc[lambda df: df.feature_set.isin(CHANNELS)].copy()
    outcomes = {key: label for key, label in OUTCOMES.items() if key in set(selected.outcome)}
    fig, axes = manuscript.setup(len(outcomes), 3, 87 if len(outcomes) == 1 else 154, sharey=True, squeeze=False)
    selected["plot_reference"] = selected.prevalence if metric == "auprc" else .5
    for i, (outcome, title) in enumerate(outcomes.items()):
        frame = selected[selected.outcome.eq(outcome)]
        (reference,) = frame.plot_reference.unique()
        low = min(reference, frame[f"{metric}_lower"].min())
        high = max(reference, frame[f"{metric}_upper"].max())
        pad = (high - low) * .09
        for j, (family, (short, _)) in enumerate(FAMILIES.items()):
            ax = axes[i, j]
            for row in frame[frame.model_type.eq(family)].itertuples():
                y = tuple(CHANNELS).index(row.feature_set) + (-.14 if row.branch == "non_windowed" else .14)
                ink = style.color("purple") if row.feature_set == "eeg_only" else None
                manuscript.point(ax, *(getattr(row, metric + suffix) for suffix in ("", "_lower", "_upper")),
                                 y, style.BRANCHES[row.branch], color=ink)
            ax.axvline(reference, color=style.color("reference"), ls="--", lw=.8)
            ax.set(yticks=range(len(CHANNELS)), yticklabels=list(CHANNELS.values()), ylim=(6.65, -.65),
                   xlim=(max(0, low - pad), min(1, high + pad)), xlabel="Average precision" if metric == "auprc" else "AUROC")
            ax.tick_params(axis="y", labelleft=j == 0)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis="x", color=style.color("grid"), lw=.6)
            manuscript.panel(ax, i, j, 3, f"{title} · {short}")
    manuscript.branch_legend(fig, "Outcome prevalence" if metric == "auprc" else "Chance (0.50)")
    return fig, selected


def decision_curves(study, branch, *, paired):
    key = "dca_comparisons" if paired else "dca"
    table = study["comparisons"][key].loc[lambda df: df.branch.eq(branch)]
    pairs = EEG_BASELINE_PAIRS[-1], EEG_INCREMENT_PAIRS[0]
    table = _pairs(table, pairs) if paired else table[table.feature_set.isin(CALIBRATION_INPUTS)].copy()
    data = table.assign(threshold_percent=100 * table.threshold_probability)
    endpoints = ("delta", "lower", "upper") if paired else ("model", "treat_all")
    for column in endpoints:
        data[f"{column}_per_100"] = 100 * data[column]
    outcomes = {key: label for key, label in OUTCOMES.items() if key in set(data.outcome)}
    fig, axes = manuscript.setup(len(outcomes), 3, 77 if len(outcomes) == 1 else 145, squeeze=False)
    colors = (style.color("green"), style.color("purple"))
    for i, (outcome, title) in enumerate(outcomes.items()):
        frame = data[data.outcome.eq(outcome)]
        if paired:
            low, high = min(0, frame.lower_per_100.min()), max(0, frame.upper_per_100.max())
            pad = .08 * (high - low)
            low, high = low - pad, high + pad
        else:
            low = min(-.25 if outcome == "any_aki" else -1, frame.model_per_100.min() - .15)
            high = max(frame.model_per_100.max(), frame.treat_all_per_100.max()) * 1.09
        for j, (family, (short, _)) in enumerate(FAMILIES.items()):
            ax = axes[i, j]
            selected = frame[frame["candidate_model" if paired else "model_type"].eq(family)]
            if paired:
                for (ref, candidate), ink in zip(pairs, colors, strict=True):
                    values = selected[selected.reference_feature_set.eq(ref) & selected.candidate_feature_set.eq(candidate)]
                    values = values.sort_values("threshold_percent")
                    ax.plot(values.threshold_percent, values.delta_per_100, color=ink, lw=1.15)
                    ax.fill_between(values.threshold_percent, values.lower_per_100, values.upper_per_100, color=ink, alpha=.15)
            else:
                for feature in CALIBRATION_INPUTS:
                    values = selected[selected.feature_set.eq(feature)].sort_values("threshold_percent")
                    ax.plot(values.threshold_percent, values.model_per_100, **input_line(feature))
                x, treat_all = values.threshold_percent, values.treat_all_per_100
                ax.plot(x, treat_all, color=style.color("reference"), ls="-.", lw=.8)
                if treat_all.min() < low < treat_all.max():
                    cross = np.interp(low, treat_all.iloc[::-1], x.iloc[::-1])
                    ax.annotate("", xy=(cross, low + .015 * (high - low)), xytext=(cross, low + .14 * (high - low)),
                                arrowprops={"arrowstyle": "-|>", "lw": .7, "color": style.color("reference")})
            ax.axhline(0, color=style.color("reference"), ls="--", lw=.7)
            ax.set(xlim=(1, 30) if outcome == "any_aki" else (5, 60), ylim=(low, high))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            if i == len(outcomes) - 1:
                ax.set_xlabel("Threshold probability (%)")
            if j == 0:
                ax.set_ylabel("Δ net benefit / 100" if paired else "Net benefit / 100")
            manuscript.panel(ax, i, j, 3, f"{title} · {short}")
    if paired:
        manuscript.legend(fig, [Line2D([0], [0], color=ink, label=CONTRASTS[pair])
                                for pair, ink in zip(pairs, colors, strict=True)] + [_reference("Zero difference")])
    else:
        _input_legend(fig, CALIBRATION_INPUTS, _reference("Treat all", linestyle="-."), _reference("Treat none"), columns=3)
    return fig, data
