"""Render publication panels from selected pooled predictions and saved intervals."""

from dataclasses import replace
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from reporting.cohort_flow import CohortStage, _draw_cohort_flow_fallback
from reporting import publication_style as style
from .revision_data import BRANCHES, CHANNELS, FAMILIES, INPUTS, OUTCOMES, IDENTITY, metric_row, verify_contrast


def save(fig, out, stem, data):
    fig.savefig(out / f"{stem}.png", dpi=style.PNG_DPI, facecolor="white")
    data.to_csv(out / f"{stem}.plotted.csv", index=False, float_format="%.17g")
    plt.close(fig)


def cohort_selection(path, roster, out):
    flow = json.loads(path.read_text())
    labels = (
        "VitalDB operations", "General anesthesia", "ASA-PS eligibility",
        "All four waveforms available", "Shared AKI/ICU eligibility", "Final analysis cohort",
    )
    stages = [
        replace(CohortStage(**stage), title=label, detail=None)
        for stage, label in zip(flow["stages"], labels, strict=True)
    ]
    stages[2] = replace(stages[2], removal_reason="ASA-PS V–VI")
    stages[-1] = replace(stages[-1], removal_reason="No waveform-feature record")
    total = roster["operations"]
    if (stages[-1].count, stages[-1].n_patients) != (total, roster["patients"]):
        raise ValueError("Cohort flow and selected main study have different populations")
    outcomes = []
    for key, label in OUTCOMES.items():
        saved = flow["final_outcomes"][key]
        events = roster["outcomes"][key]["events"]
        if (saved["eligible_operations"], saved["positive_operations"]) != (total, events):
            raise ValueError(f"Cohort flow and selected main study disagree on {label}")
        outcomes.append(f"{label}: {events:,} operations ({events / total:.1%})")

    fig = _draw_cohort_flow_fallback(stages, None, None, publication=True)
    width, height = fig.get_size_inches()
    footer = .66
    fig.set_size_inches(width, height + footer)
    fig.axes[0].set_position([0, footer / (height + footer), 1, height / (height + footer)])
    fig.text(.5, .41 / (height + footer), "Outcomes (may overlap)", ha="center", va="center",
             fontsize=8.7, color=style.color("ink"))
    fig.text(.5, .21 / (height + footer), "     ".join(outcomes), ha="center", va="center",
             fontsize=style.STYLE["figure"]["title_pt"], color=style.color("ink"))
    save(fig, out, "Figure_1", pd.DataFrame(stages))


def setup(rows, columns, height, **kwargs):
    fig, axes = plt.subplots(rows, columns, figsize=(style.WIDTH_MM / 25.4, height / 25.4),
                             layout="constrained", **kwargs)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    return fig, axes


def panel(ax, row, col, columns, label):
    ax.set_title(r"$\bf{" + chr(97 + row * columns + col) + "}$  " + label, loc="left")


def point(ax, point_value, lower, upper, y, encoding, *, color=None):
    color = color or style.INPUT_COLORS["preop_and_all_waveforms"]
    if not np.isfinite([point_value, lower, upper]).all() or upper < lower:
        raise ValueError("The estimate and interval endpoints must be finite and ordered")
    ax.errorbar((lower + upper) / 2, y, xerr=(upper - lower) / 2,
                fmt="none", ecolor=color, capsize=2, lw=.9)
    ax.plot(point_value, y, marker=encoding["marker"], color=color,
            mfc=color if encoding["filled"] else "white", ls="none", ms=4)


def legend(fig, handles):
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)


def branch_legend(fig, reference):
    color = style.INPUT_COLORS["preop_and_all_waveforms"]
    handles = [
        Line2D([0], [0], color=color, marker=style.BRANCHES[key]["marker"],
               mfc=color if style.BRANCHES[key]["filled"] else "white", ls="none", label=name)
        for key, name in BRANCHES.items()
    ]
    handles.append(Line2D([0], [0], color=style.color("reference"), ls="--", label=reference))
    legend(fig, handles)


def plot_contrast(ax, metrics, family, outcome, branch, metric, y, encoding, *, color=None):
    row = verify_contrast(metrics, family, outcome, branch)
    key = f"delta_{metric}"
    values = [100 * row[c] for c in (key, key + "_lower", key + "_upper")]
    point(ax, *values, y, encoding, color=color)
    return {"outcome": outcome, "family": family, "branch": branch, "metric": metric,
            "estimate_pp": values[0], "lower_pp": values[1], "upper_pp": values[2],
            "bootstrap_draw_hash": row.bootstrap_draw_hash,
            "candidate_hash": row.prediction_content_hash,
            "reference_hash": row.reference_prediction_content_hash}


def incremental(metrics, out):
    fig, axes = setup(2, 2, 140)
    panels = [(ri, ci, outcome, metric, name, title)
              for ri, (outcome, title) in enumerate(OUTCOMES.items())
              for ci, (metric, name) in enumerate((("auprc", "AP"), ("auroc", "AUROC")))]
    records = []
    for ri, ci, outcome, metric, name, title in panels:
        ax = axes[ri, ci]
        for fi, family in enumerate(FAMILIES):
            for bi, branch in enumerate(BRANCHES):
                records.append(plot_contrast(ax, metrics, family, outcome, branch, metric,
                                             fi + (-.14 if bi == 0 else .14), style.BRANCHES[branch]))
        ax.axvline(0, color=style.color("reference"), ls="--", lw=.8)
        ax.set(yticks=range(len(FAMILIES)), yticklabels=[f[0] for f in FAMILIES.values()],
               ylim=(2.55, -.55), xlabel=f"Δ{name} (percentage points)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="x", color=style.color("grid"), lw=.6)
        panel(ax, ri, ci, 2, title)
    branch_legend(fig, "Zero difference")
    save(fig, out, "Figure_2", pd.DataFrame(records))


def cohort_incremental(main, broader, out):
    fig, axes = setup(2, 2, 140, sharex="col")
    cohorts = (
        ("main_shared", "Main cohort", main, "#385A92", "o"),
        ("icu_waveform_sensitivity", "Broader ICU cohort", broader, "#A34D27", "s"),
    )
    records = []
    for ri, (branch, branch_label) in enumerate(BRANCHES.items()):
        for ci, (metric, name) in enumerate((("auprc", "AP"), ("auroc", "AUROC"))):
            ax = axes[ri, ci]
            for fi, family in enumerate(FAMILIES):
                for si, (scope, _, metrics, color, marker) in enumerate(cohorts):
                    record = plot_contrast(ax, metrics, family, "icu_admission", branch, metric,
                                           fi + (-.14 if si == 0 else .14),
                                           {"marker": marker, "filled": True}, color=color)
                    records.append({"cohort": scope, **record})
            ax.axvline(0, color=style.color("reference"), ls="--", lw=.8)
            ax.set(yticks=range(len(FAMILIES)), yticklabels=[f[0] for f in FAMILIES.values()], ylim=(2.55, -.55))
            if ri == 1:
                ax.set_xlabel(f"Δ{name} (percentage points)")
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="x", color=style.color("grid"), lw=.6)
            panel(ax, ri, ci, 2, f"{branch_label} · {name}")
    handles = [Line2D([0], [0], color=color, marker=marker, ls="none", label=label)
               for _, label, _, color, marker in cohorts]
    handles.append(Line2D([0], [0], color=style.color("reference"), ls="--", label="Zero difference"))
    legend(fig, handles)
    save(fig, out, "Figure_S7", pd.DataFrame(records))


def channels(metrics, out):
    records = []
    for metric, name, stem in (("auprc", "Average precision", "Figure_S5"), ("auroc", "AUROC", "Figure_S6")):
        fig, axes = setup(2, 3, 142, sharey=True)
        records.clear()
        for ri, (outcome, title) in enumerate(OUTCOMES.items()):
            subset = metrics[metrics.outcome.eq(outcome) & metrics.feature_set.isin(CHANNELS)]
            prevalence = subset.prevalence.unique()
            if len(subset) != 30 or len(prevalence) != 1:
                raise ValueError(f"Expected 30 matched channel rows and one prevalence for {outcome}")
            reference = prevalence[0] if metric == "auprc" else .5
            low = min(reference, subset[metric].min(), subset[metric + "_lower"].min())
            high = max(reference, subset[metric].max(), subset[metric + "_upper"].max())
            pad = (high - low) * .09
            for ci, (family, (short, _)) in enumerate(FAMILIES.items()):
                ax = axes[ri, ci]
                for bi, branch in enumerate(BRANCHES):
                    for yi, feature in enumerate(CHANNELS):
                        row = metric_row(subset, family, outcome, branch, feature)
                        values = [row[key] for key in (metric, metric + "_lower", metric + "_upper")]
                        point(ax, *values, yi + (-.14 if bi == 0 else .14),
                              style.BRANCHES[branch])
                        records.append({"outcome": outcome, "family": family, "branch": branch,
                                        "feature_set": feature, "metric": metric, "estimate": values[0],
                                        "lower": values[1], "upper": values[2], "reference": reference,
                                        "prediction_hash": row.prediction_content_hash,
                                        "bootstrap_draw_hash": row.bootstrap_draw_hash})
                ax.axvline(reference, color=style.color("reference"), ls="--", lw=.8)
                ax.set(yticks=range(len(CHANNELS)), yticklabels=list(CHANNELS.values()),
                       ylim=(4.65, -.65), xlim=(max(0, low-pad), min(1, high+pad)), xlabel=name)
                ax.grid(axis="x", color=style.color("grid"), lw=.6)
                ax.xaxis.set_major_locator(MaxNLocator(4))
                panel(ax, ri, ci, 3, f"{title} · {short}")
        branch_legend(fig, "Outcome prevalence" if metric == "auprc" else "Chance (0.50)")
        save(fig, out, stem, pd.DataFrame(records))


def input_line(key):
    return dict(color=style.INPUT_COLORS[key], ls=style.INPUT_STYLES[key], lw=1.15)


def input_legend(fig, keys, reference):
    legend(fig, [Line2D([0], [0], **input_line(k), label=INPUTS[k]) for k in keys] + [
        Line2D([0], [0], color=style.color("reference"), ls="--", label=reference),
    ])


def discrimination(studies, metrics, out):
    modes = {
        "Figure_3": ("auroc", roc_curve, "False-positive rate", "True-positive rate", "Chance"),
        "Figure_S3": ("auprc", precision_recall_curve, "Recall", "Precision", "Outcome prevalence"),
    }
    for stem, (metric, curve, xname, yname, reference) in modes.items():
        fig, axes = setup(2, 3, 155, sharex=True, sharey=True)
        records = []
        for ri, (outcome, title) in enumerate(OUTCOMES.items()):
            for ci, (family, (short, _)) in enumerate(FAMILIES.items()):
                ax = axes[ri, ci]
                scores, roster = [], None
                for feature in INPUTS:
                    p = studies[family].predictions(family, outcome, "non_windowed", feature)
                    if roster is not None and not p[IDENTITY].equals(roster):
                        raise ValueError("Whole-operation comparisons have different evaluation identities")
                    roster = p[IDENTITY]
                    y, risk = p.y_true, p.y_prob_calibrated
                    xvals, yvals, _ = curve(y, risk)
                    if metric == "auprc":
                        xvals, yvals = yvals, xvals
                    score_fn = average_precision_score if metric == "auprc" else roc_auc_score
                    estimate = score_fn(y, risk)
                    expected = metric_row(metrics, family, outcome, "non_windowed", feature)
                    if not (np.isclose(estimate, expected[metric], rtol=0, atol=1e-12)
                            and np.isclose(brier_score_loss(y, risk), expected.brier, rtol=0, atol=1e-12)):
                        raise ValueError(f"Table 2 and selected predictions disagree for {family}, {outcome}, {feature}")
                    scores.append(f"{estimate:.3f}")
                    ax.plot(xvals, yvals, **input_line(feature))
                    records.append(pd.DataFrame({
                        "outcome": outcome, "family": family, "input": feature, "x": xvals, "y": yvals,
                        "metric": metric, "estimate": estimate, "prediction_hash": expected.prediction_content_hash,
                    }))
                if metric == "auprc":
                    ax.axhline(y.mean(), color=style.color("reference"), ls="--", lw=.7)
                else:
                    ax.plot([0, 1], [0, 1], color=style.color("reference"), ls="--", lw=.7)
                ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
                ax.text(.03, .05 if metric == "auroc" else .09, " / ".join(scores),
                        transform=ax.transAxes, fontsize=8, va="bottom",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1})
                panel(ax, ri, ci, 3, f"{title} · {short}")
                if ci == 0:
                    ax.set_ylabel(yname)
                if ri == 1:
                    ax.set_xlabel(xname)
        input_legend(fig, INPUTS, reference)
        save(fig, out, stem, pd.concat(records, ignore_index=True))


def calibration(studies, out):
    fig = plt.figure(figsize=(style.WIDTH_MM / 25.4, 175 / 25.4), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    inputs = ("preop_only", "preop_and_all_waveforms")
    bins, histograms = [], []
    for ri, (outcome, title) in enumerate(OUTCOMES.items()):
        for ci, (family, (short, _)) in enumerate(FAMILIES.items()):
            sub = grid[ri, ci].subgridspec(2, 1, height_ratios=(3.2, 1))
            ax = fig.add_subplot(sub[0])
            hist = fig.add_subplot(sub[1], sharex=ax)
            inset = ax.inset_axes([.55, .09, .42, .40]) if ri == 0 else None
            for feature in inputs:
                p = studies[family].predictions(family, outcome, "non_windowed", feature)
                risk = p.y_prob_calibrated
                grouped = p.groupby(pd.qcut(risk, 10, duplicates="drop", labels=False))
                dots = grouped.agg(mean_risk=("y_prob_calibrated", "mean"),
                                   observed_rate=("y_true", "mean"), count=("y_true", "size"))
                bins.append(dots.rename_axis("bin").reset_index().assign(outcome=outcome, family=family, feature_set=feature))
                dots_kwargs = {**input_line(feature), "marker": style.INPUT_MARKERS[feature], "ms": 3}
                ax.plot(dots.mean_risk, dots.observed_rate, **dots_kwargs)
                if inset is not None:
                    inset.plot(dots.mean_risk, dots.observed_rate, **dots_kwargs)
                edges = np.linspace(0, 1, 41)
                counts, _ = np.histogram(risk, edges)
                histograms.append(pd.DataFrame({"outcome": outcome, "family": family, "feature_set": feature,
                                                "bin_lower": edges[:-1], "bin_upper": edges[1:], "count": counts}))
                hist.stairs(counts, edges, **input_line(feature))
            ax.plot([0, 1], [0, 1], color=style.color("reference"), ls="--", lw=.7)
            ax.set(xlim=(0, 1), ylim=(0, 1), yticks=[0, .5, 1])
            ax.tick_params(labelbottom=False)
            hist.set(xlim=(0, 1), xticks=[0, .5, 1], xlabel="Predicted risk")
            hist.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            if inset is not None:
                inset.plot([0, .35], [0, .35], color=style.color("reference"), ls="--", lw=.7)
                inset.set(xlim=(0, .35), ylim=(0, .35), xticks=[0, .35], yticks=[0, .35], title="AKI detail")
                inset.tick_params(labelsize=8, pad=1, length=1.5)
            if ci == 0:
                ax.set_ylabel("Observed risk")
                hist.set_ylabel("Operations")
            panel(ax, ri, ci, 3, f"{title} · {short}")
        hist_axes = [ax for ax in fig.axes if ax.get_xlabel() == "Predicted risk"][-3:]
        shared_max = max(ax.get_ylim()[1] for ax in hist_axes)
        for ax in hist_axes:
            ax.set_ylim(0, shared_max)
    input_legend(fig, inputs, "Perfect calibration")
    save(fig, out, "Figure_S4", pd.concat(bins, ignore_index=True))
    pd.concat(histograms, ignore_index=True).to_csv(out / "Figure_S4.prediction_support.csv", index=False)


def decision_curves(absolute, paired, metrics, out):
    inputs = ("preop_only", "preop_and_all_waveforms")
    for stem, frame in (("Figure_S8", absolute), ("Figure_S9", paired)):
        fig, axes = setup(2, 3, 145)
        rows = []
        for ri, (outcome, title) in enumerate(OUTCOMES.items()):
            basis = frame[frame.outcome.eq(outcome) & frame.branch.eq("non_windowed")]
            if stem == "Figure_S8":
                basis = basis[basis.feature_set.isin(inputs)]
                lo = min(-.25 if ri == 0 else -1, 100*basis.model.min()-.15)
                hi = max(100*basis.model.max(), 100*basis.treat_all.max())*1.09
            else:
                basis = basis[basis.reference_model.eq(basis.candidate_model)
                              & basis.reference_feature_set.eq(inputs[0]) & basis.candidate_feature_set.eq(inputs[1])]
                lo, hi = min(0, 100*basis.lower.min()), max(0, 100*basis.upper.max())
                pad = .08*(hi-lo)
                lo, hi = lo-pad, hi+pad
            for ci, (family, (short, _)) in enumerate(FAMILIES.items()):
                ax = axes[ri, ci]
                limits = (.01, .30) if ri == 0 else (.05, .60)
                if stem == "Figure_S8":
                    z = basis[basis.model_type.eq(family)]
                    for feature in inputs:
                        values = z[z.feature_set.eq(feature)].sort_values("threshold_probability")
                        expected = metric_row(metrics, family, outcome, "non_windowed", feature)
                        if not values.prediction_content_hash.eq(expected.prediction_content_hash).all():
                            raise ValueError(f"Stale absolute decision curve for {outcome}, {family}, {feature}")
                        ax.plot(100*values.threshold_probability, 100*values.model, **input_line(feature))
                    x, treat_all = 100*values.threshold_probability, 100*values.treat_all
                    ax.plot(x, treat_all, color=style.color("reference"), ls="-.", lw=.8)
                    if treat_all.min() < lo:
                        cross = np.interp(lo, treat_all.iloc[::-1], x.iloc[::-1])
                        ax.annotate("", xy=(cross, lo+.015*(hi-lo)), xytext=(cross, lo+.14*(hi-lo)),
                                    arrowprops={"arrowstyle": "-|>", "lw": .7, "color": style.color("reference")})
                else:
                    z = basis[basis.candidate_model.eq(family)].sort_values("threshold_probability")
                    expected = verify_contrast(metrics, family, outcome, "non_windowed")
                    if not (z.candidate_prediction_hash.eq(expected.prediction_content_hash).all()
                            and z.reference_prediction_hash.eq(expected.reference_prediction_content_hash).all()
                            and z.bootstrap_draw_hash.eq(expected.bootstrap_draw_hash).all()):
                        raise ValueError(f"Stale paired decision curve for {outcome}, {family}")
                    color = style.INPUT_COLORS[inputs[1]]
                    ax.plot(100*z.threshold_probability, 100*z.delta, color=color, lw=1.15)
                    ax.fill_between(100*z.threshold_probability, 100*z.lower, 100*z.upper, color=color, alpha=.19)
                expected_grid = round(100*(limits[1]-limits[0])) + 1
                if len(z) != expected_grid * (2 if stem == "Figure_S8" else 1):
                    raise ValueError(f"Unexpected decision-curve coverage in {stem}, {outcome}, {family}")
                rows.append(z)
                ax.axhline(0, color=style.color("reference"), ls="--", lw=.7)
                ax.set(xlim=[100*x for x in limits], ylim=(lo, hi))
                ax.xaxis.set_major_locator(MaxNLocator(4))
                if ri == 1:
                    ax.set_xlabel("Threshold probability (%)")
                if ci == 0:
                    ax.set_ylabel("Net benefit / 100" if stem == "Figure_S8" else "Δ net benefit / 100")
                panel(ax, ri, ci, 3, f"{title} · {short}")
        if stem == "Figure_S8":
            handles = [Line2D([0], [0], **input_line(k), label=INPUTS[k]) for k in inputs]
            legend(fig, handles + [Line2D([0], [0], color=style.color("reference"), ls="-.", label="Treat all"),
                                   Line2D([0], [0], color=style.color("reference"), ls="--", label="Treat none")])
        save(fig, out, stem, pd.concat(rows, ignore_index=True))
