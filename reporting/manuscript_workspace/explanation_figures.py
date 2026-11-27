"""Shared layouts for the manuscript's EBM explanation figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

if __package__:
    from reporting import publication_style as style
else:
    import publication_style as style


def _canvas(height):
    return plt.figure(figsize=(style.WIDTH_MM / 25.4, height / 25.4))


def _title(axis, index, label):
    axis.set_title(r"$\bf{" + chr(97 + index) + "}$  " + label, loc="left", pad=6)


def _clean(axis, *, grid=False):
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    if grid:
        axis.grid(axis="y", color=style.color("grid"), lw=style.GRID_WIDTH)


def _fold_legend(fig, *, mean=False, absent=False):
    handles = [
        Line2D([], [], color=color, ls="none" if mean else line, lw=style.CURVE_WIDTH,
               marker=marker if mean else None, ms=3, label=f"Fold {index + 1}")
        for index, (color, line, marker) in enumerate(zip(style.FOLD_COLORS, style.FOLD_STYLES, style.FOLD_MARKERS))
    ]
    if mean:
        handles.insert(0, Patch(facecolor=style.color("mean"), edgecolor="none", label="Mean"))
    if absent:
        handles.append(Line2D([], [], color=style.color("ink"), ls="none", marker="o", ms=3,
                              markerfacecolor="none", label="Term absent"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .01), ncol=len(handles),
               frameon=False, handlelength=2, columnspacing=1.1)


def term_importance_figure(data, panels, *, show_absent=False):
    """Plot means and individual outer folds for the supplied ranked terms."""
    fig = _canvas(182)
    axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
    fig.subplots_adjust(left=.47, right=.975, bottom=.135, top=.945, hspace=.55)
    for index, (outcome, label, upper) in enumerate(panels):
        axis = axes[index]
        _title(axis, index, label)
        labels = []
        for row, (_, folds) in enumerate(data.loc[data.outcome.eq(outcome)].groupby("display_rank", sort=True)):
            y = 9 - row
            labels.append(folds.display_label.iloc[0])
            axis.barh(y, folds.mean_native_importance.iloc[0], height=.55, color=style.color("mean"),
                      edgecolor="none", zorder=1)
            for fold in folds.itertuples():
                k = int(fold.fold_number) - 1
                color = style.FOLD_COLORS[k]
                axis.plot(fold.native_importance, y + (k - 2) * .105, color=color, marker=style.FOLD_MARKERS[k],
                          markerfacecolor=color if getattr(fold, "term_present", True) else "none",
                          ls="none", ms=3.5, zorder=3)
        axis.set(yticks=np.arange(9, 9 - len(labels), -1), yticklabels=labels, ylim=(-.65, 9.65),
                 xlim=(0, upper), xlabel="Importance (log-odds)")
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.tick_params(axis="y", length=0, pad=5)
        _clean(axis)
        axis.spines["left"].set_visible(False)
        axis.grid(axis="x", color=style.color("grid"), lw=style.GRID_WIDTH)
    _fold_legend(fig, mean=True, absent=show_absent)
    return fig


def response_figure(data, support, panel_column, panels):
    """Pair each fold step function with its outer-training histogram."""
    fig = _canvas(118)
    grid = fig.add_gridspec(1, len(panels), left=.10, right=.985, bottom=.28, top=.91, wspace=.46)
    for index, (panel, heading, xlabel, locator) in enumerate(panels):
        pair = grid[index].subgridspec(2, 1, height_ratios=[3.7, 1], hspace=.035)
        axis = fig.add_subplot(pair[0])
        histogram = fig.add_subplot(pair[1], sharex=axis)
        rows = data[data[panel_column].eq(panel)]
        bins = support[support[panel_column].eq(panel)]
        reference = rows.reference.iloc[0]
        _title(axis, index, heading)
        for fold, values in rows.groupby("fold"):
            values = values.sort_values("value")
            k = int(fold)
            axis.step(values.value, values.raw_contribution_minus_reference, where="post",
                      color=style.FOLD_COLORS[k], ls=style.FOLD_STYLES[k], lw=style.CURVE_WIDTH)
        axis.axhline(0, color=style.color("reference"), ls="--", lw=style.REFERENCE_WIDTH)
        axis.axvline(reference, color=style.color("reference"), ls=":", lw=style.REFERENCE_WIDTH)
        axis.set_ylabel("Δ raw log-odds")
        axis.tick_params(axis="x", labelbottom=False)
        _clean(axis, grid=True)
        axis.yaxis.set_major_locator(MaxNLocator(4))
        histogram.bar(bins.bin_lower, bins["count"], width=bins.bin_upper - bins.bin_lower, align="edge",
                      color=style.color("support"), linewidth=0)
        reference_text = f"{reference:.3g}".replace("-", "−")
        histogram.set(xlim=(bins.bin_lower.min(), bins.bin_upper.max()), ylabel="Training n",
                      xlabel=f"{xlabel}\nReference = {reference_text}")
        histogram.yaxis.set_major_locator(MaxNLocator(2, integer=True))
        histogram.xaxis.set_major_locator(locator)
        _clean(histogram)
    _fold_legend(fig)
    return fig


def heldout_waterfalls(cases, *, rows=2, height=215):
    """Show calibrated intercepts, sequential group contributions, and final predictions."""
    values = np.asarray([0, *[value for case in cases for value in (
        case["intercept"], case["final"], *case["starts"], *case["ends"],
    )]])
    span = np.ptp(values)
    xlim = (values.min() - .10 * span, values.max() + .10 * span)
    fig = _canvas(height)
    grid = fig.add_gridspec(rows, 2, left=.155, right=.985, top=.890, bottom=.155, wspace=.62, hspace=.56)
    contribution_ink = style.color("signed_contribution")
    prediction_ink = style.INPUT_COLORS["preop_and_all_waveforms"]
    for index, case in enumerate(cases):
        axis = fig.add_subplot(grid[index // 2, index % 2])
        axis.spines[["top", "right"]].set_visible(False)
        labels = ["Intercept", *case["labels"], "Prediction"]
        positions = [0, 1.4, 2.8, 4.2, *(5.4 + i for i in range(len(case["labels"]) - 3))]
        positions.append(positions[-1] + 1.1)
        axis.barh(0, case["intercept"], left=0, height=.64, facecolor=style.color("baseline"),
                  edgecolor=style.color("baseline_edge"), linewidth=.65, zorder=3)
        for row, (start, end) in enumerate(zip(case["starts"], case["ends"], strict=True), 1):
            axis.plot([start, start], [positions[row - 1] + .34, positions[row] - .34],
                      color=style.color("connector"), lw=.65, zorder=2)
            axis.barh(positions[row], abs(end - start), left=min(start, end), height=.64,
                      facecolor=contribution_ink if end >= start else "white", edgecolor=contribution_ink,
                      linewidth=.85, zorder=3)
        final = case["final"]
        axis.plot([final, final], [positions[-2] + .34, positions[-1] - .34],
                  color=style.color("connector"), lw=.65, zorder=2)
        axis.barh(positions[-1], final, left=0, height=.64, facecolor=prediction_ink,
                  edgecolor=style.color("combined_edge"), linewidth=.65, zorder=3)
        axis.set(yticks=positions, yticklabels=labels, ylim=(positions[-1] + .55, -.65), xlim=xlim)
        axis.set_xticks([tick for tick in MaxNLocator(4).tick_values(*xlim) if xlim[0] <= tick <= xlim[1]])
        axis.tick_params(axis="y", length=0, pad=3)
        axis.axvline(0, color=style.color("zero"), lw=style.REFERENCE_WIDTH, zorder=1)
        axis.grid(axis="x", color=style.color("grid"), linewidth=style.GRID_WIDTH, zorder=0)
        axis.set_xlabel("Calibrated log-odds", labelpad=4)
        title = r"$\bf{" + chr(97 + index) + "}$  " + case["title"]
        axis.text(0, 1.20, title, transform=axis.transAxes, fontsize=style.STYLE["figure"]["title_pt"],
                  fontweight="normal", va="bottom")
        axis.text(0, 1.07, case["summary"], transform=axis.transAxes, fontsize=style.STYLE["figure"]["body_pt"],
                  va="bottom", linespacing=1.25)
    handles = [Patch(facecolor=contribution_ink, edgecolor=contribution_ink, label="Raises risk"),
               Patch(facecolor="white", edgecolor=contribution_ink, label="Lowers risk"),
               Patch(facecolor=style.color("baseline"), edgecolor=style.color("baseline_edge"), label="Intercept"),
               Patch(facecolor=prediction_ink, edgecolor=style.color("combined_edge"), label="Prediction")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .010), ncol=2, frameon=False,
               handlelength=1.5, columnspacing=2.)
    return fig, xlim
