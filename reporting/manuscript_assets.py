from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from docx import Document
from docx.enum.section import WD_ORIENT
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from reporting.publication_style import (
    DEFAULT_FIGURE_FORMATS, JPEG_QUALITY, JPG_DPI, PNG_DPI, TIFF_COMPRESSION, TIFF_DPI,
    WORD_TABLE_PT, apply_document_style, color, style_context,
)

from reporting.table_rendering import (
    _add_dataframe_to_doc, _clean_cell, _table_widths, dataframe_to_html,
    dataframe_to_markdown, export_dataframe_bundle as _export_dataframe_bundle,
    numeric_display_columns, symbol_display_columns, write_markdown_sections,
)


JPEG_DPI = JPG_DPI


def save_figure_bundle(
    fig: plt.Figure,
    destination: Path,
    *,
    formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
    png_dpi: int = PNG_DPI,
    close: bool = True,
) -> List[Path]:
    """Save a figure in manuscript-ready variants."""

    base = destination.with_suffix("") if destination.suffix else destination
    saved_paths: List[Path] = []
    for fmt in dict.fromkeys(str(value).lower().lstrip(".") for value in formats):
        output_path = base.with_suffix(f".{fmt}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight"}
        if fmt.lower() == "png":
            save_kwargs["dpi"] = png_dpi
        elif fmt in {"jpg", "jpeg"}:
            save_kwargs.update(dpi=JPEG_DPI, facecolor="white", transparent=False,
                               pil_kwargs={"quality": JPEG_QUALITY, "subsampling": 0})
        elif fmt in {"tif", "tiff"}:
            save_kwargs.update(dpi=TIFF_DPI, facecolor="white", transparent=False,
                               pil_kwargs={"compression": TIFF_COMPRESSION})
        # Keep vector text editable and embedded without changing caller geometry.
        with style_context():
            fig.savefig(output_path, **save_kwargs)
        if fmt in {"tif", "tiff"}:
            from PIL import Image
            with Image.open(output_path) as raster:
                rgb = raster.convert("RGB")
            rgb.save(output_path, compression=TIFF_COMPRESSION, dpi=(TIFF_DPI, TIFF_DPI))
        saved_paths.append(output_path)
    if close:
        plt.close(fig)
    return saved_paths


def _render_pdf_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, *, notes=None, styles=None) -> None:
    # Wide machine-readable tables remain complete, spread over readable panels.
    if len(df.columns) > 7:
        identities = {"model", "model_name", "outcome", "branch", "feature set", "feature_set", "inputs", "pipeline"}
        identifiers = [c for c in df if str(c).lower() in identities]
        if not identifiers:
            identifiers = [c for c in df if c in {"Preop", "AWP", "CO2", "ECG", "Pleth"}] or list(df.columns[:2])
        identifiers = identifiers[:5]
        values = [c for c in df if c not in identifiers]
        panel_width = max(1, 7 - len(identifiers))
        panel_count = (len(values) + panel_width - 1) // panel_width
        for start in range(0, len(values), panel_width):
            panel_values = values[start:start + panel_width]
            panel_title = f"{title}\nPanel {start // panel_width + 1}/{panel_count}: {panel_values[0]} to {panel_values[-1]}"
            if len(df) <= 18:
                panel_title += f"\nRows 1–{len(df)} of {len(df)}"
            _render_pdf_table_page(pdf, panel_title, df[identifiers + panel_values], notes=notes, styles=styles)
        return
    if len(df) > 18:
        for start in range(0, len(df), 18):
            page_title = f"{title}\nRows {start + 1}–{min(start + 18, len(df))} of {len(df)}"
            _render_pdf_table_page(pdf, page_title, df.iloc[start:start + 18], notes=notes, styles=styles)
        return
    widths = _table_widths(df)
    def wrap(value, fraction):
        text = _clean_cell(value).replace("<br>", "\n").replace(" (", "\n(")
        return "\n".join(textwrap.fill(line, max(12, int(fraction * 135)), break_long_words=False) for line in text.splitlines())
    rows = [[wrap(value, widths[j]) for j, value in enumerate(row)] for row in df.itertuples(index=False, name=None)]
    headers = [wrap(value, widths[j]) for j, value in enumerate(df.columns)]
    heights = [.18 * max(value.count("\n") + 1 for value in row) + .14 for row in [headers] + rows]
    note_lines = textwrap.wrap(str(notes or ""), 140)
    title_lines = "\n".join(textwrap.fill(line, 105, break_long_words=False)
                            for line in title.splitlines())
    title_height = .22 * (title_lines.count("\n") + 1)
    fig_height = max(3, sum(heights) + .8 + title_height + .16 * len(note_lines))
    fig, ax = plt.subplots(figsize=(11.7, fig_height))
    ax.axis("off")
    ax.set_position([.03, (.35 + .16 * len(note_lines)) / fig_height, .94, sum(heights) / fig_height])
    fig.text(.03, 1 - .25 / fig_height, title_lines, fontsize=12, fontweight="bold", va="top")
    table = ax.table(
        cellText=rows, colLabels=headers, colWidths=widths,
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_height(heights[row] / sum(heights))
        cell.set_edgecolor(color('table_rule'))
        # Keep closed cell paths so facecolor shading renders, then draw only
        # horizontal rules over the filled table.
        cell.set_linewidth(0)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(color('table_header'))
        else:
            style = (styles or {}).get(df.index[row - 1], {}).get(df.columns[column], {})
            cell.set_facecolor(style.get("bg", "white"))
            cell.set_text_props(weight="bold" if style.get("bold") else "normal")
        if column < 2:
            cell.set_text_props(ha="left")
    boundary = 1.0
    for height in [0, *heights]:
        boundary -= height / sum(heights)
        ax.plot([0, 1], [boundary, boundary], transform=ax.transAxes,
                color=color('table_rule'), linewidth=.35, clip_on=False)
    if notes:
        fig.text(.03, .18 / fig_height, "\n".join(note_lines), fontsize=8, va="bottom")
    pdf.savefig(fig, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


def write_docx_sections(
    path: Path,
    *,
    title: str,
    sections: Sequence[Mapping[str, object]],
) -> Path:
    """Write a DOCX document containing one or more table sections."""

    path.parent.mkdir(parents=True, exist_ok=True)
    document = apply_document_style(Document(), body_size=WORD_TABLE_PT)
    for section in document.sections:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = section.page_height, section.page_width

    document.add_heading(title, level=0)
    for section in sections:
        if str(section["heading"]) != title:
            document.add_heading(str(section["heading"]), level=1)
        text = section.get("text")
        if text:
            document.add_paragraph(str(text))
        df = section.get("dataframe")
        if isinstance(df, pd.DataFrame):
            _add_dataframe_to_doc(document, df, section.get("styles"))
        if section.get("notes"):
            document.add_paragraph(str(section["notes"]))
        document.add_paragraph()
    document.save(path)
    return path


def write_pdf_sections(
    path: Path,
    *,
    title: str,
    sections: Sequence[Mapping[str, object]],
) -> Path:
    """Write a PDF document containing one or more table sections."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with style_context(), PdfPages(path) as pdf:
        for section in sections:
            heading = str(section["heading"])
            df = section.get("dataframe")
            text = section.get("text")
            title_text = title if heading == title else f"{title} | {heading}"
            if text and not isinstance(df, pd.DataFrame):
                fig, ax = plt.subplots(figsize=(12, 3.5))
                ax.axis("off")
                ax.set_title(title_text, fontsize=14, fontweight="bold", pad=18)
                ax.text(0.02, 0.95, str(text), va="top", ha="left", wrap=True)
                pdf.savefig(fig, bbox_inches="tight", pad_inches=.2)
                plt.close(fig)
                continue
            if isinstance(df, pd.DataFrame):
                _render_pdf_table_page(pdf, title_text, df, notes=section.get("notes"), styles=section.get("styles"))
    return path


export_dataframe_bundle = partial(
    _export_dataframe_bundle, write_docx=write_docx_sections, write_pdf=write_pdf_sections,
)


def export_table_bundle(
    base_path: Path,
    *,
    title: str,
    main_dataframe: pd.DataFrame,
    delta_dataframe: Optional[pd.DataFrame] = None,
    summary_text: Optional[str] = None,
    main_styles: Optional[dict] = None,
    delta_styles: Optional[dict] = None,
) -> Dict[str, str]:
    """Export a results table bundle with optional delta companion table."""

    base = base_path.with_suffix("") if base_path.suffix else base_path
    sections = [{"heading": title, "dataframe": main_dataframe, "text": summary_text, "styles": main_styles}]
    outputs: Dict[str, str] = {}

    main_csv = base.with_name(f"{base.name}_main").with_suffix(".csv")
    main_csv.parent.mkdir(parents=True, exist_ok=True)
    main_dataframe.to_csv(main_csv, index=False)
    outputs["csv_main"] = str(main_csv)

    if delta_dataframe is not None and not delta_dataframe.empty:
        delta_csv = base.with_name(f"{base.name}_delta").with_suffix(".csv")
        delta_dataframe.to_csv(delta_csv, index=False)
        outputs["csv_delta"] = str(delta_csv)
        sections.append({"heading": "Delta vs Reference", "dataframe": delta_dataframe, "styles": delta_styles})

    md_path = write_markdown_sections(base.with_suffix(".md"), title=title, sections=sections)
    docx_path = write_docx_sections(base.with_suffix(".docx"), title=title, sections=sections)
    pdf_path = write_pdf_sections(base.with_suffix(".pdf"), title=title, sections=sections)
    outputs["md"] = str(md_path)
    outputs["docx"] = str(docx_path)
    outputs["pdf"] = str(pdf_path)
    return outputs


def _asset_group(relative_path: Path) -> tuple[str, str]:
    top = relative_path.parts[0]
    if top == "reports":
        return "core_reports", "report"
    if top == "figures":
        return "manuscript_figures", "figure"
    if top == "tables":
        if relative_path.suffix == ".parquet":
            return "metadata_supporting", "supporting_data"
        return "manuscript_tables", "table"
    return "metadata_supporting", "metadata"


def _asset_id_and_variant(relative_path: Path) -> tuple[str, Optional[str]]:
    stem = relative_path.stem
    if stem.endswith("_main"):
        return stem[:-5], "main"
    if stem.endswith("_delta"):
        return stem[:-6], "delta"
    return stem, None


def _title_from_asset_id(asset_id: str, asset_type: str) -> str:
    if asset_id == "report":
        return "Consolidated Results Report"
    if asset_id == "metrics_summary":
        return "Metrics Summary"
    if asset_id == "preop_descriptives":
        return "Preoperative Descriptive Statistics"
    if asset_id == "missingness_table":
        return "Missingness Table"
    if asset_id == "cohort_flow":
        return "Cohort Flow Diagram"
    if asset_id.startswith("results_"):
        return "Results Table: " + asset_id.removeprefix("results_").replace("_", " ").title()
    if asset_type == "figure":
        return asset_id.replace("_", " ").title()
    return asset_id.replace("_", " ").title()


def _prefer_manifest_path(current: str, candidate: str) -> tuple[str, str]:
    current_parts = Path(current).parts
    candidate_parts = Path(candidate).parts
    if len(candidate_parts) < len(current_parts):
        return candidate, current
    if len(candidate_parts) > len(current_parts):
        return current, candidate
    if candidate < current:
        return candidate, current
    return current, candidate


def refresh_paper_bundle(paper_dir: Path) -> dict:
    """Scan the paper directory and regenerate the manifest + README index."""

    paper_dir = paper_dir.resolve(strict=False)
    assets: Dict[str, dict] = {}
    for file_path in sorted(paper_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(paper_dir)
        if relative.as_posix() in {"manifest.json", "README.md"}:
            continue

        group, asset_type = _asset_group(relative)
        asset_id, csv_variant = _asset_id_and_variant(relative)
        entry = assets.setdefault(
            asset_id,
            {
                "id": asset_id,
                "title": _title_from_asset_id(asset_id, asset_type),
                "type": asset_type,
                "group": group,
                "paths": {},
            },
        )
        rel_text = relative.as_posix()
        if csv_variant is not None and relative.suffix == ".csv":
            entry.setdefault("csv_variants", {})[csv_variant] = rel_text
        else:
            fmt = relative.suffix.lstrip(".")
            current = entry["paths"].get(fmt)
            if current is None:
                entry["paths"][fmt] = rel_text
            elif current != rel_text:
                preferred, alternate = _prefer_manifest_path(current, rel_text)
                entry["paths"][fmt] = preferred
                alt_list = entry.setdefault("alternate_paths", {}).setdefault(fmt, [])
                if alternate not in alt_list:
                    alt_list.append(alternate)

    ordered_assets = sorted(assets.values(), key=lambda item: (item["group"], item["title"], item["id"]))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_dir": str(paper_dir),
        "groups": {
            "core_reports": "Core reports",
            "manuscript_tables": "Manuscript-facing tables",
            "manuscript_figures": "Manuscript-facing figures",
            "metadata_supporting": "Metadata and supporting artifacts",
        },
        "assets": ordered_assets,
    }

    manifest_path = paper_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Catch22 Paper Bundle",
        "",
        f"Generated: {manifest['generated_at']}",
        "",
    ]
    grouped_assets: Dict[str, List[dict]] = {
        "core_reports": [],
        "manuscript_tables": [],
        "manuscript_figures": [],
        "metadata_supporting": [],
    }
    for asset in ordered_assets:
        grouped_assets.setdefault(asset["group"], []).append(asset)

    for group_key, heading in manifest["groups"].items():
        lines.append(f"## {heading}")
        lines.append("")
        items = grouped_assets.get(group_key, [])
        if not items:
            lines.append("- None")
            lines.append("")
            continue
        for asset in items:
            format_bits = [f"{fmt}: `{path}`" for fmt, path in sorted(asset["paths"].items())]
            alternate_paths = asset.get("alternate_paths", {})
            for fmt, paths in sorted(alternate_paths.items()):
                format_bits.extend(f"{fmt}-alt: `{path}`" for path in sorted(paths))
            csv_variants = asset.get("csv_variants", {})
            format_bits.extend(f"csv-{variant}: `{path}`" for variant, path in sorted(csv_variants.items()))
            lines.append(f"- {asset['title']} ({asset['id']}): " + ", ".join(format_bits))
        lines.append("")

    readme_path = paper_dir / "README.md"
    readme_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return manifest


def export_core_comparisons(metrics: pd.DataFrame, tables_dir: Path) -> List[Path]:
    """Compact manuscript comparisons plus explicitly separated supplements."""
    for identity in ("cohort_scope", "validation_scheme", "analysis_status"):
        if identity in metrics and metrics[identity].nunique(dropna=False) > 1:
            raise ValueError(f"Publication comparisons cannot mix {identity}")
    from reporting.feature_set_display import CORE_FEATURE_SET_ORDER
    names = {"preop_only": "Preoperative only", "all_waveforms": "All waveforms",
             "preop_and_all_waveforms": "Preoperative + all waveforms"}
    model_rank = {"Logistic regression": 0, "XGBoost": 1, "EBM": 2}
    files = []
    for outcome, group in metrics.groupby("Outcome"):
        ordered = group.copy()
        ordered["_model_rank"] = ordered.Model.map(model_rank).fillna(99)
        ordered["_feature_rank"] = ordered["Feature Set Key"].map({f: i for i, f in enumerate(CORE_FEATURE_SET_ORDER)}).fillna(99)
        ordered = ordered.sort_values(["Branch", "_model_rank", "_feature_rank", "Feature Set Key"])
        main_mask = ordered.Branch.eq("non_windowed") & ordered["Feature Set Key"].isin(CORE_FEATURE_SET_ORDER)
        for section, frame in (("main", ordered.loc[main_mask]), ("supplement", ordered.loc[~main_mask])):
            if frame.empty:
                continue
            identity = pd.DataFrame({"Model": frame.Model, "Branch": frame.Branch,
                                     "Feature set": frame["Feature Set Key"].map(names).fillna(frame["Feature Set Label"])})
            for column in ("n", "n_patients", "prevalence"):
                if column in frame:
                    identity[{"n": "Operations", "n_patients": "Patients", "prevalence": "Prevalence"}[column]] = frame[column]
            metric_names = ("AUPRC", "AUROC", "Brier Score", "Sensitivity", "Specificity", "Precision", "F1 Score", "Accuracy")
            table = pd.concat([identity, frame[[c for c in metric_names if c in frame]]], axis=1)
            # The complete CSV retains operating points and denominators. The
            # paper table concentrates on discrimination, calibration and gain.
            publication = identity[["Model", "Feature set"] if section == "main" else ["Model", "Branch", "Feature set"]].copy()
            if "Branch" in publication:
                publication["Branch"] = publication.Branch.map({"non_windowed": "Whole case", "windowed": "Windowed"}).fillna(publication.Branch)
            labels = {"auprc": "AP", "auroc": "AUROC", "brier": "Brier", "delta_auprc": "Δ AP"}
            for key, label in labels.items():
                if key not in frame:
                    continue
                def value(row, key=key):
                    if key == "delta_auprc" and row["Feature Set Key"] == "preop_only":
                        return "Reference"
                    point = f"{row[key]:.3f}"
                    if f"{key}_lower" in row and pd.notna(row[f"{key}_lower"]) and pd.notna(row[f"{key}_upper"]):
                        point += f" ({row[f'{key}_lower']:.3f} to {row[f'{key}_upper']:.3f})"
                    return point.replace("-", "−")
                publication[label] = frame.apply(value, axis=1)
            from results_recreation.results_analysis import METRIC_DISPLAY_MAP, get_delta_style_info, get_style_info
            styles = {}
            detailed_styles = {}
            for _, family in frame.groupby("Model"):
                for index, cells in get_style_info(family).items():
                    detailed_styles[index] = cells
                    styles[index] = {labels[key]: cells[old] for key, old in
                        (("auprc", "AUPRC"), ("auroc", "AUROC"), ("brier", "Brier Score"))
                        if old in cells}
            status = frame.analysis_status.iloc[0] if "analysis_status" in frame else "legacy"
            outcome_label = {"any_aki": "Acute kidney injury within 72 hours", "icu_admission": "ICU admission", "mortality": "In-hospital mortality"}.get(outcome, outcome)
            notes = "AP, average precision; AUROC, area under the receiver operating characteristic curve. The AP no-skill reference is the event prevalence. Lower Brier scores are better. Intervals are 95% patient-cluster bootstrap intervals when available, conditional on the saved fitted models. Δ AP compares each input set with the same model family's clinical-only reference. Shading and bold identify descriptive within-family ranks, not evidence of statistical superiority. Full operating-point metrics and denominators are retained in the companion CSV and consolidated report."
            if metrics.attrs.get("report_notes"):
                notes = " ".join([*metrics.attrs["report_notes"], notes])
            common_population = "n" in frame and all(frame[c].nunique(dropna=False) == 1 for c in ("n", "n_patients", "prevalence") if c in frame)
            if common_population:
                population = f"{int(frame['n'].iloc[0]):,} operations"
                if "n_patients" in frame and pd.notna(frame.n_patients.iloc[0]):
                    population += f" from {int(frame.n_patients.iloc[0]):,} patients"
                if "prevalence" in frame and pd.notna(frame.prevalence.iloc[0]):
                    population += f"; event prevalence {frame.prevalence.iloc[0]:.1%}"
                notes = population + ". " + notes
            elif "n" in frame:
                notes = "Populations differ across rows; use the companion CSV for each denominator. " + notes
            if status != "definitive":
                notes += f" Analysis status: {status}; these results are not definitive manuscript evidence."
            base = tables_dir / f"manuscript_{outcome}_{section}"
            files.extend(Path(value) for value in export_dataframe_bundle(
                base, title=f"{outcome_label}: {'whole-case' if section == 'main' else 'supplementary'} comparisons [{status}]",
                dataframe=table, display_dataframe=publication, notes=notes, styles=styles).values())
            files.extend(Path(value) for value in export_dataframe_bundle(
                tables_dir / f"manuscript_{outcome}_{section}_detailed",
                title=f"{outcome_label}: complete performance metrics [{status}]",
                dataframe=table, notes=notes, styles=detailed_styles).values())
            gain = frame.loc[frame["Feature Set Key"].eq("preop_and_all_waveforms")]
            if section == "main" and not gain.empty and {"delta_auprc", "delta_auprc_lower", "delta_auprc_upper"}.issubset(gain):
                fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
                for position, (_, row) in enumerate(gain.iterrows()):
                    ax.hlines(position, row.delta_auprc_lower, row.delta_auprc_upper, color=color("green"), linewidth=1.8)
                    ax.plot(row.delta_auprc, position, "o", color=color("green"), markersize=6)
                ax.axvline(0, color=color("reference"), linewidth=.9, linestyle="--")
                ax.set(yticks=range(len(gain)), yticklabels=gain.Model.tolist(),
                       xlabel="Change in AP: clinical + waveforms versus clinical only",
                       title=f"{outcome_label} [{status}]")
                ax.invert_yaxis()
                ax.spines[["top", "right"]].set_visible(False)
                destination = tables_dir.parent / "figures" / f"incremental_ap_{outcome}_main"
                files.extend(save_figure_bundle(fig, destination))
                source = gain[["Model", "delta_auprc", "delta_auprc_lower", "delta_auprc_upper"]].copy()
                source["analysis_status"] = status
                source["interval"] = "95% patient-cluster paired bootstrap; conditional on fitted predictions"
                source["comparison"] = "Same family, clinical + all waveforms minus clinical only"
                source.to_csv(destination.with_suffix(".csv"), index=False)
                files.append(destination.with_suffix(".csv"))
            delta_columns = [f"Δ {column}" for column in metric_names if f"Δ {column}" in frame]
            if delta_columns:
                delta = pd.concat([identity, frame[delta_columns]], axis=1)
                files.extend(Path(value) for value in export_dataframe_bundle(
                    tables_dir / f"manuscript_{outcome}_{section}_delta",
                    title=f"{outcome}: paired differences versus same-family preoperative-only reference",
                    dataframe=delta, notes=notes,
                    styles=get_delta_style_info(frame, {f"Δ {label}": f"delta_{key}"
                                                        for key, label in METRIC_DISPLAY_MAP.items()})).values())
    return files
