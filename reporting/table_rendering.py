"""Table rendering shared by research reports and the offline manuscript workspace."""

from __future__ import annotations

from html import unescape
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
import pandas as pd

if __package__:
    from .publication_style import WORD_FONT, WORD_TABLE_PT, color
else:
    from publication_style import WORD_FONT, WORD_TABLE_PT, color


def _clean_cell(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unescape(text)
    return text.replace("&nbsp;", " ").replace("\n", "<br>")


def dataframe_to_markdown(df: pd.DataFrame, styles=None, *, bold_empty=True) -> str:
    """Render a dataframe as a plain Markdown table without optional deps."""

    rows = []
    for index, row in zip(df.index, df.itertuples(index=False, name=None)):
        cells = []
        row_styles = (styles or {}).get(index, {})
        for column, value in zip(df.columns, row):
            cell = _clean_cell(value)
            if (bold_empty or cell.strip()) and row_styles.get(column, {}).get("bold"):
                cell = f"**{cell}**"
            cells.append(cell)
        rows.append(cells)
    headers = [_clean_cell(column) for column in df.columns]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _format_row(cells: Sequence[str]) -> str:
        padded = [cell.ljust(widths[idx]) for idx, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def numeric_display_columns(display: pd.DataFrame) -> set:
    """Recognize supplied numeric/interval strings while retaining text identities."""
    identities = {"model", "model_name", "outcome", "branch", "feature set", "feature_set",
                  "input set", "feature set key", "pipeline", "inputs", "inputs text", "caseid", "subjectid"}
    metric_labels = {"ap", "auprc", "auroc", "brier", "brier score", "sensitivity",
                     "specificity", "precision", "f1", "f1 score", "accuracy"}
    result = set()
    for column in display.columns:
        label = str(column).lower().removeprefix("δ ").strip()
        values = [str(value).strip() for value in display[column].dropna()
                  if str(value).strip() not in {"", "—", "Reference"}]
        prepared = bool(values) and all(re.fullmatch(
            r"[+−-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+−-]?\d+)?%?(?:\s*\(.*\))?", value)
            for value in (v.replace(",", "") for v in values))
        if label not in identities and (label in metric_labels or pd.api.types.is_numeric_dtype(display[column]) or prepared):
            result.add(column)
    return result


def symbol_display_columns(display: pd.DataFrame) -> set:
    return {column for column in display if set(display[column].dropna().astype(str)) <= {"✓", "—", ""}
            and display[column].astype(str).eq("✓").any()}


def _table_widths(df):
    weights = [max(10, min(30, max([len(str(c))] + [len(str(v).split(" (")[0]) for v in df[c]]))) for c in df]
    return [weight / sum(weights) for weight in weights]


def _add_dataframe_to_doc(document: Document, df: pd.DataFrame, styles=None) -> None:
    table = document.add_table(rows=1 + len(df), cols=len(df.columns))
    table.style = None
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH", "left", "right", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single" if edge in {"top", "bottom", "insideH"} else "nil")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), color("table_rule").lstrip("#"))
        borders.append(border)
    table._tbl.tblPr.append(borders)
    table.autofit = False
    repeat_header = OxmlElement("w:tblHeader")
    table.rows[0]._tr.get_or_add_trPr().append(repeat_header)
    for row in table.rows:
        keep_row = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(keep_row)
    widths = _table_widths(df)
    available = document.sections[-1].page_width - document.sections[-1].left_margin - document.sections[-1].right_margin

    numeric_columns = numeric_display_columns(df)
    symbol_columns = symbol_display_columns(df)
    header_cells = table.rows[0].cells
    for idx, column in enumerate(df.columns):
        header_cells[idx].width = int(available * widths[idx])
        table.columns[idx].width = int(available * widths[idx])
        header_cells[idx].text = str(column)
        if header_cells[idx].paragraphs and header_cells[idx].paragraphs[0].runs:
            header_run = header_cells[idx].paragraphs[0].runs[0]
            header_run.bold = True
            header_run.font.name = WORD_FONT
            header_run.font.size = Pt(WORD_TABLE_PT)
        header_cells[idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if column in symbol_columns else WD_ALIGN_PARAGRAPH.RIGHT if column in numeric_columns else WD_ALIGN_PARAGRAPH.LEFT

    for row_idx, row in enumerate(df.itertuples(index=False, name=None), start=1):
        cells = table.rows[row_idx].cells
        for col_idx, value in enumerate(row):
            cells[col_idx].width = int(available * widths[col_idx])
            cells[col_idx].text = _clean_cell(value).replace("<br>", "\n")
            cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if df.columns[col_idx] in symbol_columns else WD_ALIGN_PARAGRAPH.RIGHT if df.columns[col_idx] in numeric_columns else WD_ALIGN_PARAGRAPH.LEFT
            style = (styles or {}).get(df.index[row_idx - 1], {}).get(df.columns[col_idx], {})
            if style.get("top_rule"):
                cell_borders = OxmlElement("w:tcBorders")
                top = OxmlElement("w:top")
                for key, value in {"val": "single", "sz": "6", "color": color("reference").lstrip("#")}.items():
                    top.set(qn("w:" + key), value)
                cell_borders.append(top)
                cells[col_idx]._tc.get_or_add_tcPr().append(cell_borders)
            if style.get("bg"):
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), style["bg"].lstrip("#"))
                cells[col_idx]._tc.get_or_add_tcPr().append(shading)
            for paragraph in cells[col_idx].paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1
                for run in paragraph.runs:
                    run.font.name = WORD_FONT
                    run.font.size = Pt(WORD_TABLE_PT)
                    run.bold = bool(style.get("bold"))


def write_markdown_sections(
    path: Path,
    *,
    title: str,
    sections: Sequence[Mapping[str, object]],
    bold_empty=True,
) -> Path:
    """Write a Markdown document containing one or more table sections."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for section in sections:
        heading = str(section["heading"])
        df = section.get("dataframe")
        text = section.get("text")
        if heading != title:
            lines.extend([f"## {heading}", ""])
        if text:
            lines.append(str(text))
            lines.append("")
        if isinstance(df, pd.DataFrame) and not df.empty:
            lines.append(dataframe_to_markdown(df, section.get("styles"), bold_empty=bold_empty))
            lines.append("")
        elif isinstance(df, pd.DataFrame):
            lines.append("_No rows._")
            lines.append("")
        if section.get("notes"):
            lines.extend([str(section["notes"]), ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def dataframe_to_html(
    display: pd.DataFrame, *, title: str, notes: str | None = None,
    styles: dict | None = None,
) -> str:
    """Render one styled reading table without creating duplicate format files."""
    # Preserve prepared display strings and raw numeric precision consistently
    # with Markdown; pandas' default HTML formatter otherwise rounds to 6 places.
    styler = display.style.hide(axis="index").format(
        lambda value: "" if pd.isna(value) else str(value), escape="html")
    numeric_columns = numeric_display_columns(display)
    symbol_columns = symbol_display_columns(display)
    column_styles = []
    for position, column in enumerate(display.columns):
        numeric = column in numeric_columns
        props = [("text-align", "center" if column in symbol_columns else "right" if numeric else "left")]
        if numeric:
            props.append(("font-variant-numeric", "tabular-nums"))
        column_styles.append({"selector": f"th.col_heading.col{position}, td.col{position}", "props": props})
    styler.set_table_styles(column_styles)
    if styles:
        styler = styler.apply(lambda column: [
            ("background-color:" + styles.get(i, {}).get(column.name, {}).get("bg", "white") + ";")
            + ("font-weight:bold;" if styles.get(i, {}).get(column.name, {}).get("bold") else "")
            + ("border-top:1px solid " + color("reference") + ";" if styles.get(i, {}).get(column.name, {}).get("top_rule") else "")
            for i in column.index], axis=0)
    from html import escape
    html = '<!doctype html><meta charset="utf-8"><style>body{font:11pt Georgia,serif;color:#222;margin:2em;max-width:1100px}h1{font-size:15pt}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ddd}thead th{border-top:1px solid #777;border-bottom:1px solid #777}p{font-size:9pt;line-height:1.5}@media print{body{margin:0;max-width:none}thead{display:table-header-group}tfoot{display:table-footer-group}tr{break-inside:avoid}th,td{padding:5px 7px}}</style>'
    html = html.replace('font:11pt Georgia,serif', 'font:11pt ' + WORD_FONT + ',sans-serif').replace('color:#222', 'color:' + color('ink')).replace('#ddd', color('table_rule')).replace('#777', color('reference'))
    html += f"<h1>{escape(title)}</h1>" + styler.to_html() + f"<p>{escape(notes or '')}</p>"
    return html


def export_dataframe_bundle(
    base_path: Path,
    *,
    title: str,
    dataframe: pd.DataFrame,
    include_csv: bool = True,
    display_dataframe: pd.DataFrame | None = None,
    notes: str | None = None,
    styles: dict | None = None,
    write_docx: Callable[..., Path],
    write_pdf: Callable[..., Path],
    bold_empty=True,
) -> dict[str, str]:
    """Export a single table to manuscript-friendly formats."""

    base = base_path.with_suffix("") if base_path.suffix else base_path
    display = dataframe if display_dataframe is None else display_dataframe
    sections = [{"heading": title, "dataframe": display, "notes": notes, "styles": styles}]
    outputs = {}
    if include_csv:
        csv_path = base.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)
    outputs["md"] = str(write_markdown_sections(
        base.with_suffix(".md"), title=title, sections=sections, bold_empty=bold_empty,
    ))
    for suffix, writer in (("docx", write_docx), ("pdf", write_pdf)):
        outputs[suffix] = str(writer(base.with_suffix(f".{suffix}"), title=title, sections=sections))
    html = dataframe_to_html(display, title=title, notes=notes, styles=styles)
    base.with_suffix(".html").write_text(html, encoding="utf-8")
    outputs["html"] = str(base.with_suffix(".html"))
    return outputs
