"""Build the November 21 publication displays without modifying any source workspace."""

import argparse
from copy import deepcopy
import logging
from pathlib import Path
import re
import shutil

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
import pandas as pd

from reporting import publication_style as style
from . import revision_figures as figures
from .revision_data import FAMILIES, INPUTS, Study, digest, verify_metric_sources, write_json
from .revision_tables import LABS, build_cohort, build_predictors, expanded_performance


MAIN = ("Figure 1", "Table 1", "Table 2", "Figure 2", "Figure 3", "Figure 4", "Figure 5")
SUPPLEMENT = ("Table S1", "Table S2", *(f"Figure S{i}" for i in range(3, 11)))
CARRIED = {
    "Figure 4": "Figure4_EBM_global_importance", "Figure 5": "Figure5_EBM_selected_functions",
    "Figure S10": "S11_EBM_heldout_cases",
}
OLD = {
    "Figure 1": "Main Figure 1", "Table 1": "Main Table 1", "Table 2": "Main Table 2",
    "Figure 2": "Main Figure 2 and old Figure S5", "Figure 3": "Old Figure S2",
    "Figure 4": "Main Figure 4", "Figure 5": "Main Figure 5", "Table S1": "New",
    "Table S2": "New", "Figure S3": "Old Figure S1", "Figure S4": "Old Main Figure 3",
    "Figure S5": "Old Figure S3", "Figure S6": "Old Figure S4", "Figure S7": "Old Figure S6",
    "Figure S8": "Old Figure S7", "Figure S9": "Old Figure S8",
    "Figure S10": "Old Figure S10 (held-out cases, previously numbered S11)",
}
FIGURE_TITLES = {
    "Figure 1": "Study cohort selection",
    "Figure 2": "Incremental discrimination from waveforms",
    "Figure 3": "Whole-operation receiver operating characteristic curves",
    "Figure 4": "Global EBM feature importance",
    "Figure 5": "EBM feature-response functions",
    "Figure S3": "Whole-operation precision–recall curves",
    "Figure S4": "Whole-operation calibration",
    "Figure S5": "Absolute average precision by waveform channel",
    "Figure S6": "Absolute AUROC by waveform channel",
    "Figure S7": "Added discrimination for ICU admission by cohort",
    "Figure S8": "Decision curves",
    "Figure S9": "Paired net benefit differences",
    "Figure S10": "Held-out EBM prediction explanations",
}
TERMS = {
    "AKI": "acute kidney injury",
    "ALT": "alanine aminotransferase",
    "AP": "average precision",
    "aPTT": "activated partial thromboplastin time",
    "ASA-PS": "American Society of Anesthesiologists physical status",
    "AST": "aspartate aminotransferase",
    "AUROC": "area under the receiver operating characteristic curve",
    "BMI": "body mass index",
    "BUN": "blood urea nitrogen",
    "CKD-EPI": "Chronic Kidney Disease Epidemiology Collaboration",
    "CO₂": "carbon dioxide",
    "CRP": "C-reactive protein",
    "DFA": "detrended fluctuation analysis",
    "EBM": "explainable boosting machine",
    "ECG": "electrocardiography",
    "eGFR": "estimated glomerular filtration rate",
    "ICU": "intensive care unit",
    "LR": "logistic regression",
    "PaCO₂": "arterial partial pressure of carbon dioxide",
    "PaO₂": "arterial partial pressure of oxygen",
    "PPG": "photoplethysmography",
    "PT": "prothrombin time",
    "SD": "standard deviation",
    "SpO₂": "peripheral oxygen saturation",
    "WBC": "white blood cell count",
}


def abbreviation_key(*terms):
    return "; ".join(f"{term}, {TERMS[term]}" for term in terms) + "."


OUTCOME_KEY = ("AKI", "ICU")
MODEL_KEY = (*OUTCOME_KEY, "LR", "EBM")
RESPONSE_NOTE = (
    "Curves show fold-specific changes in model log-odds relative to the indicated reference values, before recalibration. "
    "Histograms show feature distributions from the first outer training fold. "
)
WHISKER_NOTE = "Whiskers indicate 95% CIs. "
CHANNEL_NOTE = "Waveform-only models. " + WHISKER_NOTE
CAPTIONS = {
    "Figure 1": abbreviation_key("ASA-PS", *OUTCOME_KEY),
    "Figure 2": (
        "Combined minus matched preoperative-only performance. "
        + WHISKER_NOTE + abbreviation_key(*MODEL_KEY, "AP", "AUROC")
    ),
    "Figure 3": (
        "Pooled held-out predictions. AUROCs follow legend order. "
        + abbreviation_key(*MODEL_KEY, "AUROC")
    ),
    "Figure 4": (
        "Importance represents the mean absolute contribution to model log-odds, weighted by the training distribution. "
        "Bars show means across five folds; symbols show individual folds. "
        + abbreviation_key(*OUTCOME_KEY, "EBM", "ASA-PS", "DFA", "ECG", "PPG")
    ),
    "Figure 5": (
        "Combined whole-operation models. " + RESPONSE_NOTE + abbreviation_key(*OUTCOME_KEY, "EBM", "eGFR")
    ),
    "Figure S3": (
        "Pooled held-out predictions. AP follows legend order. "
        + abbreviation_key(*MODEL_KEY, "AP")
    ),
    "Figure S4": (
        "Up to ten equal-frequency bins of pooled held-out predictions. "
        + abbreviation_key(*MODEL_KEY)
    ),
    "Figure S5": (
        CHANNEL_NOTE + abbreviation_key(*MODEL_KEY, "ECG", "PPG", "CO₂")
    ),
    "Figure S6": (
        CHANNEL_NOTE + abbreviation_key(*MODEL_KEY, "AUROC", "ECG", "PPG", "CO₂")
    ),
    "Figure S7": (
        "Combined minus matched preoperative-only performance within each cohort. "
        + WHISKER_NOTE + abbreviation_key("ICU", "LR", "EBM", "AP", "AUROC")
    ),
    "Figure S8": (
        "Whole-operation net benefit per 100 operations. Arrows mark treat-all values below the plotted range. "
        + abbreviation_key(*MODEL_KEY)
    ),
    "Figure S9": (
        "Combined minus matched preoperative-only net benefit per 100 operations (whole-operation models). "
        "Shading indicates pointwise 95% CIs. "
        + abbreviation_key(*MODEL_KEY)
    ),
    "Figure S10": (
        "Cases nearest the indicated percentiles of whole-operation combined-model risk. Contributions add to the intercept on the "
        "calibrated log-odds scale. Preoperative-only risks use separate fits. "
        + abbreviation_key(*OUTCOME_KEY, "EBM", "ASA-PS", "ECG", "PPG").removesuffix(".")
        + "; BUN, blood urea nitrogen (mg/dL); Hb, hemoglobin (g/dL); WBC, white blood cell count (10³/µL)."
    ),
}


def select_tables(frozen, revised, name, family_field, family_name):
    base = frozen.table(name)
    if revised is None:
        return base
    replacement = revised.table(name)
    if not replacement[family_field].eq(family_name).all():
        raise ValueError(f"Unexpected model family in revised LR table {name}")
    unchanged = base[~base[family_field].eq(family_name)]
    return pd.concat([unchanged, replacement], ignore_index=True)


def document(title):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(.56), Inches(.57)
    section.left_margin, section.right_margin = Inches(.72), Inches(.72)
    style.apply_document_style(doc, body_size=10)
    doc.styles["Normal"].paragraph_format.space_after = Pt(4)
    doc.core_properties.title = title
    return doc


def heading(doc, title, page_break=False):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.page_break_before = page_break
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(12)
    return p


def note(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_before = Pt(4)
    for run in p.runs:
        run.font.size = Pt(9)
    return p


def add_table(doc, widths):
    table = doc.add_table(rows=0, cols=len(widths))
    table.autofit = False
    table.style = "Table Grid"
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH", "left", "right", "insideV"):
        border = OxmlElement("w:" + edge)
        horizontal = edge in ("top", "bottom", "insideH")
        border.set(qn("w:val"), "single" if horizontal else "nil")
        if horizontal:
            border.set(qn("w:sz"), "4")
            border.set(qn("w:color"), "D4D8DC")
        borders.append(border)
    table._tbl.tblPr.append(borders)
    for col, width in zip(table.columns, widths):
        col.width = Inches(width)
    return table


def row(table, values, *, header=False, band=False, merge=(), font_size=9, compact=False, bold_columns=()):
    cells = table.add_row().cells
    tr_pr = cells[0]._tc.getparent().get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))
    if header:
        tr_pr.append(OxmlElement("w:tblHeader"))
    hidden = set()
    for start, stop in merge:
        cells[start].merge(cells[stop])
        hidden.update(range(start + 1, stop + 1))
    cells = [cell for i, cell in enumerate(cells) if i not in hidden]
    for i, (cell, value) in enumerate(zip(cells, values)):
        cell.text = str(value)
        if header or band:
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "EEF0F1")
            cell._tc.get_or_add_tcPr().append(shade)
        for p in cell.paragraphs:
            p.paragraph_format.space_before = Pt(0 if compact else 1.5)
            p.paragraph_format.space_after = Pt(0 if compact else 1.5)
            if compact:
                p.paragraph_format.line_spacing = Pt(font_size + 1.1)
            p.paragraph_format.keep_with_next = header or band
            for run in p.runs:
                run.font.size = Pt(font_size)
                run.bold = header or band or i in bold_columns


def add_frame(doc, frame, widths):
    table = add_table(doc, widths)
    row(table, frame.columns, header=True)
    for items in frame.itertuples(index=False, name=None):
        row(table, items)
    return table


def add_predictors(doc, predictors):
    heading(doc, "Table S1. Predictors and feature extraction")

    widths = (1.46, 1.71, 1.81, 2.08)
    table = add_table(doc, widths)
    row(table, ("Predictor", "Model input", "Unit or coding", "VitalDB clinical-table exceptions"),
        band=True, compact=True)
    clinical = predictors["clinical"]
    categorical = clinical["Type or unit"].str.startswith("Categorical.")
    for section, fields in (("Categorical preoperative predictors", clinical[categorical]),
                            ("Numeric preoperative predictors", clinical[~categorical])):
        row(table, [section], band=True, merge=((0, 3),), compact=True)
        for label, field, unit, source in fields.itertuples(index=False, name=None):
            if source == "VitalDB clinical table":
                source = ""
            elif field in LABS:
                source = f"Lab records ({LABS[field]})"
            row(table, (label.removeprefix("Preoperative "), field,
                        unit.removeprefix("Categorical. ").removeprefix("Continuous, "), source),
                compact=True)
    row(table, ["Waveform sources and preprocessing"], band=True, merge=((0, 3),))
    row(table, predictors["channels"].columns, band=True)
    for values in predictors["channels"].itertuples(index=False, name=None):
        row(table, values)
    row(table, ["Crop from operation start to end. Reject segments with >5% nonfinite samples. Interpolate gaps "
                "before zero-phase Butterworth filtering (design N = 4) and polyphase antialiasing resampling."],
        merge=((0, 3),))
    row(table, ["Feature construction"], band=True, merge=((0, 3),), compact=True)
    row(table, ("", "Whole-operation", "Windowed"), band=True, merge=((2, 3),), compact=True)
    for values in (
        ("Sampling rate", "10 Hz for all waveforms", "100 Hz for ECG/PPG; 62.5 Hz for CO₂/airway pressure"),
        ("Feature extraction", "Catch24 on the complete operative waveform",
         "Catch24 on complete 10-second windows with approximately 5-second overlap"),
        ("Aggregation", "None", "Mean, sample SD, minimum and maximum across windows"),
        ("Features per channel", "24", "96"),
        ("Total waveform features", "96", "384"),
    ):
        row(table, values, merge=((2, 3),), compact=True)
    row(table, ["Window stride is 500 samples at 100 Hz and 312 at 62.5 Hz. "
                "Single-window SD is missing. Undefined descriptors propagate to aggregates."],
        merge=((0, 3),), compact=True)
    row(table, ["Catch24 descriptors"], band=True, merge=((0, 3),))
    row(table, ["Canonical descriptor", "Description"], band=True, merge=((0, 1), (2, 3)))
    for key, value in predictors["descriptors"].itertuples(index=False, name=None):
        row(table, [key, value], merge=((0, 1), (2, 3)), font_size=8.5, compact=True)
    note(doc, "Before model encoding, the preoperative feature set contained 38 predictors. Whole-operation "
              "and windowed waveform representations contributed 96 and 384 features, respectively. "
              "Combined models included both preoperative and waveform features. The number of encoded model "
              "inputs varied across folds and model families because of categorical encoding and missingness indicators.")
    note(doc, "Catch22 descriptors use internally standardized signals. "
              + abbreviation_key("ALT", "ASA-PS", "AST", "aPTT", "CKD-EPI",
                                 "PaCO₂", "PaO₂", "PT", "SD", "SpO₂"))


def add_image(doc, label, image, caption, *, page_break=True):
    heading(doc, f"{label}. {FIGURE_TITLES[label]}", page_break=page_break)
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    # A 4,110-pixel publication export provides 600 dpi at 6.85 inches.
    p.add_run().add_picture(str(image), width=Inches(6.35 if label == "Figure S10" else 6.85))
    caption_para = note(doc, caption)
    caption_para.paragraph_format.keep_together = True


def main_book(source, performance, ecg_counts, art, captions, out):
    title1 = next(p.text for p in source.paragraphs if p.text.startswith("Table 1."))
    doc = document("Main figures and tables")
    add_image(doc, "Figure 1", art["Figure 1"], captions["Figure 1"], page_break=False)
    heading(doc, title1, page_break=True)
    original = deepcopy(source.tables[0]._tbl)
    doc.element.body.insert(len(doc.element.body)-1, original)
    cohort_table = doc.tables[0]
    for entry in cohort_table.rows:
        label_cell = entry.cells[0]
        if label_cell.text == "eGFR <60 mL/min/1.73 m²":
            entry._tr.getparent().remove(entry._tr)
        elif label_cell.text.startswith("ASA "):
            run = label_cell.paragraphs[0].runs[0]
            run.text = run.text.replace("ASA ", "ASA-PS ", 1)
    procedure = next(entry for entry in cohort_table.rows if entry.cells[0].text == "Procedure type")
    for paragraph in procedure.cells[2].paragraphs:
        for run in paragraph.runs:
            if run.text:
                run.bold = False
    total = sum(ecg_counts.values())

    def count_pct(count):
        return f"{count:,} ({count / total:.1%})"

    ecg_rows = [("Preoperative ECG finding", "", count_pct(ecg_counts["Missing"]))]
    ecg_rows.extend((f" {label}", count_pct(ecg_counts[label]), "")
                    for label in ("Normal sinus rhythm", "Abnormal", "Unknown")
                    if label != "Unknown" or ecg_counts[label])
    for values in ecg_rows:
        row(cohort_table, values, compact=True)
        entry = cohort_table.rows[-1]
        for cell in entry.cells[1:]:
            cell.paragraphs[0].alignment = procedure.cells[2].paragraphs[0].alignment
        procedure._tr.addprevious(entry._tr)
    note(doc, "Median (first, third quartiles) or n (%). "
              + abbreviation_key(*OUTCOME_KEY, "ASA-PS", "BMI", "ECG", "eGFR", "BUN", "WBC", "CRP"))
    heading(doc, "Table 2. Performance by input configuration", page_break=True)
    table = add_table(doc, (.78, .67, 2.40, 1.07, 1.07, 1.07))
    row(table, performance.columns, header=True, compact=True)
    score_columns = ["AP", "AUROC", "Brier score"]
    for _, group in performance.groupby(["Outcome", "Model"], sort=False):
        scores = group[score_columns].apply(lambda col: col.str.split("\n").str[0].astype(float))
        best = [scores.AP.max(), scores.AUROC.max(), scores["Brier score"].min()]
        for i, values in enumerate(group.itertuples(index=False, name=None)):
            shown = values if i == 0 else ("", "", *values[2:])
            bold = tuple(3+j for j, (value, top) in enumerate(zip(scores.iloc[i], best)) if value == top)
            row(table, shown, font_size=8.5, compact=True, bold_columns=bold)
    note(doc, "Estimates (95% CIs). "
              "Bold marks the highest AP and AUROC and lowest Brier score within each outcome and model. "
              + abbreviation_key(*MODEL_KEY, "AP", "AUROC"))
    for label in MAIN[3:]:
        add_image(doc, label, art[label], captions[label])
    doc.save(out / "Main_Figures_and_Tables.docx")


def supplement_book(predictors, cohort, art, captions, out):
    doc = document("Supplementary figures and tables")
    add_predictors(doc, predictors)
    heading(doc, "Table S2. Main and broader ICU cohorts", page_break=True)
    labels = {
        "Source baseline creatinine absent or invalid": "No finite positive baseline creatinine",
        "No finite positive creatinine result by 72 h after operation end":
            "No finite positive postoperative creatinine by 72 h",
        "AKI ascertainment unavailable under study criteria": "Ineligible for AKI ascertainment",
    }
    add_frame(doc, cohort.assign(Characteristic=cohort["Characteristic"].replace(labels)), (3.84, 1.61, 1.61))
    note(doc, "Median (first, third quartiles) or n (%). " + abbreviation_key(*OUTCOME_KEY, "BMI", "ASA-PS"))
    for label in SUPPLEMENT[2:]:
        add_image(doc, label, art[label], captions[label])
    doc.save(out / "Supplementary_Figures_and_Tables.docx")


def verify_books(output, captions):
    inventory = {}
    for filename, entries in (("Main_Figures_and_Tables.docx", MAIN),
                              ("Supplementary_Figures_and_Tables.docx", SUPPLEMENT)):
        doc = Document(output / filename)
        labels = []
        paragraphs = doc.paragraphs
        for index, paragraph in enumerate(paragraphs):
            match = re.match(r"^((?:Figure|Table) S?\d+)\. ", paragraph.text)
            if match:
                label = match.group(1)
                labels.append(label)
                if label in FIGURE_TITLES:
                    lead = paragraph.runs[0]
                    drawing, caption = paragraphs[index + 1:index + 3]
                    if (lead.text != f"{label}. {FIGURE_TITLES[label]}" or not lead.bold
                            or len(drawing._p.xpath(".//wp:inline")) != 1
                            or caption.text != captions[label]):
                        raise ValueError(f"Expected title, artwork and note for {label} in {filename}")
        figures = sum(label.startswith("Figure") for label in entries)
        if (tuple(labels) != entries or len(doc.inline_shapes) != figures
                or len(doc.tables) != len(entries) - figures
                or doc.element.xpath(".//w:footerReference")):
            raise ValueError(f"Invalid publication inventory or visible navigation in {filename}")
        resolutions = {}
        for label, shape in zip((label for label in entries if label.startswith("Figure")),
                                doc.inline_shapes, strict=True):
            embedded = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
            part = doc.part.related_parts[embedded]
            x_dpi = part.image.px_width / shape.width.inches
            y_dpi = part.image.px_height / shape.height.inches
            if part.content_type != "image/png" or min(x_dpi, y_dpi) < style.PNG_DPI - .001:
                raise ValueError(f"{label} must be a PNG at {style.PNG_DPI} dpi or higher at its placed size")
            resolutions[label] = {"pixels": [part.image.px_width, part.image.px_height],
                                  "placed_dpi": [round(x_dpi, 2), round(y_dpi, 2)]}
        inventory[filename] = {"sha256": digest(output / filename), "display_order": labels,
                               "embedded_figures": figures, "embedded_tables": len(doc.tables),
                               "figure_resolution": resolutions}
    return inventory


def assemble(args):
    response_path = args.explainability / (CARRIED["Figure 5"] + ".plotted.csv")
    response = pd.read_csv(response_path, usecols=["panel", "feature"])
    if set(response.loc[response.panel.eq("D"), "feature"]) != {"preop_egfr_ckdepi_2021"}:
        raise ValueError(f"Expected the approved eGFR response for Figure 5C in {response_path}")

    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "display_sources"
    assets.mkdir(exist_ok=True)
    core = Study(args.studies / "manuscript-core-v1", "main_shared")
    main = Study(args.studies / "manuscript-supplementary-v1", "main_shared")
    broader = Study(args.studies / "manuscript-optimized-v1", "icu_waveform_sensitivity")
    revised = {scope: Study(args.lr / scope, scope) if args.lr else None
               for scope in ("main_shared", "icu_waveform_sensitivity")}
    if args.lr:
        for study in revised.values():
            if any(config["model_type"] != "logistic_regression" or config["n_trials"] != 50
                   or config.get("class_weighting") != "fold_training_class_weight_balanced"
                   for config in study.configs.values()):
                raise ValueError("LR replacement requires a dedicated balanced 50-trial study")
    selections = {
        scope: {family: (revised[scope] or study) if family == "logistic_regression" else study
                for family in FAMILIES}
        for scope, study in (("main_shared", main), ("icu_waveform_sensitivity", broader))
    }
    main_lr, broader_lr = revised.values()
    metrics = select_tables(main, main_lr, "metrics_summary", "model_name", "Logistic regression")
    broad_metrics = select_tables(broader, broader_lr, "metrics_summary", "model_name", "Logistic regression")
    prediction_evidence = verify_metric_sources(selections["main_shared"], metrics)
    prediction_evidence.update(verify_metric_sources(selections["icu_waveform_sensitivity"], broad_metrics, broader=True))
    absolute = select_tables(core, main_lr, "decision_curve_summary", "model_type", "logistic_regression")
    paired = select_tables(core, main_lr, "decision_curve_comparisons", "candidate_model", "logistic_regression")
    performance, baseline_evidence = expanded_performance(metrics, selections["main_shared"])

    style.apply_mpl_style()
    flow_counts = args.workspace / "03_FIGURES/paper/inputs/main/figure_1_counts.json"
    figures.cohort_selection(flow_counts, core.manifest["branch_rosters"]["non_windowed"], assets)
    figures.incremental(metrics, assets)
    figures.cohort_incremental(metrics, broad_metrics, assets)
    figures.channels(metrics, assets)
    figures.discrimination(selections["main_shared"], metrics, assets)
    figures.calibration(selections["main_shared"], assets)
    figures.decision_curves(absolute, paired, metrics, assets)

    predictors = build_predictors(selections, args.workspace)
    if len(predictors["clinical"]) != 38:
        raise ValueError("Selected fits do not match the publication's 38-raw-predictor input contract")
    cohort, cohort_sources, ecg_counts = build_cohort(core, broader, args.workspace)
    performance.to_csv(assets / "Table_2_display.csv", index=False)
    metrics[metrics.feature_set.isin(INPUTS)].to_csv(assets / "Table_2_source_metrics.csv", index=False)
    for key in ("clinical", "channels", "descriptors"):
        predictors[key].to_csv(assets / f"Table_S1_{key}.csv", index=False)
    cohort.to_csv(assets / "Table_S2_display.csv", index=False)

    author = Document(args.main_docx)
    art = {}
    captions = dict(CAPTIONS)
    for label in MAIN + SUPPLEMENT:
        if not label.startswith("Figure"):
            continue
        stem = CARRIED.get(label)
        if stem:
            for suffix in (".png", ".plotted.csv", ".training_support.csv"):
                source_file = args.explainability / (stem + suffix)
                if suffix != ".training_support.csv" or source_file.exists():
                    shutil.copyfile(source_file, assets / (label.replace(" ", "_") + suffix))
        art[label] = assets / (label.replace(" ", "_") + ".png")
    for label, caption in captions.items():
        (assets / (label.replace(" ", "_") + ".caption.txt")).write_text(caption + "\n")
    main_book(author, performance, ecg_counts, art, captions, args.output)
    supplement_book(predictors, cohort, art, captions, args.output)

    protected = [core, main, broader, Study(args.studies / "manuscript-icu-supplementary-v1", "icu_waveform_sensitivity")]
    selected_metrics = ((main, "metrics_summary"), (broader, "metrics_summary"),
                        (core, "decision_curve_summary"), (core, "decision_curve_comparisons"))
    if args.lr:
        selected_metrics += ((main_lr, "metrics_summary"), (broader_lr, "metrics_summary"),
                             (main_lr, "decision_curve_summary"), (main_lr, "decision_curve_comparisons"))
    table_paths = [study.root / "tables" / (name + ".csv") for study, name in selected_metrics]
    dictionaries = [args.workspace / "04_DATA/raw_source" / name
                    for name in ("clinical_parameters.csv", "lab_parameters.csv")]
    evidence = {
        "lr_protocol": "balanced, 50 trials" if args.lr else "frozen unweighted, 20 trials",
        "style_profile_sha256": style.STYLE_SHA256,
        "manifests": [study.evidence() for study in protected + [s for s in revised.values() if s]],
        "inputs": {str(p): digest(p)
                   for p in (args.main_docx, flow_counts, *cohort_sources.values(), *dictionaries, *table_paths)},
        "main": list(MAIN), "supplement": list(SUPPLEMENT),
        "previous_display": OLD, "figure_titles": FIGURE_TITLES, "captions": captions,
        "baseline_pairs": baseline_evidence, "table_1_preoperative_ecg_counts": ecg_counts,
        "table_2_rows": len(performance),
        "predictor_dimensions": predictors["dimensions"], "fitted_input_evidence": predictors["audited"],
        "prediction_evidence": prediction_evidence, "validated_books": verify_books(args.output, captions),
        "figure_2_entries": 24, "figure_s5_entries": 60, "figure_s6_entries": 60, "figure_s7_entries": 24,
        "figures": {label: {"file": str(file), "sha256": digest(file)} for label, file in art.items()},
    }
    write_json(args.output / "display_manifest.json", evidence)
    print(f"Completed {len(MAIN)} main and {len(SUPPLEMENT)} supplementary displays with {len(performance)} performance rows.")


def main():
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "studies", "main-docx", "explainability", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--lr", type=Path, help="Root of the validated balanced-LR studies, containing a directory per cohort")
    assemble(parser.parse_args())


if __name__ == "__main__":
    main()
