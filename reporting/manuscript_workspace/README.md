# Manuscript workspace code

## Revised Word books

Run from the repository in `aki_prediction_project`. Set `workspace` to the
unpacked November 4 workspace, `study_root` to the directory containing study
runs, and `main_display_docx` to the original author file
`Main_Figures_and_Tables (2).docx`. Use fresh directories for `assets` and
`display_output`.

```bash
assets=/path/to/ebm-display-assets
templates=reporting/manuscript_workspace/templates/05_CODE
export PYTHONPATH="$PWD:$PWD/reporting${PYTHONPATH:+:$PYTHONPATH}"
python "$templates/export_icu_ebm_responses.py" --study "$study_root/manuscript-core-v1" --output "$assets/icu_ebm_response_sources.json"
python "$templates/rebuild_narrative_figures.py" --workspace "$workspace" --ebm-revision-output "$assets"
python "$templates/rebuild_case_figure.py" --source "$workspace/03_FIGURES/heldout_cases" --revision-output "$assets"
python -m reporting.manuscript_workspace.revise_publication \
  --workspace "$workspace" --studies "$study_root" --main-docx "$main_display_docx" \
  --explainability "$assets" --output "$display_output" \
  --lr "$study_root/manuscript-lr-balanced-50-v1"
```

Omit `--lr` to reproduce the frozen unweighted LR displays. The rebuild uses
saved results and writes Word books, 600 dpi PNGs, source tables and a manifest.
Render PDF proofs separately with lossless image compression and image
downsampling disabled.

To reproduce the retired capnogram figure, add `--workspace "$workspace"` to
the exporter and `--include-retired-response` to the narrative figure rebuild.

## Earlier portable workspace

`templates/` mirrors the code locations in the complete workspace. From the
repository root, use `--check` to compare the code or `--apply` to update it.
Then run the portable rebuild in the bundled writing environment with Ghostscript
and Poppler available.

```bash
python -m reporting.manuscript_workspace /path/to/manuscript-workspace --check
python -m reporting.manuscript_workspace /path/to/manuscript-workspace --apply
python /path/to/manuscript-workspace/05_CODE/rebuild_publication_workspace.py --package
```

The portable rebuild preserves author files, uses the bundled case CSVs and
performs no fitting. See the [style profile](../publication_style.json) and
[portable style guide](templates/05_CODE/PUBLICATION_STYLE.md).
