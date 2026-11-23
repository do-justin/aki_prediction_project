# Results artifacts

Saved artifact layout and reporting commands. Scientific definitions are in
[METHODS.md](METHODS.md); execution and study paths are in [README.md](README.md).

The frozen grids contain 180 main-cohort, 90 broader-ICU and 18 exploratory
mortality configurations. The main core was fitted at revision `fcff7c7`.
Supplementary studies reuse verified core references. The revised display
books select balanced LR from its separate study.

## Layout and integrity

Under `$AKI_ARTIFACT_ROOT`, the main roots are:

```text
data/processed/
results/catch22/experiments/<cohort_scope>/<validation_scheme>/
results/catch22/paper/<cohort_scope>/<validation_scheme>/
logs/
```

Scopes are `main_shared`, `icu_waveform_sensitivity`, and
`mortality_waveform_sensitivity`; validation schemes are
`nested_cv` and diagnostic `holdout`. Explicit results/paper overrides are base
roots unless already scoped. Historical roots and model basenames are retained.

Each experiments root contains `run_manifest.json`: requested configurations,
study/code/environment identity, diagnostic status, frozen branch rosters,
source hashes, resource budgets, and execution history. Required configurations
must exist before strict reporting succeeds. Family stages extend the manifest;
partial coverage cannot be presented as a complete three-family comparison.

A model configuration lives at:

```text
models/<family>/<outcome>/<branch>/<feature_set>/
  predictions/train_oof.csv
  predictions/test.csv
  artifacts/calibration.json
  artifacts/threshold.json
  artifacts/metadata.json
  artifacts/validation.json
  artifacts/folds/repeat_00/outer_00/
```

Nested `test.csv` pools outer-test predictions. Training OOF predictions are
model-development diagnostics, not publication estimates. Each outer-fold
bundle stores its fitted model, preprocessor, calibrator, threshold, parameters,
train/test predictions, and integrity metadata. XGBoost uses `model.json`;
LR/EBM use `model.pkl`; preprocessing uses `preprocessor.pkl`. Resume requires
matching fingerprints and every promised artifact, not just a prediction file.

Clinical/preoperative schemas are version 4, merged features version 5, and
waveform extraction protocol version 3. Old waveform/merged artifacts require
fresh extraction; changing a statistics-only protocol requires regenerating
statistics, not fitting models again. Historical artifacts are never relabeled
or upgraded in place.

Step 02's per-branch extraction sidecars record cohort IDs/operative bounds via
a roster hash, extraction source hashes, channel/filter specifications, rates,
substitutions, library versions, anti-alias resampling and missing-value policies.
They identify the original signal-processing contract in
[METHODS.md](METHODS.md#waveform-processing). Production finite descriptors are
preserved; the duration pilot's fallback-masked aggregates are separate diagnostic
artifacts and must not replace the frozen modeling inputs.

## Regenerate from saved artifacts

Use the [scoped saved-result commands](README.md#future-analyses-and-saved-result-reporting)
after selecting the existing study and processed-data paths. These consume saved
data and models without fitting. The default reporting policy is 5,000
unstratified patient-cluster draws, seed 0. The report-only launcher records
changed bootstrap settings before regeneration; direct metrics commands must
agree with the manifest policy. Standalone explanations require the appropriate
`COHORT_SCOPE`.

Strict reports align unique cases and verify patient, truth, and fold identity.
They require the same family's clinical reference and reject missing requested
runs, altered predictions, stale intervals, mixed study identities, or a pooled
prediction file inconsistent with its verified fold union. Explicit `--legacy`
reading supports historical artifacts but does not certify them as definitive.

## Study reporting exports

See the [display rebuild guide](reporting/manuscript_workspace/README.md) for
the revised Word books and the earlier portable workspace.

The scoped paper tree contains tables, figures, reports, metadata, and a
`manifest.json` linking exported files. The principal outputs are:

| Artifact | Purpose |
| --- | --- |
| `tables/manuscript_<outcome>_main.{docx,pdf,html,md}` | Whole-case core comparison across all three families |
| Companion CSVs and `tables/metrics_summary.*` | Full metrics, denominators, and inspectable values |
| `tables/results_*_{main,delta}.*` | Detailed and supplementary results |
| `figures/incremental_ap_<outcome>_main.{svg,png,csv}` | Paired combined-versus-clinical AP differences |
| ROC/PR/calibration figures | Pooled outer-test curves and scores; calibration axes cover 0–1 |
| `reports/report.{md,docx,pdf}` | Consolidated working/supplementary report |

Study model tables prioritize AP, AUROC, Brier score, and paired AP differences.
Supplementary tables retain branch identity and all requested classification
metrics. Shading and selective bolding are descriptive, not significance marks.
Fold mean/SD curves are optional supplemental output, explicitly not confidence
intervals. Diagnostic status remains visible throughout exports.

## Comparisons and decision curves

Scoped results and paper tables include:

| File | Contents |
| --- | --- |
| `paired_comparisons.csv` | Directed core/ablation differences with paired intervals |
| `decision_curve_summary.csv` | Model, treat-all, and treat-none net benefit with pointwise intervals |
| `decision_curve_comparisons.csv` | Paired net-benefit differences and intervals |
| `delong_comparisons.csv` | Design eligibility, numerical status, raw/Holm-adjusted p-values |
| `comparison_inventory.json` | Coverage, hashes, actual bootstrap-draw identities, and calculation policy |

`comparison_primary` contains the primary whole-operation AP contrasts and
`comparison_families` the matched-input family contrasts. Uncertainty
protocol 3 records requested and valid replicate counts. DeLong exports
retain `unsupported_design` or `degenerate` status where applicable.
`dca_delta_*` shows paired differences with pointwise bands and
`dca_families_*` compares families.
[Statistical definitions](METHODS.md#evaluation-and-statistical-analysis)

## Descriptives and explanations

Table 1 uses source-value median/IQR, explicit missingness, clinical sections,
and operation/patient denominators. Cohort flow, missingness, and descriptives
share the frozen scope/branch selectors. Whole-case outputs use the base names
`preop_descriptives`, `missingness_table`, and `cohort_flow`; windowed outputs
append `_windowed` to avoid overwriting another population. Flow counts live in
`metadata/cohort_flow_counts*.json`.

Fold explanations are under `figures/fold_explanations/`, mirroring model/fold
identity. They verify saved bundles and modeling-data hashes before export:

- LR: coefficients, intercept, training-SD units, and fold-specific encodings.
  Individual one-hot coefficients are not reference-category odds ratios.
- XGBoost: held-out SHAP contributions, importance, beeswarms, and base values
  on the raw uncalibrated log-odds scale.
- EBM: native term importance, held-out contributions, response plots, and
  separate missing/unknown-bin scores.

Explanations describe fitted associations rather than causal mechanisms;
logistic recalibration remains separate. Legacy single-model SHAP/EBM HTML
exports are opt-in. Smoke outputs live only under their disposable `SMOKE_ROOT`.
Aeon and archived outputs are outside the manuscript pathway.

## Window-duration pilot and feature audit

The frozen pilot completed November 1, 2026. Its artifact root is
`/media/volume/catch22/data/aki_prediction_project/pilots/window-duration-v2`:

- `REPORT.md`: feature audit, 475 matched baseline channels, five extraction
  rejections and 120 paired synthetic challenges.
- `control/manifest.json` and `code/`: frozen sample, inputs, methods and software.
- `channels/`, `grid_diagnostics.json`, `pilot.npz`: channel checkpoints, numerical
  diagnostics and raw/audited aggregates.
- `summary/{primary,finite_only,matched,first,middle,final}/`: recorded summaries,
  descriptor metrics and availability.

See [Methods](METHODS.md#window-duration-rationale-and-retrospective-sensitivity-analysis)
for interpretation and the [pilot record](.agent/workflows/window-duration-pilot.md)
for the numerical rules and open saved-model audit.

## Optimized extension and protected references

The broader ICU and exploratory mortality core batches each completed 18
configurations, with 180 verified fold bundles across the two batches. Their
results/paper namespaces are separate from the main core. The extension study's
`control/reused_inventory.json` records original paths, fingerprints and hashes
for the protected 36 main runs. Reuse verifies their frozen provenance; new
checkpoints require current-source matching. Missing or corrupt references stop
execution instead of triggering refits. The relaunch check selected zero fits.

Execution records comprise `control/execution_plan.json`,
`control/active_scheduler.json`, per-job logs/resource samples and
`control/supervisor_status.json`. The active-scheduler file identifies the latest
execution epoch; earlier resource logs and code snapshots are retained.
Orchestration overlays record source commits and hashes separately from
scientific fingerprints. Per-fold HPO histories are in `artifacts/trials/`.
The saved equivalence report records the agreement required before extension
jobs were admitted. Mortality DCA retains its 0.1–5% grid and exploratory status.
Computational validation does not establish admission-level outcome adjudication
or clinical validation.

The source-label audit confirmed 52 mortality-positive operations in 47
patients among 5,765 operations and 5,543 patients in each branch.
[Mortality interpretation](METHODS.md#operation-level-mortality-and-admission-linkage)
· [Archived label checks](archive/docs/manuscript-reorientation-2026-11-04.md)

Fold metadata records floating prediction dtypes. Checkpoint reload uses those
dtypes and round-trip CSV parsing so partial resume preserves native XGBoost
float32 test predictions and float64 training predictions exactly. Legacy
metadata remains readable; protected source artifacts are never rewritten.
