# AKI Prediction Project

Postoperative AKI and ICU-admission prediction from VitalDB intraoperative
Catch24 waveform features and preoperative clinical data. The manuscript path
uses logistic regression, XGBoost and EBM. The completed paper grid contains
the main AKI/ICU analyses, broader-ICU sensitivity analysis and exploratory
mortality models.

- [METHODS.md](METHODS.md): scientific definitions and validation protocol
- [RESULTS.md](RESULTS.md): saved artifacts and reporting commands
- [Current handoff](.agent/workflows/manuscript-reorientation.md): completed runs, open work and manuscript exports
- [AGENTS.md](AGENTS.md): repository working rules

## Active code

| Location | Purpose |
| --- | --- |
| `run_catch22.py` | Canonical preparation, experiment and reporting launcher |
| `data_preparation/` | [Steps 01–05](data_preparation/README.md), cohort eligibility and waveform processing |
| `model_creation/` | Patient-grouped HPO, training, calibration and threshold selection |
| `results_recreation/` | Saved-prediction metrics, paired comparisons and bootstrap estimates |
| `reporting/` | Cohort flow, descriptives, figures, tables and reports |
| `reporting/manuscript_workspace/` | Code-only templates for the portable manuscript workspace |
| `scripts/` | CPU execution, window-duration evidence and INSPIRE linkage tools |
| `tests/` | Isolated offline regression tests and synthetic fixtures |
| `experimental/aeon/` | [Experimental Aeon commands](experimental/aeon/README.md); implementations retain their documented module paths |
| `archive/` | Superseded notebooks, experiments and utilities; excluded from the current workflow |

## Environment and verification

Linux x86_64 / Python 3.12 CPU is supported. Activate the validated environment:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate aki_prediction_project
bash scripts/cpu/verify-offline.sh
```

For a new environment:

```bash
conda env create -f environment.yml
conda activate aki_prediction_project
python -m pip install -r requirements-test.txt
```

Runtime pins are in `requirements-cpu.txt`; exact replay and dependency-change
procedures are in [locks/README.md](locks/README.md). Preserve validated environments
when experimenting. Use `xgboost-cpu`, never both XGBoost distributions. CUDA,
Aeon, AutoGluon and notebooks are outside the supported CPU profile.

The verification script runs native checks and the full synthetic test suite in
temporary directories. For a focused check, use `python -m pytest -q tests/test_<name>.py`.
The [real-data smoke test](data_preparation/README.md#real-data-smoke) uses VitalDB.

## Manuscript figures and tables

The [display rebuild guide](reporting/manuscript_workspace/README.md) covers the
current Word books and the earlier portable workspace. Git tracks the rebuild
code. Research data, author documents and generated packages are kept separately.

## Existing research artifacts

On this VM, study roots are under
`/media/volume/catch22/data/aki_prediction_project/studies/`:

| Study | Contents |
| --- | --- |
| `manuscript-core-v1` | Original 36 main-core configurations and sealed processed features |
| `manuscript-optimized-v1` | Numerical equivalence gate, broader-ICU core and exploratory mortality core |
| `manuscript-supplementary-v1` | Full 180-configuration main grid, reusing the main core |
| `manuscript-icu-supplementary-v1` | Full 90-configuration broader-ICU grid, reusing its core |
| `manuscript-lr-balanced-50-v1` | Completed display-scoped balanced LR revision with 28 main and four broader-ICU configurations |

Source clinical/laboratory CSVs are under
`/media/volume/catch22/projects/aki_prediction_project/data/raw`.
Study manifests and `control/launch-status.json` record completion. Retain the
research volume when shelving compute. Mortality supplements are unstarted.

## Future analyses and saved-result reporting

Use a fresh study root for an authorized new analysis. Path overrides select the
study and its processed inputs; historical artifact names and study identities
must remain intact. A read-only planning example is:

```bash
export RAW_DIR=/media/volume/catch22/projects/aki_prediction_project/data/raw
export PROCESSED_DIR=/media/volume/catch22/data/aki_prediction_project/studies/manuscript-core-v1/data/processed
export AKI_ARTIFACT_ROOT=/media/volume/catch22/data/aki_prediction_project/studies/new-study
python -m run_catch22 experiments --preset core --prep skip \
  --cohort-scopes icu_waveform_sensitivity mortality_waveform_sensitivity \
  --reuse-from /media/volume/catch22/data/aki_prediction_project/studies/manuscript-optimized-v1 \
  --dry-run
```

`--prep auto` prepares absent inputs or validates existing ones; `--prep skip`
only validates. `--prep force` requires an empty processed directory. Reuse and
resume require matching source, cohort, preprocessing, model and validation
fingerprints. See `python -m run_catch22 experiments --help` for configuration,
admission and reporting options. The [historical study log](archive/docs/manuscript-reorientation-2026-11-04.md)
documents the optimized supervisor.

For balanced LR reproduction, see `python scripts/cpu/run-balanced-lr.py --help`.

For saved-result work, select the existing study roots before running:

```bash
python -m run_catch22 experiments --preset core --report-only \
  --cohort-scopes icu_waveform_sensitivity mortality_waveform_sensitivity
COHORT_SCOPE=mortality_waveform_sensitivity python -m reporting.fold_explanations
```

Study reporting can recompute metrics and bootstrap estimates.
[Reporting inputs and safeguards](RESULTS.md#regenerate-from-saved-artifacts)
