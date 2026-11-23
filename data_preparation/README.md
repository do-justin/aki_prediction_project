# Data preparation pipeline

This directory contains the production Catch22 preprocessing stages. Run the
steps in order; each stage depends on artifacts from the previous one.

Key settings live in `data_preparation/inputs.py`; waveform rates and filters
are in `WAVEFORM_SPECS` in `waveform_processing.py`. Steps 01–05 write metadata
sidecars that are validated before artifact reuse; Step 02 records a separate
extraction fingerprint for each enabled branch.

## Stages

Run these commands in order in the activated project environment:

| Stage | Command | Output |
| --- | --- | --- |
| Cohort | `python -m data_preparation.step_01_cohort_construction` | Shared cohort and flow metadata |
| Waveform features | `python -m data_preparation.step_02_catch_22` | Catch24 features for the enabled branches |
| Preoperative data | `python -m data_preparation.step_03_preop_prep` | Preoperative table and outcome-specific patient-grouped holdout splits |
| Wide features | `python -m data_preparation.step_04_intraop_prep` | Wide waveform feature tables |
| Merge | `python -m data_preparation.step_05_data_merge` | Modeling tables for the prepared branches |

The launcher prepares both waveform branches. Standalone Step 02 defaults to
whole-case extraction; set `GENERATE_WINDOWED_FEATURES=True` to enable windows in
a fresh output root. Reuse existing study features instead of regenerating them.
The manuscript retains 10-second windows with a nominal five-second stride;
the [duration pilot](../METHODS.md#window-duration-rationale-and-retrospective-sensitivity-analysis)
did not change production extraction.

## Production definitions

See [waveform processing](../METHODS.md#waveform-processing) for filtering and
resampling and [feature engineering](../METHODS.md#feature-engineering) for
branch-specific extraction and missingness. Use a fresh output root when the
Step 02 extraction fingerprint changes or is missing. Cohort and outcome
definitions are implemented in `step_01_cohort_construction.py` and
`aki_cohort.py`.

## Real-data smoke

Run this research diagnostic only when requested, with a fresh or empty
`SMOKE_ROOT`. It reads source data and contacts VitalDB:

```bash
python -m run_catch22 smoke
```

It runs Steps 01–05 on a trimmed cohort and writes to the isolated smoke root.
Use the root README’s offline verification for ordinary code checks.
