# EEG feasibility study

This worktree runs the separate VitalDB EEG feasibility analysis. The [original handoff](VitalDB_EEG_Codex_Handoff.md) records the study design and evidence for each QC choice. The EEG study uses the established preoperative variables, AKI and ICU labels, and Catch24 model families. It adds timestamped native EEG and restricts comparisons to a shared five-modality roster in both branches.

## Extractor upgrade

This revision uses `pycatch22 1.0.2` in a separate locked Conda environment and study root. The original `0.4.5` extraction was paused after 35 operation checkpoints. Its code, environment, artifacts and pause record remain under the original branch and study root.

The upgrade retains 128 Hz EEG, the signal-quality protocol, the original waveform sources, and the grouped validation design. It recomputes all five waveform groups with the native 1.0.2 descriptor definitions and order. Explicit `mask_nans=False` preserves undefined descriptors and their window aggregation behavior. Descriptor values can change across versions, including periodicity and embedding distance. Historical feature, model, and split artifacts are version-specific. Verified raw track responses and the frozen track catalog can be copied independently.

In the outcome-blinded comparison, whole-operation Catch24 on one 4-hour-25-minute 128 Hz record took 21 seconds with 1.0.2 and 19 minutes 40 seconds in the 0.4.5 pilot. Three 10-minute excerpts showed 9–14× speedups. The new routine can exit early after the first qualifying periodicity peak, so the gain varies by waveform. The other pinned dependencies passed the lock audit and carry over to this revision.

## Execution checklist

- [x] Preserve the manuscript baseline (`ab39e021468c692fd31e9b3d62f7637d24c4cfaa`) and historical core fitting revision (`fcff7c7c9794e0ea8405814c68176760a2eb9afc`). The local baseline tag is `joms-manuscript-pre-eeg-2026-11-25`. The verified recovery copy is at `/home/exouser/codex-handoffs/joms-manuscript-pre-eeg-2026-11-25`, with a file and display-source inventory, Git bundles, binary replay inputs, and the author document at revision 3160. The release checksum catalog identifies the read-only waveform files.
- [x] Implement the isolated EEG extractor upgrade. The original branch and study remain available. The four original waveforms use the pinned VitalDB loader's numerical gridding and a frozen public-track cache. EEG reads native local packets and retains the actual observation clock for SQI.
- [ ] Complete and inspect the blinded pilot. Review `pilot/pilot_cases.csv`, the raw/cleaned examples, QC records, and the protocol fingerprint. Review timestamps, units, gap shape, and resources before preparing the definitive cohort.
- [ ] Complete both feature branches and the joint split preflight. Review `data/processed/eeg_attrition.json` and `control/eeg_split_preflight.json`. A split failure in any combination blocks fitting for the entire study.
- [ ] Complete isolated diagnostic fits and reporting in `diagnostic/`. The diagnostic marker binds to the exact code and validated split gate.
- [ ] Complete all 414 configurations and rebuild all displays and companion exports. Confirm that each scope's run manifest is validated and the EEG display manifest traces to the saved artifacts.
- [ ] Review Word and PDF books, rerun protected-source checksum checks, and record definitive resource use and paired EEG effects.

## Commands

Run these commands from the EEG upgrade worktree. The launcher uses the study root below by default and resolves every write within it.

```bash
source /home/exouser/miniforge3/etc/profile.d/conda.sh
conda activate aki_prediction_project_eeg_pycatch102
export RAW_DIR=/media/volume/catch22/projects/aki_prediction_project/data/raw
export AKI_ARTIFACT_ROOT=/media/volume/catch22/data/aki_prediction_project/studies/eeg-feasibility-pycatch22-1.0.2-v1

python -B -m run_catch22 experiments --study-profile eeg --dry-run
python -B -m run_catch22 eeg-pilot
python -B -m run_catch22 experiments --study-profile eeg --preflight-only
python -B -m run_catch22 smoke --study-profile eeg
python -B -m run_catch22 experiments --study-profile eeg --prep skip --resume
python -B -m run_catch22 experiments --study-profile eeg --report-only
```

After the pilot and candidate commit, use `bash experimental/eeg/run_study.sh` in a dedicated tmux session to execute or resume the remaining stages. The supervisor records its current stage and exit status in `control/eeg-supervisor.status`. The launcher logs to `logs/experiment_log.txt`. A preflight status of `blocked` records a preparation, source, or operational failure. A status of `infeasible` records a scientifically unsupported split. Resumption uses the same code, artifact root, and protocol.

Extraction defaults to one worker per available CPU, bounded by affinity, CPU quota, and remaining operations. `AKI_EXTRACTION_WORKERS` can lower that count. `AKI_EEG_DOWNLOAD_WORKERS` independently limits concurrent VitalDB downloads and defaults to four.

`results/catch22/eeg_report/` contains the manuscript-layout EEG Word and PDF books, vector figure assets, source tables, paired comparisons, curve data, and numerical explanation exports. The PDF books and standalone table proofs are exported from Word with LibreOffice Writer. Set `AKI_EEG_SOFFICE` to the Writer executable if `soffice` is outside `PATH`. The figure assets remain available as standalone vector PDFs.

Companion fold explanations export in separate processes. The default uses the available CPU allocation and budgets 1 GiB of available RAM per process. Each parallel worker uses one native compute thread. Set `AKI_REPORT_CPU_BUDGET` to lower the limit. To rebuild displays directly from validated saved fits and comparison tables, run the following command with the study paths set above. Use `--fold-workers 1` for a serial export.

```bash
python -B -m reporting.eeg_study --artifact-root "$AKI_ARTIFACT_ROOT" \
  --processed-dir "$AKI_ARTIFACT_ROOT/data/processed" --fold-workers 8
```

A local benchmark on 120 copied folds across both cohorts, both branches, and all three model families took 143.9 seconds with one worker, 25.5 seconds with eight, and 22.6 seconds with 16. All 400 export files matched after decompressing the gzip CSVs. These timings cover fold export and worker startup.

A completed display manifest records the fold exporter fingerprint and output checksums. Later rebuilds reuse fold explanations when the saved modeling inputs, fold bundle, exporter, and output checksums match. The first rebuild of an older report populates this fingerprint. Missing or damaged exports are regenerated individually. Numerical fold CSVs use gzip level 6.
On the 120-fold benchmark copy, a fresh eight-worker export took 24.0 seconds and verified reuse took 0.55 seconds. The 400 reused files matched byte for byte.

Bootstrap reporting indexes saved draws by configuration for confidence intervals, deltas, and comparisons. It retains the existing draw order, inputs, and provenance checks.

## Fixed study settings

The experimental scope uses the handoff's 23 named configurations and the three cohort–outcome combinations. All 414 configurations are fit independently, including the branch-specific clinical baselines. The shared whole-operation core runs first. Fit LR, XGBoost and EBM with 50 trials, training-partition balancing, and deterministic patient-grouped 5/3/5 validation. Report 5,000 patient-clustered bootstrap draws with seed 0 from the saved outer-test predictions.

The experimental manifests use `eeg_main_shared_v1` and `eeg_icu_waveform_sensitivity_v1` with the existing internal scope keys. Sidecars carry selected EEG source, QC, descriptor availability, both-branch success, immutable inputs and split provenance. The model registry selects clinical variables from the approved whitelist and waveform descriptors from the exact five groups.
