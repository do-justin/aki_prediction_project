# EEG feasibility study

This worktree runs the separate VitalDB EEG feasibility analysis. The [original handoff](VitalDB_EEG_Codex_Handoff.md) records the study design and evidence for each QC choice. The EEG study uses the established preoperative variables, AKI and ICU labels, and Catch24 model families. It adds timestamped native EEG and restricts comparisons to a shared five-modality roster in both branches.

## Execution checklist

- [x] Preserve the manuscript baseline (`ab39e021468c692fd31e9b3d62f7637d24c4cfaa`) and historical core fitting revision (`fcff7c7c9794e0ea8405814c68176760a2eb9afc`). The local baseline tag is `joms-manuscript-pre-eeg-2026-11-25`. The verified recovery copy is at `/home/exouser/codex-handoffs/joms-manuscript-pre-eeg-2026-11-25`, with a file and display-source inventory, Git bundles, binary replay inputs, and the author document at revision 3160. The release checksum catalog identifies the read-only waveform files.
- [x] Implement and verify the isolated EEG profile. The original extraction files retain their tracked hashes. The four original waveforms use the pinned VitalDB loader's numerical gridding and a frozen public-track cache. EEG reads native local packets and retains the actual observation clock for SQI.
- [ ] Complete and inspect the blinded pilot. Review `pilot/pilot_cases.csv`, the raw/cleaned examples, QC records, and the protocol fingerprint. Review timestamps, units, gap shape, and resources before preparing the definitive cohort.
- [ ] Complete both feature branches and the joint split preflight. Review `data/processed/eeg_attrition.json` and `control/eeg_split_preflight.json`. A split failure in any combination blocks fitting for the entire study.
- [ ] Complete isolated diagnostic fits and reporting in `diagnostic/`. The diagnostic marker binds to the exact code and validated split gate.
- [ ] Complete all 414 configurations and rebuild all displays and companion exports. Confirm that each scope's run manifest is validated and the EEG display manifest traces to the saved artifacts.
- [ ] Review Word and vector PDF books, rerun protected-source checksum checks, and record definitive resource use and paired EEG effects.

## Commands

Run these commands from the EEG worktree in `aki_prediction_project`. The launcher uses the study root below by default and resolves every write within it.

```bash
source /home/exouser/miniforge3/etc/profile.d/conda.sh
conda activate aki_prediction_project
export RAW_DIR=/media/volume/catch22/projects/aki_prediction_project/data/raw
export AKI_ARTIFACT_ROOT=/media/volume/catch22/data/aki_prediction_project/studies/eeg-feasibility-v1

python -B -m run_catch22 experiments --study-profile eeg --dry-run
python -B -m run_catch22 eeg-pilot
python -B -m run_catch22 experiments --study-profile eeg --preflight-only
python -B -m run_catch22 smoke --study-profile eeg
python -B -m run_catch22 experiments --study-profile eeg --prep skip --resume
python -B -m run_catch22 experiments --study-profile eeg --report-only
```

After the pilot and candidate commit, use `bash experimental/eeg/run_study.sh` in a dedicated tmux session to execute or resume the remaining stages. The supervisor records its current stage and exit status in `control/eeg-supervisor.status`. The launcher logs to `logs/experiment_log.txt`. A preflight status of `blocked` records a preparation, source, or operational failure. A status of `infeasible` records a scientifically unsupported split. Resumption uses the same code, artifact root, and protocol.

Extraction defaults to one worker per available CPU, bounded by affinity, CPU quota, and remaining operations. `AKI_EXTRACTION_WORKERS` can lower that count. `AKI_EEG_DOWNLOAD_WORKERS` independently limits concurrent VitalDB downloads and defaults to four.

`results/catch22/eeg_report/` contains the EEG Word and PDF books, source tables, figure assets, paired comparisons, curve data, and numerical explanation exports. PDF and Word composition use the same figures, tables, and ordering. PDF proofs of the Word files require a DOCX renderer for pagination review.

## Fixed study settings

The experimental scope uses the handoff's 23 named configurations and the three cohort–outcome combinations. All 414 configurations are fit independently, including the branch-specific clinical baselines. The shared whole-operation core runs first. Fit LR, XGBoost and EBM with 50 trials, training-partition balancing, and deterministic patient-grouped 5/3/5 validation. Report 5,000 patient-clustered bootstrap draws with seed 0 from the saved outer-test predictions.

The experimental manifests use `eeg_main_shared_v1` and `eeg_icu_waveform_sensitivity_v1` with the existing internal scope keys. Sidecars carry selected EEG source, QC, descriptor availability, both-branch success, immutable inputs and split provenance. The model registry selects clinical variables from the approved whitelist and waveform descriptors from the exact five groups.
