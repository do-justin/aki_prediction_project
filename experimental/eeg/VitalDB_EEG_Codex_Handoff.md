# VitalDB EEG feasibility study: Codex implementation handoff

**Prepared for Justin Do | November 25, 2026, America/Los_Angeles**

**Purpose:** Implement and execute a separate EEG proof-of-concept study while preserving the current Journal of Medical Systems manuscript and its complete reproduction path.

This is a task specification to give Codex in the research repository. It synthesizes the conversation and a fresh read-only review of the repository, manuscript, supplemental package, and EEG methods sources. No branch, tag, experiment, or manuscript change was made while preparing this handoff.

## 1. Assignment and scope

You are working in the VitalDB postoperative-risk project. Implement the EEG extension described below, verify it, run the complete specified experiment grid in the configured research environment, and regenerate the existing main and supplemental display families from the new results. Work through the stages autonomously when their prerequisites pass. The user accepts a substantial, resumable CPU run using existing JetStream credits; a one-to-two-day estimate is not a deadline or a guarantee.

The scientific question is whether adding one raw frontal EEG waveform to the existing four intraoperative waveforms improves discrimination of postoperative AKI and ICU admission, alone and in combination with preoperative clinical variables. Retain the existing channel-agnostic Catch24 approach and compare both whole-operation and windowed representations.

This study is an **independent exploratory feasibility analysis**. Its results may later inform the manuscript or a separate project. Do not assume they will become a supplement, replace the manuscript results, or justify changing its current conclusions.

### Required scope

- Preserve the original manuscript code, data products, fits, predictions, figures, tables, and exact rebuild instructions.
- Create a separate Git branch and worktree and a separate experimental study/output root.
- Add one canonical EEG modality, with a preferred source channel and a deterministic fallback.
- Run both `non_windowed` and `windowed` branches.
- Run the shared AKI-eligible cohort for AKI and ICU admission, plus the broader ICU cohort.
- Run LR, XGBoost, and EBM with the current validation machinery, balanced training, and 50 HPO trials for every family.
- Include single-waveform models, combined models, five-waveform ablations, and the original four-waveform ablations.
- Rebuild all current main and supplemental display families and retain exhaustive numerical/source exports.
- Use saved predictions and model bundles for reporting; do not refit models to draw a figure.

### Outside this task

Do not add mortality, severe-AKI modeling, length of stay, delirium, new external datasets, A-line features, foundation models, unsupervised training, deep learning, spectrogram models, AutoGluon, TabPFN, Aeon, MIL, or longer window grids. Do not invent a postoperative delirium label. Do not add processed BIS/SR/SEF or a separate spectral-feature model to this initial grid. Those are possible later questions, not prerequisites for this proof of concept.

Do not merge the experiment into `master`, replace the manuscript, or publish results as part of this task. Keep manuscript-facing decisions available for Justin and his mentors.

## 2. Sources of truth and verified starting point

Read the live checkout before editing. Respect its `AGENTS.md`, `METHODS.md`, and applicable narrow rules. The task-specific experimental overrides below apply only to the new study.

| Source | Verified state or role |
| --- | --- |
| Canonical repository | [do-justin/aki_prediction_project](https://github.com/do-justin/aki_prediction_project); the older StealthX051 owner redirects |
| Default branch | `master`, not `main`; do not rename it |
| Last verified master SHA | `ab39e021468c692fd31e9b3d62f7637d24c4cfaa` |
| Original manuscript core fitting revision | `fcff7c7`, as documented in RESULTS; resolve and preserve its complete SHA locally |
| Implemented scientific contract | [METHODS.md at the verified revision](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/METHODS.md) |
| Operational commands | [README.md](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/README.md) |
| Existing studies and artifacts | [RESULTS.md](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/RESULTS.md) |
| Coding and environment rules | [AGENTS.md](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/AGENTS.md) |
| Current native manuscript | [VitalDB Catch22 Draft](https://docs.google.com/document/d/1OzXIW9U9Xpceko9R8132QJFRpJ6Kwlep8noYZYA110A/edit) |
| Current display package | [VitalDB_Revised_Display_Package_11_22_26](https://drive.google.com/drive/folders/1IVprEq-RxtGG9k-QrgNLK8b8_LwbRs5D) |
| Project status and mentor comments | [VitalDB Waveform Project Status](https://docs.google.com/document/d/1ihwbwJUilHMOc2YfoX3h5h0vaaynMHn_wEcJLzMY26I/edit); its numerical examples are older than the current manuscript |

The current repository and manuscript contain **three model families and 5 outer / 3 HPO / 5 calibration folds**. Older project notes saying two models or 5 inner HPO folds are stale. Historical LR runs used unweighted fitting and 20 trials; the revised display-selected LR runs use balanced fitting and 50 trials. Use the latter settings consistently in this new experiment.

The reviewed manuscript reports 3,571 operations / 3,432 patients in its shared cohort, with 189 AKI-positive operations and 834 ICU-positive operations. Its broader ICU cohort contains 5,765 operations / 5,543 patients. These are **historical manuscript reference counts**, not expected EEG-study denominators or constants to insert into new reports.

If the live checkout has advanced, record the actual starting SHA, inspect relevant changes, and preserve the appropriate manuscript baseline. Do not reset a newer checkout to this document's SHA or overwrite uncommitted work.

## 3. Preserve the manuscript before changing the experiment

### 3.1 Preserve the code and the run, not just the current branch

A tag on today's master does not capture all manuscript fitting provenance. Current displays combine several studies and a later LR revision.

Inventory these repository-documented study roots:

1. `manuscript-core-v1`
2. `manuscript-optimized-v1`
3. `manuscript-supplementary-v1`
4. `manuscript-icu-supplementary-v1`
5. `manuscript-lr-balanced-50-v1`

The documented parent is `/media/volume/catch22/data/aki_prediction_project/studies/`. The documented raw-data location is `/media/volume/catch22/projects/aki_prediction_project/data/raw`. These locations were read from the repository, not inspected on JetStream during preparation of this handoff. Resolve the actual mounted paths before acting.

Create a preservation manifest covering:

- Each study's run manifest, scientific configuration, code revision, environment identity, and source hashes.
- Original processed cohort tables and both feature branches.
- Operation/patient rosters, nested split plans and caches, training/calibration artifacts, outer-fold bundles, predictions, models, thresholds, and metrics.
- Current main/supplemental author documents, figure assets, source tables, display-selection mapping, and report-building revision.
- Environment specifications, exact locks, relevant package versions, and the successful commands needed to reproduce the run and rebuild displays.
- The relationships between reused core outputs and subsequent supplemental/revised-display outputs.

Generate a file inventory with sizes and checksums. Preserve a recoverable snapshot/copy or equivalent existing backup, preferably outside the experimental working volume. Checksums detect changes; they do not restore deleted files. Avoid writable hardlinks between an experimental destination and protected manuscript files.

Use the existing environment and backup facilities; do not build a new infrastructure project. Do not claim a verified complete freeze if a required study, fitting revision, environment, or display source cannot be located. Document the exact gap and complete any safe remaining inventory work.

An annotated local baseline tag such as `joms-manuscript-pre-eeg-2026-11-25` is appropriate after the baseline is verified. Check for an existing tag and never move it. Use the configured Git identity. Preserve the relevant older fitting commits as well as the current reporting commit.

### 3.2 Separate the experimental work

Create a sibling worktree on a branch such as `experiment/eeg-feasibility`, based on the verified manuscript baseline. Inspect existing branches/worktrees first and reuse a matching experimental worktree only after confirming its provenance.

Suggested new study identity:

- Study ID: `eeg-feasibility-v1`
- Study root: `/media/volume/catch22/data/aki_prediction_project/studies/eeg-feasibility-v1`
- Protocol: `eeg-qc-v1`
- Manifest cohort identities: `eeg_main_shared_v1` and `eeg_icu_waveform_sensitivity_v1`
- Experimental documentation: a scoped directory such as `experimental/eeg/`

Use the existing launcher layout and supported `AKI_ARTIFACT_ROOT`, `PROCESSED_DIR`, and raw-source configuration. Isolate **processed inputs, extraction checkpoints, merged tables, split caches, model outputs, reporting outputs, temporary files, and logs**. A different branch alone does not isolate these.

Keep the existing internal `cohort_scope` keys `main_shared` and `icu_waveform_sensitivity` under the EEG study/profile. The `eeg_*` names above are experiment-specific manifest identities, not replacements for those internal enum values. Preserve the scope registry and existing schema/report integrations.

The launcher intentionally clears several per-file environment overrides. Renaming only the EEG CSV is insufficient. Resolve all effective paths from the launcher environment and assert that no writable path equals, contains, or resolves through a symlink into a protected manuscript tree.

Keep the default four-waveform profile and its CLI/artifact semantics unchanged. Add an explicit EEG study profile or narrowly scoped experiment option. Do not globally redefine what existing feature-set names mean.

The experimental run may read unchanged raw source files. It may reuse verified immutable four-waveform features when extraction/source identities match exactly, but must construct separately identified experimental inputs and rosters. Do not reuse old models or predictions merely because feature names match.

### 3.3 Preserve the software environment

The supported repository environment is Linux x86_64, Python 3.12, CPU Conda `aki_prediction_project`. Use the pinned `xgboost-cpu` distribution and current lock procedure. Do not install a second XGBoost distribution, introduce a parallel venv, or mutate the validated manuscript environment in place.

If additional packages are genuinely needed, create a separate candidate environment, verify it, and freeze its exact successful state for the experiment. Most EEG processing should be possible with existing NumPy/SciPy/pycatch22 dependencies.

## 4. Experimental population and prediction task

### 4.1 Preserve clinical definitions

Prediction is at operation completion. Use only the operative waveform interval and the established preoperative predictors. Retain repeat operations; the prediction unit is the operation and the splitting/resampling unit is `subjectid`.

Preserve the current general-anesthesia restriction, ASA V–VI exclusion with missing ASA retained, operation-time validation, preoperative feature construction, and outcome phenotyping. Do not add a complete-tabular-case requirement or a 30% missingness exclusion rule.

Preserve the implemented AKI definition: finite positive source baseline creatinine ≤4 mg/dL, at least one finite positive postoperative creatinine strictly after operation end and within 72 hours, and an observed increase of ≥0.3 mg/dL within 48 hours or ≥1.5-fold within 72 hours. This is the existing creatinine-based, study-specific ascertainment window; do not silently relabel it as full seven-day KDIGO ascertainment.

Preserve ICU labeling from a finite nonnegative source ICU length of stay: positive means ICU admission, zero means none. Do not rename this outcome “unplanned ICU admission.”

### 4.2 Use two populations and three population–outcome combinations

| Manifest cohort identity | Internal scope key | Eligibility beyond waveform qualification | Models/outcomes |
| --- | --- | --- | --- |
| `eeg_main_shared_v1` | `main_shared` | Existing AKI eligibility and valid AKI label, plus valid ICU label | AKI and ICU admission |
| `eeg_icu_waveform_sensitivity_v1` | `icu_waveform_sensitivity` | Valid ICU label; no creatinine/AKI ascertainment requirement | ICU admission |

The regular/shared cohort and its ICU analysis are not two different populations. The expanded ICU cohort is the second population. Keep their artifacts and denominators separate.

### 4.3 Require five usable modalities for the primary paired experiment

The experimental default is:

1. The original four source waveforms are present, allowing the established ECG substitution.
2. One selected raw EEG source is available under the deterministic EEG fallback rule.
3. All five signals pass their declared channel-level quality/length requirements and produce a feature record in **both** branches.
4. Outcome eligibility is applied for the requested scope.
5. Freeze the resulting operation roster before fitting any models.

**This is intentionally stricter than the existing manuscript rule.** The manuscript requires four source tracks but permits partial channel extraction failures and channel-feature NaNs. Do not describe this experimental complete-modality requirement as an unchanged cohort rule.

Channel-level success does not require every Catch24 value to be finite. Preserve undefined descriptor values and aggregates as NaN. Distinguish an absent/failed channel from a valid feature record containing mathematically undefined descriptors.

For every direct comparison, use the same roster for preoperative-only, four-waveform, five-waveform, single-channel, combined, and ablation models. Also use the same roster in both branches. Retrain all comparators on that roster; an older model trained on the broader manuscript cohort is not the paired four-waveform comparator.

Do not mix operations lacking usable EEG into some training folds through native missing-value handling. This primary experiment estimates added EEG information among operations meeting the five-modality qualification rule. A later study of optional EEG availability across the broader four-waveform population would be a separately specified question.

Use the same `opstart`–`opend` interval for all channels. Do not restrict the four original channels to EEG-clean timestamps, require identical accepted windows across modalities, or splice their signals to match EEG availability. Those changes would alter their established feature extraction.

### 4.4 Audit attrition explicitly

Before fitting, report counts of operations, unique patients, positive operations, positive patients, and outcome prevalence at each relevant stage:

- Original clinical/source-track eligibility.
- EEG1 availability, fallback availability, and selected source.
- SQI availability and valid time coverage.
- Source nonfinite, SQI, amplitude, and combined QC failures.
- Whole-operation and windowed channel-extraction failures.
- The strict five-channel/both-branch intersection.
- Shared AKI/ICU and broader ICU eligibility.

Use separate first-exclusion flow categories and an additional overlapping-reasons audit so one operation is not counted twice in the flowchart. Include comparisons between retained and excluded operations and the original manuscript population, with selection interpreted descriptively.

Do not assume device availability equals raw-waveform coverage. Do not relax quality thresholds or cohort rules to improve event counts or model performance. If the strict cohort cannot support the required grouped nested splits, save the feasibility report and complete the implementation/report scaffolding without launching an invalid definitive grid.

## 5. EEG processing: exact protocol and evidence provenance

### 5.1 What the sources establish

The following is a **prespecified adaptation assembled from published components**. No cited paper validates this exact combined pipeline for whole-operation Catch24 prediction of AKI or ICU admission.

| Component | Primary source | Supported use and boundary |
| --- | --- | --- |
| Raw EEG tracks and metadata | VitalDB official parameter/API documentation [1–2] | `BIS/EEG1_WAV` and `BIS/EEG2_WAV`, 128 Hz, µV; `BIS/SQI` is numeric quality data. Check actual loader output and timestamps. |
| 0.5–50 Hz Butterworth band-pass; 100 µV amplitude criterion | Seo et al., 2021 [3] | BIS-recorded EEG during dexmedetomidine sedation under spinal anesthesia. The authors excluded artifactual segments exceeding ±100 µV. They did not specify our sample-mask algorithm, filter order, or interpolation rule. Their SQI >90 criterion was stated for BIS and SEF95. |
| SQI <50 exclusion | Fritz et al., 2016 [4] | Applied to processed BIS suppression-ratio data. Using the threshold to mask raw EEG is an adaptation. This citation is Fritz 2016, not the previously misattributed Willingham citation. |
| Fourth-order, zero-phase 0.5–50 Hz precedent | Sukriti et al., 2026 [5] | A 24-subject depth-of-anesthesia study. Its other steps differ from ours. It supports a filter precedent, not reproduction of an entire pipeline. |
| Exact filter implementation semantics | SciPy `butter` and `sosfiltfilt` [6–7] | Report API design N, SOS, frequencies, sampling rate, and padding explicitly; avoid an ambiguous claim about effective filter order. |
| ≤5% allowance, linear interpolation and endpoint fill | Existing project pipeline [R1–R2] | Compatibility choice. It is not established by the EEG references as a universally valid missingness threshold. |

### 5.2 Channel selection

Canonical modeling group: `eeg`.

- Preferred source: `BIS/EEG1_WAV`.
- Fallback source: `BIS/EEG2_WAV`.
- Select one source for the entire operation and both branches.
- Mirror the existing ECG loader fallback: attempt EEG2 when EEG1 is unavailable, returns no/empty data, or cannot be loaded. Record the reason.
- A nonempty EEG1 recording that later fails QC does not silently trigger EEG2 substitution under this version.
- Do not choose a channel based on outcomes, feature importance, held-out performance, or window-by-window quality.
- Store the actual source channel separately from the canonical feature prefix.

This channel hierarchy is our deterministic implementation choice, not a published preference for EEG1.

### 5.3 Sampling, units, and timing

Load EEG at its verified native rate, expected 128 Hz. Verify physical units and any gain/offset conversion before amplitude screening. Convert to µV explicitly when necessary and record the conversion. Never apply a 100 µV cutoff to values still in volts or millivolts.

Assert the expected rate before using the filter specification below. If source metadata contradicts 128 Hz, resolve that discrepancy in the pilot; do not silently reinterpret the samples or use a hardcoded sampling frequency against a different source grid.

Preserve native timestamps and the existing case-relative operation boundaries. Inspect the loader for implicit interpolation/resampling so a regridded value is not incorrectly counted as an observed raw sample.

Read numeric SQI with its actual timestamps. VitalDB omits missing numeric rows, so a row-index join or an assumed uninterrupted 1-Hz vector is inappropriate [2].

For this experiment, use a **bounded previous-observation hold**, not linear SQI interpolation:

- Initial implementation default: `sqi_max_age_seconds = 1.0`, an explicit engineering choice to be checked against acquisition metadata during the outcome-blinded pilot.
- At an EEG timestamp, use the most recent SQI observation at or before it only if its age is within the frozen bound.
- No backward filling before the first known SQI and no indefinite extension across gaps.
- Missing, stale, nonfinite, or out-of-range quality values are unknown quality, not good quality.
- Resolve duplicate/nonmonotone numeric timestamps deterministically with an audit; do not silently discard contradictory records.

The one-second hold age is not a second “published SQI cutoff.” If verified recording metadata demonstrates a different nominal acquisition cadence, document the evidence and correct this engineering parameter **before outcome modeling**, then freeze it. Do not estimate a more permissive hold age separately for each case or tune it to retain a desired number of events.

### 5.4 One fixed cleaning sequence

Implement the following order for EEG:

1. Load the fixed preferred/fallback channel and timestamped SQI.
2. Validate native rate/units and construct the complete expected operative EEG sample grid using the established boundary convention. Map observed samples onto that grid.
3. Align SQI to native EEG timestamps using the bounded hold above.
4. Build separate masks for nonfinite EEG, unknown SQI, SQI <50, and raw EEG amplitude outside ±100 µV.
5. Form their **union**; overlapping defects count once.
6. Reject the channel if the union exceeds 5% of expected operative samples. Exactly 5% is allowed. Reject records without sufficient finite usable support.
7. Fill the remaining masked EEG samples once, using the existing interpolation semantics: linear interior interpolation and nearest finite endpoint fill at leading/trailing gaps.
8. Filter the resulting finite operative signal once with the specified Butterworth SOS forward/backward implementation.
9. Derive both feature branches from that same cleaned signal.

Threshold boundaries are deliberate: SQI =50 is accepted; raw amplitude =±100 µV is accepted; `abs(x) >100` is flagged. Preserve the masks and provenance even though the waveform values are subsequently filled.

For the new EEG path, samples before the first available EEG observation, after the last observation, or absent within the recording count as missing on the expected operative grid. Do not shorten the interval to the returned array's span or compute the unusable fraction only over observed coverage. Preserve the original four-channel implementation; this explicit EEG coverage accounting belongs in the new protocol and its manifest.

This protocol masks **raw samples before filtering**. Seo et al. described rejection of artifactual segments and did not specify this operation. Samplewise masking, its timing, the union rule, and subsequent interpolation are our stated adaptations [3–4]. Do not claim exact replication or invent a published epoch length.

Do not implement multiple primary filter/SQI/amplitude recipes or select a recipe by predictive performance. Do not silently add notch filtering, ICA, artifact subspace reconstruction, EMG regression, winsorization, channel averaging, or waveform z-scoring.

### 5.5 Filter implementation

Use the existing SciPy SOS path:

```python
sos = scipy.signal.butter(
    N=4,
    Wn=[0.5, 50.0],
    btype="bandpass",
    fs=128.0,  # verified EEG source rate
    output="sos",
)
clean_eeg = scipy.signal.sosfiltfilt(
    sos,
    interpolated_eeg,
    padtype="odd",
    padlen=None,
)
```

Record the actual rate, parameters, edge handling, and SciPy version. Check minimum signal length before filtering. Do not filter separately inside each 10-second window.

Describe this as Butterworth **design `N=4` with forward/backward SOS filtering**. For a band-pass design, SciPy's resulting order is `2*N`; forward/backward application changes the magnitude response again [6–7]. Do not claim that the final effective response is simply a fourth-order filter with final −3 dB edges.

The full operative record is available at the operation-end prediction time. This retrospective zero-phase operation does not establish a real-time causal-filter deployment pipeline.

### 5.6 Preserve both feature branches

Leave the original filters unchanged: PPG 0.1–10 Hz band-pass, ECG 0.5–40 Hz band-pass, CO₂ 8 Hz low-pass, and airway pressure 12 Hz low-pass, through the existing design N=4 zero-phase SOS path. Their native source rates remain 500 Hz for ECG/PPG and 62.5 Hz for CO₂/airway pressure. Do not apply the EEG-specific SQI/amplitude masks to these signals.

| Signal | Whole-operation branch | Windowed branch |
| --- | --- | --- |
| ECG and PPG | Existing 10 Hz representation | Existing 100 Hz representation |
| CO₂ and airway pressure | Existing 10 Hz representation | Existing 62.5 Hz representation |
| EEG | **128 Hz** | **128 Hz** |

Do not apply the existing global 10-Hz full-series downsampling to EEG. A 10-Hz signal cannot retain the specified EEG passband. Add a channel-specific override while leaving the original four channels unchanged.

- Whole-operation: Catch24 over the cleaned entire operative EEG, 24 features.
- Windowed: complete 10-second windows, five-second stride; at 128 Hz, 1,280 samples per window and 640-sample stride.
- Window aggregates: mean, sample SD (`ddof=1`), minimum, maximum for each descriptor, 96 EEG features.
- Five-channel totals before preoperative encoding: 120 whole-operation features and 480 window-aggregated features.
- Use `pycatch22.catch22_all(..., catch24=True)` and existing descriptor names.
- No additional waveform standardization before Catch24. Its mean and SD remain in processed waveform units.
- Preserve the existing rule that undefined per-window descriptors propagate to their corresponding aggregates. Do not switch silently to `nanmean`/`nanstd`.
- Preserve an undefined aggregate SD as NaN if only one valid window contributes.

The comparison between full and windowed **multimodal pipelines** still differs in original-channel sample rates and feature counts. Do not interpret it as a pure experiment isolating only temporal aggregation.

### 5.7 Low amplitude, undefined descriptors, and gap auditing

Low amplitude can be physiological. Do not define “bad EEG” by anesthetic suppression, high suppression ratio, or a clinically broad minimum-amplitude rule.

The current numerical SD guard is `1e-6` in signal units. For EEG v1, retain that existing guard after conversion to µV and audit its exclusions. Do not broaden it into a clinical suppression threshold. An entire EEG record failing the guard does not qualify as a usable modality merely because its mean and SD can be calculated; preserve those diagnostic summaries and the failure reason in the QC sidecar.

Apply the existing numerical guard to individual EEG windows as well. A numerically degenerate window is excluded under that established rule; a channel with no qualifying windows fails the windowed branch. Within otherwise qualifying recordings/windows, undefined Catch24 descriptors remain NaN and do not cause case exclusion. This fixes the low-variance policy before cohort construction rather than allowing case-by-case decisions. Preserve the original four-channel behavior.

Keep source missingness separate from quality-related masking. Save at least:

- Selected source channel, load/fallback status, sampling rate, unit conversion.
- Operative expected/observed sample counts and duration.
- Nonfinite, low-SQI, unknown-SQI, and amplitude-exceedance fractions.
- Union unusable/interpolated fraction.
- Interior gap count and maximum duration.
- Leading and trailing filled duration.
- Retained/failed windows and per-descriptor missingness.
- Channel-level pass/fail reason and branch extraction status.

The 5% allowance does not cap individual gap duration. At a two-hour duration it could encompass six continuous minutes. Inspect gap shapes and representative raw/cleaned signals without using outcome associations to choose thresholds. Include short, long, low-amplitude, artifact-heavy, and near-threshold records in the pilot.

Do not automatically discard every window touching interpolation or introduce an undocumented maximum-gap cutoff. If the pilot exposes a serious signal-integrity problem, record the evidence and version the protocol before definitive fitting rather than concealing the change. A small synthetic-gap check on otherwise usable real signals can assess feature sensitivity; it is an engineering audit, not a claim of clinical validation.

## 6. Exact feature-set grid

Let `P` denote the unchanged preoperative feature set. Let `W4 = {PPG, ECG, CO2, AWP}`, and `W5 = W4 ∪ {EEG}`.

Use the **23-set exact union** below. This formalizes the user's request to run all existing analyses plus the EEG extension. The earlier 19-set proposal omitted the four original preoperative-plus-three-waveform ablations; retaining those adds four sets, not a powerset search.

The following IDs are proposed explicit experiment IDs. Adapt them to the repository's naming conventions if necessary, but persist an exact membership map and keep four-versus-five parent sets unambiguous.

| # | Proposed ID | Predictor membership |
| --- | --- | --- |
| 1 | `preop_only` | P |
| 2 | `pleth_only` | PPG |
| 3 | `ecg_only` | ECG |
| 4 | `co2_only` | CO₂ |
| 5 | `awp_only` | AWP |
| 6 | `eeg_only` | EEG |
| 7 | `original_four_waveforms` | W4 |
| 8 | `all_five_waveforms` | W5 |
| 9 | `preop_and_pleth` | P + PPG |
| 10 | `preop_and_ecg` | P + ECG |
| 11 | `preop_and_co2` | P + CO₂ |
| 12 | `preop_and_awp` | P + AWP |
| 13 | `preop_and_eeg` | P + EEG |
| 14 | `preop_and_all_five` | P + W5 |
| 15 | `preop_and_five_minus_pleth` | P + W5 − PPG |
| 16 | `preop_and_five_minus_ecg` | P + W5 − ECG |
| 17 | `preop_and_five_minus_co2` | P + W5 − CO₂ |
| 18 | `preop_and_five_minus_awp` | P + W5 − AWP |
| 19 | `preop_and_original_four` | P + W4; exactly the five-channel ablation omitting EEG |
| 20 | `preop_and_four_minus_pleth` | P + W4 − PPG |
| 21 | `preop_and_four_minus_ecg` | P + W4 − ECG |
| 22 | `preop_and_four_minus_co2` | P + W4 − CO₂ |
| 23 | `preop_and_four_minus_awp` | P + W4 − AWP |

Rows 20–23 are the historical compatibility block. Do not count `preop_and_five_minus_eeg` again as a separate scientific fit from row 19.

| Scope | Outcomes | Branches | Families | Feature sets | Nominal configurations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared EEG cohort | 2 | 2 | 3 | 23 | 276 |
| Broader EEG ICU cohort | 1 | 2 | 3 | 23 | 138 |
| **Total** | | | | | **414** |

The 19-set version would have 342 nominal configurations. This handoff chooses 414 to preserve every original feature identity. These are configurations, not counts of all fitted estimators; nested tuning and calibration require many additional fits.

Deduplicate truly identical preoperative-only branch jobs only after verifying identical rosters, folds, feature data, preprocessing, HPO budget, weights, seeds, and model settings. Retain a manifest entry for every requested nominal configuration and a pointer to its verified canonical fit. Do not reuse a main-cohort fit for the broader ICU cohort.

## 7. Comparisons and validation

### 7.1 Prespecify the EEG comparisons

Within each model family, outcome, cohort, and branch, compute paired comparisons using identical cases:

1. **P + W5 versus P + W4:** incremental EEG information beyond the existing combined predictor.
2. **W5 versus W4:** incremental EEG information within waveform-only prediction.
3. **P + W5 versus P:** overall increment over the clinical baseline.
4. **P + EEG versus P**, and EEG alone: contextual measures of EEG information.

Retain the existing core and ablation contrasts as well. Every five-waveform ablation should be compared both with the complete P + W5 model and, where the existing reports do so, with P. The compatibility four-waveform ablations should also have a paired comparison with P + W4.

Treat the whole-operation contrasts in the shared cohort as the principal feasibility comparisons; windowed, broader-ICU, individual-channel, and ablation findings are exploratory. Do not choose a winning primary model after viewing its performance.

### 7.2 Preserve grouped nested validation

- Five outer folds, three inner HPO folds, five training-only calibration cross-fitting folds; one repeat, base seed 42, and the existing outer-fold seed offsets.
- Group every split by `subjectid`, including EBM internal validation/bagging. Never split windows as independent observations.
- Build experimental split plans after the final cohort is frozen. Do not require a restricted original split if it no longer supports the analysis.
- Cache one deterministic plan per cohort/outcome and reuse it across all feature sets and both branches. Outcomes may retain different stratified plans.
- Preserve the existing grouped stratification search over up to 64 deterministic seed candidates, based on fold class balance rather than predictive performance.
- Verify class support in outer, inner, calibration, and EBM internal partitions. No silent fold reduction.
- Fit all learned preprocessing only on each training partition.
- Keep outcome definitions, inclusion flags, IDs, split fields, EEG availability, SQI statistics, selected-channel names, gap statistics, and extraction-status fields out of predictors.
- Preserve the approved preoperative whitelist and existing categorical encoding. Use NaN natively for XGBoost/EBM and training-derived imputation/indicators/scaling for LR.
- Use 50 seeded HPO trials for **each** family, with the current search spaces and inner-fold raw-score AP objective.
- Recompute class imbalance handling within every fitting partition. LR uses `class_weight="balanced"`; XGBoost applies its class ratio once through `scale_pos_weight`; EBM follows its current fold-local weighting.
- Preserve the existing training-only logistic recalibration and threshold selection. Recalibration remains unweighted unless the current validated contract explicitly says otherwise.
- Apply locked calibrators/thresholds to untouched outer-test data. Save each operation's one outer-test prediction.

Do not confuse Step 03's outcome-specific holdout split columns with the definitive nested split cache. Preserve the former contract where required, but use nested CV for the definitive experiment.

### 7.3 Metrics and uncertainty

Primary metric: **average precision (AP)**, using the repository's implementation, not trapezoidal PR area under an ambiguous label. Also export AUROC, Brier score, calibration, sensitivity, specificity, precision, F1, accuracy, prevalence, and denominators.

Use saved pooled outer-test predictions. Preserve 5,000 patient-clustered bootstrap resamples, seed 0, and paired differences. Resample patients with all their operations and preserve repeated cluster multiplicity. Verify case ID, subject ID, label, fold, and prediction provenance before pairing.

Compute EEG-added deltas directly; do not infer them by subtracting two separately reported confidence intervals. Use the same bootstrap draws for each paired contrast. Include ΔAP, ΔAUROC, and ΔBrier, with directions labeled.

Report bootstrap intervals as conditional on the saved predictions/fits. They do not redo HPO or remove every source of shared-training dependence in pooled CV. Preserve the repository's `unsupported_design` behavior for ordinary DeLong testing of pooled-CV/repeated-patient data; do not fabricate conventional p-values.

Ablation intervals are nominal exploratory intervals, not multiplicity-adjusted discoveries. Report the complete grid even if EEG adds little or nothing.

Decision curves retain the existing fixed threshold ranges: AKI 0.01–0.30 and ICU 0.05–0.60, in 0.01 steps. Use saved calibrated outer-test probabilities, paired patient resampling, and pointwise bands. Add an EEG-increment comparison of P + W5 versus P + W4 without selecting thresholds after seeing results.

## 8. Implementation map and failure modes

Prefer minimal modifications to existing modules. Verify actual function signatures and supported flags before writing commands; proposed names in this handoff are not claims that the current CLI already supports them.

| Location | Required change or inspection |
| --- | --- |
| `data_preparation/inputs.py` | Explicit experimental modality/profile; EEG preferred/fallback mapping; per-channel whole-operation sampling rate; isolated paths |
| `data_preparation/waveform_processing.py` | EEG filter/rate/unit specification, fixed cleaning helper, provenance; retain old-channel behavior |
| `data_preparation/step_01_cohort_construction.py` | Fifth-source eligibility and exclusion accounting under the EEG profile |
| `data_preparation/cohort_scopes.py` | Preserve existing internal scope enums; associate distinct EEG manifest identities with the experimental profile/rosters |
| `data_preparation/step_02_catch_22.py` | Timestamped SQI alignment, masks/union/interpolation, EEG128 full branch, both branches, QC sidecars and fingerprint |
| `data_preparation/step_04_intraop_prep.py` | Canonical EEG pivot and feature dimensions; preserve descriptor NaNs |
| `data_preparation/step_05_data_merge.py` | Experimental matched rosters, both-branch intersection, distinct cohort identities |
| `model_creation/utils.py` | Register EEG as a waveform prefix; explicit W4/W5 membership; keep audit metadata out of preop |
| `run_catch22.py` | Experimental preset/allowlist, 23-set grid, both scopes/branches, balanced LR50, isolated study layout |
| `model_creation/experiment_queue.py`, `split_plan.py`, `split_cache.py`, `validation.py` | Reuse grouped/resumable machinery; verify EEG/profile fingerprints and exact comparability |
| `reporting/feature_set_display.py`, `metadata/feature_components.yml`, `metadata/display_dictionary.json` | EEG labels, units, four/five parent sets, 120/480 feature dimensions |
| `results_recreation/` | Paired EEG-added contrasts from saved predictions, complete numerical exports |
| `reporting/manuscript_workspace/` | Parameterized experiment inputs and regenerated scientific content using current visual style |

### Specific hazards to prevent

1. **Changing old feature meaning:** adding EEG to the prefix registry currently changes dynamic `all_waveforms` membership. Preserve default four-channel semantics and define explicit experimental aliases.
2. **Predictor leakage:** unregistered EEG/QC columns can be classified as preoperative candidates by exclusion-based selection. Keep QC in sidecars and assert the preop whitelist.
3. **False cache reuse:** new masks, sampling rates, rosters, features, trial budgets, or weighting require distinct compatible identities. Never edit a fingerprint to force an old checkpoint to load.
4. **Partial-channel inclusion drift:** current branch tables can retain incomplete channels. Enforce the new strict roster in one explicit, audited place before all experimental feature selection.
5. **Old reporting evidence:** the publication builder hardcodes manuscript study roots and carries prebuilt EBM displays. A new output directory alone does not make its content experimental.

Fingerprint at least source identities, code/environment versions, modality profile, native/full/window rates, filter design/padding, SQI alignment/cutoff, amplitude-mask stage/cutoff, union/interpolation rules, fallback policy, low-variance handling, feature/aggregation semantics, cohort roster, outcome definition, split plan, HPO/weighting/calibration settings, and diagnostic/definitive mode.

Use an explicit extraction/protocol schema version. Preserve previous immutable identities instead of retroactively rewriting them.

## 9. Recreate the complete reporting package

The current verified package has **two main tables, four analytical main figures plus a cohort-flow figure, and ten supplemental displays**. Reproduce these families using EEG-study data. Their current titles below provide the inventory; adapt titles/captions where five-versus-four comparisons require it.

| Display | Current family | EEG-study requirements |
| --- | --- | --- |
| Figure 1 | Study cohort selection | New five-channel/QC/both-branch flow and both outcome scopes |
| Table 1 | Cohort characteristics | New shared cohort, current descriptive conventions, explicit missingness |
| Table 2 | Performance by input configuration | Matched P, W4, W5, P+W4, P+W5, both branches/families/outcomes; use continuation panels if needed |
| Figure 2 | Incremental discrimination | Add explicit P+W5 minus P+W4 and W5 minus W4 panels alongside core clinical-baseline contrasts |
| Figure 3 | Whole-operation ROC curves | New pooled outer-test predictions, matched four/five comparisons |
| Figure 4 | Global EBM feature importance | New experiment's fitted bundles and correct feature/source mapping |
| Figure 5 | EBM feature-response functions | New fold-specific curves and feature distributions |
| Table S1 | Predictors and feature extraction | EEG source, processing provenance, sample rates, 120/480 dimensions, unchanged clinical variables |
| Table S2 | Main and broader ICU cohorts | New experimental counts; add EEG inclusion/exclusion context in a companion audit |
| Figure S3 | Whole-operation PR curves | New probabilities, AP, correct prevalence baseline |
| Figure S4 | Whole-operation calibration | New predictions; show the meaningful full risk range, optional zoom only |
| Figure S5 | Absolute AP by waveform channel | Include EEG-only and complete modality sets with clear cohort/branch labels |
| Figure S6 | Absolute AUROC by waveform channel | Same exact feature identities and cohorts as S5 |
| Figure S7 | Added discrimination for ICU by cohort | Shared versus broader ICU, with both clinical-baseline and EEG-added contrasts distinguished |
| Figure S8 | Decision curves | New calibrated predictions and fixed ranges |
| Figure S9 | Paired net-benefit differences | Existing core comparison plus EEG-added comparison |
| Figure S10 | Held-out EBM prediction explanations | New saved outer-test cases and corresponding fold bundles |

The numbered manuscript-style package may emphasize whole-operation results as the current package does. The **complete analysis export must also contain windowed ROC/PR/calibration, metrics, ablations, and available model-explanation outputs**, plus broader-ICU versions where relevant. Do not interpret a compact Word/PDF book as permission to omit the rest of the requested analysis.

Preserve current colors, typography, display ordering, source-table conventions, and publication-quality export settings where suitable. Use `reporting/publication_style.json` and existing templates. Make legends explicit about original four versus all five waveforms. Avoid putting all 23 models on an unreadable single curve panel.

### Reporting safeguards

- Regenerate flow counts and Table 1 from the new cohort. Do not copy old counts from the author Word document or portable workspace.
- Rebuild EBM importance, response functions, and case explanations from experimental fits. The current builder carries old Figures 4, 5, and S10; replace this behavior in the experiment.
- Inspect the builder's hardcoded study names, LR override, fixed predictor count, selected-feature assumptions, and approved eGFR-response assertion.
- Do not force EEG into a top-importance plot if it is not among the displayed top features. A separate clearly labeled EEG response panel is acceptable if selected by a documented, outcome-independent rule.
- For held-out examples, use the existing deterministic rule: cases nearest the 10th and 90th percentiles of combined whole-operation EBM risk, separately by outcome. Apply it to the new P+W5 model; state the rule and deterministic tie-breaking. Do not search for visually favorable EEG cases.
- Use the matching outer-fold model/calibrator for each case. Do not mix raw model log-odds contributions with calibrated probabilities without the existing documented transformation.
- Preserve training-only distributions/reference values for feature-response plots; record selected feature IDs and folds.
- Preserve approved units and descriptions. PPG waveform mean is not SpO₂; Catch24 names that resemble clinical indices are not validated clinical measurements.
- Keep existing preoperative ECG model encoding; descriptive ECG may be collapsed as normal versus abnormal under the current display rules.
- Preserve source tables, PNG/PDF assets, document outputs, and a display manifest linking each panel to cohort, branch, family, feature set, model bundle, predictions, and code revision.
- Verify every plotted/table value against its saved numerical source.
- Do not merely rename old figure files, copy old model explanations, or use training OOF curves as outer-test performance.

Export a complete ablation table and concise comparison plots even if the existing numbered display package does not have a dedicated ablation figure. Label the four-original compatibility block separately from five-waveform ablations.

## 10. Staged execution

### Stage A: inspect and freeze

Read the source documents and narrow rules, inventory the actual runtime/studies, verify baseline provenance, preserve recoverable artifacts and environment, create the separate worktree, and resolve protected versus writable paths.

Deliverables: preservation manifest, exact baseline/fitting/reporting SHAs, source-study map, environment record, experiment branch/root, and any concrete unresolved preservation gap.

### Stage B: implement and run an outcome-blinded extraction pilot

Implement the EEG path and experimental configuration. Use a small deterministic case sample chosen to cover duration, signal availability, fallback, SQI gaps, amplitude excursions, and low-amplitude recordings. Do not choose cleaning settings by outcomes.

Verify native sample rate/units, SQI cadence/support, masking/interpolation order, edge behavior, low-variance handling, descriptor completeness, and fallback provenance. Save representative raw/cleaned plots and QC tables.

Benchmark both feature branches, especially whole-operation Catch24 at 128 Hz, with representative short/median/long records. Record wall time, peak memory, process count, and failed/time-limited extractions. Do not assume extraction is cheap because model fitting is already optimized.

Freeze the exact protocol/configuration after resolving genuine metadata/implementation issues and before definitive modeling. Record every deviation from this handoff with its reason and supporting audit.

### Stage C: targeted verification and real-data smoke

Run meaningful tests of the touched behavior, then representative real-data diagnostic model runs for LR, XGBoost, and EBM, covering both branches and both cohort scopes. Small diagnostic settings may be used for smoke tests only; label their outputs and exclude them from definitive results.

Exercise the actual preprocessing, training, recalibration, thresholding, prediction saving, resumption, and reporting paths. Do not launch the full grid merely because isolated EEG extraction works.

Commit the successfully tested implementation and freeze its environment/configuration before definitive execution. Record that exact experimental SHA. If a scientific or numerical fix is needed later, create an auditable new revision and invalidate affected artifacts rather than mixing incompatible results under one unchanged identity.

### Stage D: full extraction and cohort/split freeze

Extract both branches once into the new study. Reuse only proven-compatible immutable four-waveform feature artifacts. Build the strict common roster, outcome scopes, attrition tables, and final feature matrices.

Generate and verify grouped split plans. Check support for every required fold and model-internal partition. Save final operation, patient, outcome, feature-membership, and split hashes.

Complete a measured compute/storage estimate from extraction and model-smoke timings. Estimate extraction, HPO/model fitting, bootstrap/explanation generation, and reporting separately. State uncertainty; do not promise completion in one or two days.

### Stage E: run the full requested grid

Run the principal matched core models first as an end-to-end checkpoint, then continue to the remaining singles and ablations, and the broader ICU grid. This staging changes execution order only; it does not reduce the final 23-set scope.

Use the existing resumable CPU scheduler, bounded process/thread settings, and scientific fingerprint checks. Avoid nested parallelism that oversubscribes cores or memory. Use local caches and the existing data-fetch behavior rather than repeatedly downloading tracks for every model job.

Save status per configuration: planned, running, completed, verified reuse, failed with reason. Isolate pilot artifacts. Do not silently replace failed definitive jobs with reduced-fold, reduced-trial, or bounded-round diagnostics.

If execution continues beyond an interactive session, use the existing supported durable job mechanism and provide the actual job/session identifier, log location, completed/remaining counts, and resume command. Do not claim a job is running unless it was actually started and observed.

### Stage F: rebuild, validate, and summarize

Regenerate metrics and all display families from saved definitive artifacts. Audit panel provenance, denominators, units, captions, and uncertainty. Render the final Word/PDF outputs and inspect their layout.

Recheck protected manuscript checksums/paths and record any expected changes outside immutable scientific content separately. Provide a completion report with the experiment's code/environment versions, cohort/event counts, grid completeness, paired EEG effects, preprocessing retention, compute consumption, and material limitations.

Do not use favorable results as a reason to modify the original manuscript automatically.

## 11. Required verification cases

Use the existing test framework. Add focused tests that address these scientific or operational risks; avoid broad refactors and tests that only restate configuration values.

1. **Legacy preservation:** unchanged four-channel sample inputs produce identical features within the existing numerical tolerance under the default profile; default feature memberships and paths remain unchanged.
2. **Fallback:** EEG1 present, absent/empty, load failure, EEG2 absent, and EEG1 nonempty-but-QC-failed behave exactly as specified; both branches use the same selected source.
3. **Units and sampling:** known µV/mV/V input conversion, unexpected metadata failure, EEG128 retention, and original four-channel rates.
4. **SQI timing:** irregular timestamps, leading unknown time, stale gaps, exact threshold 50, invalid numeric values, and a quality change at a timestamp boundary.
5. **Masks and interpolation:** overlapping defects count once; exactly 5% passes, >5% fails; exact amplitude boundary; all-invalid recordings; interior versus leading/trailing fill; original missingness retained separately from artifact masks.
6. **Filter integrity:** plausible synthetic passband/stopband behavior and finite output with documented padding; no accidental 10-Hz EEG resampling.
7. **Feature semantics:** whole24/window96 per EEG group, 1,280/640 EEG windows, defined sample SD, partial undefined descriptors, and physiological low-amplitude handling.
8. **Roster integrity:** all compared feature sets and both branches share the frozen roster; strict channel success does not become an all-descriptors-finite requirement.
9. **Leakage:** no patient crosses a split; all preprocessing is fold local; preop contains neither EEG nor QC/outcome/identity fields.
10. **Reuse/resume:** changing any relevant EEG/QC/cohort/rate/HPO/weighting configuration invalidates incompatible caches; identical interrupted jobs resume without duplicating outer-test predictions.
11. **Grid membership:** 23 unique intended predictor sets; no double fit for the EEG-omitted alias; distinct W4/W5 ablations.
12. **Reporting provenance:** all experiment displays resolve to new experimental predictions/rosters/bundles; no carried old figure/count escapes detection.

A failed optional cosmetic check should not block valid computation. A failed cohort, leakage, unit, provenance, or isolation check must be repaired before definitive modeling.

## 12. Deliverables and definition of done

Produce:

1. Committed experimental implementation on the separate branch/worktree, with a concise change summary.
2. Manuscript preservation manifest and verified reproduction instructions.
3. EEG protocol/configuration and a source-to-decision evidence table.
4. QC/coverage/fallback/gap/attrition report, including selection effects and separate operation/patient counts.
5. Frozen experimental rosters, grouped split plans, feature-membership map, and environment/source/code identities.
6. Complete saved definitive results for the 414 nominal configurations, with verified canonical-fit pointers where deduplicated.
7. Full metrics, paired comparisons, bootstrap summaries, decision curves, and exhaustive ablation exports.
8. All main and supplemental display families rebuilt from experimental data, plus windowed/broader-cohort companion outputs.
9. Models, outer-fold bundles, calibrators, thresholds, explanations, and reproducible figure/source tables.
10. Exact commands for preparation, smoke tests, full execution, resumption, and report-only rebuilding, verified against the implemented CLI.
11. A final status report distinguishing completed, reused, failed, and intentionally excluded work.

The primary manuscript must remain reproducible and unchanged. The new study must be independently identifiable and reproducible. Complete execution does not require a favorable EEG effect.

If access to JetStream, raw data, original artifacts, or compute is unavailable, complete all code, tests that can run, configuration, and launch/report commands available in the accessible environment. State the specific blocker and remaining command/stage. Do not invent cohort counts, results, a successful freeze, or a running job.

## 13. Decision ledger

This ledger distinguishes conversation decisions from details formalized in this handoff.

| Status | Decisions |
| --- | --- |
| User direction | Preserve the manuscript; separate EEG proof of concept; preferred EEG channel with fallback; both whole/window branches; lightweight preprocessing; retain consideration of the existing 5% interpolation architecture; all cohorts, single waveforms, ablations, and familiar displays; substantial resumable compute acceptable |
| Formalized experimental defaults | Strict five-channel/both-branch usability; 23-set exact union; matched rosters/folds; EEG128 in both branches; composite QC order with SQI50 and raw100µV masks; balanced50-trial LR/XGB/EBM; explicit EEG-added paired contrasts |
| Engineering parameter requiring metadata confirmation | Bounded SQI hold age, initially 1.0 s; actual units/rate and loader behavior; these must be resolved and recorded before fitting |
| Empirical feasibility checks | Qualified coverage/event counts, grouped-fold support, gap distributions, whole128Hz runtime/memory, actual runtime mounts and archive completeness |
| Not decided by this experiment | Manuscript inclusion, replacement of primary results, claim of clinical utility, external validity, causal EEG interpretation, or novelty/“first study” claims |

## 14. References and implementation sources

### EEG methods and official documentation

1. VitalDB. Open Dataset overview and parameter list. [Official overview](https://vitaldb.net/dataset/?query=overview). Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. *Scientific Data*. 2022;9:279. doi:[10.1038/s41597-022-01411-5](https://doi.org/10.1038/s41597-022-01411-5).
2. VitalDB. Web API OpenDataset. [Official timestamp and missing-row documentation](https://vitaldb.net/docs/?documentId=API%2FWeb_API_OpenDataset.md).
3. Seo KH, Kim K, Lee SK, Cho J, Hong JH. Changes in electroencephalographic power and bicoherence spectra according to depth of dexmedetomidine sedation in patients undergoing spinal anesthesia. *International Journal of Medical Sciences*. 2021;18(10):2117–2127. doi:[10.7150/ijms.54677](https://doi.org/10.7150/ijms.54677). PMID:[33859518](https://pubmed.ncbi.nlm.nih.gov/33859518/). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8040410/).
4. Fritz BA, Kalarickal PL, Maybrier HR, et al. Intraoperative electroencephalogram suppression predicts postoperative delirium. *Anesthesia & Analgesia*. 2016;122(1):234–242. doi:[10.1213/ANE.0000000000000989](https://doi.org/10.1213/ANE.0000000000000989). PMID:[26418126](https://pubmed.ncbi.nlm.nih.gov/26418126/). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4684753/).
5. Sukriti, Kriplani C, Kumar S, Singh A. Predicting depth of anaesthesia from single-channel EEG using a deep TCN-BiLSTM-attention model with EWMA. *Scientific Reports*. 2026;16:25470. doi:[10.1038/s41598-026-54608-8](https://doi.org/10.1038/s41598-026-54608-8). PMID:[42243301](https://pubmed.ncbi.nlm.nih.gov/42243301/). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13473519/).
6. SciPy. `scipy.signal.butter`. [Official API documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html). Record the installed version used by the experiment.
7. SciPy. `scipy.signal.sosfiltfilt`. [Official API documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).

### Repository implementation anchors at the reviewed revision

- **R1:** [data_preparation/step_02_catch_22.py](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/data_preparation/step_02_catch_22.py): missingness gate, interpolation, branches, extraction.
- **R2:** [data_preparation/waveform_processing.py](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/data_preparation/waveform_processing.py): signal specifications, filtering, resampling, fallback.
- **R3:** [data_preparation/inputs.py](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/data_preparation/inputs.py): mandatory channels, rate/window settings, paths.
- **R4:** [model_creation/utils.py](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/model_creation/utils.py): feature prefixes, membership and preoperative separation.
- **R5:** [run_catch22.py](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/run_catch22.py): launcher, study layout, grid and manifests.
- **R6:** [reporting/manuscript_workspace/revise_publication.py](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/reporting/manuscript_workspace/revise_publication.py): authoritative current display inventory and historical carry-forward behavior.
- **R7:** [reporting/manuscript_workspace/README.md](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/reporting/manuscript_workspace/README.md): existing report reproduction workflow.
- **R8:** [.agent/rules/catch22-scientific-integrity.md](https://github.com/do-justin/aki_prediction_project/blob/ab39e021468c692fd31e9b3d62f7637d24c4cfaa/.agent/rules/catch22-scientific-integrity.md): scientific guardrails.

### Current display sources

- [Main figures and tables](https://drive.google.com/file/d/16Iu0rQmHLIysgq-FUTjreGWHrAFnqmYa/view).
- [Supplementary figures and tables](https://drive.google.com/file/d/1QUkec2bdKT_JL6AhTg98xmkeqrSaFAMz/view).

These sources establish the reviewed starting point. Verify their live state when implementing; preserve provenance rather than silently substituting newer content.
