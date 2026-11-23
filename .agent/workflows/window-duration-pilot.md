# Completed Catch24 duration pilot and manuscript decision

Retain the original 10-second windows with a nominal five-second stride. The
main AKI models preceded this outcome-blinded pilot. Forty seconds is an
exploratory alternative. The duration search is closed. See [Methods](../../METHODS.md#window-duration-rationale-and-retrospective-sensitivity-analysis)
for the manuscript interpretation and [Results](../../RESULTS.md#window-duration-pilot-and-feature-audit)
for the frozen artifacts.

## Recorded engineering definitions

- Sampled 120 distinct patients from the main-shared AKI outer-fold-0 training
  partition with `default_rng(0)` on sorted patient IDs, one operation per
  patient. Channel rejection did not trigger replacement sampling.
- At each tested length `L` and rate `fs`, used `int(L*fs)` samples per window
  and stride `s = int(L*fs/2)`. The four origins were `0`, `s//4`, `s//2`, and
  `(3*s)//4`. Used complete operative windows with the production filters,
  flat-window rule and aggregations. The `pycatch22==0.4.5` audit masked verified
  finite fallbacks in its qualification view only. Observed flags involved the
  embedding-distance descriptor in CO₂ and airway pressure. Ordinary zeros and
  all raw descriptors were preserved.
- Availability required finite mean, SD, minimum and maximum at all four
  origins. Each channel needed at least 22 of 24 descriptors available in
  114 of 120 patients. Coverage was measured per descriptor.
- For each descriptor, channel, reducer and length, the origin statistic was
  each patient's across-origin range divided by the between-patient IQR of
  the patients' origin medians. Its empirical 90th percentile had to be at
  most 0.25 in at least 114 complete patients. A denominator at or below
  `100*eps_float64*max(abs(patient origin-medians))` was unevaluable. Every
  channel/reducer combination needed 20 of 24 passes.
- The shortest candidate needed both gates at `L` and `2L`. Doubling could add
  at most one passing descriptor per channel for availability and per
  channel/reducer for stability. The same length needed at least 80% selection
  across 2,000 paired patient bootstrap draws (seed 0).

## Findings and limits

Under the all-origin rule, embedding-distance availability rose from 47 to
118 of 120 airway-pressure operations and from 85 to 119 CO₂ operations between
10 and 40 seconds. No candidate passed, including in the raw finite-only and
matched-count diagnostics. Longer windows did not consistently stabilize the
SD or extrema. Grid shifts ranged from about 3.75 to 240 seconds across lengths.
Matched counts were equalized across origins within each length before quality
exclusions. Overlapping grids are dependent and extrema can capture brief events
or artifacts. These checks measure computability and placement sensitivity.

The synthetic audit used 120 paired 1,200-second traces with oscillatory/AR(1)
generators, two rates, two noise levels and a central 60-second change. At
lengths of 80 seconds or more, no window fits fully inside the episode. Missing
contrasts were retained and no clinical attenuation or latency criterion was
applied. The same pilot patients can enter held-out partitions of other folds.
Existing model performance had already been inspected.

V2 reused the original V1 sample and 12 hash-verified filtered signals. A later
scheduler adjustment retained 14 completed channel outputs and left numerical
processing unchanged. See `control/resource-adjustments/20261101T052748Z/` under
the pilot root in Results.

## Pending saved-model interpretation audit

Review `CO_Embed2_Dist_tau_d_expfit_meandiff` in saved windowed AKI models across
all folds and families. Include all CO₂ and airway-pressure reducers and their
missingness indicators. Verify data and bundle hashes. Compare results within
each family's explanation scale without refitting. Importance is evidence for
prioritizing a source or interpretation check. Feature removal and retraining
require a separate analysis.

## Related evidence

[Idrobo-Ávila et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11100950/)
used the same segmentation for signal-quality assessment.
[Finseth et al., 2023](https://doi.org/10.1109/ACCESS.2023.3254134) compared 10–40
seconds for physiological stress classification.
[Zhou et al., 2022](https://www.jmir.org/2022/7/e34669/) studied stability in
minute-sampled wearable heart rates. [Lubba et al., 2019](https://link.springer.com/article/10.1007/s10618-019-00647-x)
developed Catch22. The pilot tolerances were engineering conventions.
