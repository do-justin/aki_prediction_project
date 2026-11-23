---
description: Run the Catch22 pipeline
---

# Catch22 execution

Use [README.md](../../README.md) for the environment, source paths, study
roots and launcher commands. Scientific settings are in
[METHODS.md](../../METHODS.md); artifact contracts are in
[RESULTS.md](../../RESULTS.md). Do not duplicate commands here.

- Readiness-only requests: offline tests, CLI help, resource inventory, and
  `experiments --preset core --dry-run`; no data preparation or model fitting.
- Research execution requires a run request. Start with the authorized scope;
  use the gated core queue, which mixes model families, before supplementary
  grids. The [pilot record](window-duration-pilot.md) covers saved-model
  duration follow-up.
- Requested real-data smoke uses a fresh disposable root and live VitalDB access.
- Never overwrite historical artifacts or bypass failed provenance checks.
- Use run manifests for study status and the [manuscript handoff](manuscript-reorientation.md)
  for publication decisions.
