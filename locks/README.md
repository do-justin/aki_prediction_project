# CPU environment replay

These locks record the Linux x86_64 / CPython 3.12.16 environment prepared on
October 30, 2026, from commit `b75138f`, for Catch24, XGBoost, EBM and pytest.
The original GPU environment is separate. Scientific equivalence checks are
recorded in the [historical study log](../archive/docs/manuscript-reorientation-2026-11-04.md).

For ordinary recreation, use the [Conda setup](../README.md#environment-and-verification).
The captured wheels below support exact binary replay.

- `conda-linux-64.explicit.txt`: 27 exact Conda URLs/builds with SHA256 hashes.
- `requirements-linux-64.lock.txt`: 66 pinned pip distributions; also constrains
  ordinary installs from the root manifests.
- `requirements-linux-64.wheels.txt`: the same 66 pins with captured wheel hashes.
- `installed-python-linux-64.txt`: 70 Python distribution versions, including
  Conda-owned packaging tools. This uses package metadata rather than `pip freeze`
  so Conda's build-time `file://` paths are not treated as installable inputs.

The wheelhouse is stored outside Git. On the current CPU VM it is at
`/home/exouser/codex-handoffs/catch22-cpu-2026-10-30/wheelhouse`.
Retain this directory with the locks: `pycatch22==0.4.5` was compiled locally;
rebuilding its source can change the wheel hash. A clone containing only the
locks cannot promise exact binary replay without these wheels. Conda packages
must also remain available upstream or in the Conda cache. The pip install is
offline; Conda creation may contact conda-forge for uncached packages.

From the repository root, select a new prefix and the retained wheelhouse:

```bash
AKI_REBUILD_PREFIX="$HOME/miniforge3/envs/aki_prediction_project_rebuild_final" \
AKI_WHEELHOUSE=/home/exouser/codex-handoffs/catch22-cpu-2026-10-30/wheelhouse \
  bash scripts/cpu/rebuild-from-locks.sh
```

The destination must not exist. The script uses `CONDA_EXE` or `conda` on PATH;
`AKI_CONDA_EXE` overrides that choice. It preserves locks, wheels and existing
environments. It checks wheel hashes, exact Conda builds and distribution
versions, then runs `pip check`, CLI help, native checks and the offline suite.
Evidence is retained in a new `/tmp/aki-cpu-rebuild.*` directory printed at
startup. Copy evidence elsewhere before temporary storage is cleaned.

For subsequent verification, activate the environment and run:

```bash
bash scripts/cpu/verify-offline.sh
```

Outputs go to a fresh `/tmp/aki-cpu-offline.*` directory. To keep evidence in a
durable location, set `AKI_VERIFY_ROOT` to a new directory whose parent exists.
Existing directories are refused because pytest clears its temporary base.
All project input/output roots are overridden inside the script, and storage
enforcement is disabled only for those synthetic validation subprocesses.
Neither script runs the smoke launcher or accesses original research storage.

Native checks cover the synthetic Windows Python 3.9 Catch24 reference
(`rtol=atol=1e-10`), CPU XGBoost and SHAP, the EBM factory and explanations,
and Plotly, CSV, Markdown, Word and PDF exports. Scientific reproduction also
requires provenance, cohort/split/schema and saved-result comparisons.

## Verification recorded October 30, 2026

In `aki_prediction_project_rebuild_verified`, all 27 Conda builds, 66 wheel
hashes and 70 distribution versions matched. Native checks and all 81 then-current
tests passed, with a maximum Catch24 discrepancy of `6.661338147750939e-16`.
The ordinary Conda setup also matched the distribution snapshot and passed both
checks using package hosts with no local wheelhouse or pip cache. Evidence is
under `outputs/cpu-validation/2026-10-30/`.

## Changing dependencies

1. Keep the validated environment and its Git revision intact.
   Work on a separate branch and create a new Conda candidate from the edited
   manifests. Install Conda packages first and pip packages last. Avoid updating
   the validated mixed Conda/pip environment in place.
2. Change direct pins and affected transitive constraints together, recording why
   each change is needed. Never install both XGBoost distributions. Treat changes
   to Python, pycatch22 or EBM as requiring additional compatibility and numerical
   checks. Update the Python-version expectation in `verify_native.py` if needed.
3. Capture the candidate's exact Conda export with `conda list --explicit --sha256`.
   Regenerate the pip-only version lock. If maintaining exact binary replay,
   capture compatible wheels and refresh the wheel hashes together.
   Regenerate the installed-distribution snapshot from package metadata. Keep
   Conda-owned packaging tools out of the pip lock; do not copy `file://` build
   paths from `pip freeze`. Review all manifest/lock differences together.
4. Recreate from the manifests in a fresh environment and run native checks and
   the complete offline suite before promoting the candidate. Version changes
   may alter predictions even when tests pass: compare features, splits, predictions,
   calibration and thresholds on fixed, provenance-checked research inputs before
   replacing runs.
5. Retain the old and new Git revisions, verification logs, input hashes and run
   settings. Keep patient data out of Git; retain wheels for exact binary replay.

Lock refresh is manual. Exact replay uses the retained wheelhouse and available
Conda packages; ordinary setup uses the pinned manifests and package hosts.

This procedure follows the upstream guidance for
[pip repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/)
and [using pip with Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#using-pip-in-an-environment).
