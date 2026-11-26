#!/usr/bin/env bash
set -euo pipefail

worktree="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$worktree"
project_conda="${AKI_CONDA_INIT:-/home/exouser/miniforge3/etc/profile.d/conda.sh}"
source "$project_conda"
conda activate aki_prediction_project
export AKI_ARTIFACT_ROOT="${AKI_ARTIFACT_ROOT:-/media/volume/catch22/data/aki_prediction_project/studies/eeg-feasibility-v1}"
export RAW_DIR="${RAW_DIR:-/media/volume/catch22/projects/aki_prediction_project/data/raw}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export AKI_EEG_DOWNLOAD_WORKERS="${AKI_EEG_DOWNLOAD_WORKERS:-4}"

if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Commit the verified EEG implementation before running this supervisor.\n' >&2
    exit 1
fi

python -B -c 'from experimental.eeg.runtime import configure_environment; configure_environment()'
control="$AKI_ARTIFACT_ROOT/control"
mkdir -p "$control"
exec 9>"$control/eeg-supervisor.lock"
if ! flock -n 9; then
    printf 'An EEG supervisor already owns this study root.\n' >&2
    exit 1
fi
phase=preflight
phase_started=$SECONDS
status_file="$control/eeg-supervisor.status"
record() {
    printf 'phase=%s\nstatus=%s\ncode=%s\nutc=%s\n' "$phase" "$1" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "$status_file.tmp"
    mv "$status_file.tmp" "$status_file"
}
record_timing() {
    printf '%s\t%s\t%s\t%s\n' "$phase" "$(($SECONDS - phase_started))" "$1" "$(date -u +%FT%TZ)" >> "$control/eeg-phase-timing.tsv"
}
finish() {
    exit_status=$?
    record_timing "$exit_status"
    if [[ $exit_status -eq 0 ]]; then
        record completed
    else
        record "exit_$exit_status"
    fi
}
trap finish EXIT

record running
python -B -m run_catch22 experiments --study-profile eeg --preflight-only
record_timing 0
phase=diagnostic
phase_started=$SECONDS
record running
python -B -m run_catch22 smoke --study-profile eeg
record_timing 0
phase=definitive_and_reporting
phase_started=$SECONDS
record running
python -B -m run_catch22 experiments --study-profile eeg --prep skip --resume
record_timing 0
phase=preservation_recheck
phase_started=$SECONDS
record running
python -B -m experimental.eeg.verify_preservation
