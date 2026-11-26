from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest
from vitaldb import dataset

from data_preparation import eeg_extraction, eeg_study
from data_preparation.artifact_metadata import file_sha256, metadata_path_for
from data_preparation.inputs import MANDATORY_WAVEFORMS
from data_preparation.step_04_intraop_prep import process_waveform_file
from experimental.eeg.runtime import _require_isolation


def test_frozen_catalog_initializes_fresh_processed_directory(monkeypatch, tmp_path):
    frozen = tmp_path / "pilot-tracks.csv"
    frozen.write_text("caseid,tname,tid\n9,BIS/EEG1_WAV,track123\n")
    processed = tmp_path / "new-study" / "data" / "processed"
    monkeypatch.setenv("AKI_EEG_TRACK_CATALOG", str(frozen))

    catalog = eeg_study.track_catalog(processed, create=True)

    assert eeg_study.available_cases(catalog, ("BIS/EEG1_WAV",)) == {9}
    assert (processed / eeg_study.TRACK_CATALOG_NAME).read_bytes() == frozen.read_bytes()


@pytest.mark.parametrize("content", ("Time,Value\n0,1\n,2\n0.04,3\n", "Time,Value\n0,1\n0.02,2\n0.04,3\n"))
def test_frozen_original_track_reader_preserves_vitaldb_grid(monkeypatch, tmp_path, content):
    path = tmp_path / "track.csv.gz"
    path.write_bytes(gzip.compress(content.encode(), mtime=0))
    real_read_csv = pd.read_csv

    def use_local_source(source, *args, **kwargs):
        return real_read_csv(path if str(source).startswith("https://api.vitaldb.net/") else source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", use_local_source)
    monkeypatch.setattr(eeg_extraction, "_cache_dir", tmp_path, raising=False)
    monkeypatch.setattr(eeg_extraction, "_track_sources", {}, raising=False)
    expected = dataset.load_trk("track", interval=0.01)
    actual = eeg_extraction._cached_track("track", interval=0.01)
    np.testing.assert_array_equal(actual, expected)
    assert eeg_extraction._track_sources == {"track": file_sha256(path)}


def test_pivot_qualifies_complete_records_with_undefined_descriptors(monkeypatch, tmp_path):
    monkeypatch.setenv("AKI_STUDY_PROFILE", "eeg")
    channels = [*MANDATORY_WAVEFORMS, eeg_study.EEG_WAVEFORM]
    source, output = (tmp_path / name for name in ("long.csv", "wide.csv"))
    features = [(9, wave, 11.0 if wave != eeg_study.EEG_WAVEFORM else None) for wave in channels]
    features.extend((10, wave, 11.0) for wave in MANDATORY_WAVEFORMS)
    pd.DataFrame(features, columns=["caseid", "waveform", "example"]).to_csv(source, index=False)
    eeg_study.write_json(metadata_path_for(source), {
        "extraction_protocol_version": eeg_extraction.EEG_EXTRACTION_PROTOCOL_VERSION,
        "extraction_fingerprint": {"profile": "eeg"},
        "data_sha256": file_sha256(source),
        "requested_caseids": [9, 10],
    })

    process_waveform_file(source, output, "Full Features")
    meta = json.loads(metadata_path_for(output).read_text())
    assert meta["complete_caseids"] == [9]
    assert meta["extraction_diagnostics"]["missing_channels_by_caseid"] == {"10": [eeg_study.EEG_WAVEFORM]}
    assert pd.read_csv(output)["BIS_EEG1_WAV_example"].isna().all()


def test_attrition_first_exclusions_and_grouped_events(tmp_path):
    ledger = pd.DataFrame({
        "caseid": [1, 2, 3, 4, 5], "subjectid": [7, 7, 8, 9, 10],
        "eligible_any_aki": [1, 1, 0, 1, 1], "aki_label": [1, 1, None, 0, 1],
        "eligible_icu_admission": [1, 1, 1, 1, 1], "y_icu_admit": [1, 0, 1, 0, 1],
        "eligible_clinical": [True, True, True, True, False],
        "has_four_waveform_tracks": [True, True, True, True, False],
        "eeg_extraction_candidate": [True, True, True, True, False],
    })
    eeg_study.write_csv(tmp_path / eeg_study.SOURCE_COHORT_NAME, ledger)
    qc = pd.DataFrame({"caseid": [1, 2, 3, 4], "status": ["passed", "passed", "passed", "failed"],
                       "branch_status_full": ["success", "success", "success", "qc_failure"],
                       "branch_status_windowed": ["success", "success", "success", "qc_failure"],
                       "mask_counts_sqi_unknown": [0, 0, 4, 10], "mask_counts_raw_amplitude_high": [0, 0, 0, 5]})
    eeg_study.write_csv(tmp_path / eeg_study.QC_NAME, qc)

    eeg_study.write_attrition(tmp_path, {1, 2, 3}, branch_complete={"non_windowed": {1, 2, 3}, "windowed": {1, 2, 3}})
    result = json.loads((tmp_path / "eeg_attrition.json").read_text())
    main = result["scopes"]["main_shared"]["outcomes"]["any_aki"]
    assert [main["eligible_operations"], main["positive_operations"], main["positive_patients"]] == [2, 2, 1]
    assert result["scopes"]["icu_waveform_sensitivity"]["operations"] == 3
    first = pd.read_csv(tmp_path / "eeg_operation_attrition.csv").set_index("caseid")["first_exclusion"]
    assert first[4] == "eeg_source_or_qc" and first[5] == "clinical"
    assert result["overlapping_reasons"]["mask_counts_sqi_unknown"]["operations"] == 2
    assert result["overlapping_reasons"]["mask_counts_raw_amplitude_high"]["operations"] == 1


def test_isolation_rejects_protected_tree_and_linked_outputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    original = raw / "protected.txt"
    original.write_text("before")
    root = tmp_path / "eeg"
    root.mkdir()
    with pytest.raises(ValueError, match="protected path"):
        _require_isolation(tmp_path, {"processed": root}, raw)
    (root / "via-link").symlink_to(original)
    with pytest.raises(ValueError, match="writable path alias"):
        _require_isolation(root, {"processed": root}, raw)
    assert original.read_text() == "before"
