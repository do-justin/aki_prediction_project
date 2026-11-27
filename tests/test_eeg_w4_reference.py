import pytest

from data_preparation import eeg_extraction as extraction
from data_preparation.artifact_metadata import ArtifactCompatibilityError, file_sha256, metadata_path_for


def test_w4_reference_requires_disjoint_amplitude_study(tmp_path, monkeypatch):
    monkeypatch.setenv("AKI_EEG_W4_REFERENCE", str(tmp_path))
    monkeypatch.delenv("AKI_EEG_QC_PROTOCOL", raising=False)
    with pytest.raises(ValueError, match="amplitude-only"):
        extraction._legacy_reference()

    monkeypatch.setenv("AKI_EEG_QC_PROTOCOL", extraction.AMPLITUDE_PROTOCOL)
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(tmp_path / "nested"))
    with pytest.raises(ValueError, match="disjoint"):
        extraction._legacy_reference()


def test_verified_w4_reuse_recomputes_eeg_and_rejects_changed_legacy_result(tmp_path, monkeypatch):
    study = tmp_path / "strict"
    reference = study / "data/processed"
    checkpoints = reference / "eeg_extraction_checkpoints"
    checkpoints.mkdir(parents=True)
    raw = tmp_path / "raw"
    native = extraction.raw_recording_path(raw, 1)
    native.parent.mkdir(parents=True)
    native.write_text("native recording")
    release = {native.relative_to(raw).as_posix(): file_sha256(native)}
    monkeypatch.setattr(extraction, "RAW_DIR", raw)
    monkeypatch.setattr(extraction, "descriptor_names", lambda: ["descriptor"])
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(tmp_path / "new"))
    monkeypatch.setenv("AKI_EEG_W4_REFERENCE", str(study))
    monkeypatch.setenv("AKI_EEG_QC_PROTOCOL", extraction.AMPLITUDE_PROTOCOL)

    saved_fingerprints = {
        mode: {
            "mode": mode,
            "eeg_code_sha256": {"eeg_extraction.py": extraction.STRICT_TRACK_READER_SHA256},
            "eeg_protocol": {"protocol": "eeg-qc-v1"},
            "track_catalog_sha256": "original catalog",
        }
        for mode in extraction.MODES
    }
    extraction.write_json(checkpoints / "fingerprint.json", saved_fingerprints)
    sources = {"one": "checksum"}
    track_inventory = reference / "eeg_track_source_hashes.json"
    extraction.write_json(track_inventory, sources)
    old_results = {
        channel: {
            mode: {"caseid": 1, "waveform": channel,
                   **({"descriptor": 1} if mode == "full" else {"descriptor_mean": 2})}
            for mode in extraction.MODES
        }
        for channel in extraction.CHANNELS
    }
    saved = {"caseid": 1, "eeg_recording_sha256": file_sha256(native), "sources": sources,
             "qc": {"caseid": 1}, "results": old_results,
             "timings": {channel: {} for channel in extraction.MANDATORY_WAVEFORMS}}
    checkpoint = checkpoints / "1.json"
    extraction.write_json(checkpoint, saved)
    for mode, (features, errors) in extraction._branch_output_paths(reference).items():
        rows = [old_results[channel][mode] for channel in extraction.CHANNELS]
        feature_frame, error_frame = extraction._branch_frames(mode, rows, [], ["descriptor"])
        extraction.write_csv(features, feature_frame)
        extraction.write_csv(errors, error_frame)
        extraction.write_json(metadata_path_for(features), {
            "requested_caseids": [1], "extraction_fingerprint": saved_fingerprints[mode],
            "source_track_hashes_sha256": file_sha256(track_inventory), "data_sha256": file_sha256(features),
        })
    resolved, attestation = extraction._legacy_reference()
    assert resolved == reference
    fingerprints = {
        mode: dict(prior, eeg_protocol={"protocol": extraction.AMPLITUDE_PROTOCOL},
                   original_four_reference=attestation)
        for mode, prior in saved_fingerprints.items()
    }
    checked = extraction._verify_legacy_reference(reference, [1], fingerprints, release)
    monkeypatch.setattr(extraction, "_w4_reference", reference)
    monkeypatch.setattr(extraction, "_w4_checkpoint_hashes", checked, raising=False)
    new_eeg = {mode: {"caseid": 1, "waveform": extraction.EEG_WAVEFORM, "descriptor": 42}
               for mode in extraction.MODES}
    monkeypatch.setattr(extraction, "extract_eeg_case", lambda *args: (new_eeg, {}, {"caseid": 1}))
    actual = extraction._extract_operation((1, 0, 10, file_sha256(native)))
    assert actual["results"][extraction.EEG_WAVEFORM] == new_eeg
    assert all(actual["results"][channel] == old_results[channel] for channel in extraction.MANDATORY_WAVEFORMS)
    assert actual["sources"] == sources

    saved["results"][extraction.MANDATORY_WAVEFORMS[0]]["full"]["descriptor"] = 99
    extraction.write_json(checkpoint, saved)
    with pytest.raises(ArtifactCompatibilityError, match="checkpoint changed"):
        extraction._extract_operation((1, 0, 10, file_sha256(native)))
    with pytest.raises(ArtifactCompatibilityError, match="do not reconstruct"):
        extraction._verify_legacy_reference(reference, [1], fingerprints, release)
