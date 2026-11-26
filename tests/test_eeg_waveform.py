"""Test EEG timing, screening and Catch24 behavior from synthetic packets."""

import base64
import json
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

from data_preparation import eeg


ORIGIN = 4_102_444_800.0


def _waveform(values, *, start=0., rate=128., unit="uV", fmt=1, gain=1., offset=0.):
    return SimpleNamespace(
        type=1, srate=rate, unit=unit, fmt=fmt, gain=gain, offset=offset,
        recs=[{"dt": ORIGIN + start, "val": np.asarray(values)}],
    )


def _sqi(times, values):
    return SimpleNamespace(
        type=2, unit="%", recs=[{"dt": ORIGIN + dt, "val": value} for dt, value in zip(times, values)]
    )


def _recording(monkeypatch, tmp_path, tracks, *, unreadable_primary=False):
    path = eeg.raw_recording_path(tmp_path, 1)
    path.parent.mkdir()
    path.touch()
    calls = []

    def read(source, *, track_names):
        assert source == str(path)
        calls.append(tuple(track_names))
        if unreadable_primary and eeg.EEG_WAVEFORM in track_names:
            raise OSError("primary packet decoding failed")
        return SimpleNamespace(dtstart=ORIGIN, trks={name: tracks[name] for name in track_names if name in tracks})

    monkeypatch.setattr(eeg.vitaldb, "VitalFile", read)
    return calls


def _signal(seconds=15):
    times = np.arange(seconds * eeg.NATIVE_RATE) / eeg.NATIVE_RATE
    return 30 * np.sin(2 * np.pi * 12 * times)


@pytest.mark.parametrize("preferred", ["absent", "empty", "unreadable"])
def test_fallback_selects_one_recording_for_both_branches(monkeypatch, tmp_path, preferred):
    tracks = {eeg.EEG_FALLBACK: _waveform(_signal()), eeg.SQI_TRACK: _sqi(range(15), [90] * 15)}
    if preferred == "empty":
        tracks[eeg.EEG_WAVEFORM] = _waveform([])
    calls = _recording(monkeypatch, tmp_path, tracks, unreadable_primary=preferred == "unreadable")
    result, _, qc = eeg.extract_eeg_case(1, 0., 15., tmp_path)
    assert qc["selected_source"] == eeg.EEG_FALLBACK
    assert qc["fallback_reason"] == preferred
    assert qc["branch_status"] == {"full": "success", "windowed": "success"}
    assert result["full"]["waveform"] == result["windowed"]["waveform"] == eeg.EEG_WAVEFORM
    assert len(result["full"]) == 24 + 2
    assert len(result["windowed"]) == 96 + 2
    assert len(calls) == (3 if preferred == "unreadable" else 1)
    json.dumps(qc, allow_nan=False)


def test_nonempty_preferred_remains_selected_when_qc_fails(monkeypatch, tmp_path):
    tracks = {
        eeg.EEG_WAVEFORM: _waveform(np.full(1920, 101.)),
        eeg.EEG_FALLBACK: _waveform(_signal()),
        eeg.SQI_TRACK: _sqi(range(15), [90] * 15),
    }
    _recording(monkeypatch, tmp_path, tracks)
    result, _, qc = eeg.extract_eeg_case(1, 0, 15, tmp_path)
    assert qc["selected_source"] == eeg.EEG_WAVEFORM
    assert qc["fallback_reason"] is None
    assert qc["track_status"][eeg.EEG_FALLBACK] == "present"
    assert qc["mask_counts"]["raw_amplitude_high"] == 1920
    assert {record["error"] for record in result.values()} == {"unusable_fraction_above_5_pct"}


def test_absence_and_missing_recording_are_distinct(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError):
        eeg.load_clean_eeg(2, 0, 15, tmp_path)
    _recording(monkeypatch, tmp_path, {})
    result, _, qc = eeg.extract_eeg_case(1, 0, 15, tmp_path)
    assert set(qc["track_status"].values()) == {"absent"}
    assert result["full"]["error"] == "no_eeg_source"


@pytest.mark.parametrize("unit,factor,fmt,gain,offset", [
    ("µV", 1., 1, 999., 500.), ("mV", .001, 1, 1., 0.), ("V", .000001, 1, 1., 0.),
    ("uV", 1., 3, 2., 5.),
])
def test_native_conversion_keeps_rates_and_packets(monkeypatch, tmp_path, unit, factor, fmt, gain, offset):
    physical = np.rint(_signal())
    encoded = physical * factor if fmt < 3 else np.rint((physical - offset) / gain).astype(np.int16)
    expected = physical if fmt < 3 else encoded * gain + offset
    waveform = _waveform(encoded, start=.25 / 128, unit=unit, fmt=fmt, gain=gain, offset=offset)
    _recording(monkeypatch, tmp_path, {eeg.EEG_WAVEFORM: waveform, eeg.SQI_TRACK: _sqi(range(15), [80] * 15)})
    result = eeg.load_clean_eeg(1, 0, 15, tmp_path)
    assert result.clean is not None
    assert result.qc["native_rate_hz"] == 128.
    assert result.qc["expected_samples"] == len(result.clean) == 1920
    np.testing.assert_allclose(result.raw, expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.observed_times - result.times, .25 / 128, rtol=0, atol=1e-13)


@pytest.mark.parametrize("metadata,reason", [
    ({"rate": 256.}, "unexpected_waveform_type_or_rate"),
    ({"unit": "unknown"}, "unexpected_waveform_unit"),
])
def test_metadata_discrepancy_fails_selected_track(monkeypatch, tmp_path, metadata, reason):
    tracks = {
        eeg.EEG_WAVEFORM: _waveform(_signal(), **metadata),
        eeg.EEG_FALLBACK: _waveform(_signal()),
        eeg.SQI_TRACK: _sqi(range(15), [90] * 15),
    }
    _recording(monkeypatch, tmp_path, tracks)
    result = eeg.load_clean_eeg(1, 0, 15, tmp_path)
    assert result.clean is None
    assert result.qc["selected_source"] == eeg.EEG_WAVEFORM
    assert result.qc["failure_reason"] == reason


def test_sqi_uses_source_timestamps_and_does_not_bridge_invalid_rows():
    track = _sqi([1, 0.5, 0.5, 2, 2, 4, 5, 6], [60, 50, 50, 75, 90, 101, np.inf, 85])
    timestamps = np.array([0, .5, .75, 1., 1.99, 2., 2.1, 3.75, 4, 4.5, 5, 6, 7, 7.01])
    aligned, audit = eeg._align_sqi(track, ORIGIN, timestamps)
    np.testing.assert_equal(aligned, [np.nan, 50, 50, 60, 60, np.nan, np.nan, np.nan,
                                      np.nan, np.nan, np.nan, 85, 85, np.nan])
    assert audit["sqi_duplicate_timestamps"] == 2
    assert audit["sqi_conflicting_timestamps"] == 1
    assert audit["sqi_nonmonotone_timestamps"] == 1
    assert audit["sqi_invalid_values"] == 2


def test_actual_waveform_sample_time_can_precede_grid_quality_change(monkeypatch, tmp_path):
    wave = _waveform(_signal(), start=-.25 / 128)
    _recording(monkeypatch, tmp_path, {
        eeg.EEG_WAVEFORM: wave, eeg.SQI_TRACK: _sqi([-.5, 0., *range(1, 15)], [49, *([90] * 15)]),
    })
    result = eeg.load_clean_eeg(1, 0, 15, tmp_path)
    assert result.clean is not None
    assert result.times[0] == 0
    assert result.observed_times[0] == -.25 / 128
    assert result.sqi[:2].tolist() == [49, 90]
    assert result.qc["mask_counts"]["sqi_low"] == 1


def test_packet_half_ties_and_conflicting_collisions():
    packets = [
        {"dt": ORIGIN + 3.5 / 128, "val": np.array([3., 4., 5.])},
        {"dt": ORIGIN + 4.5 / 128, "val": np.array([20.])},
        {"dt": ORIGIN + 5.5 / 128, "val": np.array([5.])},
    ]
    raw, times, audit = eeg._map_waveform(packets, ORIGIN, 2, 8, 1., 0., 1.)
    np.testing.assert_equal(raw, [np.nan, np.nan, 3., np.nan, 5., np.nan])
    assert times[2] == 3.5 / 128
    assert not np.isfinite(times[3])
    assert audit["waveform_collision_slots"] == 2
    assert audit["waveform_conflict_slots"] == 1


@pytest.mark.parametrize("add_masked_sample", [False, True])
def test_qc_union_boundary_and_linear_fill(monkeypatch, add_masked_sample):
    raw = np.linspace(0, 60, 100)
    raw[[0, 51, -1]] = np.nan
    raw[50] = 101
    raw[49] = 100
    raw[48] = -100
    observed = np.arange(100) / 128
    observed[[0, -1]] = np.nan
    sqi = np.full(100, 50.)
    sqi[[50, 51, 95]] = 49
    sqi[0] = np.nan
    if add_masked_sample:
        sqi[40] = 49

    masks, qc = eeg._mask_audit(raw, observed, sqi)
    assert qc["mask_counts"]["source_missing"] == 2
    assert qc["mask_counts"]["source_nonfinite"] == 1
    assert qc["mask_counts"]["raw_amplitude_high"] == 1
    assert qc["mask_counts"]["sqi_unknown"] == 1
    assert qc["mask_counts"]["unusable_union"] == (6 if add_masked_sample else 5)
    assert not masks["unusable_union"][48:50].any()
    monkeypatch.setattr(eeg.scipy.signal, "sosfiltfilt", lambda _, signal, **kwargs: signal)
    filtered, reason = eeg._clean_signal(raw, masks["unusable_union"], qc)
    if add_masked_sample:
        assert filtered is None and reason == "unusable_fraction_above_5_pct"
    else:
        assert reason is None
        assert filtered[0] == raw[1] and filtered[-1] == raw[-2]
        np.testing.assert_allclose(filtered[50:52], np.linspace(raw[49], raw[52], 4)[1:-1])
        assert qc["interpolated_fraction"] == .05
        assert qc["max_interior_gap_seconds"] == 2 / 128
        assert qc["trailing_filled_seconds"] == 1 / 128


def test_missing_packet_edges_count_against_expected_duration(monkeypatch, tmp_path):
    packets = _waveform(_signal()[40:-40], start=40 / 128)
    _recording(monkeypatch, tmp_path, {eeg.EEG_WAVEFORM: packets, eeg.SQI_TRACK: _sqi(range(15), [90] * 15)})
    result = eeg.load_clean_eeg(1, 0., 15., tmp_path)
    assert result.qc["observed_samples"] == 1840
    assert result.qc["expected_samples"] == 1920
    assert result.qc["mask_counts"]["source_missing"] == 80
    assert result.qc["leading_filled_seconds"] == result.qc["trailing_filled_seconds"] == 40 / 128


def test_filter_attains_passband_at_native_rate_and_retains_small_physiological_signal(monkeypatch, tmp_path):
    times = np.arange(20 * 128) / 128
    raw = .04 * sum(np.sin(2 * np.pi * hz * times) for hz in (0.1, 12, 59))
    _recording(monkeypatch, tmp_path, {
        eeg.EEG_WAVEFORM: _waveform(raw), eeg.SQI_TRACK: _sqi(range(20), [90] * 20),
    })
    result = eeg.load_clean_eeg(1, 0, 20, tmp_path)
    interior = (times >= 5) & (times < 15)
    assert result.clean is not None
    assert result.qc["filter_default_pad_samples"] == 27

    def amplitude(hz):
        return 2 * abs(np.mean(result.clean[interior] * np.exp(-2j * np.pi * hz * times[interior])))

    assert amplitude(12) == pytest.approx(.04, rel=.001)
    assert max(amplitude(.1), amplitude(59)) < .0002


def test_numerical_flatline_is_audited_and_not_extracted(monkeypatch, tmp_path):
    _recording(monkeypatch, tmp_path, {
        eeg.EEG_WAVEFORM: _waveform(np.full(1920, 1e-10)), eeg.SQI_TRACK: _sqi(range(15), [90] * 15),
    })
    result, _, qc = eeg.extract_eeg_case(1, 0, 15, tmp_path)
    assert result["full"]["error"] == "below_numerical_sd"
    assert qc["clean_sd_uv"] < 1e-6
    assert qc["clean_mean_uv"] == pytest.approx(0., abs=1e-12)


def test_window_stride_and_undefined_descriptor_propagation(monkeypatch, tmp_path):
    observed = []
    data = iter(([1., 2.], [1., np.inf], [3., 4.], [5., 6.]))

    def catch24(values, *, catch24):
        assert catch24
        observed.append(len(values))
        return {"names": ["known", "undefined"], "values": next(data)}

    monkeypatch.setattr(eeg.pycatch22, "catch22_all", catch24)
    raw = _signal(20)
    _recording(monkeypatch, tmp_path, {
        eeg.EEG_WAVEFORM: _waveform(raw), eeg.SQI_TRACK: _sqi(range(20), [90] * 20),
    })
    records, _, qc = eeg.extract_eeg_case(1, 0, 20, tmp_path)
    assert observed == [2560, 1280, 1280, 1280]
    windowed = records["windowed"]
    assert windowed["known_mean"] == 3
    assert windowed["known_std"] == 2
    assert all(np.isnan(windowed[f"undefined_{f}"]) for f in ("mean", "std", "min", "max"))
    assert qc["undefined_window_descriptor_counts"] == {"known": 0, "undefined": 1}


def test_one_qualifying_window_preserves_nan_sample_sd_and_masks(monkeypatch, tmp_path):
    raw = _signal(10)
    raw[200] = np.inf
    _recording(monkeypatch, tmp_path, {
        eeg.EEG_WAVEFORM: _waveform(raw), eeg.SQI_TRACK: _sqi(range(10), [90] * 10),
    })
    records, _, qc = eeg.extract_eeg_case(1, 0, 10, tmp_path)
    assert qc["total_windows"] == qc["retained_windows"] == 1
    assert np.isnan(records["windowed"]["DN_Mean_std"])
    mask_bytes = zlib.decompress(base64.b64decode(qc["masks"]["data"]["source_nonfinite"]))
    mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8), bitorder="little")
    assert np.flatnonzero(mask[:qc["expected_samples"]]).tolist() == [200]


def test_absent_quality_is_unknown_for_the_entire_operative_grid(monkeypatch, tmp_path):
    _recording(monkeypatch, tmp_path, {eeg.EEG_WAVEFORM: _waveform(_signal())})
    result = eeg.load_clean_eeg(1, 0, 15, tmp_path)
    assert result.clean is None
    assert result.qc["sqi_status"] == "absent"
    assert result.qc["valid_sqi_samples"] == 0
    assert result.qc["mask_counts"]["unusable_union"] == 1920


def test_filter_and_window_lengths_have_independent_lower_bounds(monkeypatch, tmp_path):
    _recording(monkeypatch, tmp_path, {
        eeg.EEG_WAVEFORM: _waveform(_signal()), eeg.SQI_TRACK: _sqi(range(15), [90] * 15),
    })
    too_short = eeg.load_clean_eeg(1, 0, 27 / 128, tmp_path)
    assert too_short.qc["failure_reason"] == "insufficient_filter_length"
    records, _, qc = eeg.extract_eeg_case(1, 0, 9, tmp_path)
    assert "error" not in records["full"]
    assert records["windowed"]["error"] == "insufficient_window_length"
    assert qc["total_windows"] == 0


def test_window_guard_excludes_degenerate_windows_without_excluding_descriptor_nans(monkeypatch):
    signal = np.r_[np.zeros(1280), _signal(5)]
    loaded = eeg.EEGCleanResult(signal, signal, signal, signal, signal, {}, {"status": "passed"})
    monkeypatch.setattr(eeg, "load_clean_eeg", lambda *args: loaded)
    records, _, qc = eeg.extract_eeg_case(1, 0, 15, "unused")
    assert qc["total_windows"] == 2
    assert qc["retained_windows"] == qc["failed_windows"] == 1
    assert qc["branch_status"]["windowed"] == "success"
    assert np.isnan(records["windowed"]["DN_Mean_std"])
