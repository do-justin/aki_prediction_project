"""The EEG copy of the manuscript retains its approved clinical response panels."""

import matplotlib.pyplot as plt
import pandas as pd

from reporting import eeg_manuscript_explanations as manuscript


def test_response_figure_uses_fixed_manuscript_terms_instead_of_importance_rank(monkeypatch):
    columns = ["outcome", "feature", "fold", "reference"]
    terms = [("any_aki", "preop_hb", 0, 13.1), ("icu_admission", "age", 0, 61.),
             ("icu_admission", "preop_egfr_ckdepi_2021", 0, 95.8)]
    support = pd.DataFrame(terms, columns=columns).assign(bin_lower=0., bin_upper=100., count=1)
    responses = pd.DataFrame(terms, columns=columns).assign(value=50., raw_log_odds_minus_reference=0.)
    generic = responses.assign(feature="BIS_EEG1_WAV_DN_Spread_std")

    def fake_explanation_sources(study, branch, *, manuscript_ranges):
        assert branch == "non_windowed"
        assert manuscript_ranges == {"any_aki": {"preop_hb": (.01, .99)},
                                     "icu_admission": {"age": (.05, .95), "preop_egfr_ckdepi_2021": (.05, .95)}}
        return {"responses": generic, "manuscript_responses": responses, "manuscript_support": support}

    monkeypatch.setattr(manuscript, "explanation_sources", fake_explanation_sources)
    sources = manuscript.manuscript_explanation_sources({}, "non_windowed")
    figure, plotted = manuscript.response_figure(sources)
    try:
        assert plotted.feature.tolist() == ["preop_hb", "age", "preop_egfr_ckdepi_2021"]
        assert "Age (years)" in figure.axes[3].get_xlabel()
        assert all(axis.get_ylabel() == "Training n" for axis in figure.axes[1::2])
    finally:
        plt.close(figure)
