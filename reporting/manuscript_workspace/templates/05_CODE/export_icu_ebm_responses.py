"""Export the approved ICU response panels from the frozen combined EBM."""

import argparse
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.artifact_metadata import file_sha256
from model_creation.validation import is_outer_fold_checkpoint_complete


OUTCOME = 'icu_admission'
BRANCH = 'non_windowed'
FEATURE_SET = 'preop_and_all_waveforms'
REVISION = 'fcff7c7c9794e0ea8405814c68176760a2eb9afc'
PANELS = {
    'figure5c': {
        'feature': 'preop_egfr_ckdepi_2021',
        'definition': 'Preoperative creatinine-based eGFR (2021 CKD-EPI equation).',
        'unit': 'mL/min/1.73 m²',
    },
    'figure_s10c': {
        'feature': 'Primus_CO2_DN_Mean',
        'definition': 'Arithmetic mean of all samples in the filtered, resampled 10 Hz whole-operation capnogram.',
        'unit': 'nominal mmHg in the VitalDB source waveform',
        'unit_source': 'https://vitaldb.net/dataset/?query=overview',
    },
}


def load_training_folds(study, raw, dataset_hash):
    fold_root = (study / 'results/catch22/experiments/main_shared/nested_cv/models/ebm'
                 / OUTCOME / BRANCH / FEATURE_SET / 'artifacts/folds/repeat_00')
    folds = []
    for fold_index in range(5):
        fold = fold_root / f'outer_{fold_index:02d}'
        assert is_outer_fold_checkpoint_complete(fold), fold
        metadata_path = fold / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        fingerprint = metadata['validation_fingerprint']
        assert fingerprint['dataset_hash'] == dataset_hash
        fingerprint_keys = ('code_revision', 'model_type', 'outcome', 'branch',
                            'feature_set', 'cohort_scope')
        assert tuple(fingerprint[k] for k in fingerprint_keys) == (
            REVISION, 'ebm', OUTCOME, BRANCH, FEATURE_SET, 'main_shared'
        )
        assert (metadata['repeat_id'], metadata['outer_fold_id']) == (0, fold_index)
        with (fold / 'model.pkl').open('rb') as stream:
            model = pickle.load(stream)
        with (fold / 'preprocessor.pkl').open('rb') as stream:
            preprocessor = pickle.load(stream)
        train = pd.read_csv(fold / 'train_oof.csv', usecols=['caseid'])
        test = pd.read_csv(fold / 'test_predictions.csv', usecols=['caseid'])
        assert train.caseid.is_unique and set(train.caseid).isdisjoint(test.caseid)
        columns = list(preprocessor.numeric_columns_) + list(preprocessor.categorical_columns_)
        transformed = preprocessor.transform(raw.loc[train.caseid, columns])
        fold_provenance = {
            'fold': fold.name,
            'metadata_sha256': file_sha256(metadata_path),
            'model_sha256': metadata['model_sha256'],
            'preprocessor_sha256': metadata['preprocessor_sha256'],
        }
        folds.append((model, transformed, fold_provenance))
    return folds


def export_term(panel, folds, importance):
    feature = panel['feature']
    observed = [trained[feature].to_numpy(dtype=float) for _, trained, _ in folds]
    finite = [values[np.isfinite(values)] for values in observed]
    quantiles = [np.quantile(values, [0.05, 0.50, 0.95]) for values in finite]
    lower, reference, upper = map(float, np.median(quantiles, axis=0))
    assert lower < reference < upper
    support, curve, fold_checks = [], [], []

    for fold_index, (model, training, fold_provenance) in enumerate(folds):
        term = model.term_names_.index(feature)
        feature_index, = model.term_features_[term]
        cuts = np.asarray(model.bins_[feature_index][0], dtype=float)
        scores = model.term_scores_[term]
        values = np.unique(np.r_[lower, cuts[(cuts > lower) & (cuts < upper)], reference, upper])
        probes = np.zeros((len(values), model.n_features_in_))
        probes[:, feature_index] = values
        actual = model.eval_terms(probes)[:, term]
        replayed = scores[np.searchsorted(cuts, values, side='right') + 1]
        np.testing.assert_array_equal(actual, replayed)
        reference_score = actual[values == reference][0]
        relative = actual - reference_score

        common = {'outcome': OUTCOME, 'feature': feature, 'fold': fold_index}
        curve.extend(common | {'feature_set': FEATURE_SET, 'reference': reference,
                                'value': float(value), 'raw_contribution_minus_reference': float(score)}
                     for value, score in zip(values, relative, strict=True))
        if fold_index == 0:
            counts, edges = np.histogram(finite[0], bins=np.linspace(lower, upper, 26))
            assert counts.sum() == np.sum((finite[0] >= lower) & (finite[0] <= upper))
            support.extend(common | {'bin_lower': float(left), 'bin_upper': float(right),
                                     'count': int(count), 'observed_training_n': len(finite[0])}
                           for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True))

        fold_checks.append(fold_provenance | {
            'train_n': len(training), 'finite_training_n': len(finite[fold_index]),
            'missing_training_n': len(training) - len(finite[fold_index]),
            'training_quantiles_q05_q50_q95': quantiles[fold_index].tolist(),
            'missing_bin_raw_score_minus_reference': float(scores[0] - reference_score),
            'lower_raw_score_minus_reference': float(relative[0]),
            'upper_raw_score_minus_reference': float(relative[-1]),
            'native_probe_discrepancy': float(np.max(np.abs(actual - replayed))),
        })

    provenance = panel | {
        'outcome': OUTCOME, 'branch': BRANCH, 'feature_set': FEATURE_SET,
        'reference_value': reference, 'reference_rule': 'Median of five outer-training medians',
        'range': [lower, upper], 'range_rule': 'Medians of five outer-training 5th and 95th percentiles',
        'mean_native_importance': float(importance.loc[feature]),
        'importance_rank_all_terms': int(importance.index.get_loc(feature)) + 1,
        'training_support_fold': 'outer_00',
        'plotted_training_count': sum(record['count'] for record in support),
        'folds': fold_checks,
    }
    return {'provenance': provenance, 'curve': curve, 'training_support': support}


def export(study, workspace, output):
    processed = study / 'data/processed/aki_features_master_wide.csv'
    identity = study / 'control/study-identity.json'
    preoperative_source = study / 'code/data_preparation/step_03_preop_prep.py'
    assert json.loads(identity.read_text())['code_revision'] == REVISION
    sources = [processed, identity, preoperative_source]
    panel_keys = ['figure5c']
    if workspace is not None:
        source_archive = workspace / '06_SOURCE_DOCUMENTS/waveform_descriptor_definitions/pycatch22-0.4.5.tar.gz'
        with tarfile.open(source_archive) as release:
            source = 'pycatch22-0.4.5/src/C/'
            mean = release.extractfile(source + 'DN_Mean.c').read().decode()
            wrapper = release.extractfile(source + 'catch22_wrap.c').read().decode()
            assert 'm /= size;' in mean and 'm += a[i];' in mean
            assert 'python_wrapper_double(args, &DN_Mean, 0)' in wrapper
        sources.append(source_archive)
        panel_keys.append('figure_s10c')
    source_hashes = {str(path): file_sha256(path) for path in sources}

    raw = pd.read_csv(processed, low_memory=False).set_index('caseid')
    folds = load_training_folds(study, raw, source_hashes[str(processed)])
    fold_importance = [pd.Series(model.term_importances(), index=model.term_names_)
                       for model, _, _ in folds]
    importance = pd.concat(fold_importance, axis=1).mean(axis=1).sort_values(ascending=False)
    provenance = {
        'cohort': 'main_shared', 'study': study.name, 'code_revision': REVISION,
        'repeat': 'repeat_00', 'source_sha256': source_hashes, 'models_fitted': 0,
    }
    panels = {key: export_term(PANELS[key], folds, importance) for key in panel_keys}
    assert all(file_sha256(Path(path)) == digest for path, digest in source_hashes.items())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'provenance': provenance, 'panels': panels}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, help='Also export the retired capnogram panel from a complete workspace')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.study, args.workspace, args.output)
