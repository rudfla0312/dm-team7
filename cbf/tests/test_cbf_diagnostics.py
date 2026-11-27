"""Deterministic paired-resampling and failure-reason checks (outside notebook)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbf_diagnostics import paired_validation_intervals, failure_summary


def test_paired_bootstrap_repeated():
    grid = [
        {'feature_mode': 'idf', 'k': 50, 'p': 2},
        {'feature_mode': 'idf', 'k': 20, 'p': 1},
        {'feature_mode': 'raw', 'k': 50, 'p': 2},
    ]
    ndcg = {('idf', 50, 2): [.8, .4, .2, 0],
            ('idf', 20, 1): [.7, .3, .2, .1],
            ('raw', 50, 2): [.75, .4, .1, .0]}
    recall = {key: values for key, values in ndcg.items()}
    first = paired_validation_intervals(grid, ndcg, recall, ['a', 'b', 'c', 'd'], 200, 42)
    again = paired_validation_intervals(grid, ndcg, recall, ['a', 'b', 'c', 'd'], 200, 42)
    assert first == again
    assert len(first['comparisons']) == 2
    assert np.isclose(first['comparisons'][0]['mean_delta_ndcg@10'], .025)


def test_failure_reason_buckets():
    diseases = ['a', 'b', 'c', 'd']
    similarity = np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)

    def no_neighbors(*args):
        return []

    counts = failure_summary(diseases, {'a': 1, 'b': 0, 'c': 0, 'd': 0},
                             [True, False, True, True], similarity, None,
                             diseases, {d: i for i, d in enumerate(diseases)}, no_neighbors)
    assert counts == {'no_gene_features': 1,
                      'no_positive_fitting_neighbors': 2, 'recommended': 1}


if __name__ == '__main__':
    test_paired_bootstrap_repeated()
    test_failure_reason_buckets()
    print('Paired bootstrap and failure-reason tests: PASS')
