"""Descriptive diagnostics for the frozen CBF validation and test protocol."""

from collections import Counter

import numpy as np


def paired_validation_intervals(grid, ndcg_by_setting, recall_by_setting, queries,
                                bootstrap_samples=2000, seed=42):
    """Paired resampling by validation disease, not by individual indication edge."""
    best, second = grid[0], grid[1]
    matching_raw = next(config for config in grid if
                        config['feature_mode'] == 'raw' and
                        config['k'] == best['k'] and config['p'] == best['p'])
    rng = np.random.default_rng(seed)
    comparisons = []
    for label, challenger in [('runner_up', second), ('matched_raw_vs_idf', matching_raw)]:
        key_best = (best['feature_mode'], best['k'], best['p'])
        key_challenger = (challenger['feature_mode'], challenger['k'], challenger['p'])
        delta = np.asarray(ndcg_by_setting[key_best]) - np.asarray(ndcg_by_setting[key_challenger])
        recall_delta = (np.asarray(recall_by_setting[key_best]) -
                        np.asarray(recall_by_setting[key_challenger]))
        sample = rng.integers(0, len(queries), size=(bootstrap_samples, len(queries)))
        sampled_mean = delta[sample].mean(axis=1)
        low, high = np.percentile(sampled_mean, [2.5, 97.5])
        comparisons.append({
            'comparison': label,
            'selected': {'feature_mode': best['feature_mode'], 'k': best['k'], 'p': best['p']},
            'challenger': {'feature_mode': challenger['feature_mode'],
                           'k': challenger['k'], 'p': challenger['p']},
            'n_paired_validation_diseases': len(queries),
            'mean_delta_ndcg@10': float(delta.mean()),
            'mean_delta_recall@10': float(recall_delta.mean()),
            'ndcg@10_paired_bootstrap_ci95': [float(low), float(high)],
            'positive_difference_diseases': int((delta > 0).sum()),
            'negative_difference_diseases': int((delta < 0).sum()),
            'equal_difference_diseases': int((delta == 0).sum()),
        })
    return {
        'method': 'paired percentile bootstrap resampling validation diseases',
        'seed': seed,
        'bootstrap_samples': bootstrap_samples,
        'significance_note': (
            'Exploratory intervals only: configurations were selected on these same '
            'Validation diseases, and multiple comparisons were explored. '
            'A CI excluding zero is not a post-selection significance guarantee.'),
        'comparisons': comparisons,
    }


def failure_summary(queries, n_recs, gene_available, similarities, fit, diseases,
                    disease_index, neighbor_fn):
    """Categorize unrankable test queries without reading Test truth for model decisions."""
    reason_counts = Counter()
    for disease in queries:
        if int(n_recs[disease]) > 0:
            reason_counts['recommended'] += 1
        elif not bool(gene_available[disease_index[disease]]):
            reason_counts['no_gene_features'] += 1
        elif not neighbor_fn(disease, similarities, fit, diseases, disease_index):
            reason_counts['no_positive_fitting_neighbors'] += 1
        else:
            reason_counts['all_neighbor_drugs_excluded'] += 1
    return dict(sorted(reason_counts.items()))
