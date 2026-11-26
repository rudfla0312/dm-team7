"""Measure the effect of removing only the Louvain community boundary."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from run_own_algorithms import evaluate, matrix, read, write_scores
from louvain_helpers import (build_graph, communities_for, feature_similarities,
                             group_weights, sorted_similarity_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    split = args.split_dir / 'team_split.csv'
    conflict_path = args.split_dir / 'contraindication_conflicts.csv'
    records, conflicts = read(split), read(conflict_path)
    fit = [r for r in records if r['split'] in {'train', 'validation'}]
    diseases = sorted({r['disease_id'] for r in fit})
    test = {r['disease_id'] for r in records if r['split'] == 'test'}
    queries = [i for i, disease in enumerate(diseases) if disease in test]
    sim = feature_similarities(args.processed, diseases)['idf_cosine']
    neighbors = sorted_similarity_rows(diseases, sim)
    graph, _ = build_graph(diseases, neighbors, 30)
    communities = communities_for(graph, 1.0, 42)
    group = {d: i for i, community in enumerate(communities) for d in community}
    retention = {}
    for k in (10, 30, 50):
        counts = [(sum(group[other] == group[diseases[i]] for other, _ in neighbors[i][:k]),
                   min(k, len(neighbors[i]))) for i in queries if neighbors[i]]
        retention[k] = {'queries': len(counts), 'retained': sum(a for a, _ in counts),
                        'available': sum(b for _, b in counts),
                        'macro': float(np.mean([a / b for a, b in counts]))}

    indications, drugs = matrix(fit, diseases)
    weights_by_variant = {
        'within_community': group_weights(sim, communities, diseases, 2),
        'all_positive_neighbors': csr_matrix(np.square(sim)),
    }
    scores = {}
    for name, weights in weights_by_variant.items():
        scores_path = args.output / f'{name}_scores.csv'
        write_scores(scores_path, (weights @ indications).toarray(), queries,
                     diseases, drugs, fit, conflicts, True)
        scores[name] = evaluate('test', scores_path, args.output / name, split, conflict_path)

    with (args.processed / 'disease_gene_t03.csv').open(encoding='utf-8-sig', newline='') as f:
        genes = Counter(r['disease_id'] for r in csv.DictReader(f)
                        if float(r['evidence_score']) >= .3)
    subgroup = []
    for name in scores:
        rows = read(args.output / name / 'per_disease.csv')
        for low, high in ((0, 0), (1, 10), (11, float('inf'))):
            bucket = [r for r in rows if low <= genes[r['disease_id']] <= high]
            subgroup.append({'variant': name, 'genes': f'{low}-{high}', 'n': len(bucket),
                             'with_recommendations': sum(int(r['n_recommended_up_to_100']) > 0 for r in bucket),
                             'recall@10': float(np.mean([float(r['recall@10']) for r in bucket])) if bucket else None})

    report = {'communities': len(communities), 'retention': retention,
              'metrics': scores, 'by_gene_count': subgroup}
    (args.output / 'boundary_comparison.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
