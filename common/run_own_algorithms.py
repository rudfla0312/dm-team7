"""Run Louvain and normalized SVD using the fixed team split and evaluator."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix

HERE = Path(__file__).resolve().parent
from louvain_helpers import (build_graph, communities_for, feature_similarities,
                             group_weights, sorted_similarity_rows)


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def matrix(rows, diseases):
    dix = {d: i for i, d in enumerate(diseases)}
    drugs = sorted({r['drug_id'] for r in rows})
    tix = {d: i for i, d in enumerate(drugs)}
    rr = [dix[r['disease_id']] for r in rows]
    cc = [tix[r['drug_id']] for r in rows]
    return coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(diseases), len(drugs))).tocsr(), drugs


def write_scores(path, S, query_rows, diseases, drugs, fit, conflicts, positive_only):
    # Mask BEFORE retaining top 100; zeros do not count as native Louvain recommendations.
    forbidden = defaultdict(set)
    for r in fit + conflicts:
        forbidden[r['disease_id']].add(r['drug_id'])
    tix = {d: i for i, d in enumerate(drugs)}
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['disease_id', 'drug_id', 'score'])
        for i in query_rows:
            d = diseases[i]
            row = np.asarray(S[i], dtype=float).copy()
            for drug in forbidden[d]:
                j = tix.get(drug)
                if j is not None:
                    row[j] = -np.inf
            valid = np.isfinite(row)
            if positive_only:
                valid &= row > 0
            ids = np.flatnonzero(valid)
            # Stable score ties use drug ID: drugs are in sorted order.
            ids = ids[np.lexsort((ids, -row[ids]))[:100]]
            writer.writerows((d, drugs[j], format(row[j], '.17g')) for j in ids)


def evaluate(stage, scores, out, split, conflicts):
    subprocess.run([sys.executable, str(HERE / 'evaluate.py'), '--split', str(split),
                    '--conflicts', str(conflicts), '--scores', str(scores), '--stage', stage,
                    '--output-dir', str(out)], check=True,
                   stdout=subprocess.DEVNULL)
    with (out / 'summary.json').open(encoding='utf-8') as f:
        return json.load(f)


def run_svd(fit, diseases, queries, conflicts, stage, out, split_path, conflict_path, ranks):
    start = time.perf_counter()
    R, drugs = matrix(fit, diseases)
    dr = np.maximum(np.asarray(R.sum(axis=1)).ravel(), 1.)
    dc = np.maximum(np.asarray(R.sum(axis=0)).ravel(), 1.)
    X = csr_matrix(R.multiply(dr[:, None]**-.5).multiply(dc[None, :]**-.3))
    gram = (X @ X.T).toarray()
    _, u = eigh(gram, subset_by_index=[len(diseases)-max(ranks), len(diseases)-1], check_finite=False)
    u = u[:, ::-1]
    z = np.asarray(X.T @ u)
    results = []
    for k in ranks:
        model_start = time.perf_counter()
        scores = (u[:, :k] @ z[:, :k].T) * dc[None, :]**.3
        score_path = out / f'svd_{stage}_rank{k}_scores.csv'
        write_scores(score_path, scores, queries, diseases, drugs, fit, conflicts, False)
        metrics = evaluate(stage, score_path, out / f'svd_{stage}_rank{k}', split_path, conflict_path)
        results.append((k, metrics))
        print(f'SVD {stage} rank={k}: Recall@10={metrics["recall@10"]:.4f}', flush=True)
    return results, time.perf_counter()-start


def run_louvain(fit, diseases, queries, conflicts, stage, out, split_path, conflict_path, configs, sims):
    start = time.perf_counter()
    R, drugs = matrix(fit, diseases)
    result = []
    for mode, k, res, power in configs:
        model_start = time.perf_counter()
        sim = sims[mode]
        rows = sorted_similarity_rows(diseases, sim)
        graph, _ = build_graph(diseases, rows, k)
        communities = communities_for(graph, res, 42)
        W = group_weights(sim, communities, diseases, power)
        S = (W @ R).toarray()
        name = f'{mode}_k{k}_r{res}_p{power}'
        score_path = out / f'louvain_{stage}_{name}_scores.csv'
        write_scores(score_path, S, queries, diseases, drugs, fit, conflicts, True)
        metrics = evaluate(stage, score_path, out / f'louvain_{stage}_{name}', split_path, conflict_path)
        result.append(((mode, k, res, power), metrics))
        print(f'Louvain {stage} {name}: Recall@10={metrics["recall@10"]:.4f}, groups={len(communities)}', flush=True)
    return result, time.perf_counter()-start


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--processed', type=Path, required=True, help='contains disease_gene_t03.csv')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    sp = args.split_dir / 'team_split.csv'
    cp = args.split_dir / 'contraindication_conflicts.csv'
    rows, conflicts = read(sp), read(cp)
    diseases = sorted({r['disease_id'] for r in rows})
    idx = {d:i for i,d in enumerate(diseases)}
    train = [r for r in rows if r['split']=='train']
    trainval = [r for r in rows if r['split'] in ('train', 'validation')]
    val_queries = sorted({idx[r['disease_id']] for r in rows if r['split']=='validation'})
    test_queries = sorted({idx[r['disease_id']] for r in rows if r['split']=='test'})
    ranks = [50, 100, 200, 300, 400]
    configs = [(mode, 30, 1.0, power) for mode in ('score_cosine', 'idf_cosine') for power in (1, 2)]
    sv, _ = run_svd(train, diseases, val_queries, conflicts, 'validation', args.output, sp, cp, ranks)
    sims = feature_similarities(args.processed, diseases)
    lv, _ = run_louvain(train, diseases, val_queries, conflicts, 'validation', args.output, sp, cp, configs, sims)
    choose = lambda results: max(results, key=lambda x: (x[1]['recall@10'], x[1]['ndcg@10']))
    best_svd = choose(sv)[0]
    best_louvain = choose(lv)[0]
    print(f'Validation selection: SVD rank={best_svd}; Louvain={best_louvain}', flush=True)
    st, _ = run_svd(trainval, diseases, test_queries, conflicts, 'test', args.output, sp, cp, [best_svd])
    lt, _ = run_louvain(trainval, diseases, test_queries, conflicts, 'test', args.output, sp, cp, [best_louvain], sims)
    with (args.output / 'comparison.json').open('w', encoding='utf-8') as f:
        json.dump({'svd_validation': sv, 'louvain_validation': lv, 'svd_test': st, 'louvain_test': lt}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
