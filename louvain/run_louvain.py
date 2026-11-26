"""Run Louvain using the fixed team split and evaluator."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
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
    subprocess.run([sys.executable, str(HERE.parent / 'common' / 'evaluate.py'), '--split', str(split),
                    '--conflicts', str(conflicts), '--scores', str(scores), '--stage', stage,
                    '--output-dir', str(out)], check=True,
                   stdout=subprocess.DEVNULL)
    with (out / 'summary.json').open(encoding='utf-8') as f:
        return json.load(f)



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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True,
                        help="contains disease_gene_t03.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    split_path = args.split_dir / "team_split.csv"
    conflict_path = args.split_dir / "contraindication_conflicts.csv"
    rows, conflicts = read(split_path), read(conflict_path)
    diseases = sorted({row["disease_id"] for row in rows})
    index = {disease: i for i, disease in enumerate(diseases)}
    train = [row for row in rows if row["split"] == "train"]
    trainval = [row for row in rows if row["split"] in ("train", "validation")]
    validation_queries = sorted({index[row["disease_id"]] for row in rows
                                 if row["split"] == "validation"})
    test_queries = sorted({index[row["disease_id"]] for row in rows
                           if row["split"] == "test"})

    configurations = [(mode, 30, 1.0, power)
                      for mode in ("score_cosine", "idf_cosine")
                      for power in (1, 2)]
    similarities = feature_similarities(args.processed, diseases)
    validation, _ = run_louvain(
        train, diseases, validation_queries, conflicts, "validation",
        args.output, split_path, conflict_path, configurations, similarities)
    best_configuration = max(validation, key=lambda item: (
        item[1]["recall@10"], item[1]["ndcg@10"]))[0]
    print(f"Validation selection: Louvain={best_configuration}", flush=True)
    test, _ = run_louvain(
        trainval, diseases, test_queries, conflicts, "test",
        args.output, split_path, conflict_path, [best_configuration], similarities)
    with (args.output / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump({"louvain_validation": validation, "louvain_test": test},
                  stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
