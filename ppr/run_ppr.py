"""Select PPR on common validation scores, refit on train+validation, and evaluate test."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
COMMON = ROOT.parent / "common"
ALPHAS = [.5, .65, .75, .8, .85, .9, .95]
BUDGETS = [.5, .75, .9, .95]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(fit, genes, targets):
    edges = [(("disease", d), ("drug", m)) for d, m in sorted(fit)]
    edges.extend((("disease", r["disease_id"]), ("gene", r["gene_id"])) for r in genes)
    edges.extend((("drug", r["drug_id"]), ("gene", r["gene_id"])) for r in targets)
    edges = sorted(set(edges))
    nodes = sorted({n for edge in edges for n in edge})
    index = {node: i for i, node in enumerate(nodes)}
    a, b = np.array([(index[a], index[b]) for a, b in edges]).T
    adjacency = sparse.csr_matrix((np.ones(2 * len(edges)), (np.r_[a, b], np.r_[b, a])),
                                  shape=(len(nodes), len(nodes)))
    transition = (sparse.diags(1 / np.asarray(adjacency.sum(axis=1)).ravel()) @ adjacency).tocsr()
    return transition, index, np.array([kind for kind, _ in nodes])


def balance(matrix, types, budget):
    result = matrix.copy()
    for i in np.flatnonzero(types == "disease"):
        start, end = result.indptr[i:i+2]
        neighbors = types[result.indices[start:end]]
        drug, gene = neighbors == "drug", neighbors == "gene"
        if drug.any() and gene.any():
            values = result.data[start:end]
            values[drug] = budget / drug.sum()
            values[gene] = (1 - budget) / gene.sum()
    assert np.allclose(np.asarray(result.sum(axis=1)).ravel(), 1)
    return result


def score_ppr(matrix, index, catalog, diseases, excluded, alpha, workers):
    transpose = matrix.T.tocsr()
    catalog_index = np.array([index["drug", drug] for drug in catalog])

    def solve(batch):
        starts = np.array([index["disease", d] for d in batch])
        columns = np.arange(len(batch))
        x = np.zeros((len(index), len(batch)))
        x[starts, columns] = 1
        for _ in range(5000):
            y = alpha * (transpose @ x)
            y[starts, columns] += 1 - alpha
            if np.max(np.abs(y-x).sum(axis=0)) < 1e-10:
                break
            x = y
        else:
            raise RuntimeError("PPR did not converge")
        residual = alpha * (transpose @ y)
        residual[starts, columns] += 1 - alpha
        assert np.max(np.abs(residual-y).sum(axis=0)) < 1e-10
        assert np.allclose(y.sum(axis=0), 1)
        rows = []
        for j, disease in enumerate(batch):
            scores = y[catalog_index, j]
            count = 0
            for i in np.argsort(-scores, kind="stable"):
                drug = catalog[i]
                if scores[i] <= 0:
                    break
                if drug not in excluded[disease]:
                    rows.append({"disease_id": disease, "drug_id": drug, "score": float(scores[i])})
                    count += 1
                    if count == 100:
                        break
        return rows

    batches = [diseases[i:i+24] for i in range(0, len(diseases), 24)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for batch in pool.map(solve, batches) for r in batch]


def evaluate(stage, label, rows, runtime, output, data_dir):
    directory = output / stage / label
    directory.mkdir(parents=True)
    path = directory / "scores.csv"
    # Empty scores are valid for methods that cannot recommend.
    if rows:
        write(path, rows)
    else:
        path.write_text("disease_id,drug_id,score\n")
    subprocess.run([sys.executable, str(COMMON / "evaluate.py"),
                    "--split", str(data_dir / "splits/team_split.csv"),
                    "--conflicts", str(data_dir / "splits/contraindication_conflicts.csv"),
                    "--scores", str(path), "--stage", stage,
                    "--runtime-seconds", str(runtime),
                    "--output-dir", str(directory / "evaluation")],
                   check=True, capture_output=True, text=True)
    summary = json.loads((directory / "evaluation/summary.json").read_text())
    print(stage, label, "Recall@10", round(summary["recall@10"], 6),
          "NDCG@10", round(summary["ndcg@10"], 6), flush=True)
    return {"setting": label, **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="prepared inputs containing splits/, features/ and verified_hashes.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "run_results",
                        help="new output directory; existing directories are not overwritten")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    data_dir = args.data_dir.resolve()
    expected_hashes = json.loads((data_dir / "verified_hashes.json").read_text())
    paths = {
        "team_split.csv": data_dir / "splits/team_split.csv",
        "contraindication_conflicts.csv": data_dir / "splits/contraindication_conflicts.csv",
        "disease_gene.csv": data_dir / "features/disease_gene.csv",
        "drug_gene.csv": data_dir / "features/drug_gene.csv",
    }
    for name, path in paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_hashes[name]:
            raise ValueError(f"Input hash mismatch: {path}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    split = read(data_dir / "splits/team_split.csv")
    genes = read(data_dir / "features/disease_gene.csv")
    targets = read(data_dir / "features/drug_gene.csv")
    conflicts = read(data_dir / "splits/contraindication_conflicts.csv")
    # Lock the search protocol before inspecting test scores.
    protocol = {"alphas": ALPHAS, "budgets": BUDGETS,
                "search": "select basic alpha first; select weighted budget at that alpha",
                "selection": "validation macro NDCG@10, Recall@10 tie break, then label",
                "score_policy": "positive PPR scores only; prefilter known and conflict pairs; top 100",
                "test_fit": "train+validation", "test_retuning": False,
                "workers": args.workers, "tolerance": 1e-10,
                "runtime_scope": "graph construction + reweighting + inference; exclude data reads and evaluator",
                "data_hashes": json.loads((data_dir / "verified_hashes.json").read_text())}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    selected = {}
    all_metrics = []
    for stage in ("validation", "test"):
        fit_labels = {"train"} if stage == "validation" else {"train", "validation"}
        fit = {(r["disease_id"], r["drug_id"]) for r in split if r["split"] in fit_labels}
        truth = {(r["disease_id"], r["drug_id"]) for r in split if r["split"] == stage}
        assert not fit & truth
        known = defaultdict(set)
        for d, drug in fit:
            known[d].add(drug)
        excluded = defaultdict(set, {d: set(v) for d, v in known.items()})
        for r in conflicts:
            excluded[r["disease_id"]].add(r["drug_id"])
        diseases = sorted({d for d, _ in truth})
        catalog = sorted({drug for _, drug in fit})
        begin = time.perf_counter()
        matrix, index, types = build(fit, genes, targets)
        build_seconds = time.perf_counter() - begin
        for d, drug in truth:
            if ("drug", drug) in index:
                assert matrix[index["disease", d], index["drug", drug]] == 0
        counts = Counter(drug for _, drug in fit)
        popular = sorted(catalog, key=lambda m: (-counts[m], m))
        begin = time.perf_counter()
        rows = []
        for d in diseases:
            eligible = [m for m in popular if m not in excluded[d]][:100]
            rows.extend({"disease_id": d, "drug_id": m, "score": counts[m]} for m in eligible)
        all_metrics.append(evaluate(stage, "popularity", rows, time.perf_counter()-begin, output, data_dir))
        configs = [(a, None) for a in ALPHAS] if stage == "validation" else [
            (selected["alpha"], None), (selected["alpha"], selected["budget"])]
        basic_metrics = []
        for alpha, budget in configs:
            label = f"basic_alpha_{alpha}" if budget is None else f"weighted_alpha_{alpha}_budget_{budget}"
            begin = time.perf_counter()
            graph = matrix if budget is None else balance(matrix, types, budget)
            rows = score_ppr(graph, index, catalog, diseases, excluded, alpha, args.workers)
            summary = evaluate(stage, label, rows, build_seconds+time.perf_counter()-begin, output, data_dir)
            all_metrics.append(summary)
            basic_metrics.append((alpha, summary))
        if stage == "validation":
            best = sorted(basic_metrics, key=lambda x: (-x[1]["ndcg@10"], -x[1]["recall@10"], x[0]))[0]
            selected["alpha"] = best[0]
            weighted_metrics = []
            for budget in BUDGETS:
                begin = time.perf_counter()
                graph = balance(matrix, types, budget)
                rows = score_ppr(graph, index, catalog, diseases, excluded, selected["alpha"], args.workers)
                label = f"weighted_alpha_{selected['alpha']}_budget_{budget}"
                summary = evaluate(stage, label, rows, build_seconds+time.perf_counter()-begin, output, data_dir)
                all_metrics.append(summary)
                weighted_metrics.append((budget, summary))
            best = sorted(weighted_metrics, key=lambda x: (-x[1]["ndcg@10"], -x[1]["recall@10"], x[0]))[0]
            selected["budget"] = best[0]
            (output / "selected_config.json").write_text(json.dumps(selected, indent=2)+"\n")
        (output / "metrics.json").write_text(json.dumps(all_metrics, indent=2)+"\n")
    write(output / "comparison.csv", all_metrics)


if __name__ == "__main__":
    main()
