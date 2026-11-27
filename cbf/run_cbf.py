"""Tune strict disease-gene CBF on common Validation, then evaluate fixed Test once."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cbf_diagnostics import failure_summary, paired_validation_intervals
from cbf_helpers import (FEATURE_MODES, NEIGHBOR_COUNTS, SIMILARITY_POWERS,
                         disease_drugs, make_fitting_data, nearest_diseases,
                         profiles_and_similarities, rank_drugs, read_csv)


ROOT = Path(__file__).resolve().parent
COMMON_EVALUATOR = ROOT.parent / "common" / "evaluate.py"
SEED = 42
EXPECTED = {
    "team_split.csv": "7514f8a689313d6f01827f1ddffe9b527334784132eb4a36edf406d65cb2d380",
    "contraindication_conflicts.csv": "fdb96c5bc6774ab1ea0c63ed247cc89edcc7565f127b21ec572af7355bf385ce",
}


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_input_files(data_dir):
    """Reject mismatched team splits and features before any model fitting."""
    files = {
        "team_split.csv": data_dir / "splits" / "team_split.csv",
        "contraindication_conflicts.csv": data_dir / "splits" / "contraindication_conflicts.csv",
        "disease_gene.csv": data_dir / "features" / "disease_gene.csv",
    }
    hashes = {key: sha256(path) for key, path in files.items()}
    for key, expected in EXPECTED.items():
        if hashes[key] != expected:
            raise ValueError(f"Team SHA-256 mismatch for {key}")
    manifest = data_dir / "verified_hashes.json"
    if manifest.is_file():
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        for key, actual in hashes.items():
            if key in recorded and recorded[key] != actual:
                raise ValueError(f"Prepared input was modified: {key}")
    return files, hashes


def confirm_team_splits(split, conflicts, eligible_path):
    """Check relation counts and the exact per-disease BLAKE2b split algorithm."""
    counts = split["split"].value_counts().to_dict()
    if counts != {"train": 35068, "validation": 11290, "test": 11477}:
        raise ValueError(f"Unexpected team split counts: {counts}")
    if len(split) != 57835 or len(conflicts) != 339:
        raise ValueError("Team indication or conflict count changed")
    rows = list(split[["disease_id", "drug_id"]].itertuples(index=False, name=None))
    blocked = set(conflicts[["disease_id", "drug_id"]].itertuples(index=False, name=None))
    if len(set(rows)) != len(rows) or blocked.intersection(rows):
        raise ValueError("Duplicate fitting pairs or contraindication conflicts in split")
    if eligible_path.is_file():
        eligible = read_csv(eligible_path, ("disease_id", "drug_id"))
        if set(rows) != set(eligible[["disease_id", "drug_id"]].itertuples(index=False, name=None)):
            raise ValueError("Eligible indication pairs differ from shared split")
    for disease, group in split.groupby("disease_id"):
        drugs = sorted(group["drug_id"], key=lambda drug: (
            hashlib.blake2b(f"{SEED}|{disease}|{drug}".encode(), digest_size=8).digest(), drug))
        n = len(drugs)
        target = max(1, round(.2 * n))
        n_test = min(target, max(0, n - 2))
        n_validation = min(target, max(0, n - n_test - 2))
        labels = {drug: "test" if i < n_test else "validation" if i < n_test + n_validation
                  else "train" for i, drug in enumerate(drugs)}
        if any(labels[drug] != label for drug, label in group[["drug_id", "split"]].itertuples(index=False, name=None)):
            raise ValueError(f"Frozen per-disease split mismatch: {disease}")
    return counts


def write_scores(path, queries, fitting, similarity, diseases, disease_index, conflicts, k, p):
    """Write at most 100 positive CBF scores per query, after exclusion and stable ties."""
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("disease_id", "drug_id", "score"))
        for disease in queries:
            neighbors = nearest_diseases(disease, similarity, fitting, diseases, disease_index)
            ranked = rank_drugs(disease, neighbors, fitting, conflicts, k, p)
            writer.writerows((disease, drug, format(score, ".17g")) for drug, score in ranked)



def evaluate(stage, score_path, folder, data_dir, runtime=None):
    """Invoke the unchanged team evaluator; keep its summary.json authoritative."""
    command = [sys.executable, str(COMMON_EVALUATOR),
               "--split", str(data_dir / "splits/team_split.csv"),
               "--conflicts", str(data_dir / "splits/contraindication_conflicts.csv"),
               "--scores", str(score_path), "--stage", stage, "--output-dir", str(folder)]
    if runtime is not None:
        command.extend(["--runtime-seconds", str(runtime)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads((folder / "summary.json").read_text(encoding="utf-8"))


def metric_at_10(ranked, truth):
    """Local Validation score uses exactly the shared evaluator's macro relevance formula."""
    drugs = [drug for drug, _ in ranked[:10]]
    recall = sum(drug in truth for drug in drugs) / len(truth)
    dcg = sum(1 / math.log2(i + 2) for i, drug in enumerate(drugs) if drug in truth)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(truth))))
    return recall, dcg / idcg


def choose_configuration(queries, true_pairs, fitting, similarities, diseases, disease_index, conflicts):
    """Compare all 12 configurations on Train-only Validation, never on Test."""
    truth = disease_drugs(true_pairs)
    neighbor_cache = {
        mode: {d: nearest_diseases(d, similarities[mode], fitting, diseases, disease_index)
               for d in queries}
        for mode in FEATURE_MODES
    }
    configurations = []
    ndcg_by_setting, recall_by_setting = {}, {}
    for mode, k, p in product(FEATURE_MODES, NEIGHBOR_COUNTS, SIMILARITY_POWERS):
        recall_sum = ndcg_sum = 0.0
        ndcg_values, recall_values = [], []
        recommended = 0
        for disease in queries:
            ranked = rank_drugs(disease, neighbor_cache[mode][disease], fitting, conflicts, k, p)
            recall, ndcg = metric_at_10(ranked, truth[disease])
            recall_sum += recall
            ndcg_sum += ndcg
            ndcg_values.append(ndcg)
            recall_values.append(recall)
            recommended += bool(ranked)
        ndcg_by_setting[mode, k, p] = ndcg_values
        recall_by_setting[mode, k, p] = recall_values
        n = len(queries)
        configurations.append({"feature_mode": mode, "k": k, "p": p,
                               "recall@10": recall_sum / n, "ndcg@10": ndcg_sum / n,
                               "query_coverage": recommended / n})
    # Equal NDCG: higher Recall; then simpler p, smaller k, Raw rather than IDF.
    configurations.sort(key=lambda r: (-r["ndcg@10"], -r["recall@10"],
                                       r["p"], r["k"], FEATURE_MODES.index(r["feature_mode"])))
    return configurations, ndcg_by_setting, recall_by_setting


def verify_evaluator(score_file, summary, heldout, candidate_catalog):
    """Independently reproduce the common Recall/NDCG and Coverage for the exported scores."""
    predictions = defaultdict(list)
    with score_file.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            predictions[row["disease_id"]].append((row["drug_id"], float(row["score"])))
    truths = disease_drugs(heldout)
    measured = defaultdict(float)
    listed = {k: set() for k in (10, 50, 100)}
    measured["query_coverage"] = sum(bool(predictions[d]) for d in truths) / len(truths)
    for disease, truth in truths.items():
        ordered = sorted(predictions[disease], key=lambda item: (-item[1], item[0]))
        for k in (10, 50, 100):
            top = [drug for drug, _ in ordered[:k]]
            hits = sum(drug in truth for drug in top)
            dcg = sum(1 / math.log2(i + 2) for i, drug in enumerate(top) if drug in truth)
            idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(truth))))
            measured[f"recall@{k}"] += hits / len(truth) / len(truths)
            measured[f"ndcg@{k}"] += dcg / idcg / len(truths)
            listed[k].update(top)
    for k in (10, 50, 100):
        measured[f"catalog_coverage@{k}"] = len(listed[k]) / len(candidate_catalog)
    for name, value in measured.items():
        if not math.isclose(value, summary[name], abs_tol=1e-10, rel_tol=0):
            raise AssertionError(f"Shared evaluator disagrees for {name}: {value} vs {summary[name]}")


def write_popularity(path, queries, fit, conflicts):
    """Popularity is a separate benchmark, never a fallback inside CBF."""
    counts = Counter(fit["drug_id"])
    popular = sorted(counts, key=lambda drug: (-counts[drug], drug))
    known = disease_drugs(fit)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("disease_id", "drug_id", "score"))
        for disease in queries:
            excluded = known.get(disease, set()) | conflicts.get(disease, set())
            valid = [drug for drug in popular if drug not in excluded][:100]
            writer.writerows((disease, drug, counts[drug]) for drug in valid)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "run_results")
    args = parser.parse_args()
    data_dir, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new or empty results directory")
    files, hashes = verify_input_files(data_dir)
    split = read_csv(files["team_split.csv"], ("disease_id", "drug_id", "split"))
    conflicts = read_csv(files["contraindication_conflicts.csv"], ("disease_id", "drug_id"))
    split_counts = confirm_team_splits(
        split, conflicts, data_dir / "splits/team_eligible_indications.csv")
    blocked = conflicts.groupby("disease_id")["drug_id"].agg(set).to_dict()
    diseases = sorted(split["disease_id"].unique())
    disease_index = {d: i for i, d in enumerate(diseases)}
    train = split[split["split"] == "train"]
    validation = split[split["split"] == "validation"]
    test = split[split["split"] == "test"]
    trainval = split[split["split"].isin(("train", "validation"))]
    validation_queries = sorted(validation["disease_id"].unique())
    test_queries = sorted(test["disease_id"].unique())
    output.mkdir(parents=True, exist_ok=True)

    model_start = time.perf_counter()
    sims, has_gene, global_n, profile_shape = profiles_and_similarities(
        files["disease_gene.csv"], diseases)
    profile_seconds = time.perf_counter() - model_start
    tuning_start = time.perf_counter()
    fitting_train = make_fitting_data(train, disease_index)
    grid, validation_ndcg, validation_recall = choose_configuration(
        validation_queries, validation, fitting_train, sims, diseases, disease_index, blocked)
    tuning_seconds = time.perf_counter() - tuning_start
    pd.DataFrame(grid).to_csv(output / "validation_comparison.csv", index=False)
    # Validation-only sensitivity plot: never consult the test split when tuning.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparisons = pd.DataFrame(grid)
    for mode in FEATURE_MODES:
        for power in SIMILARITY_POWERS:
            line = comparisons[(comparisons["feature_mode"] == mode) &
                               (comparisons["p"] == power)].sort_values("k")
            ax.plot(line["k"], line["ndcg@10"], marker="o", label=f"{mode}, p={power}")
    ax.set(xlabel="Neighbor count k", ylabel="Validation NDCG@10",
           title="CBF hyperparameter sensitivity")
    ax.set_xticks(NEIGHBOR_COUNTS)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "cbf_validation_sensitivity.png", dpi=160)
    plt.close(fig)
    paired = paired_validation_intervals(
        grid, validation_ndcg, validation_recall, validation_queries, seed=SEED)
    save_json(output / "validation_paired_bootstrap.json", paired)
    winner = grid[0]
    selected = {"feature_mode": winner["feature_mode"], "k": winner["k"], "p": winner["p"]}
    save_json(output / "selected_config.json", selected)
    print("Validation selection:", selected, "NDCG@10=", round(winner["ndcg@10"], 6), flush=True)

    validation_scores = output / "validation_scores.csv"
    write_scores(validation_scores, validation_queries, fitting_train,
                 sims[selected["feature_mode"]], diseases, disease_index, blocked,
                 selected["k"], selected["p"])
    validation_summary = evaluate("validation", validation_scores, output / "validation", data_dir)
    if not math.isclose(validation_summary["ndcg@10"], winner["ndcg@10"], abs_tol=1e-12):
        raise AssertionError("Local validation tuning does not match team evaluation")
    verify_evaluator(validation_scores, validation_summary, validation, fitting_train["catalog"])

    final_start = time.perf_counter()
    fitting_test = make_fitting_data(trainval, disease_index)
    test_scores = output / "test_scores.csv"
    write_scores(test_scores, test_queries, fitting_test,
                 sims[selected["feature_mode"]], diseases, disease_index, blocked,
                 selected["k"], selected["p"])
    fit_and_score_seconds = time.perf_counter() - final_start
    runtime = profile_seconds + fit_and_score_seconds
    test_summary = evaluate("test", test_scores, output / "test", data_dir, runtime)
    if test_summary["ignored_score_rows"]:
        raise AssertionError("Invalid CBF scores were ignored by the shared evaluator")
    verify_evaluator(test_scores, test_summary, test, fitting_test["catalog"])
    print("CBF final Test: Recall@10=", round(test_summary["recall@10"], 6),
          "NDCG@10=", round(test_summary["ndcg@10"], 6), flush=True)

    baseline_start = time.perf_counter()
    popularity_scores = output / "popularity_scores.csv"
    write_popularity(popularity_scores, test_queries, trainval, blocked)
    popularity_runtime = time.perf_counter() - baseline_start
    popularity_summary = evaluate("test", popularity_scores, output / "popularity_test",
                                  data_dir, popularity_runtime)
    per_disease = pd.read_csv(output / "test/per_disease.csv", dtype={"disease_id": str})
    fit_degrees = trainval.groupby("disease_id")["drug_id"].nunique()
    per_disease["has_gene_profile"] = per_disease["disease_id"].map(
        lambda disease: bool(has_gene[disease_index[disease]]))
    per_disease["n_fit_indications"] = per_disease["disease_id"].map(fit_degrees).fillna(0).astype(int)
    per_disease["fit_degree_group"] = pd.cut(
        per_disease["n_fit_indications"], [-1, 2, 5, 20, float("inf")],
        labels=["0-2", "3-5", "6-20", "21+"])
    summary_rows = []
    for group_name in ("has_gene_profile", "fit_degree_group"):
        for value, subgroup in per_disease.groupby(group_name, observed=True):
            summary_rows.append({"grouping": group_name, "group": str(value),
                                 "n_diseases": len(subgroup),
                                 "recall@10": subgroup["recall@10"].mean(),
                                 "ndcg@10": subgroup["ndcg@10"].mean(),
                                 "query_coverage": (subgroup["n_recommended_up_to_100"] > 0).mean()})
    pd.DataFrame(summary_rows).to_csv(output / "test_by_group.csv", index=False)

    # Post-hoc error analysis; this is not consulted when selecting parameters.
    n_recs = dict(zip(per_disease["disease_id"], per_disease["n_recommended_up_to_100"]))
    reasons = failure_summary(test_queries, n_recs, has_gene,
                              sims[selected["feature_mode"]], fitting_test,
                              diseases, disease_index, nearest_diseases)
    if sum(reasons.values()) != len(test_queries):
        raise AssertionError("Test failure counts do not cover every query")
    if reasons.get("recommended", 0) != test_summary["diseases_with_any_recommendation"]:
        raise AssertionError("Failure diagnostics disagree with team coverage")
    outside = int(test_summary["positives_outside_candidates"])
    positive_pairs = int(test_summary["positive_pairs"])
    macro_oracle = float((1 - per_disease["truth_outside_candidates"] /
                          per_disease["n_truth"]).mean())
    diagnostics = {
        "test_diseases": len(test_queries), "failure_reasons": reasons,
        "candidate_drugs": test_summary["candidate_drugs"],
        "positives_outside_candidates": outside,
        "positive_pairs": positive_pairs,
        "unrankable_positive_pair_fraction": outside / positive_pairs,
        "oracle_macro_recall_ceiling_from_catalog_only": macro_oracle,
        "macro_recall_ceiling_note": (
            "This is an upper bound from catalog availability only, ignoring rank-K "
            "capacity and model scoring. It is not an achievable CBF score."),
        "catalog_coverage@10": test_summary["catalog_coverage@10"],
        "popularity_catalog_coverage@10": popularity_summary["catalog_coverage@10"],
        "comparison_note": "Catalog coverage is breadth of distinct recommended drugs, not clinical diversity.",
    }
    save_json(output / "test_diagnostics.json", diagnostics)
    comparison = {
        "validation": {"fitting_pairs": len(train), "evaluation_pairs": len(validation),
                       "evaluation_diseases": validation_summary["evaluation_diseases"],
                       "candidate_drugs": validation_summary["candidate_drugs"],
                       "ndcg@10": validation_summary["ndcg@10"]},
        "test": {"fitting_pairs": len(trainval), "evaluation_pairs": len(test),
                 "evaluation_diseases": test_summary["evaluation_diseases"],
                 "candidate_drugs": test_summary["candidate_drugs"],
                 "ndcg@10": test_summary["ndcg@10"]},
        "ndcg@10_difference_test_minus_validation": (
            test_summary["ndcg@10"] - validation_summary["ndcg@10"]),
        "interpretation": (
            "Test uses more fitting indications and a larger candidate catalog, and "
            "Validation/Test involve different diseases and held-out pairs. "
            "More fitting data is a plausible contributor to the higher Test score, "
            "but this comparison alone does not establish causality or significance. "
            "Test was not used for parameter selection."),
    }
    save_json(output / "validation_test_context.json", comparison)

    protocol = {
        "algorithm": "strict CBF: disease-gene cosine similarity and fitting indication transfer",
        "seed": SEED, "split_counts": split_counts,
        "team_input_sha256": hashes,
        "feature_threshold": .3,
        "feature_idf_formula": "ln((N+1)/(df+1))+1; N=unique diseases in full disease_gene input",
        "idf_disease_count_N": global_n,
        "profile_shape": list(profile_shape),
        "grid": {"weighting": FEATURE_MODES, "k": NEIGHBOR_COUNTS, "p": SIMILARITY_POWERS},
        "selection": "Validation macro NDCG@10; Recall@10 ties; then p, k, Raw before IDF",
        "selected": selected,
        "fit_rule": "Validation=train only; Test=train+validation; Test evaluated once",
        "score_policy": "positive score; fitting catalog; filter own indications and 339 conflicts; top 100; drug-ID tie",
        "popularity_fallback": False,
        "drug_gene_used_in_cbf_scoring": False,
        "runtime_scope": "profile construction + final fitting and top-100 CSV export; excludes raw CSV preparation, validation tuning, evaluator",
        "runtime_seconds": runtime,
        "validation_tuning_seconds": tuning_seconds,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "test_not_used_for_parameter_selection": True,
        "feature_scope": "Transductive: IDF N and df include all diseases in provided gene data, including evaluation diseases; no held-out indication labels are used for IDF",
        "similarity_storage": "dense pairwise disease similarity array, O(D^2) memory and work; D=2327",
        "statistical_comparison": "paired validation bootstrap; exploratory post-selection intervals, not confirmatory hypothesis test",
    }
    save_json(output / "protocol.json", protocol)
    # Publish a compact results/ snapshot with the same layout committed to GitHub.
    # Detailed scores and per-disease CSVs stay outside this summary directory.
    published = output / "results"
    (published / "test").mkdir(parents=True)
    for name in ("selected_config.json", "protocol.json", "validation_comparison.csv",
                 "validation_paired_bootstrap.json", "validation_test_context.json",
                 "cbf_validation_sensitivity.png"):
        shutil.copyfile(output / name, published / name)
    for name in ("test_by_group.csv", "test_diagnostics.json"):
        shutil.copyfile(output / name, published / "test" / name)
    save_json(published / "test" / "cbf_summary.json", test_summary)
    save_json(published / "test" / "popularity_summary.json", popularity_summary)
    print("Published results:", published, flush=True)


if __name__ == "__main__":
    main()
