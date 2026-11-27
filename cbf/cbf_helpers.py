"""Disease-gene profiles and neighbor-based drug scoring for strict CBF."""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize


FEATURE_MODES = ("raw", "idf")
NEIGHBOR_COUNTS = (10, 20, 50)
SIMILARITY_POWERS = (1, 2)


def read_csv(path, columns):
    """Read a CSV and preserve entity IDs as strings."""
    frame = pd.read_csv(path, dtype={name: str for name in columns}, keep_default_na=False)
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"{path}: required columns {columns}")
    return frame


def disease_drugs(frame):
    """Return each disease's observed drug IDs for a fitting split."""
    return frame.groupby("disease_id")["drug_id"].agg(set).to_dict()


def profiles_and_similarities(gene_csv, diseases):
    """Build raw/IDF cosine similarities with IDF over *all* feature-file diseases."""
    genes = read_csv(gene_csv, ("disease_id", "gene_id", "evidence_score"))
    genes["evidence_score"] = pd.to_numeric(genes["evidence_score"], errors="raise")
    values = genes["evidence_score"].to_numpy()
    if not np.isfinite(values).all() or (values < .3).any():
        raise ValueError("Disease-gene features must be finite and have evidence_score >= 0.3")
    if len(genes) != 810225:
        raise ValueError(f"Expected 810225 prepared gene relations, got {len(genes)}")
    # A unique feature prevents duplicate evidence edges from being added together in COO.
    genes = (genes.sort_values("evidence_score", ascending=False, kind="stable")
             .drop_duplicates(("disease_id", "gene_id")))
    full_n = int(genes["disease_id"].nunique())
    all_genes = pd.Index(sorted(genes["gene_id"].unique()))
    disease_index = {d: i for i, d in enumerate(diseases)}
    usable = genes[genes["disease_id"].isin(disease_index)]
    rr = np.array([disease_index[d] for d in usable["disease_id"]], dtype=np.int32)
    cc = all_genes.get_indexer(usable["gene_id"]).astype(np.int32)
    matrix = coo_matrix(
        (usable["evidence_score"].to_numpy(dtype=np.float64), (rr, cc)),
        shape=(len(diseases), len(all_genes)),
    ).tocsr()
    has_gene = np.diff(matrix.indptr) > 0
    df = (genes.groupby("gene_id")["disease_id"].nunique()
          .reindex(all_genes).to_numpy(dtype=np.float64))
    # Match the report: ln((N+1)/(df+1))+1, with N from the whole feature input.
    idf = np.log((full_n + 1) / (df + 1)) + 1
    weighted = {"raw": matrix, "idf": matrix.multiply(idf).tocsr()}
    similarities = {}
    for name, features in weighted.items():
        # Apply IDF *before* L2 normalization; matrix multiplication is cosine similarity.
        unit = normalize(features, norm="l2", axis=1).tocsr()
        sim = (unit @ unit.T).toarray()
        np.fill_diagonal(sim, 0.0)
        similarities[name] = sim
    return similarities, has_gene, full_n, matrix.shape


def make_fitting_data(pairs, disease_index):
    """Return train-only disease-drug relations, fitting candidate drugs and source rows."""
    known = disease_drugs(pairs)
    sources = np.array(sorted(disease_index[d] for d in known), dtype=np.int32)
    return {"known": known, "catalog": set(pairs["drug_id"]), "sources": sources}


def nearest_diseases(query, similarities, fitting, diseases, disease_index, limit=50):
    """Find positive-similarity fitting diseases, excluding the query itself."""
    row = disease_index[query]
    source_rows = fitting["sources"]
    weights = similarities[row, source_rows]
    keep = (source_rows != row) & (weights > 0)
    source_rows, weights = source_rows[keep], weights[keep]
    # Source rows are disease-ID sorted, so stable score sorting breaks ties by ID.
    order = np.argsort(-weights, kind="stable")[:limit]
    return [(diseases[source_rows[j]], float(weights[j])) for j in order]


def rank_drugs(query, neighbors, fitting, conflicts, k, power, top=100):
    """Transfer fitting indications from top-k neighbors; never use popularity fallback."""
    forbidden = fitting["known"].get(query, set()) | conflicts.get(query, set())
    score = defaultdict(float)
    for neighbor, similarity in neighbors[:k]:
        for drug in fitting["known"][neighbor]:
            if drug not in forbidden:
                score[drug] += similarity ** power
    # Filter *before* the top-100 cut; identical scores break by drug ID.
    return sorted(score.items(), key=lambda entry: (-entry[1], entry[0]))[:top]
