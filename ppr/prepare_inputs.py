"""Adapt local input columns, then run the team's unchanged preparation scripts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
COMMON = ROOT.parent / "common"
EXPECTED = {
    "team_split.csv": "7514f8a689313d6f01827f1ddffe9b527334784132eb4a36edf406d65cb2d380",
    "contraindication_conflicts.csv": "fdb96c5bc6774ab1ea0c63ed247cc89edcc7565f127b21ec572af7355bf385ce",
}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data",
                        help="new directory for prepared inputs")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    adapted = out / "adapted"
    adapted.mkdir()
    raw = read(args.data_dir / "indication.csv")
    conflicts = {
        (row["to"], row["from"])
        for row in pq.read_table(args.parquet, columns=["from", "to", "relation", "properties"]).to_pylist()
        if row["relation"] == "INDICATION"
        and any(a["relation"] == "CONTRAINDICATION"
                for a in ((row["properties"] or {}).get("relation_assertions") or []))
    }
    pairs = {(r["disease_id"], r["drug_id"]) for r in raw}
    assert len(raw) == len(pairs) == 58174 and len(conflicts & pairs) == 339
    for row in raw:
        row["also_asserted_contraindication"] = str((row["disease_id"], row["drug_id"]) in conflicts)
    write(adapted / "indication.csv", raw)
    genes = read(args.data_dir / "disease_gene_t03.csv")
    if "score" in genes[0]:
        for row in genes:
            row["evidence_score"] = row.pop("score")
    write(adapted / "disease_gene.csv", genes)
    subprocess.run([sys.executable, str(COMMON / "prepare_splits.py"),
                    "--indications", str(adapted / "indication.csv"),
                    "--output-dir", str(out / "splits"), "--seed", "42"], check=True)
    subprocess.run([sys.executable, str(COMMON / "prepare_features.py"),
                    "--disease-gene", str(adapted / "disease_gene.csv"),
                    "--drug-gene", str(args.data_dir / "drug_gene.csv"),
                    "--output-dir", str(out / "features")], check=True)
    # The team's documented conflict hash uses disease/drug sorted row order.
    path = out / "splits/contraindication_conflicts.csv"
    rows = sorted(read(path), key=lambda r: (r["disease_id"], r["drug_id"]))
    path.unlink()
    write(path, rows)
    hashes = {}
    for name, expected in EXPECTED.items():
        digest = hashlib.sha256((out / "splits" / name).read_bytes()).hexdigest()
        assert digest == expected, (name, digest)
        hashes[name] = digest
    for path in sorted((out / "features").glob("*.csv")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "verified_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("Both team hashes match.", flush=True)


if __name__ == "__main__":
    main()
