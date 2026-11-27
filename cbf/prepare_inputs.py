"""Prepare verified team inputs for CBF using the existing common scripts."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMON = ROOT.parent / "common"
EXPECTED_HASHES = {
    "team_split.csv": "7514f8a689313d6f01827f1ddffe9b527334784132eb4a36edf406d65cb2d380",
    "contraindication_conflicts.csv": "fdb96c5bc6774ab1ea0c63ed247cc89edcc7565f127b21ec572af7355bf385ce",
}


def digest(path):
    """Compute the SHA-256 digest of the exact file bytes."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def adapt_gene_columns(source, destination):
    """Convert 'score' to the common 'evidence_score' column without changing rows."""
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if not {"disease_id", "gene_id"}.issubset(columns):
            raise ValueError("Disease-gene input has missing IDs")
        if "evidence_score" in columns:
            renamed = columns
        elif "score" in columns:
            renamed = ["evidence_score" if field == "score" else field for field in columns]
        else:
            raise ValueError("Disease-gene input requires score or evidence_score")
        with destination.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=renamed)
            writer.writeheader()
            for row in reader:
                if "score" in row and "evidence_score" not in row:
                    row["evidence_score"] = row.pop("score")
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="folder with the provided shared CSVs and gene/drug-gene inputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new or empty prepared-input directory")
    (output / "splits").mkdir(parents=True)
    (output / "features").mkdir()
    (output / "adapted").mkdir()
    for name, expected in EXPECTED_HASHES.items():
        original = source / name
        actual = digest(original)
        if actual != expected:
            raise ValueError(f"Team hash mismatch: {name} (found {actual})")
        shutil.copyfile(original, output / "splits" / name)
    eligible = source / "team_eligible_indications.csv"
    if eligible.is_file():
        shutil.copyfile(eligible, output / "splits" / eligible.name)
    adapted = output / "adapted" / "disease_gene.csv"
    adapt_gene_columns(source / "disease_gene_t03.csv", adapted)
    subprocess.run([
        sys.executable, str(COMMON / "prepare_features.py"),
        "--disease-gene", str(adapted), "--drug-gene", str(source / "drug_gene.csv"),
        "--output-dir", str(output / "features"), "--threshold", "0.3",
    ], check=True)
    files = [output / "splits" / name for name in EXPECTED_HASHES]
    files += sorted((output / "features").glob("*.csv"))
    if eligible.is_file():
        files.append(output / "splits" / eligible.name)
    hashes = {path.name: digest(path) for path in files}
    (output / "verified_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print("CBF inputs prepared; official split/conflict hashes match.", flush=True)


if __name__ == "__main__":
    main()
