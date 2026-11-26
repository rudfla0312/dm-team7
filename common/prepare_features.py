"""Prepare shared feature tables without filtering diseases by gene availability."""
import argparse
import csv
import math
from pathlib import Path


def filter_csv(source, dest, fields, predicate):
    with source.open(encoding='utf-8-sig', newline='') as src:
        reader = csv.DictReader(src)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f'{source}: required columns {fields}')
        with dest.open('x', encoding='utf-8', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
            n = 0
            for row in reader:
                if predicate(row):
                    writer.writerow(row)
                    n += 1
    return n


def valid_gene(row, threshold):
    try:
        score = float(row['evidence_score'])
    except (ValueError, TypeError):
        raise ValueError(f"invalid evidence_score: {row['evidence_score']!r}")
    if not math.isfinite(score):
        raise ValueError('non-finite evidence score')
    return score >= threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--disease-gene', type=Path, required=True)
    p.add_argument('--drug-gene', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.3)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = [out/'disease_gene.csv', out/'drug_gene.csv']
    if any(x.exists() for x in paths):
        raise FileExistsError('빈 출력 폴더를 사용하세요.')
    print('disease-gene:', filter_csv(args.disease_gene, paths[0], ['disease_id', 'gene_id', 'evidence_score'], lambda r: valid_gene(r, args.threshold)))
    print('drug-gene:', filter_csv(args.drug_gene, paths[1], ['drug_id', 'gene_id'], lambda r: True))
    print('유전자 유무에 따른 질병 제외는 하지 않습니다.')


if __name__ == '__main__':
    main()
