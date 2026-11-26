"""Build the shared indication split from the provided processed CSV.

The conflict flag comes from the processed data; see README for provenance.
"""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indications', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    rows = read_csv(args.indications)
    required = {'disease_id', 'drug_id', 'also_asserted_contraindication'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'입력 파일에 필요한 열: {required}')
    pairs = [(r['disease_id'], r['drug_id']) for r in rows]
    if any(not d or not drug for d, drug in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError('빈 ID 또는 중복 질병–약물 쌍이 있습니다.')
    if len(rows) != 58174:
        raise ValueError(f'원본 58,174쌍이 필요합니다: {len(rows)}')
    flags = {r['also_asserted_contraindication'].strip().lower() for r in rows}
    if not flags <= {'true', 'false'}:
        raise ValueError(f'알 수 없는 금기 충돌 값: {flags}')
    conflicts = [r for r in rows if r['also_asserted_contraindication'].strip().lower() == 'true']
    clean = [r for r in rows if r['also_asserted_contraindication'].strip().lower() == 'false']
    if len(conflicts) != 339 or len(clean) != 57835:
        raise ValueError(f'금기 충돌/정제 관계 수 확인: {len(conflicts)}/{len(clean)}')
    grouped = defaultdict(list)
    for r in clean:
        grouped[r['disease_id']].append(r)
    split = []
    for disease in sorted(grouped):
        def key(r):
            digest = hashlib.blake2b(f"{args.seed}|{disease}|{r['drug_id']}".encode(), digest_size=8).digest()
            return digest, r['drug_id']
        records = sorted(grouped[disease], key=key)
        n = len(records)
        target = max(1, round(.2*n))
        nt = min(target, max(0, n-2))
        nv = min(target, max(0, n-nt-2))
        for i, row in enumerate(records):
            label = 'test' if i < nt else 'validation' if i < nt+nv else 'train'
            split.append({'disease_id': disease, 'drug_id': row['drug_id'], 'split': label})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    names = ['team_eligible_indications.csv', 'team_split.csv', 'contraindication_conflicts.csv']
    if any((out / n).exists() for n in names):
        raise FileExistsError('출력 파일이 이미 있습니다. 빈 출력 폴더를 사용하세요.')
    write_csv(out/names[0], list(rows[0]), clean)
    write_csv(out/names[1], ['disease_id', 'drug_id', 'split'], split)
    write_csv(out/names[2], ['disease_id', 'drug_id'], conflicts)
    for label in ('train', 'validation', 'test'):
        selected = [r for r in split if r['split'] == label]
        print(f"{label}: {len(selected):,} pairs, {len({r['disease_id'] for r in selected}):,} diseases")
    print(f'clean: {len(clean):,}; conflicts: {len(conflicts):,}; seed: {args.seed}')


if __name__ == '__main__':
    main()
