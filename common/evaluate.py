"""Evaluate disease-drug scores with the common split and candidate rules."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read(path, required):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: requires {required}')
        return list(reader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--conflicts', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--stage', choices=['validation', 'test'], required=True)
    p.add_argument('--fit', choices=['train', 'trainval'], help='선택 사항: validation은 train, test는 trainval로 고정')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--runtime-seconds', type=float, help='별도로 측정한 학습+추론 시간 (초)')
    args = p.parse_args()
    required_fit = 'train' if args.stage == 'validation' else 'trainval'
    if args.fit is not None and args.fit != required_fit:
        p.error(f'{args.stage} 평가에는 --fit {required_fit}을 사용해야 합니다.')
    args.fit = required_fit
    if args.runtime_seconds is not None and (not math.isfinite(args.runtime_seconds) or args.runtime_seconds < 0):
        p.error('runtime-seconds는 0 이상의 유한 숫자여야 합니다.')
    rows = read(args.split, ['disease_id', 'drug_id', 'split'])
    conflicts = read(args.conflicts, ['disease_id', 'drug_id'])
    seen = set()
    known = defaultdict(set)
    positives = defaultdict(set)
    candidates = set()
    fit_sets = {'train'} if args.fit == 'train' else {'train', 'validation'}
    for r in rows:
        disease, drug, label = r['disease_id'], r['drug_id'], r['split']
        if not disease or not drug or label not in {'train', 'validation', 'test'}:
            raise ValueError('분할 파일에 빈 ID 또는 잘못된 split이 있습니다.')
        if (disease, drug) in seen:
            raise ValueError(f'분할 파일에 중복 쌍: {disease}, {drug}')
        seen.add((disease, drug))
        if label in fit_sets:
            known[disease].add(drug)
            candidates.add(drug)
        if label == args.stage:
            positives[disease].add(drug)
    excluded = defaultdict(set)
    for r in conflicts:
        excluded[r['disease_id']].add(r['drug_id'])
        if (r['disease_id'], r['drug_id']) in seen:
            raise ValueError('금기 충돌 쌍이 분할에 포함되어 있습니다.')
    if not positives or not candidates:
        raise ValueError('평가 정답 또는 학습 후보 약물이 없습니다.')
    rankings = defaultdict(list)
    score_pairs = set()
    ignored = 0
    for r in read(args.scores, ['disease_id', 'drug_id', 'score']):
        d, drug = r['disease_id'], r['drug_id']
        if not d or not drug or (d, drug) in score_pairs:
            raise ValueError(f'빈 ID 또는 중복 점수 쌍: {d}, {drug}')
        score_pairs.add((d, drug))
        try:
            score = float(r['score'])
        except ValueError:
            raise ValueError(f'숫자가 아닌 점수: {r["score"]}')
        if not math.isfinite(score):
            raise ValueError('점수는 유한한 숫자여야 합니다.')
        if d not in positives or drug not in candidates or drug in known[d] or drug in excluded[d]:
            ignored += 1
            continue
        rankings[d].append((drug, score))
    ks = (10, 50, 100)
    per_disease = []
    distinct = {k: set() for k in ks}
    unrankable_truths = 0
    total_truths = 0
    for d in sorted(positives):
        truth = positives[d]
        total_truths += len(truth)
        unrankable_truths += len(truth - candidates)
        ordered = sorted(rankings[d], key=lambda pair: (-pair[1], pair[0]))
        recs = [drug for drug, _ in ordered[:max(ks)]]
        result = {'disease_id': d, 'n_truth': len(truth), 'truth_outside_candidates': len(truth-candidates), 'n_recommended_up_to_100': len(recs)}
        for k in ks:
            top = recs[:k]
            hits = sum(x in truth for x in top)
            dcg = sum(1.0/math.log2(i+2) for i, x in enumerate(top) if x in truth)
            idcg = sum(1.0/math.log2(i+2) for i in range(min(k, len(truth))))
            result[f'recall@{k}'] = hits / len(truth)
            result[f'ndcg@{k}'] = dcg / idcg
            distinct[k].update(top)
        per_disease.append(result)
    n = len(per_disease)
    summary = {'stage': args.stage, 'fit': args.fit, 'evaluation_diseases': n, 'candidate_drugs': len(candidates), 'positive_pairs': total_truths, 'positives_outside_candidates': unrankable_truths, 'diseases_with_any_recommendation': sum(r['n_recommended_up_to_100'] > 0 for r in per_disease), 'ignored_score_rows': ignored, 'runtime_seconds': args.runtime_seconds}
    summary['query_coverage'] = summary['diseases_with_any_recommendation'] / n
    for k in ks:
        for metric in ('recall', 'ndcg'):
            name = f'{metric}@{k}'
            summary[name] = sum(r[name] for r in per_disease)/n
        summary[f'catalog_coverage@{k}'] = len(distinct[k])/len(candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir/'summary.json', args.output_dir/'per_disease.csv']
    if any(path.exists() for path in paths):
        raise FileExistsError('빈 평가 출력 폴더를 사용하세요.')
    with paths[0].open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with paths[1].open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_disease[0]))
        writer.writeheader()
        writer.writerows(per_disease)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
