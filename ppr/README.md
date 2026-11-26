# PPR 기반 약물 재창출 후보 추천

질병–약물, 질병–유전자, 약물–유전자 관계로 그래프를 만들고, 대상 질병에서 시작하는 Personalized PageRank로 후보 약물의 순위를 계산한다.

## 파일 구성

- `prepare_inputs.py` — 입력 열 정리, 공통 전처리 실행, 분할 파일 해시 확인
- `run_ppr.py` — Validation 설정 선택, Train + Validation 재학습, Test 평가
- `requirements.txt` — 실행에 사용한 패키지 버전
- `results/` — 설정 비교와 최종 평가 요약
- `preliminary/` — 공통 평가 도입 전의 예비 실험 노트북

공통 전처리와 평가는 저장소 최상위 `common/`의 코드를 사용한다.

## 데이터 준비

[OptimusKG V3.2](https://doi.org/10.7910/DVN/IYNGEV)의 팀 공통 입력을 사용한다.

필요한 파일은 다음과 같다.

- `indication.csv` — `disease_id`, `drug_id`를 포함하는 적응증
- `disease_gene_t03.csv` — `disease_id`, `gene_id`, `score` 또는 `evidence_score`
- `drug_gene.csv` — `drug_id`, `gene_id`
- 적응증·금기 충돌 확인에 사용하는 원본 Parquet — `from`, `to`, `relation`, `properties` 열과 `relation_assertions`

입력 CSV는 팀이 준비한 전처리 파일이다. 아래 스크립트는 원본 전체 다운로드와 세 CSV의 최초 추출을 수행하지 않는다.

질병–유전자 관계는 근거 점수 0.3 이상을 사용한다. 적응증과 금기가 겹치는 339쌍을 제거한 뒤, 공통 seed 42 분할을 적용한다.

| 구분 | 관계 수 |
|---|---:|
| Train | 35,068 |
| Validation | 11,290 |
| Test | 11,477 |

입력 데이터와 개별 추천 결과 파일은 저장소에 포함하지 않는다.

## 실행 방법

기존 기록은 Python 3.13, NumPy 2.4.6, SciPy 1.18.0, PyArrow 26.0.1로 생성했다. 이번 저장소 정리본의 재실행 검증은 Python 3.14와 동일한 패키지 버전으로 수행했다. 저장소 최상위 폴더에서 실행한다.

```bash
python -m pip install -r ppr/requirements.txt

python ppr/prepare_inputs.py \
  --data-dir /path/to/processed \
  --parquet /path/to/drug_disease.parquet

python ppr/run_ppr.py \
  --output-dir /path/to/new_results \
  --workers 4
```

경로는 각자 환경에 맞게 바꾼다. 입력 준비 결과는 기본적으로 `ppr/data/`에 저장한다. 다른 위치를 쓰려면 입력 준비 명령에 `--output-dir /path/to/prepared`를, PPR 실행 명령에 `--data-dir /path/to/prepared`를 지정한다.

이미 준비한 공통 입력은 `splits/`, `features/`, `verified_hashes.json` 구조로 두고 `run_ppr.py --data-dir`로 사용할 수 있다. 실행 전에 네 입력 파일의 실제 해시를 기록된 해시와 비교한다.

입력 준비와 PPR 실행 모두 새 출력 폴더를 사용한다. 기존 폴더는 덮어쓰지 않는다. PPR 실행의 기본 출력은 `ppr/run_results/`이며, 저장소의 기록용 `ppr/results/`와 구분한다.

## 추천 방법

기본 PPR은 연결된 이웃으로 동일한 확률로 이동한다.

관계 가중 PPR은 질병에 약물과 유전자 이웃이 모두 있을 때, 약물 방향과 유전자 방향에 서로 다른 이동 비중을 준다. 최종 선택된 약물 방향 비중은 0.95다.

각 평가 단계의 학습 데이터에 등장한 약물을 후보로 사용한다. 대상 질병에 이미 알려진 약물과 금기 충돌 쌍을 제외하고, 양수 점수 후보를 상위 100개까지 저장한다.

## 설정 선택과 평가

먼저 Train으로 그래프를 만들고, Validation의 질병별 평균 NDCG@10으로 설정을 선택한다. 동점이면 Recall@10을 비교한다.

- 이동 확률 α 후보 — 0.50, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95
- 약물 방향 비중 후보 — 0.50, 0.75, 0.90, 0.95

기본 PPR의 α를 먼저 선택한 뒤, 해당 α에서 약물 방향 비중을 비교했다. 선택된 값은 α=0.65, 약물 방향 비중=0.95다.

설정을 고정한 후 Train + Validation으로 그래프를 다시 만들고 Test를 평가했다. Test는 설정 선택에 사용하지 않았다.

## 최종 Test 결과

| 방법 | Recall@10 | NDCG@10 | Query coverage |
|---|---:|---:|---:|
| Popularity | 0.060486 | 0.057270 | 1.000000 |
| 기본 PPR | 0.301107 | 0.266193 | 0.998796 |
| 관계 가중 PPR | 0.311628 | 0.283385 | 0.998796 |

평가 대상은 1,661개 질병이다. PPR은 그중 1,659개에 추천을 생성했다. 추천하지 못한 질병도 성능 평균에 0점으로 포함했다.

관계 가중 PPR이 기본 PPR보다 Recall@10과 NDCG@10에서 높은 결과를 보였다. 약물 방향 비중 0.95는 비교한 범위의 최댓값이며, 그보다 큰 값은 이번 탐색에서 확인하지 않았다.

## 결과 파일

저장소의 `results/`에는 기존 최종 실행의 요약을 모았다.

- `validation_comparison.csv` — Validation 설정별 성능
- `selected_config.json` — 최종 선택 설정
- `protocol.json` — 탐색 조건, 입력 해시, 시간 측정 범위
- `test/` — 기본 PPR, 관계 가중 PPR, Popularity의 최종 평가 요약

새로 실행하면 지정한 출력 폴더에 `comparison.csv`와 `metrics.json`으로 전체 Validation·Test 요약을 저장한다. 단계별 하위 폴더에는 `scores.csv`, `evaluation/summary.json`, `evaluation/per_disease.csv`를 생성한다. 실행 시간은 환경과 실행마다 달라질 수 있다.

`preliminary/`의 결과는 데이터 분할과 평가 조건이 다르므로 최종 비교표에는 위 공통 평가 결과를 사용한다.

## 출처 및 AI 사용

데이터는 OptimusKG V3.2를 사용했다. 공통 전처리·평가 코드는 팀에서 공유한 코드를 사용했다. PPR은 NumPy와 SciPy 희소 행렬로 구현했다.

OpenAI Codex의 도움을 받아 2026년 11월에 PPR 코드와 문서를 작성·수정했다. 입력 파일 해시 대조, 평가 정답의 그래프 연결 제외 확인, 공통 평가기 실행으로 결과를 검증했다. 원래 코드 작성 당시의 정확한 모델 식별자는 기록하지 않았다.
