# CBF 기반 약물 재창출 후보 추천

질병–유전자 연관성으로 Cosine Similarity를 계산하고, 유사 질병의 **학습 적응증**을 이용해 약물 후보를 순위화한다. 약물–유전자 관계는 CBF 점수 계산에 사용하지 않는다.

## 파일 구성

- `prepare_inputs.py` — 팀 공통 CSV 검증, 입력 열 정리 및 공통 특징 전처리
- `cbf_helpers.py` — Raw/IDF 특징 벡터, 코사인 유사도, 이웃 질병 및 약물 점수 계산
- `run_cbf.py` — 12개 Validation 설정 비교, Train+Validation 기반 Test 평가, 결과 저장
- `cbf_diagnostics.py` — Validation paired bootstrap 및 추천 실패 분석
- `CBF_Final.ipynb` — Colab에서 동일한 Python 코드를 실행·확인하는 노트북
- `requirements.txt` — 실험에 사용한 Python 패키지 버전
- `tests/` — 추천 점수 손계산·진단 코드 테스트
- `results/` — 설정 비교·프로토콜·최종 공통 평가 요약

전처리와 평가는 저장소 최상위 `common/prepare_features.py`, `common/evaluate.py`를 그대로 호출한다. `cbf/`와 `common/`은 같은 상위 폴더에 있어야 한다.

## 데이터 준비

[OptimusKG V3.2](https://doi.org/10.7910/DVN/IYNGEV)의 팀 공통 입력을 사용한다. CSV는 로컬에 두며 GitHub에 넣지 않는다.

| 입력 파일 | 역할 |
|---|---|
| `team_split.csv` | 공통 seed 42 Train 35,068 / Validation 11,290 / Test 11,477 |
| `contraindication_conflicts.csv` | 적응증과 금기가 겹쳐 제거한 339쌍 |
| `team_eligible_indications.csv` | 정제된 적응증 57,835쌍의 무결성 대조(선택 입력) |
| `disease_gene_t03.csv` | 질병–유전자 연관성 점수 0.3 이상, 810,225쌍 |
| `drug_gene.csv` | 팀 공통 특징 입력; CBF 점수에는 사용하지 않음 |

`prepare_inputs.py`는 `score` 열을 `evidence_score`로 정리하고 팀 분할·충돌 파일의 SHA-256을 확인한다. 원본 Parquet에서 CSV를 최초 추출하는 작업은 `common/`과 팀 데이터 준비 절차를 따른다. **810,225쌍 검사와 고정 해시는 OptimusKG V3.2 팀 확정 입력에 맞춘 것**이다.

## 실행 방법

실험 환경: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0, Matplotlib 3.10.8. 저장소 **최상위 폴더**에서 실행한다.

```bash
python -m pip install -r cbf/requirements.txt

python cbf/prepare_inputs.py \
  --source-dir /path/to/team_csv \
  --output-dir /path/to/cbf_prepared

python cbf/run_cbf.py \
  --data-dir /path/to/cbf_prepared \
  --output-dir /path/to/cbf_run

python cbf/tests/test_cbf_math.py
python cbf/tests/test_cbf_diagnostics.py
```

각 출력 디렉터리는 **새 디렉터리 또는 빈 디렉터리**여야 한다. `--source-dir`에는 위 팀 공통 입력 CSV를 배치한다. 실행 후 `/path/to/cbf_run/results/`가 아래 GitHub `cbf/results/`와 **같은 폴더 구조**로 만들어진다. 세부 추천 점수와 질병별 평가 CSV는 `cbf_run/`에 별도로 보존된다.

## 추천 방법과 평가

- Raw 또는 **score × IDF** (`IDF(g)=ln((N+1)/(df(g)+1))+1`)를 적용하고 L2 정규화한 뒤 코사인 유사도를 계산한다. N과 df는 전체 질병–유전자 입력 기준이다.
- 자기 자신을 제외하고 양의 유사도와 학습 적응증이 있는 질병 상위 `k`개를 선택한다. 약물 점수는 이웃 유사도의 `p`제곱을 합산한다.
- **12개 설정:** Raw/IDF × `k={10,20,50}` × `p={1,2}`. Validation macro NDCG@10, 동점이면 Recall@10, 그다음 작은 p·k와 Raw 우선으로 선택한다.
- Validation에는 Train만, 최종 Test에는 Train+Validation만 사용한다. 추천 대상 질병의 학습 적응증과 금기 충돌은 제외하고, 양의 점수 약물만 최대 100개 추천한다. 동점은 약물 ID 순서다.
- **Strict CBF**에는 Popularity fallback을 사용하지 않는다. 추천이 없는 질병은 공통 평가기에서 0점으로 계산한다. 인기순 추천은 별도 Baseline이다.

## 최종 Test 결과

| 방법 | Recall@10 | NDCG@10 | Query coverage |
|---|---:|---:|---:|
| CBF (IDF, k=50, p=2) | **0.185250** | **0.178217** | **0.859121** |
| Popularity | 0.060486 | 0.057270 | 1.000000 |

공통 Test의 평가 질병은 1,661개다. CBF는 유전자 특징이 없는 **234개 질병**에 추천을 생성하지 못했다. Train+Validation 후보 카탈로그 밖에 있는 Test 정답 **933쌍**도 Recall의 분모에 포함된다. CBF의 Catalog Coverage@10은 **0.219415**로 Popularity의 약 **0.005185**보다 높았다.

## 결과 파일

기존 공통 Test 결과는 `results/`에 정리되어 있다.

- `selected_config.json` — Validation에서 선택된 설정
- `validation_comparison.csv`, `cbf_validation_sensitivity.png` — 12개 설정 비교와 그래프
- `validation_paired_bootstrap.json` — 동일 Validation 질병의 paired bootstrap 비교
- `validation_test_context.json` — Validation/Test 데이터 규모와 NDCG 차이
- `protocol.json` — 시드, 입력 SHA-256, 라이브러리 버전, 실행 시간 범위
- `test/cbf_summary.json`, `test/popularity_summary.json` — 최종 Test 성능
- `test/test_by_group.csv`, `test/test_diagnostics.json` — 특징 정보별 성능과 추천 실패 분석

선정된 설정의 Validation NDCG@10은 **0.146167**, Test NDCG@10은 **0.178217**이다. Test는 적응증 46,358쌍으로 재학습하며(Validation 학습 35,068쌍), 평가 질병도 달라지므로 상승 원인을 단정할 수 없다. Bootstrap 신뢰구간은 주요 비교에서 0을 포함하므로 IDF 설정의 통계적 우월성을 주장하지 않는다.

**한계:** 전체 유전자 입력에는 Test 질병도 포함되므로 IDF는 *transductive* 특징 통계다(숨긴 적응증 라벨은 사용하지 않음). 유전자 근거가 기존 임상 정보를 간접 반영할 수 있고, 조밀한 질병 유사도 행렬은 큰 데이터로 확장할 때 비용이 증가한다. 이 평가는 기존 적응증의 복원 성능이지 약물의 임상 효능 검증이 아니다.

## 출처 및 외부 코드

[1] OptimusKG, Version 3.2, Harvard Dataverse, DOI: [10.7910/DVN/IYNGEV](https://doi.org/10.7910/DVN/IYNGEV) (accessed 2026-11-05).

[2] M. J. Pazzani and D. Billsus, “Content-Based Recommendation Systems,” *The Adaptive Web*, 2007, DOI: [10.1007/978-3-540-72079-9_10](https://doi.org/10.1007/978-3-540-72079-9_10).

[3] K. Järvelin and J. Kekäläinen, “Cumulated Gain-Based Evaluation of IR Techniques,” *ACM TOIS*, 2002, DOI: [10.1145/582415.582418](https://doi.org/10.1145/582415.582418).

NumPy, SciPy, pandas, scikit-learn, Matplotlib의 공개 API를 사용한다.

`common/`은 팀 공통 전처리·평가 코드이며 `cbf/`는 CBF 담당 구현이다. 실험 버전은 `requirements.txt`와 `results/protocol.json`에 기록했다.

## AI 사용

2026년 11월 OpenAI ChatGPT를 코드 작성·수정 및 README/보고서 초안에 사용했다. 여러 세션의 정확한 모델 버전은 일괄 확인할 수 없어 특정 버전으로 단정하지 않았다. 손계산 테스트, 팀 입력 해시, 공통 평가기 교차 검증 및 반복 실행으로 결과를 확인했다.
