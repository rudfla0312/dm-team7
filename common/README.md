# 공통 데이터 준비 및 평가

이 폴더에는 팀 전체가 같이 쓰는 데이터 준비 코드와 평가 코드가 들어 있습니다. 처음에는 각자 분할 방식과 전처리가 달라서 점수를 그대로 비교하기 어려웠습니다. 그래서 최종 비교에서는 모든 방법이 아래 조건을 똑같이 쓰도록 맞췄습니다.

각 알고리즘의 구현(PPR, CBF, Louvain, SVD)은 담당자 폴더에 있고, 아래에서는 공통 데이터 준비와 평가 스크립트의 사용법을 설명합니다. 여기서 설명하는 데이터 준비·평가 스크립트 3개는 Python 표준 라이브러리만 사용하므로 따로 설치할 패키지는 없습니다.

## 데이터와 분할

모든 방법은 같은 seed 42 분할과 같은 평가 코드를 사용합니다. 원본 적응증은 58,174쌍인데, 적응증이면서 동시에 금기로도 기록된 339쌍은 어느 쪽이 맞는지 알 수 없어서 제외했고, 남은 57,835쌍을 씁니다.

| 구분 | 관계 수 | 비율 |
|---|---:|---:|
| Train | 35,068 | 약 60.6% |
| Validation | 11,290 | 약 19.5% |
| Test | 11,477 | 약 19.8% |

분할은 질병마다 따로, 정해진 해시 순서로 나눕니다. Validation과 Test는 각각 한 질병 약물 쌍의 20% 정도를 목표로 하고, 가능하면 Train에 최소 두 쌍은 남깁니다. 약물이 너무 적은 질병은 Train에만 들어갈 수도 있습니다.

유전자 정보가 없는 질병이나 범위가 넓은 질병 범주도 분할에서 빼지 않았습니다. 이런 질병에서 추천을 못 하는 방법은 아래 평가 규칙에 따라 처리됩니다.

## 실행 순서

1. Train으로 학습하고, Validation 점수로 하이퍼파라미터 같은 설정을 고릅니다.
2. 설정을 고정한 뒤 Train + Validation을 합쳐서 다시 학습합니다.
3. 이 모델로 Test 추천 점수를 새로 만들어서 공통 평가 코드에 넣습니다.

Test는 설정을 고르는 데 쓰지 않습니다. Validation 점수를 만들 때와 Test 점수를 만들 때 학습 데이터가 다르다는 점만 주의해 주세요.

## 데이터 준비

아래 명령은 저장소의 common 폴더에서 실행합니다.

아래 경로는 각자 환경에 맞게 바꿔서 실행합니다. 이미 공유받은 분할 파일이 있으면 다시 만들 필요는 없습니다. 같은 출력 폴더에 이전 결과 파일이 남아 있으면 오류가 나므로, 다시 만들 때는 빈 폴더를 쓰세요.

```bash
python prepare_splits.py --indications /path/to/indication.csv --output-dir /path/to/splits --seed 42
python prepare_features.py --disease-gene /path/to/disease_gene_t03.csv --drug-gene /path/to/drug_gene.csv --output-dir /path/to/features
```

입력 파일에 필요한 열은 다음과 같습니다.

- `indication.csv`: `disease_id`, `drug_id`, `also_asserted_contraindication`
- 질병–유전자 파일: `disease_id`, `gene_id`, `evidence_score`. 열 이름이 `score`로 되어 있으면 실행 전에 `evidence_score`로 바꿔 주세요.
- 약물–유전자 파일: `drug_id`, `gene_id`

질병–유전자 관계는 증거 점수가 일정 값 이상인 것만 남깁니다. 기본값은 0.3이며(`--threshold`로 바꿀 수 있습니다), 팀 최종 비교에서는 이 값을 사용합니다. 준비가 끝나면 질병–유전자 810,225개, 약물–유전자 36,611개가 나와야 합니다. 숫자가 다르면 입력 파일이 다른 버전일 가능성이 큽니다.

공유 분할 파일의 SHA-256은 다음과 같습니다.

```text
team_split.csv                 7514f8a689313d6f01827f1ddffe9b527334784132eb4a36edf406d65cb2d380
contraindication_conflicts.csv fdb96c5bc6774ab1ea0c63ed247cc89edcc7565f127b21ec572af7355bf385ce
```

내 컴퓨터에서 만든 파일의 해시가 위와 다르면 먼저 원본 입력 파일이 같은 것인지, 분할 생성 조건(seed 등)이 같은지 확인하세요. 그래도 다르면 행 순서나 줄바꿈 문자(CRLF/LF)가 달라서 생기는 경우가 있습니다.

정제된 데이터와 이 코드로 만든 파생 데이터 파일은 저장소에 포함하지 않았습니다. 기존에 공유한 공통 입력 파일을 사용하며, 다시 생성할 때는 필요한 입력 파일(전처리된 CSV와 충돌 표시 등)을 준비한 뒤 위 명령을 실행합니다.

## 추천 점수 파일 형식

각 알고리즘은 아래 형식의 UTF-8 CSV를 만들면 됩니다. 점수가 클수록 높은 순위이고, 같은 질병–약물 쌍은 한 번만 적어야 합니다.

```csv
disease_id,drug_id,score
DOID_1,CHEMBL1,0.85
DOID_1,CHEMBL2,0.32
```

Validation용과 Test용 파일은 따로 만듭니다.

## 평가

```bash
python evaluate.py --split /path/to/splits/team_split.csv --conflicts /path/to/splits/contraindication_conflicts.csv --scores /path/to/validation_scores.csv --stage validation --output-dir /path/to/validation_result
python evaluate.py --split /path/to/splits/team_split.csv --conflicts /path/to/splits/contraindication_conflicts.csv --scores /path/to/test_scores.csv --stage test --output-dir /path/to/test_result
```

평가에서 적용되는 규칙은 다음과 같습니다.

- **후보 약물**: Validation 단계에서는 Train에 나온 약물, Test 단계에서는 Train + Validation에 나온 약물만 후보가 됩니다.
- **제외 대상**: 각 질병의 학습 적응증과 금기 충돌 쌍은 추천에서 제외됩니다.
- **정답이 있는 질병**: 해당 단계에 정답이 있는 질병만 평가합니다. 추천을 하나도 못 한 질병도 0점으로 평균에 포함됩니다.
- **후보에 없는 정답**: 후보에 없어서 추천할 수 없었던 정답 약물도 지표의 분모에는 그대로 남습니다.
- **동점**: 약물 ID 순서로 처리합니다.
- **인기순 보완**: 주 실험에서는 쓰지 않습니다.

주 지표는 질병별 Recall@10과 NDCG@10의 평균입니다. 이 외에 @50, @100, query coverage, catalog coverage도 같이 저장합니다.

점수 파일에 없는 약물은 평가기가 임의로 점수를 채우지 않습니다. 상위 몇 개만 저장해서 파일 크기를 줄일 수는 있지만, 학습 적응증과 충돌 쌍이 빠진 뒤에도 @100까지 볼 수 있을 만큼 여유 있게 저장해야 합니다. 몇 개까지 저장했는지는 결과와 함께 적어 주세요.

실행 시간도 기록하고 싶으면 `--runtime-seconds`로 넘기면 됩니다. 이때 학습과 추론 시간을 같은 범위로 재야 방법끼리 비교할 수 있습니다.

## 결과 파일

평가기는 출력 폴더에 두 파일을 만듭니다.

- `summary.json`: 지표 평균, coverage, 실행 시간
- `per_disease.csv`: 질병별 지표

## 팀 비교 때 같이 적어 둘 것

최종 표를 만들 때는 점수만 올리지 말고 아래 내용을 같이 기록해 주세요. 나중에 보고서에서 조건을 설명할 때 필요합니다.

- 최종으로 고른 설정과 그 값을 고른 기준
- 사용한 입력 파일 (유전자 관계를 썼는지 등)
- 추천 점수를 몇 개까지 저장했는지
- 실행 환경(Python과 주요 패키지 버전)과 실행 시간 측정 범위

## 결과를 해석할 때 주의할 점

이 평가는 이미 알려진 적응증 일부를 숨겨 두고 다시 찾아내는지 보는 것입니다. 점수가 높다고 해서 새로운 약물 사용이 실제로 효과가 있다는 뜻은 아니고, 기록되지 않은 쌍이 효과가 없다는 뜻도 아닙니다. 보고서에서도 임상적으로 검증된 결과처럼 쓰지 않도록 주의합니다.
