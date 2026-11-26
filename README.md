# dm-team7

## Repository layout

- common/: shared split preparation, feature preparation and evaluation.
- cbf/: content-based filtering.
- louvain/: Louvain runner, helpers and community-boundary analysis.
- ppr/: final common-evaluation PPR code and recorded results; see [ppr/README.md](ppr/README.md).
- svd/: normalized SVD runner.

## Reproducing the shared split and Louvain/SVD results

Use the team's processed OptimusKG v3.2 inputs. The [source dataset](https://doi.org/10.7910/DVN/IYNGEV)
is available separately from this repository. Put `indication.csv`, `disease_gene_t03.csv`, and
`drug_gene.csv` in `data/`. Use the same processed release and do not commit local data or outputs.

With Python 3.12, run these commands from the repository root:

```sh
python common/prepare_splits.py --indications data/indication.csv --output-dir outputs/splits --seed 42
python common/prepare_features.py --disease-gene data/disease_gene_t03.csv --drug-gene data/drug_gene.csv --output-dir outputs/features
python -m pip install -r louvain/requirements.txt -r svd/requirements.txt
python louvain/run_louvain.py --split-dir outputs/splits --processed data --output outputs/louvain
python svd/run_svd.py --split-dir outputs/splits --processed data --output outputs/svd
python louvain/compare_louvain_boundary.py --split-dir outputs/splits --processed data --output outputs/boundary
```

The shared split has 35,068 train, 11,290 validation and 11,477 test pairs.
SHA-256 for `team_split.csv` is `7514f8a689313d6f01827f1ddffe9b527334784132eb4a36edf406d65cb2d380`.
SHA-256 for `contraindication_conflicts.csv` is `fdb96c5bc6774ab1ea0c63ed247cc89edcc7565f127b21ec572af7355bf385ce`.
Use a new output directory for each run. The feature preparation command is optional for the
Louvain runs shown above because they read the original, already thresholded disease–gene CSV directly.
SVD uses the indication split and does not read gene features; its --processed option is retained for consistent command arguments.

To evaluate a separate implementation, save scores as a CSV with
`disease_id,drug_id,score` and run:

```sh
python common/evaluate.py --split outputs/splits/team_split.csv --conflicts outputs/splits/contraindication_conflicts.csv --scores path/to/scores.csv --stage test --output-dir outputs/my_test
```

Use `--stage validation` for predictions fitted on Train only. Use `--stage test`
for predictions fitted on Train + Validation after choosing settings on Validation.
Mask fitting indications and conflict pairs before truncating any score file to 100 candidates.
The evaluator writes aggregate and per-disease results. The boundary analysis writes
`boundary_comparison.json` with retention and gene-count summaries.

OpenAI Codex assisted with the shared evaluation and experiment scripts in November 2026.
The comparison was checked against the original within-community Louvain results.
