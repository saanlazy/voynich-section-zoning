# Section Classification Model Update Report

## Environment

- scikit-learn version: 1.8.0
- Random seed: 42
- Unknown category excluded from classification: yes
- Validation: StratifiedKFold, k = 5
- Label permutation baseline: N = 1000; predictions are held fixed while true section labels are permuted to break label-prediction association.
- Empirical p-value formula: `(k + 1) / (N + 1)`

## Models Executed

- majority_class_baseline
- stratified_random_baseline
- nearest_centroid_structural
- knn3_structural
- token_count_only_baseline
- logistic_regression
- random_forest
- linear_svm

## Best Models

- Best by macro-F1: `linear_svm`
- Best by balanced accuracy: `logistic_regression`
- Best-model macro-F1 permutation p-value: `0.000999000999000999`

## Interpretation

Structural features predict conventional visual/codicological section labels above baseline. This is not semantic classification and does not identify section meaning.
