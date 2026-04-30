| model | validation | n_folios | accuracy | macro_F1 | weighted_F1 | balanced_accuracy | macro_F1_permutation_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority_class_baseline | stratified_5_fold | 205 | 0.629268 | 0.128743 | 0.486081 | 0.166667 | not applicable |
| stratified_random_baseline | stratified_5_fold | 205 | 0.429268 | 0.126413 | 0.413734 | 0.133333 | not applicable |
| nearest_centroid_structural | stratified_5_fold | 205 | 0.756098 | 0.640346 | 0.792454 | 0.716551 | 0.000999001 |
| knn3_structural | stratified_5_fold | 205 | 0.814634 | 0.554601 | 0.799654 | 0.548851 | 0.000999001 |
| token_count_only_baseline | stratified_5_fold | 205 | 0.673171 | 0.526676 | 0.723549 | 0.550881 | 0.000999001 |
| logistic_regression | stratified_5_fold | 205 | 0.84878 | 0.740507 | 0.863619 | 0.79168 | 0.000999001 |
| random_forest | stratified_5_fold | 205 | 0.902439 | 0.734052 | 0.88363 | 0.691325 | 0.000999001 |
| linear_svm | stratified_5_fold | 205 | 0.873171 | 0.741812 | 0.87737 | 0.763224 | 0.000999001 |
