# Paper 6 Advanced Experiment Summary

## 1. Purpose
The advanced experiments test whether section-conditioned structural models are required, or whether a single global model explains token formation and local arrangement across manuscript sections.

## 2. Input files used
- `data/ZL3b-n.txt`
- `data/section_metadata.csv`
- `outputs/section_pipeline/parsed_tokens_with_sections.csv`
- `outputs/section_pipeline/parsed_lines_with_sections.csv`

## 3. Advanced experiments executed
1. Global vs section-specific PCS model
2. Cross-section PCS transfer matrix
3. Inventory vs ordering decomposition
4. Section classification from structural features
5. Folio-order change-point detection
6. Unknown folio structural proximity
7. Robustness checks
8. Currier-style metadata confound check where available

## 4. Model definitions
PCS models use prefix length 2, suffix length 1, and core as the middle component. Empty-core tokens are excluded from PCS likelihood calculations. Add-alpha smoothing is used with alpha = 0.1.

## 5. Baselines
Global PCS, section-specific PCS, cross-section PCS transfer, line-internal shuffle, global shuffle, majority-class baseline, stratified random baseline, token-count-only baseline, and section-label permutation baseline.

## 6. Tables generated
Tables 9-18 are available in `outputs/advanced/tables/`.

## 7. Figures generated
Figures 9-16 are available in `outputs/advanced/figures/`.

## 8. Main results
- Best global-vs-section PCS delta: biological_balneological (0.6386).
- Best computed classifier: nearest_centroid_structural.
- Classifier macro-F1: 0.6429; label-permutation p=0.0010.
- Downsampled biological_balneological mean H(suffix|core): 0.2835.
- Unknown proximity rows generated: 22.

## 9. Strong findings
Entropy decomposition and model-comparison results provide the strongest basis for distinguishing inventory restriction from local ordering and for evaluating global versus section-conditioned PCS structure.

## 10. Moderate/descriptive findings
Classification, transfer, and change-point results are useful but should be presented as structural correlations with conventional visual/codicological labels.

## 11. Low-confidence findings
Low-certainty-label analyses and unknown-folio proximity are descriptive only.

## 12. Robustness outcomes
Robustness checks are summarized in Table 15. Results are labeled robust, descriptive, fragile, or low-confidence rather than over-interpreted.

## 13. Interpretation rules
The results do not establish decipherment, translation, source-language identification, semantic categories, or confirmed syntax.

## 14. Recommended manuscript structure
Place Experiments 1-3 in the main Results section, classification and change-point analyses as secondary Results, and unknown/Currier/robustness checks in a robustness or supplementary section.

## 15. Missing or failed items
Logistic regression, random forest, and linear SVM were not computed because earlier dependency-limited drafts lacked scikit-learn in the project environment. Manual nearest-centroid and KNN-style structural classifiers were computed instead.
