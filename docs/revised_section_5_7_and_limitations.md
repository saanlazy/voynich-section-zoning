# Revised Section 5.7, Section 6.6, and Limitations Text

## Revised Methods 5.7 paragraph

The classification experiment tested whether structural features predict conventional visual/codicological section labels above baseline. Unknown folios were excluded from the primary classification experiment and retained for separate proximity analysis. Models were evaluated under stratified 5-fold cross-validation with random seed 42. The comparison included majority-class and stratified-random baselines, a token-count-only baseline, nearest-centroid and kNN structural classifiers, logistic regression with balanced class weights, a balanced random forest with 500 trees, and a balanced linear SVM. Evaluation used accuracy, macro-F1, weighted-F1, balanced accuracy, per-class precision/recall/F1, and confusion matrices. Label-permutation tests used 1,000 permutations of section labels against fixed model predictions to estimate a macro-F1 null distribution. Because section labels are imbalanced, macro-F1 and balanced accuracy are emphasized over raw accuracy.

## Revised Results 6.6 paragraph

Structural features predicted conventional visual/codicological section labels above baseline. The best model by macro-F1 was `linear_svm`, and the best model by balanced accuracy was `logistic_regression`. The best-model macro-F1 permutation test gave p = 0.0010, using 1,000 label permutations and the empirical formula `(k + 1) / (N + 1)`. This result indicates that conventional section labels correlate with structural profiles, but it should not be interpreted as semantic classification or section meaning recovery.

## Revised Limitations paragraph

The classification experiment evaluates whether structural features predict conventional visual/codicological labels, not whether semantic categories have been identified. The unknown category was excluded from the primary classifier and analyzed separately as structural proximity only. Class imbalance remains important; therefore macro-F1, balanced accuracy, and per-class metrics should be read alongside raw accuracy. Model performance depends on the selected folio-level features, the ZL3b transcription, and the operational PCS segmentation.

## Revised Table 5 caption

Table 5. Section Classification Results. Classification performance for conventional visual/codicological section labels using structural features and baseline models. Unknown folios are excluded from the primary classification test. Macro-F1 and balanced accuracy are emphasized because section classes are imbalanced; successful prediction indicates structural correlation with labels, not semantic classification.

## Revised Figure 5 caption

Figure 5. Section Classification Confusion Matrix. Confusion matrix for the best structural classifier by macro-F1. The figure shows how structural features separate conventional visual/codicological labels, without implying semantic identification or confirmed section meaning.

## Remove from manuscript

- “Because earlier dependency-limited drafts lacked standard classifier results...”
- “standard classifier results should be refreshed with the updated classification script”
- “not computed in earlier dependency-limited drafts”
