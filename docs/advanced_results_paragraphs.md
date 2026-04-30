# Advanced Results Paragraphs

## 1. Global vs section-specific PCS model result
The model-comparison results test whether a single global PCS model is sufficient for all manuscript regions. Positive cross-entropy deltas indicate sections where section-specific PCS models fit held-out tokens better than the global model, supporting section-conditioned token-formation regimes only where the improvement is stable across splits.

## 2. Cross-section transfer result
The cross-section transfer matrix shows which section-trained PCS models explain other sections relatively well or poorly. These transfer patterns are interpreted as structural similarity or divergence in token formation, not semantic similarity.

## 3. Inventory vs ordering decomposition result
The entropy decomposition separates inventory restriction from ordering effects. This distinction is important because low section entropy may arise from a restricted token inventory even when local ordering adds little additional constraint.

## 4. Section classification result
Structural features were evaluated for their ability to predict conventional visual/codicological section labels above baseline. Predictive success supports correlation between section labels and structural profiles, but does not imply semantic categorization.

## 5. Change-point / boundary result
Folio-order change-point scores estimate whether structural discontinuities align with conventional section boundaries. Alignment supports structural zoning; non-alignment indicates that visual/codicological labels and statistical structure are not identical.

## 6. Unknown folio proximity result
Unknown folios were compared with labeled-section centroids in structural feature space. The resulting nearest-centroid labels are proximity descriptors only and should not be treated as section identifications.

## 7. Robustness result
Robustness checks distinguish results that persist under unknown exclusion, label uncertainty, downsampling, and permutation baselines from results that remain descriptive or low-confidence.

## 8. Integrated advanced interpretation
Together, the advanced experiments test whether the Voynich corpus is better described by section-conditioned structural regimes than by a single global model. The appropriate claim is section-specific structural zoning in token formation and arrangement, not decipherment or confirmed syntax.

## 9. Limitations specific to advanced experiments
The experiments depend on ZL3b tokenization, conventional section labels, smoothing choices, feature selection, and finite section sample sizes. Unknown folios and low-certainty labels are handled conservatively.

## 10. Final conclusion paragraph
The advanced results support treating the Voynich Manuscript as structurally heterogeneous across conventional manuscript regions. The evidence favors local regimes of token formation and arrangement, while leaving semantic interpretation, translation, and source-language identification outside the scope of the analysis.
