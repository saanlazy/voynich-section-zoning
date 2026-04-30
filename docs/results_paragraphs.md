# Paper 6 Results Paragraphs

## 1. Section corpus variation summary
The section-level corpus summary indicates that token counts, type-token ratios, token entropy, and token length profiles vary across conventional visual/codicological sections. The largest parsed section by token count is `herbal`, but all section labels should be interpreted as descriptive manuscript labels rather than semantic categories.

## 2. Token distribution divergence summary
Pairwise token-distribution distances show that section profiles are not uniform across the manuscript. Jensen-Shannon divergence and cosine distance provide evidence for section-specific token frequency regimes, while distinctive-token scores identify forms disproportionately concentrated within individual sections.

## 3. PCS dependency variation summary
PCS metrics indicate that token-internal component dependencies differ by section. The lowest observed H(suffix|core) occurs in `biological_balneological`, and the highest suffix-from-core top-1 accuracy occurs in `biological_balneological`, supporting section-specific variation in component constraints without assigning meaning to those sections.

## 4. Line-position constraint variation summary
Line-initial, line-final, and paragraph-initial concentration scores vary across sections. These results extend positional-constraint analysis to section-level profiles and support structural zoning at manuscript-region scale.

## 5. Inter-token variation summary
Inter-token metrics show that local sequential organization also varies by section. The lowest real bigram entropy occurs in `astronomical_zodiac`, and real/baseline contrasts should be interpreted as evidence for local ordering differences rather than syntax or translation.

## 6. Family clustering variation summary
PCS-defined family clustering differs across sections. The strongest same-core local clustering relative to shuffled baseline occurs in `unknown`, suggesting local organization of related token forms without implying thematic or semantic grouping.

## 7. Integrated structural zoning summary
The integrated feature profile combines token, PCS, positional, inter-token, and family-clustering metrics into section-level structural profiles. The resulting distance matrix and PCA visualization support the presence of local regimes of token formation and arrangement across manuscript regions.

## 8. Conservative interpretation paragraph
The results indicate section-specific variation in token organization, but they do not assign semantic labels to manuscript sections. The appropriate interpretation is that conventional visual/codicological section labels correlate with structural profiles, supporting structural zoning rather than decipherment.

## 9. Limitations paragraph
The analysis depends on ZL3b transcription conventions, automatic normalization choices, and section labels derived from explicit comments in the transcription. Unknown or low-certainty folios are left unassigned, low-count sections are marked cautiously, and the results do not establish translation, meaning, source-language identification, or confirmed syntax.
