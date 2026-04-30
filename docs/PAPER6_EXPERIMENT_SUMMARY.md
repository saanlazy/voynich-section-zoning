# Paper 6 Experiment Summary

## 1. Purpose
This experiment evaluates whether global multi-level constraints in the Voynich Manuscript vary across manuscript sections, folio groups, visual/codicological categories, and local zones where metadata permits.

## 2. Input Files
- Primary transcription: `/Users/admin/.gemini/antigravity/codex/paper6_section_zoning_work/data/ZL3b-n.txt`

## 3. Section Metadata
- Metadata file: `data/section_metadata.csv`
- Section labels are conventional visual/codicological labels extracted from explicit local comments only.
- Unknown or uncertain folios remain `unknown`.

## 4. Analyses Executed
- Section corpus summary
- Token distribution by section
- PCS component distribution by section
- Section-specific line-position effects
- Inter-token structure by section
- Token-family clustering by section
- Integrated section profile and clustering

## 5. Generated Tables
See `outputs/tables/` for Tables 1-8 in CSV, Markdown, and TXT formats.

## 6. Generated Figures
See `outputs/figures/` for Figures 1-8 in PNG and PDF formats.

## 7. Core Results, Ten-Line Summary
1. Largest parsed section by token count: herbal (11034 tokens).
2. Lowest H(suffix|core): biological_balneological (0.3823).
3. Highest suffix-from-core top-1 accuracy: biological_balneological (0.9028).
4. Lowest section bigram entropy: astronomical_zodiac (9.2390).
5. Highest next-token top-1 accuracy: astronomical_zodiac (0.7117).
6. Lowest suffix-to-next-prefix entropy: biological_balneological (3.3770).
7. Strongest same-core clustering ratio: unknown (0.8712).
8. Strongest line-initial concentration: biological_balneological (0.0800).
9. Strongest line-final concentration: herbal (0.1000).
10. Largest token-distribution divergence: astronomical_zodiac vs biological_balneological (JSD=0.7103).

## 8. Interpretive Cautions
The results should be described as section-specific structural variation, structural zoning, or local regimes of token formation and arrangement. They do not establish decipherment, translation, source-language identification, semantic categories, or confirmed syntax.

## 9. Missing / Low-Confidence Items
- Missing items: none
- Low-count sections are marked in metric tables.

## 10. Manuscript Drafting Recommendations
Use Table 8 as the principal results summary, Figure 8 as the compact zoning overview, and Figures 2-4 to connect section-level findings to the earlier PCS and inter-token framework.

## Output Counts
- CSV files: 25
- Markdown files: 17
- TXT files: 9
- PNG files: 8
- PDF files: 8
