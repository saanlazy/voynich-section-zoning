# Section-Specific Structural Zoning in the Voynich Manuscript: Global versus Section-Conditioned Models of Token Formation

**Author:** Youngsan Chang  
**Version:** 2026g / GitHub-ready source-based corrected version  
**Package status:** GitHub-ready and Zenodo-deposition-ready; DOI placeholders remain until deposition.

## Summary

This package supports a section-specific extension of the global multi-level framework presented in Chang 2026f. It tests whether token-structural constraints vary across conventional visual/codicological manuscript sections. The study evaluates section-specific structural zoning, section-conditioned structural regimes, global versus section-specific PCS models, cross-section PCS transfer, inventory/order decomposition, section classification, folio-order change-point analysis, unknown-folio structural proximity, and robustness checks.

The study does **not** claim decipherment, translation, source-language identification, confirmed syntax, or semantic section identification. Section labels such as herbal, astronomical/zodiac, biological/balneological, cosmological, pharmaceutical, and recipes/stars are treated as conventional visual/codicological labels only.

## Package Structure

| Path | Role |
| --- | --- |
| `manuscript/` | Latest available DOCX/PDF manuscript files copied without rewriting. |
| `data/raw/` | Raw EVA transcription input where available. |
| `data/metadata/` | Section metadata. |
| `data/processed/` | Processed feature matrices, classification outputs, transfer matrices, robustness outputs, and parsed corpus tables. |
| `scripts/` | Analysis, advanced experiment, classification update, and manuscript-build scripts where available. |
| `outputs/tables/` | Final body Tables 1–7 plus supplementary table assets. |
| `outputs/figures/` | Final body Figures 1–6 plus supplementary figure assets. |
| `outputs/qc/` | QC reports, validation logs, and package validation report. |
| `docs/` | Captions, result paragraphs, interpretation notes, insertion guides, and summary documentation. |

## Reproduction Procedure

Install the required Python packages listed in `requirements.txt` if present. The known project environment used:

- scikit-learn = 1.8.0
- pandas = 2.2.3
- numpy = 2.4.4
- scipy = 1.17.1
- matplotlib = 3.10.9

Typical execution order from the project root:

```bash
python scripts/paper6_section_zoning_pipeline.py
python scripts/paper6_advanced_experiments.py
python scripts/update_section_classification_models.py
```

Random seed: 42. Permutation/bootstrap/shuffle experiments use N = 1000 where available; fallbacks are documented in QC reports.


## External Required Data

`ZL3b-n.txt` is third-party source data and is not redistributed in this repository/package. To reproduce analyses from raw text, obtain the Zandbergen-Landini EVA transcription (ZL3b) from the original source and place it at:

```text
data/raw/ZL3b-n.txt
```

See `docs/DATA_SOURCE.md` for details.

## Main Inputs

- External required data: `data/raw/ZL3b-n.txt` (not redistributed; see `docs/DATA_SOURCE.md`)
- `data/metadata/section_metadata.csv`
- processed section feature matrices and classification outputs in `data/processed/`

## Main Outputs

### Final Tables 1-7

1. Section Corpus Summary
2. PCS Metrics by Section
3. Global versus Section-Specific PCS Model Comparison
4. Inventory versus Ordering Decomposition
5. Section Classification Results
6. Robustness Checks
7. Final Key Findings

### Final Figures 1-6

1. PCS Conditional Entropy by Section
2. Global versus Section-Specific PCS Model Performance
3. Cross-Section Transfer Heatmap
4. Inventory versus Ordering Contribution by Section
5. Section Classification Confusion Matrix
6. Advanced Structural Zoning Model Summary

## Main Analyses

- section-level token profile
- PCS dependency by section
- global versus section-specific PCS model comparison
- cross-section transfer
- inventory/order decomposition
- section classification
- folio-order change-point analysis
- unknown-folio proximity
- robustness checks

## AI Use Disclosure

AI-based tools assisted with code implementation, figure generation, manuscript editing, and package organization. Research design, analytical decisions, statistical interpretation, and final argumentation were conducted by the author.

## DOI / Zenodo Status

Zenodo package concept DOI: https://doi.org/10.5281/zenodo.19915433

The reproduction package concept DOI has been assigned by Zenodo. No fabricated DOI values are used.

## Citation

See `CITATION.cff`. If no DOI has been assigned yet, cite the manuscript title, author, version, and eventual Zenodo archive once available.

## Interpretation Guardrails

Allowed: section-specific structural zoning, section-conditioned structural regimes, structural similarity/divergence, local token-formation regimes.  
Disallowed: decipherment, translation, source-language identification, confirmed syntax, semantic section identification.
