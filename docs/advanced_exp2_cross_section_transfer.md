# Advanced Experiment 2

Cross-section transfer evaluates structural similarity by training a PCS model on one section and testing it on another. Better transfer is interpreted only as structural similarity in PCS token formation, not semantic similarity.

## Best transfer pairs

| source_section | target_section | delta_vs_target_own_model |
| --- | --- | --- |
| herbal | cosmological | -0.0662504 |
| cosmological | astronomical_zodiac | -0.00302219 |
| unknown | cosmological | 0.145294 |
| recipes_stars | cosmological | 0.295744 |
| herbal | astronomical_zodiac | 0.363536 |

## Weakest transfer pairs

| source_section | target_section | delta_vs_target_own_model |
| --- | --- | --- |
| astronomical_zodiac | biological_balneological | 4.92682 |
| pharmaceutical | biological_balneological | 4.12757 |
| astronomical_zodiac | recipes_stars | 3.0688 |
| cosmological | biological_balneological | 3.02925 |
| pharmaceutical | recipes_stars | 2.6893 |
