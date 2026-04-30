# Advanced Model Validation Report

- Random seed: 42
- Requested iterations: 1000
- Executed iterations: 1000
- Fallback: none
- PCS segmentation: prefix length 2, suffix length 1, core = middle component; empty core excluded.
- Smoothing: add-alpha/Laplace-style smoothing with alpha = 0.1.
- Held-out splits: 80/20, 70/30, 50/50 for Experiment 1; 80/20 for transfer matrix.
- Baselines: global PCS, section-specific PCS, cross-section PCS transfer, line-internal shuffle, global shuffle, majority-class, stratified random, token-count-only, label permutation.
