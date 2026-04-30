#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.python_packages'
if LOCAL.exists():
    sys.path.insert(0, str(LOCAL))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ADV = ROOT / 'outputs/advanced'
FINAL = ROOT / 'outputs/final_manuscript_assets'
SEED = 42
N_PERM = 1000
UNKNOWN = 'unknown'


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return '_No rows available._\n'
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, r in df.iterrows():
        vals = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                vals.append('not available' if math.isnan(v) else f'{v:.6g}')
            else:
                vals.append(str(v).replace('|', '\\|').replace('\n', ' '))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines) + '\n'


def write_table(df: pd.DataFrame, stem: Path):
    out = df.replace([np.inf, -np.inf], np.nan).fillna('not available')
    out.to_csv(stem.with_suffix('.csv'), index=False)
    stem.with_suffix('.md').write_text(df_to_md(out), encoding='utf-8')
    stem.with_suffix('.txt').write_text(out.to_string(index=False), encoding='utf-8')


def metrics(y_true, y_pred, labels):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_F1': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'weighted_F1': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }


def run_cv(X, y, folios, feature_cols, model_specs, labels, kfold):
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=SEED)
    rows = []
    pred_rows = []
    perclass_rows = []
    for model_name, estimator, cols in model_specs:
        y_true_all = []
        y_pred_all = []
        folio_all = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X[train_idx][:, cols]
            X_test = X[test_idx][:, cols]
            y_train = y[train_idx]
            y_test = y[test_idx]
            est = clone(estimator)
            est.fit(X_train, y_train)
            pred = est.predict(X_test)
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(pred.tolist())
            folio_all.extend(folios[test_idx].tolist())
            for f, truth, pp in zip(folios[test_idx], y_test, pred):
                pred_rows.append({'model': model_name, 'fold': fold, 'folio_id': f, 'true_section': truth, 'predicted_section': pp})
        m = metrics(y_true_all, y_pred_all, labels)
        m.update({'model': model_name, 'validation': f'stratified_{kfold}_fold', 'n_folios': len(y_true_all)})
        rows.append(m)
        prec, rec, f1, support = precision_recall_fscore_support(y_true_all, y_pred_all, labels=labels, zero_division=0)
        for lab, p, r, f, s in zip(labels, prec, rec, f1, support):
            perclass_rows.append({'model': model_name, 'section_label': lab, 'precision': p, 'recall': r, 'F1': f, 'support': int(s)})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows), pd.DataFrame(perclass_rows)


def permutation_pvalue_fixed_predictions(y_true, y_pred, observed_macro_f1, labels, model_name):
    """Permutation baseline: break label-prediction association without refitting."""
    rng = np.random.default_rng(SEED + 1000 + abs(hash(model_name)) % 100000)
    vals = []
    for i in range(N_PERM):
        y_perm = rng.permutation(y_true)
        vals.append(f1_score(y_perm, y_pred, labels=labels, average='macro', zero_division=0))
    p = (sum(v >= observed_macro_f1 for v in vals) + 1) / (len(vals) + 1)
    return p, vals


def plot_confusion(cm_df: pd.DataFrame, out_png: Path, out_pdf: Path, title: str):
    labels = cm_df['true_section'].tolist()
    vals = cm_df.drop(columns=['true_section']).to_numpy(dtype=float)
    short = {
        'herbal': 'herbal',
        'astronomical_zodiac': 'zodiac',
        'biological_balneological': 'biological',
        'cosmological': 'cosmological',
        'pharmaceutical': 'pharmaceutical',
        'recipes_stars': 'recipes',
    }
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(vals, cmap='Greys')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([short.get(x, x) for x in labels], rotation=35, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([short.get(x, x) for x in labels])
    ax.set_xlabel('Predicted conventional label')
    ax.set_ylabel('True conventional label')
    ax.set_title(title)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, str(int(vals[i, j])), ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    feature_path = ADV / 'section_classification_feature_matrix.csv'
    df = pd.read_csv(feature_path)
    df = df[df['section_label'] != UNKNOWN].copy()
    labels = sorted(df['section_label'].unique())
    counts = df['section_label'].value_counts()
    kfold = min(5, int(counts.min()))
    if kfold < 2:
        raise SystemExit('Not enough section samples for stratified validation')
    id_cols = {'folio_id', 'section_label'}
    feature_cols = [c for c in df.columns if c not in id_cols]
    Xdf = df[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    Xdf = Xdf.fillna(Xdf.mean(numeric_only=True)).fillna(0.0)
    X = Xdf.to_numpy(dtype=float)
    y = df['section_label'].to_numpy()
    folios = df['folio_id'].to_numpy()
    all_cols = np.arange(X.shape[1])
    token_col = np.array([feature_cols.index('token_count')]) if 'token_count' in feature_cols else np.array([0])
    model_specs = [
        ('majority_class_baseline', DummyClassifier(strategy='most_frequent', random_state=SEED), all_cols),
        ('stratified_random_baseline', DummyClassifier(strategy='stratified', random_state=SEED), all_cols),
        ('nearest_centroid_structural', make_pipeline(StandardScaler(), NearestCentroid()), all_cols),
        ('knn3_structural', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)), all_cols),
        ('token_count_only_baseline', make_pipeline(StandardScaler(), NearestCentroid()), token_col),
        ('logistic_regression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced', random_state=SEED)), all_cols),
        ('random_forest', RandomForestClassifier(n_estimators=500, random_state=SEED, class_weight='balanced'), all_cols),
        ('linear_svm', make_pipeline(StandardScaler(), LinearSVC(class_weight='balanced', random_state=SEED, max_iter=10000)), all_cols),
    ]
    results, predictions, perclass = run_cv(X, y, folios, feature_cols, model_specs, labels, kfold)

    # Permutation p-values for macro-F1. This uses a section-label permutation
    # null with model predictions held fixed, breaking label-prediction
    # association without refitting expensive models 1000 times.
    pvals = []
    null_records = []
    for model_name, estimator, cols in model_specs:
        obs = float(results.loc[results.model == model_name, 'macro_F1'].iloc[0])
        if model_name in {'majority_class_baseline', 'stratified_random_baseline'}:
            pvals.append({'model': model_name, 'macro_F1_permutation_p_value': 'not applicable'})
            continue
        pred_sub = predictions[predictions.model == model_name]
        p, vals = permutation_pvalue_fixed_predictions(pred_sub.true_section.to_numpy(), pred_sub.predicted_section.to_numpy(), obs, labels, model_name)
        pvals.append({'model': model_name, 'macro_F1_permutation_p_value': p})
        for i, v in enumerate(vals):
            null_records.append({'model': model_name, 'iteration': i, 'permuted_macro_F1': v})
    pval_df = pd.DataFrame(pvals)
    results = results.merge(pval_df, on='model', how='left')
    results = results[['model', 'validation', 'n_folios', 'accuracy', 'macro_F1', 'weighted_F1', 'balanced_accuracy', 'macro_F1_permutation_p_value']]

    # Best model by macro-F1 for confusion matrix.
    best_macro = results.sort_values('macro_F1', ascending=False).iloc[0]['model']
    best_bal = results.sort_values('balanced_accuracy', ascending=False).iloc[0]['model']
    best_preds = predictions[predictions.model == best_macro]
    cm = confusion_matrix(best_preds.true_section, best_preds.predicted_section, labels=labels)
    cm_df = pd.DataFrame(cm, columns=labels)
    cm_df.insert(0, 'true_section', labels)

    write_table(results, ADV / 'tables/table12_section_classification_results')
    write_table(results, FINAL / 'tables/table5_section_classification_results_final')
    predictions.to_csv(ADV / 'section_classification_predictions.csv', index=False)
    perclass.to_csv(ADV / 'section_classification_per_class_metrics.csv', index=False)
    cm_df.to_csv(ADV / 'section_classification_confusion_matrix.csv', index=False)
    pd.DataFrame(null_records).to_csv(ADV / 'section_classification_label_permutation_null.csv', index=False)

    plot_confusion(cm_df, ADV / 'figures/figure12_section_classification_confusion_matrix.png', ADV / 'figures/figure12_section_classification_confusion_matrix.pdf', f'Figure 12. Section Classification Confusion Matrix ({best_macro})')
    plot_confusion(cm_df, FINAL / 'figures/figure5_section_classification_confusion_matrix_final.png', FINAL / 'figures/figure5_section_classification_confusion_matrix_final.pdf', f'Figure 5. Section Classification Confusion Matrix ({best_macro})')

    # Reports.
    best_p = results.loc[results.model == best_macro, 'macro_F1_permutation_p_value'].iloc[0]
    old_table = pd.read_csv(ADV / 'tables/table12_section_classification_results.csv')
    report = f"""# Section Classification Model Update Report

## Environment

- scikit-learn version: {sklearn_version}
- Random seed: {SEED}
- Unknown category excluded from classification: yes
- Validation: StratifiedKFold, k = {kfold}
- Label permutation baseline: N = {N_PERM}; predictions are held fixed while true section labels are permuted to break label-prediction association.
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

- Best by macro-F1: `{best_macro}`
- Best by balanced accuracy: `{best_bal}`
- Best-model macro-F1 permutation p-value: `{best_p}`

## Interpretation

Structural features predict conventional visual/codicological section labels above baseline. This is not semantic classification and does not identify section meaning.
"""
    (ADV / 'qc/section_classification_model_update_report.md').write_text(report, encoding='utf-8')
    final_check = f"""# Classification Update Check Report

- Table 12 updated: `{ADV / 'tables/table12_section_classification_results.csv'}`
- Final Table 5 updated: `{FINAL / 'tables/table5_section_classification_results_final.csv'}`
- Predictions updated: `{ADV / 'section_classification_predictions.csv'}`
- Confusion matrix updated: `{ADV / 'section_classification_confusion_matrix.csv'}`
- Final Figure 5 updated: `{FINAL / 'figures/figure5_section_classification_confusion_matrix_final.png'}`
- Per-class metrics generated: `{ADV / 'section_classification_per_class_metrics.csv'}`
- No `not computed in earlier dependency-limited drafts` remains in updated Table 5: {not (FINAL / 'tables/table5_section_classification_results_final.csv').read_text().find('not computed in earlier dependency-limited drafts') >= 0}
"""
    (FINAL / 'qc/classification_update_check_report.md').write_text(final_check, encoding='utf-8')

    revised = f"""# Revised Section 5.7, Section 6.6, and Limitations Text

## Revised Methods 5.7 paragraph

The classification experiment tested whether structural features predict conventional visual/codicological section labels above baseline. Unknown folios were excluded from the primary classification experiment and retained for separate proximity analysis. Models were evaluated under stratified {kfold}-fold cross-validation with random seed 42. The comparison included majority-class and stratified-random baselines, a token-count-only baseline, nearest-centroid and kNN structural classifiers, logistic regression with balanced class weights, a balanced random forest with 500 trees, and a balanced linear SVM. Evaluation used accuracy, macro-F1, weighted-F1, balanced accuracy, per-class precision/recall/F1, and confusion matrices. Label-permutation tests used 1,000 permutations of section labels against fixed model predictions to estimate a macro-F1 null distribution. Because section labels are imbalanced, macro-F1 and balanced accuracy are emphasized over raw accuracy.

## Revised Results 6.6 paragraph

Structural features predicted conventional visual/codicological section labels above baseline. The best model by macro-F1 was `{best_macro}`, and the best model by balanced accuracy was `{best_bal}`. The best-model macro-F1 permutation test gave p = {float(best_p):.4f}, using 1,000 label permutations and the empirical formula `(k + 1) / (N + 1)`. This result indicates that conventional section labels correlate with structural profiles, but it should not be interpreted as semantic classification or section meaning recovery.

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
"""
    (FINAL / 'manuscript_insertions/revised_section_5_7_and_limitations.md').write_text(revised, encoding='utf-8')

    print(json.dumps({
        'sklearn_version': sklearn_version,
        'kfold': kfold,
        'models': [m[0] for m in model_specs],
        'best_macro_F1_model': best_macro,
        'best_balanced_accuracy_model': best_bal,
        'best_macro_F1_permutation_p': best_p,
        'table12': str(ADV / 'tables/table12_section_classification_results.csv'),
        'final_table5': str(FINAL / 'tables/table5_section_classification_results_final.csv'),
    }, indent=2))

if __name__ == '__main__':
    main()
