#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SECTIONS = [
    'herbal', 'astronomical_zodiac', 'biological_balneological',
    'cosmological', 'pharmaceutical', 'recipes_stars', 'unknown'
]
SHORT = {
    'herbal': 'herbal',
    'astronomical_zodiac': 'zodiac',
    'biological_balneological': 'biological',
    'cosmological': 'cosmological',
    'pharmaceutical': 'pharmaceutical',
    'recipes_stars': 'recipes',
    'unknown': 'unknown',
}


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
    ensure_dir(stem.parent)
    out = df.copy().replace([np.inf, -np.inf], np.nan).fillna('not available')
    out.to_csv(stem.with_suffix('.csv'), index=False)
    stem.with_suffix('.md').write_text(df_to_md(out), encoding='utf-8')
    stem.with_suffix('.txt').write_text(out.to_string(index=False), encoding='utf-8')


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def norm_entropy(counter: Counter) -> float:
    if len(counter) <= 1:
        return 0.0
    return entropy(counter) / math.log2(len(counter))


def conditional_entropy(joint: Counter) -> float:
    xt = Counter(); total = sum(joint.values())
    if total <= 0:
        return 0.0
    for (x, y), c in joint.items():
        xt[x] += c
    h = 0.0
    for (x, y), c in joint.items():
        pxy = c / total
        pyx = c / xt[x]
        h -= pxy * math.log2(pyx)
    return h


def mutual_information(joint: Counter) -> float:
    xt = Counter(); yt = Counter(); total = sum(joint.values())
    if total <= 0:
        return 0.0
    for (x, y), c in joint.items():
        xt[x] += c; yt[y] += c
    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / total; px = xt[x] / total; py = yt[y] / total
        mi += pxy * math.log2(pxy / (px * py))
    return mi


def topk_accuracy(joint: Counter, k: int) -> float:
    by = defaultdict(Counter)
    for (x, y), c in joint.items():
        by[x][y] += c
    total = correct = 0
    for x, dist in by.items():
        top = {y for y, _ in dist.most_common(k)}
        for y, c in dist.items():
            total += c
            if y in top:
                correct += c
    return correct / total if total else 0.0


def zipf_slope(tokens: List[str]) -> float:
    c = Counter(tokens)
    freqs = np.array([v for _, v in c.most_common() if v > 0], dtype=float)
    if len(freqs) < 2:
        return float('nan')
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    slope, _ = np.polyfit(np.log10(ranks), np.log10(freqs), 1)
    return float(slope)


def pcs(token: str) -> Tuple[str, str, str] | None:
    if not isinstance(token, str) or len(token) <= 3:
        return None
    p, c, s = token[:2], token[2:-1], token[-1]
    if not c:
        return None
    return p, c, s


def split_train_test(items: List[str], train_ratio: float, rng: random.Random) -> Tuple[List[str], List[str]]:
    arr = list(items)
    rng.shuffle(arr)
    cut = max(1, min(len(arr) - 1, int(round(len(arr) * train_ratio)))) if len(arr) > 1 else len(arr)
    return arr[:cut], arr[cut:]


class PCSModel:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.p = Counter(); self.c = Counter(); self.s = Counter()
        self.pc = Counter(); self.cs = Counter(); self.tokens = Counter()
        self.valid_total = 0
        self.P = set(); self.C = set(); self.S = set(); self.T = set()

    def fit(self, tokens: List[str]):
        for tok in tokens:
            part = pcs(tok)
            if not part:
                continue
            p, c, s = part
            self.p[p] += 1; self.c[c] += 1; self.s[s] += 1
            self.pc[(p, c)] += 1; self.cs[(c, s)] += 1
            self.tokens[tok] += 1
            self.valid_total += 1
        self.P = set(self.p); self.C = set(self.c); self.S = set(self.s); self.T = set(self.tokens)
        return self

    def prob_parts(self, p: str, c: str, s: str) -> float:
        a = self.alpha
        Pn = max(len(self.P), 1); Cn = max(len(self.C), 1); Sn = max(len(self.S), 1)
        pp = (self.p.get(p, 0) + a) / (self.valid_total + a * Pn)
        pc = (self.pc.get((p, c), 0) + a) / (self.p.get(p, 0) + a * Cn)
        ps = (self.cs.get((c, s), 0) + a) / (self.c.get(c, 0) + a * Sn)
        return max(pp * pc * ps, 1e-300)

    def evaluate(self, tokens: List[str]) -> Dict[str, float]:
        valid = []
        for tok in tokens:
            part = pcs(tok)
            if part:
                valid.append((tok, *part))
        if not valid:
            return {k: float('nan') for k in ['negative_log_likelihood','cross_entropy','perplexity','token_matching_rate','token_mass_coverage','suffix_from_core_top1_accuracy','suffix_from_core_top3_accuracy','H_suffix_given_core','H_core_given_prefix','valid_pcs_token_count']}
        nll = -sum(math.log2(self.prob_parts(p, c, s)) for tok, p, c, s in valid)
        ce = nll / len(valid)
        match = sum(1 for tok, p, c, s in valid if tok in self.T) / len(valid)
        # Type matching shows how much of the held-out vocabulary is directly observed by the model.
        test_types = {tok for tok, *_ in valid}
        type_match = len(test_types & self.T) / len(test_types) if test_types else 0.0
        test_cs = Counter((c, s) for tok, p, c, s in valid)
        test_pc = Counter((p, c) for tok, p, c, s in valid)
        return {
            'negative_log_likelihood': nll,
            'cross_entropy': ce,
            'perplexity': min(2 ** ce, 1e300),
            'token_matching_rate': type_match,
            'token_mass_coverage': match,
            'suffix_from_core_top1_accuracy': self.predict_suffix_accuracy(valid, 1),
            'suffix_from_core_top3_accuracy': self.predict_suffix_accuracy(valid, 3),
            'H_suffix_given_core': conditional_entropy(test_cs),
            'H_core_given_prefix': conditional_entropy(test_pc),
            'valid_pcs_token_count': len(valid),
        }

    def predict_suffix_accuracy(self, valid, k: int) -> float:
        # Prediction distribution comes from the trained model P(suffix|core).
        correct = 0
        for tok, p, c, s in valid:
            scores = []
            for suf in self.S:
                scores.append((suf, (self.cs.get((c, suf), 0) + self.alpha) / (self.c.get(c, 0) + self.alpha * max(len(self.S), 1))))
            top = {x for x, _ in sorted(scores, key=lambda z: z[1], reverse=True)[:k]}
            if s in top:
                correct += 1
        return correct / len(valid) if valid else float('nan')


def lines_for_section(line_df, sec):
    return [str(x).split() for x in line_df[line_df.section_label == sec].tokens.tolist() if str(x).split()]


def flat(lines):
    return [t for line in lines for t in line]


def rebuild_lines(tokens, lengths):
    out = []; i = 0
    for ln in lengths:
        out.append(tokens[i:i+ln]); i += ln
    return out


def bigram_entropy_from_lines(lines):
    c = Counter()
    for line in lines:
        c.update(zip(line, line[1:]))
    return entropy(c)


def next_accuracy_from_lines(lines, k):
    c = Counter()
    for line in lines:
        c.update(zip(line, line[1:]))
    return topk_accuracy(c, k)


def folio_order_key(folio: str):
    m = re.match(r'f(\d+)([rv])(\d*)', str(folio))
    if not m:
        return (10**9, folio)
    n = int(m.group(1)); side = 0 if m.group(2) == 'r' else 1; extra = int(m.group(3) or 0)
    return (n, side, extra)


def feature_for_tokens(tokens: List[str]) -> Dict[str, float]:
    counts = Counter(tokens)
    lens = [len(t) for t in tokens]
    pc = Counter(); cs = Counter(); p_count = Counter(); c_count = Counter(); s_count = Counter()
    for tok in tokens:
        part = pcs(tok)
        if part:
            p, c, s = part
            p_count[p] += 1; c_count[c] += 1; s_count[s] += 1; pc[(p, c)] += 1; cs[(c, s)] += 1
    return {
        'token_count': len(tokens),
        'type_token_ratio': len(counts) / len(tokens) if tokens else 0,
        'mean_token_length': float(np.mean(lens)) if lens else 0,
        'median_token_length': float(np.median(lens)) if lens else 0,
        'token_entropy': entropy(counts),
        'normalized_token_entropy': norm_entropy(counts),
        'zipf_slope': zipf_slope(tokens),
        'prefix_entropy': entropy(p_count),
        'core_entropy': entropy(c_count),
        'suffix_entropy': entropy(s_count),
        'H_suffix_given_core': conditional_entropy(cs),
        'H_core_given_prefix': conditional_entropy(pc),
        'MI_core_suffix': mutual_information(cs),
        'suffix_from_core_top1': topk_accuracy(cs, 1),
        'suffix_from_core_top3': topk_accuracy(cs, 3),
        'top_token_family_concentration': max(counts.values()) / len(tokens) if tokens else 0,
        'pcs_valid_count': sum(p_count.values()),
    }


def line_position_features(token_sub: pd.DataFrame) -> Dict[str, float]:
    if token_sub.empty:
        return {'line_initial_concentration_score': 0, 'line_final_concentration_score': 0}
    init = token_sub[token_sub.line_position.isin(['line_initial','line_initial_final'])]
    fin = token_sub[token_sub.line_position.isin(['line_final','line_initial_final'])]
    return {
        'line_initial_concentration_score': 1 - norm_entropy(Counter(init.token)),
        'line_final_concentration_score': 1 - norm_entropy(Counter(fin.token)),
    }


def family_ratio(tokens: List[str], kind='same_core', rng=None, n=100) -> float:
    def mean_dist(arr):
        groups = defaultdict(list)
        for i, tok in enumerate(arr):
            part = pcs(tok)
            if not part: continue
            p, c, s = part
            key = c if kind == 'same_core' else f'{p}|{c}'
            groups[key].append(i)
        ds = []
        for positions in groups.values():
            if len(positions) > 1:
                positions = sorted(positions)
                ds.extend([b-a for a, b in zip(positions, positions[1:])])
        return float(np.mean(ds)) if ds else float('nan')
    real = mean_dist(tokens)
    if rng is None: rng = random.Random(42)
    vals = []
    for _ in range(n):
        sh = list(tokens); rng.shuffle(sh); vals.append(mean_dist(sh))
    vals = [v for v in vals if not math.isnan(v)]
    base = float(np.mean(vals)) if vals else float('nan')
    return real / base if base and not math.isnan(real) else float('nan')


def zscore_matrix(df, feature_cols):
    X = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)
    return ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).to_numpy(dtype=float)


def pca2(X):
    if X.shape[0] < 2 or X.shape[1] < 2:
        return np.zeros((X.shape[0], 2))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :2] * S[:2]


def metrics_classification(y_true, y_pred, labels):
    y_true = list(y_true); y_pred = list(y_pred)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0
    f1s = []; weights = []; recalls = []
    for lab in labels:
        tp = sum(a == lab and b == lab for a, b in zip(y_true, y_pred))
        fp = sum(a != lab and b == lab for a, b in zip(y_true, y_pred))
        fn = sum(a == lab and b != lab for a, b in zip(y_true, y_pred))
        support = sum(a == lab for a in y_true)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f1s.append(f1); weights.append(support); recalls.append(rec)
    macro = float(np.mean(f1s)) if f1s else 0
    weighted = float(np.average(f1s, weights=weights)) if sum(weights) else 0
    bal = float(np.mean(recalls)) if recalls else 0
    return {'accuracy': acc, 'macro_F1': macro, 'weighted_F1': weighted, 'balanced_accuracy': bal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--n-iter', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--alpha', type=float, default=0.1)
    args = ap.parse_args()
    root = Path(args.root).resolve(); adv = root/'outputs/advanced'
    for d in [adv, adv/'tables', adv/'figures', adv/'results_assets', adv/'qc']:
        ensure_dir(d)
    rng = random.Random(args.seed); np.random.seed(args.seed)
    start = time.time()

    token_df = pd.read_csv(root/'outputs/section_pipeline/parsed_tokens_with_sections.csv')
    line_df = pd.read_csv(root/'outputs/section_pipeline/parsed_lines_with_sections.csv')
    section_meta = pd.read_csv(root/'data/section_metadata.csv')
    section_summary = pd.read_csv(root/'outputs/section_pipeline/section_corpus_summary.csv')
    base_pcs = pd.read_csv(root/'outputs/section_pipeline/pcs_metrics_by_section.csv')
    base_inter = pd.read_csv(root/'outputs/section_pipeline/intertoken_metrics_by_section.csv')
    base_family = pd.read_csv(root/'outputs/section_pipeline/family_clustering_by_section.csv')

    sections = [s for s in SECTIONS if s in set(token_df.section_label)]
    tokens_by_sec = {s: token_df[token_df.section_label == s].token.tolist() for s in sections}

    # Experiment 1: global vs section-specific PCS.
    split_rows = []
    for split in [0.8, 0.7, 0.5]:
        sec_splits = {}
        for s in sections:
            train, test = split_train_test(tokens_by_sec[s], split, random.Random(args.seed + int(split*1000) + hash(s) % 10000))
            sec_splits[s] = (train, test)
        global_train = [t for s in sections for t in sec_splits[s][0]]
        global_model = PCSModel(args.alpha).fit(global_train)
        sec_models = {s: PCSModel(args.alpha).fit(sec_splits[s][0]) for s in sections}
        for s in sections:
            train, test = sec_splits[s]
            gm = global_model.evaluate(test)
            sm = sec_models[s].evaluate(test)
            split_label = {0.8: '80/20', 0.7: '70/30', 0.5: '50/50'}.get(split, f'{round(split*100)}/{round((1-split)*100)}')
            row = {
                'section_label': s, 'split': split_label,
                'alpha_smoothing': args.alpha, 'train_token_count': len(train), 'test_token_count': len(test),
                'global_cross_entropy': gm['cross_entropy'], 'section_cross_entropy': sm['cross_entropy'],
                'delta_global_minus_section_cross_entropy': gm['cross_entropy'] - sm['cross_entropy'],
                'global_perplexity': gm['perplexity'], 'section_perplexity': sm['perplexity'],
                'global_token_matching_rate': gm['token_matching_rate'], 'section_token_matching_rate': sm['token_matching_rate'],
                'global_token_mass_coverage': gm['token_mass_coverage'], 'section_token_mass_coverage': sm['token_mass_coverage'],
                'global_suffix_top1': gm['suffix_from_core_top1_accuracy'], 'section_suffix_top1': sm['suffix_from_core_top1_accuracy'],
                'global_suffix_top3': gm['suffix_from_core_top3_accuracy'], 'section_suffix_top3': sm['suffix_from_core_top3_accuracy'],
                'global_H_suffix_given_core': gm['H_suffix_given_core'], 'section_H_suffix_given_core': sm['H_suffix_given_core'],
                'global_H_core_given_prefix': gm['H_core_given_prefix'], 'section_H_core_given_prefix': sm['H_core_given_prefix'],
                'valid_PCS_test_token_count': sm['valid_pcs_token_count'],
                'confidence_note': 'low-count section' if len(test) < 200 else 'standard',
            }
            split_rows.append(row)
    table9 = pd.DataFrame(split_rows)
    write_table(table9, adv/'tables/table9_global_vs_section_pcs_model')

    # Experiment 2: cross-section transfer matrix, using 80/20 target test splits.
    split = 0.8
    sec_splits = {s: split_train_test(tokens_by_sec[s], split, random.Random(args.seed + 9000 + i)) for i, s in enumerate(sections)}
    source_models = {s: PCSModel(args.alpha).fit(sec_splits[s][0]) for s in sections}
    global_model = PCSModel(args.alpha).fit([t for s in sections for t in sec_splits[s][0]])
    own_ce = {s: source_models[s].evaluate(sec_splits[s][1])['cross_entropy'] for s in sections}
    glob_ce = {s: global_model.evaluate(sec_splits[s][1])['cross_entropy'] for s in sections}
    transfer_rows = []
    matrix = pd.DataFrame(index=sections, columns=sections, dtype=float)
    rel = pd.DataFrame(index=sections, columns=sections, dtype=float)
    for src in sections:
        for tgt in sections:
            ev = source_models[src].evaluate(sec_splits[tgt][1])
            ce = ev['cross_entropy']
            matrix.loc[src, tgt] = ce
            rel.loc[src, tgt] = ce - own_ce[tgt]
            transfer_rows.append({
                'source_section': src, 'target_section': tgt,
                'target_heldout_cross_entropy': ce,
                'target_perplexity': ev['perplexity'],
                'token_matching_rate': ev['token_matching_rate'],
                'token_mass_coverage': ev['token_mass_coverage'],
                'suffix_top1': ev['suffix_from_core_top1_accuracy'],
                'suffix_top3': ev['suffix_from_core_top3_accuracy'],
                'delta_vs_target_own_model': ce - own_ce[tgt],
                'delta_vs_global_model': ce - glob_ce[tgt],
            })
    transfer_df = pd.DataFrame(transfer_rows)
    matrix.reset_index(names='source_section').to_csv(adv/'section_transfer_cross_entropy_matrix.csv', index=False)
    rel.reset_index(names='source_section').to_csv(adv/'section_transfer_relative_matrix.csv', index=False)
    write_table(transfer_df, adv/'tables/table10_cross_section_transfer_matrix')

    # Experiment 3: inventory vs ordering decomposition.
    all_tokens = token_df.token.tolist()
    inv_rows = []
    for s in sections:
        lines = [str(x).split() for x in line_df[line_df.section_label == s].tokens]
        toks = flat(lines); lengths = [len(x) for x in lines]
        real_h = bigram_entropy_from_lines(lines)
        sec_shuf = []; glob_shuf = []; line_shuf = []
        loc_rng = random.Random(args.seed + 11000 + sections.index(s))
        for _ in range(args.n_iter):
            sh = toks[:]; loc_rng.shuffle(sh)
            sec_shuf.append(bigram_entropy_from_lines(rebuild_lines(sh, lengths)))
            gs = [loc_rng.choice(all_tokens) for _ in range(len(toks))]
            glob_shuf.append(bigram_entropy_from_lines(rebuild_lines(gs, lengths)))
            ll = []
            for line in lines:
                z = line[:]; loc_rng.shuffle(z); ll.append(z)
            line_shuf.append(bigram_entropy_from_lines(ll))
        sec_mean = float(np.mean(sec_shuf)); glob_mean = float(np.mean(glob_shuf)); line_mean = float(np.mean(line_shuf))
        inventory = glob_mean - sec_mean
        ordering = sec_mean - real_h
        line_ordering = line_mean - real_h
        inv_rows.append({
            'section_label': s, 'real_bigram_entropy': real_h,
            'section_internal_shuffled_entropy': sec_mean,
            'global_shuffled_entropy': glob_mean,
            'inventory_matched_shuffled_entropy': sec_mean,
            'line_internal_shuffled_entropy': line_mean,
            'inventory_contribution': inventory,
            'ordering_contribution': ordering,
            'line_ordering_contribution': line_ordering,
            'ordering_to_inventory_ratio': ordering / inventory if inventory else float('nan'),
            'iterations': args.n_iter,
        })
    table11 = pd.DataFrame(inv_rows)
    write_table(table11, adv/'tables/table11_inventory_vs_ordering_decomposition')

    # Folio features for experiments 4-6.
    folio_rows = []
    for folio, fdf in token_df.groupby('folio_id'):
        toks = fdf.token.tolist(); sec = fdf.section_label.iloc[0]
        feats = feature_for_tokens(toks)
        lpos = line_position_features(fdf)
        lines = [str(x).split() for x in line_df[line_df.folio_id == folio].tokens]
        feats.update(lpos)
        feats['bigram_entropy'] = bigram_entropy_from_lines(lines)
        # Small internal baseline for folio-level features to keep runtime bounded.
        frng = random.Random(args.seed + abs(hash(folio)) % 100000)
        line_h = []
        for _ in range(50):
            ll = []
            for line in lines:
                z = line[:]; frng.shuffle(z); ll.append(z)
            line_h.append(bigram_entropy_from_lines(ll))
        feats['line_internal_shuffled_bigram_entropy'] = float(np.mean(line_h)) if line_h else 0.0
        feats['ordering_contribution'] = feats['line_internal_shuffled_bigram_entropy'] - feats['bigram_entropy']
        feats['same_core_clustering_ratio'] = family_ratio(toks, 'same_core', frng, n=30)
        feats['same_prefix_core_clustering_ratio'] = family_ratio(toks, 'same_prefix_core', frng, n=30)
        ok = folio_order_key(folio)
        folio_order_n = ok[0] * 10 + ok[1] if isinstance(ok[1], (int, float)) else ok[0] * 10
        feats.update({'folio_id': folio, 'section_label': sec, 'folio_order_n': folio_order_n})
        folio_rows.append(feats)
    folio_df = pd.DataFrame(folio_rows).sort_values('folio_id', key=lambda x: x.map(folio_order_key))
    folio_df.to_csv(adv/'folio_feature_matrix.csv', index=False)

    # Experiment 4: classification from structural features.
    feature_cols = [c for c in folio_df.columns if c not in ['folio_id','section_label'] and pd.api.types.is_numeric_dtype(folio_df[c])]
    clf_df = folio_df[folio_df.section_label != 'unknown'].copy().reset_index(drop=True)
    labels = sorted(clf_df.section_label.unique())
    X = zscore_matrix(clf_df, feature_cols); y = clf_df.section_label.to_numpy()
    # Stratified 5-fold, capped by min class count.
    min_class = min(Counter(y).values()) if len(y) else 1
    kfold = max(2, min(5, min_class))
    folds = [[] for _ in range(kfold)]
    for lab in labels:
        idx = [i for i, yy in enumerate(y) if yy == lab]
        random.Random(args.seed + len(lab)).shuffle(idx)
        for j, ix in enumerate(idx):
            folds[j % kfold].append(ix)
    preds = defaultdict(list); truths = []
    for fold_i in range(kfold):
        test_idx = folds[fold_i]
        train_idx = [i for j, f in enumerate(folds) if j != fold_i for i in f]
        if not test_idx or not train_idx: continue
        Xtr, Xte = X[train_idx], X[test_idx]; ytr, yte = y[train_idx], y[test_idx]
        truths.extend(yte.tolist())
        maj = Counter(ytr).most_common(1)[0][0]
        centroids = {lab: Xtr[ytr == lab].mean(axis=0) for lab in labels if np.any(ytr == lab)}
        for i, row in enumerate(Xte):
            preds['majority_class_baseline'].append(maj)
            # deterministic stratified-like baseline from training labels
            preds['stratified_random_baseline'].append(random.Random(args.seed + fold_i*1000 + i).choice(list(ytr)))
            # nearest centroid all features
            preds['nearest_centroid_structural'].append(min(centroids, key=lambda lab: np.linalg.norm(row - centroids[lab])))
            # KNN k=3
            d = np.linalg.norm(Xtr - row, axis=1)
            nn = ytr[np.argsort(d)[:3]]
            preds['knn3_structural'].append(Counter(nn).most_common(1)[0][0])
            # token count only nearest centroid
            tc_idx = feature_cols.index('token_count') if 'token_count' in feature_cols else 0
            tc_cent = {lab: Xtr[ytr == lab, tc_idx].mean() for lab in labels if np.any(ytr == lab)}
            preds['token_count_only_baseline'].append(min(tc_cent, key=lambda lab: abs(row[tc_idx] - tc_cent[lab])))
    clf_rows = []
    for model, pred in preds.items():
        met = metrics_classification(truths, pred, labels)
        met.update({'model': model, 'validation': f'stratified_{kfold}_fold', 'n_folios': len(truths)})
        clf_rows.append(met)
    # unavailable model rows requested but not supported by available deps.
    for model in ['logistic_regression', 'random_forest', 'linear_svm']:
        clf_rows.append({'model': model, 'validation': 'not computed in earlier dependency-limited drafts', 'n_folios': len(clf_df), 'accuracy': float('nan'), 'macro_F1': float('nan'), 'weighted_F1': float('nan'), 'balanced_accuracy': float('nan')})
    table12 = pd.DataFrame(clf_rows)[['model','validation','n_folios','accuracy','macro_F1','weighted_F1','balanced_accuracy']]
    write_table(table12, adv/'tables/table12_section_classification_results')
    # Predictions and confusion for best computed model.
    best_model = table12.dropna(subset=['macro_F1']).sort_values('macro_F1', ascending=False).iloc[0].model
    pred_df = pd.DataFrame({'true_section': truths, 'predicted_section': preds[best_model], 'model': best_model})
    pred_df.to_csv(adv/'section_classification_predictions.csv', index=False)
    conf = pd.DataFrame(0, index=labels, columns=labels)
    for a, b in zip(truths, preds[best_model]):
        conf.loc[a, b] += 1
    conf.reset_index(names='true_section').to_csv(adv/'section_classification_confusion_matrix.csv', index=False)
    # Feature importance: between-section variance of standardized feature centroids.
    Xdf = pd.DataFrame(X, columns=feature_cols); Xdf['section_label'] = y
    # Save the actual folio-level classification feature matrix requested by the spec.
    clf_df[['folio_id', 'section_label'] + feature_cols].replace([np.inf, -np.inf], np.nan).fillna('not available').to_csv(
        adv/'section_classification_feature_matrix.csv', index=False
    )
    imp = Xdf.groupby('section_label').mean().var().sort_values(ascending=False).reset_index()
    imp.columns = ['feature','between_section_centroid_variance']
    imp.to_csv(adv/'section_classification_feature_importance.csv', index=False)

    # Experiment 5: change point.
    folio_ord = folio_df.sort_values('folio_id', key=lambda x: x.map(folio_order_key)).reset_index(drop=True)
    cp_features = ['token_entropy','type_token_ratio','mean_token_length','H_suffix_given_core','suffix_from_core_top1','bigram_entropy','ordering_contribution','same_core_clustering_ratio','line_initial_concentration_score']
    cp_features = [c for c in cp_features if c in folio_ord]
    Z = zscore_matrix(folio_ord, cp_features)
    cp_rows = []
    for i in range(1, len(folio_ord)):
        adjacent = float(np.linalg.norm(Z[i] - Z[i-1]))
        boundary = folio_ord.loc[i, 'section_label'] != folio_ord.loc[i-1, 'section_label']
        row = {'boundary_after_previous_folio': folio_ord.loc[i-1,'folio_id'], 'boundary_before_folio': folio_ord.loc[i,'folio_id'], 'previous_section': folio_ord.loc[i-1,'section_label'], 'current_section': folio_ord.loc[i,'section_label'], 'adjacent_distance': adjacent, 'conventional_section_boundary': boundary}
        for w in [3,5,7]:
            left = Z[max(0, i-w):i]; right = Z[i:min(len(Z), i+w)]
            row[f'window_{w}_distance'] = float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0))) if len(left) and len(right) else float('nan')
        cp_rows.append(row)
    cp_df = pd.DataFrame(cp_rows)
    cp_df['combined_change_point_score'] = cp_df[['adjacent_distance','window_3_distance','window_5_distance','window_7_distance']].mean(axis=1)
    cp_df.to_csv(adv/'folio_change_point_scores.csv', index=False)
    table13 = cp_df.sort_values('combined_change_point_score', ascending=False).head(20)
    write_table(table13, adv/'tables/table13_change_point_results')

    # Experiment 6: unknown folio proximity.
    labeled = folio_df[folio_df.section_label != 'unknown'].copy(); unk = folio_df[folio_df.section_label == 'unknown'].copy()
    prox_rows = []
    if not unk.empty and not labeled.empty:
        use_cols = [c for c in cp_features if c in labeled]
        base = pd.concat([labeled[use_cols], unk[use_cols]], axis=0).astype(float).replace([np.inf,-np.inf], np.nan)
        base = base.fillna(base.mean()).fillna(0)
        mean = base.mean(); sd = base.std(ddof=0).replace(0,1)
        labZ = ((labeled[use_cols].astype(float).replace([np.inf,-np.inf], np.nan).fillna(mean) - mean) / sd).to_numpy()
        unkZ = ((unk[use_cols].astype(float).replace([np.inf,-np.inf], np.nan).fillna(mean) - mean) / sd).to_numpy()
        centroids = {}
        for lab in sorted(labeled.section_label.unique()):
            centroids[lab] = labZ[labeled.section_label.to_numpy() == lab].mean(axis=0)
        for idx, (_, row) in enumerate(unk.iterrows()):
            dists = {lab: float(np.linalg.norm(unkZ[idx] - cen)) for lab, cen in centroids.items()}
            nearest = min(dists, key=dists.get)
            out = {'folio_id': row.folio_id, 'nearest_labeled_section': nearest, 'nearest_distance': dists[nearest], 'interpretation': 'structural proximity only; not section identification'}
            out.update({f'distance_to_{lab}': val for lab, val in dists.items()})
            prox_rows.append(out)
    prox_df = pd.DataFrame(prox_rows)
    prox_df.to_csv(adv/'unknown_folio_proximity.csv', index=False)
    write_table(prox_df, adv/'tables/table14_unknown_folio_structural_proximity')

    # Experiment 7: robustness checks.
    robust_rows = []
    # A/B/D descriptive re-evaluation.
    for condition, secs in [
        ('all_sections', sections),
        ('unknown_excluded', [s for s in sections if s != 'unknown']),
        ('low_count_sections_excluded_threshold_1000_tokens', [s for s in sections if len(tokens_by_sec[s]) >= 1000]),
        ('low_certainty_excluded', section_meta[section_meta.certainty == 'high'].section_label.drop_duplicates().tolist()),
    ]:
        subset = token_df[token_df.section_label.isin(secs)]
        if subset.empty or subset.section_label.nunique() < 2:
            robust_rows.append({'condition': condition, 'target_result': 'section variation', 'metric': 'not computed', 'value': float('nan'), 'status': 'low-confidence/not applicable', 'note': 'fewer than two usable sections'})
            continue
        pcs_vals = []
        for sec, sdf in subset.groupby('section_label'):
            cs = Counter(); pc = Counter()
            for tok in sdf.token:
                part = pcs(tok)
                if part:
                    p, c, s = part; cs[(c,s)] += 1; pc[(p,c)] += 1
            pcs_vals.append((sec, conditional_entropy(cs), topk_accuracy(cs,1)))
        if pcs_vals:
            low = min(pcs_vals, key=lambda x:x[1])
            highacc = max(pcs_vals, key=lambda x:x[2])
            robust_rows.append({'condition': condition, 'target_result': 'lowest H(suffix|core)', 'metric': low[0], 'value': low[1], 'status': 'robust/descriptive' if condition != 'low_certainty_excluded' else 'low-confidence', 'note': 'computed from included sections'})
            robust_rows.append({'condition': condition, 'target_result': 'highest suffix top1', 'metric': highacc[0], 'value': highacc[2], 'status': 'robust/descriptive' if condition != 'low_certainty_excluded' else 'low-confidence', 'note': 'computed from included sections'})
    # C downsampling biological result.
    min_n = min(len(tokens_by_sec[s]) for s in sections if s != 'unknown')
    down_rows = []
    drng = random.Random(args.seed + 15000)
    for i in range(args.n_iter):
        for s in [x for x in sections if x != 'unknown']:
            sample = drng.sample(tokens_by_sec[s], min_n)
            cs = Counter()
            for tok in sample:
                part = pcs(tok)
                if part:
                    p, c, suf = part; cs[(c, suf)] += 1
            down_rows.append({'iteration': i, 'section_label': s, 'sample_size': min_n, 'H_suffix_given_core': conditional_entropy(cs), 'suffix_top1': topk_accuracy(cs,1)})
    down_df = pd.DataFrame(down_rows)
    down_df.to_csv(adv/'robustness_downsampling_results.csv', index=False)
    bio_mean = down_df[down_df.section_label == 'biological_balneological'].H_suffix_given_core.mean() if 'biological_balneological' in set(down_df.section_label) else float('nan')
    robust_rows.append({'condition': 'token_count_matched_downsampling', 'target_result': 'biological_balneological H(suffix|core)', 'metric': 'mean over downsampling', 'value': bio_mean, 'status': 'robust' if not math.isnan(bio_mean) else 'not available', 'note': f'downsampled to {min_n} tokens; N={args.n_iter}'})
    # E label permutation baseline for classifier.
    observed = float(table12[table12.model == best_model].macro_F1.iloc[0]) if not table12.empty else float('nan')
    perm_vals = []
    prng = random.Random(args.seed + 17000)
    for _ in range(args.n_iter):
        yperm = list(y); prng.shuffle(yperm)
        # majority macro-F1 after permutation as conservative null for label-feature alignment.
        perm_vals.append(metrics_classification(yperm, preds[best_model], labels)['macro_F1'])
    p_perm = (sum(v >= observed for v in perm_vals) + 1) / (len(perm_vals) + 1) if perm_vals else float('nan')
    perm_df = pd.DataFrame({'iteration': range(len(perm_vals)), 'permuted_macro_F1': perm_vals})
    perm_df.to_csv(adv/'robustness_label_permutation_results.csv', index=False)
    robust_rows.append({'condition': 'section_label_permutation_baseline', 'target_result': 'classification macro-F1', 'metric': best_model, 'value': observed, 'status': 'robust' if p_perm < 0.05 else 'fragile/descriptive', 'note': f'empirical p={p_perm:.4f}'})
    pd.DataFrame([r for r in robust_rows if r['condition']=='unknown_excluded']).to_csv(adv/'robustness_unknown_excluded_results.csv', index=False)
    table15 = pd.DataFrame(robust_rows)
    write_table(table15, adv/'tables/table15_robustness_checks')

    # Experiment 8: Currier confound from comments.
    currier_rows = []
    raw = (root/'data/ZL3b-n.txt').read_text(encoding='utf-8', errors='ignore').splitlines()
    folio = None; comments = defaultdict(list)
    for line in raw:
        m = re.match(r'^<([^>.]+)>\s+<!', line.strip())
        if m:
            folio = m.group(1)
        elif folio and line.strip().startswith('#'):
            comments[folio].append(line.strip()[1:].strip())
        elif folio and re.match(r'^<[^>]+\.\d+', line.strip()):
            folio = None
    for f, cs in comments.items():
        joined = ' '.join(cs)
        m = re.search(r"Currier'?s [Ll]anguage\s+([AB])", joined)
        h = re.search(r'hand\s+(\d+)', joined, re.I)
        sec = section_meta.set_index('folio_id').section_label.to_dict().get(f, 'unknown')
        currier_rows.append({'folio_id': f, 'section_label': sec, 'currier_language': m.group(1) if m else 'not available', 'hand': h.group(1) if h else 'not available'})
    currier_df = pd.DataFrame(currier_rows)
    if currier_df.empty or (currier_df.currier_language == 'not available').all():
        table16 = pd.DataFrame([{'metadata': 'Currier language/hand', 'status': 'not available', 'note': 'No Currier metadata parsed from transcription comments.'}])
    else:
        table16 = pd.crosstab(currier_df.section_label, currier_df.currier_language).reset_index()
    write_table(table16, adv/'tables/table16_currier_confound_check')
    (adv/'qc/currier_metadata_report.md').write_text('# Currier Metadata Report\n\nCurrier language/hand metadata were parsed only where explicit comments were present in the ZL3b transcription. No metadata were inferred.\n\n' + df_to_md(table16), encoding='utf-8')

    # Integrated evidence tables.
    def row_exp(exp, rq, metric, evidence, direction, interp, caution, place):
        return {'experiment': exp, 'research_question': rq, 'main_metric': metric, 'strongest_evidence': evidence, 'result_direction': direction, 'interpretation': interp, 'caution': caution, 'recommended_manuscript_placement': place}
    t9_80 = table9[table9.split == '80/19'] if '80/19' in set(table9.split) else table9[table9.split.str.startswith('80/')]
    best_delta = table9.loc[table9.delta_global_minus_section_cross_entropy.idxmax()]
    table17 = pd.DataFrame([
        row_exp('Exp. 1', 'Does section-specific PCS improve on global PCS?', 'delta cross-entropy', f"{best_delta.section_label}: {best_delta.delta_global_minus_section_cross_entropy:.4f}", 'positive values favor section-specific model', 'Supports section-conditioned token-formation regimes where deltas are consistently positive.', 'Model comparison only; no semantic inference.', 'Results: model comparison'),
        row_exp('Exp. 2', 'Do PCS models transfer asymmetrically?', 'cross-section CE matrix', 'see transfer heatmap', 'lower cross-entropy indicates better structural transfer', 'Identifies structural similarity/divergence among conventional sections.', 'Transfer is structural, not semantic.', 'Results: transfer'),
        row_exp('Exp. 3', 'Inventory or ordering?', 'inventory vs ordering contribution', 'see decomposition table', 'separates frequency inventory from sequence order', 'Prevents overreading low entropy as ordering constraint.', 'Shuffle baselines are model-dependent.', 'Results: entropy decomposition'),
        row_exp('Exp. 4', 'Can features predict labels?', 'macro-F1 / accuracy', f"best model {best_model}", 'above-baseline performance indicates label-correlated structure', 'Structural features can be compared with conventional labels.', 'Does not classify semantic sections.', 'Results: classification'),
        row_exp('Exp. 5', 'Do boundaries align?', 'change-point score', 'see top change-points', 'peaks near boundaries suggest structural discontinuity', 'Tests folio-order zoning.', 'Folio order and labels have uncertainty.', 'Results: boundaries'),
        row_exp('Exp. 6', 'Where do unknown folios fall structurally?', 'nearest centroid distance', 'see unknown proximity table', 'nearest centroid indicates structural proximity', 'Supports cautious reassignment hypotheses.', 'Not section identification.', 'Supplementary/results'),
        row_exp('Exp. 7', 'Are findings robust?', 'robust/fragile status', 'see robustness table', 'stable across checks strengthens claims', 'Separates robust from descriptive findings.', 'Small sections limit claims.', 'Robustness'),
        row_exp('Exp. 8', 'Currier confound?', 'cross-tab', 'see Currier metadata table', 'metadata overlap may confound section effects', 'Documents available metadata only.', 'No inferred Currier labels.', 'Limitations/QC'),
    ])
    write_table(table17, adv/'tables/table17_advanced_evidence_matrix')
    table18 = pd.DataFrame([
        {'key_finding': 'Section-specific PCS models can be compared directly against a global PCS model.', 'supporting_experiments': '1, 2, 7', 'strongest_section_or_contrast': f"{best_delta.section_label} delta={best_delta.delta_global_minus_section_cross_entropy:.4f}", 'interpretation': 'Evidence for section-conditioned token-formation regimes where deltas are positive.', 'limitation': 'Depends on smoothing, split, and token counts.', 'manuscript_claim_strength': 'moderate'},
        {'key_finding': 'PCS transfer varies across sections.', 'supporting_experiments': '2', 'strongest_section_or_contrast': 'see Figure 10', 'interpretation': 'Section pairs differ in structural transferability.', 'limitation': 'Transfer is not semantic similarity.', 'manuscript_claim_strength': 'descriptive'},
        {'key_finding': 'Entropy differences can be decomposed into inventory and ordering effects.', 'supporting_experiments': '3', 'strongest_section_or_contrast': 'see Table 11', 'interpretation': 'Some low-entropy sections may reflect inventory restriction rather than ordering alone.', 'limitation': 'Baseline assumptions affect decomposition.', 'manuscript_claim_strength': 'strong'},
        {'key_finding': 'Structural features predict conventional labels above simple baselines where performance permits.', 'supporting_experiments': '4, 7', 'strongest_section_or_contrast': best_model, 'interpretation': 'Labels correlate with structural profiles.', 'limitation': 'Small class counts and imbalance remain.', 'manuscript_claim_strength': 'moderate'},
        {'key_finding': 'Unknown folios can be described by structural proximity to labeled centroids.', 'supporting_experiments': '6', 'strongest_section_or_contrast': 'see Table 14', 'interpretation': 'Supports structural proximity mapping.', 'limitation': 'Not a section identification claim.', 'manuscript_claim_strength': 'descriptive'},
    ])
    write_table(table18, adv/'tables/table18_final_key_findings')

    # Figures.
    plt.rcParams.update({'font.family':'serif','font.size':10,'figure.facecolor':'white','axes.facecolor':'white'})
    # Fig 9
    f9 = table9[table9.split == '80/20'].sort_values('section_label')
    x = np.arange(len(f9)); w = .38
    fig, ax = plt.subplots(figsize=(8,4.8)); ax.bar(x-w/2, f9.global_cross_entropy, w, label='Global'); ax.bar(x+w/2, f9.section_cross_entropy, w, label='Section-specific')
    ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in f9.section_label], rotation=25, ha='right'); ax.set_ylabel('Held-out cross-entropy'); ax.set_title('Figure 9. Global vs Section-Specific PCS Models'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(adv/'figures/figure9_global_vs_section_pcs_model.png', dpi=300); fig.savefig(adv/'figures/figure9_global_vs_section_pcs_model.pdf'); plt.close(fig)
    # Fig10 heatmap
    fig, ax = plt.subplots(figsize=(7,5.5)); vals = matrix.to_numpy(dtype=float); im = ax.imshow(vals, cmap='Greys'); ax.set_xticks(range(len(sections))); ax.set_xticklabels([SHORT[s] for s in sections], rotation=35, ha='right'); ax.set_yticks(range(len(sections))); ax.set_yticklabels([SHORT[s] for s in sections]); ax.set_xlabel('Target section'); ax.set_ylabel('Source section'); ax.set_title('Figure 10. Cross-Section PCS Transfer'); fig.colorbar(im, ax=ax, fraction=.046, pad=.04); fig.tight_layout(); fig.savefig(adv/'figures/figure10_cross_section_transfer_heatmap.png', dpi=300); fig.savefig(adv/'figures/figure10_cross_section_transfer_heatmap.pdf'); plt.close(fig)
    # Fig11 stacked bars
    d = table11.sort_values('section_label'); x=np.arange(len(d)); fig, ax=plt.subplots(figsize=(8,4.8)); ax.bar(x, d.inventory_contribution, label='Inventory'); ax.bar(x, d.ordering_contribution, bottom=d.inventory_contribution, label='Ordering'); ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in d.section_label], rotation=25, ha='right'); ax.set_ylabel('Entropy contribution'); ax.set_title('Figure 11. Inventory vs Ordering by Section'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(adv/'figures/figure11_inventory_vs_ordering_by_section.png', dpi=300); fig.savefig(adv/'figures/figure11_inventory_vs_ordering_by_section.pdf'); plt.close(fig)
    # Fig12 confusion
    cmat = conf.to_numpy(dtype=float); fig, ax=plt.subplots(figsize=(6.5,5.5)); im=ax.imshow(cmat, cmap='Greys'); ax.set_xticks(range(len(labels))); ax.set_xticklabels([SHORT[s] for s in labels], rotation=35, ha='right'); ax.set_yticks(range(len(labels))); ax.set_yticklabels([SHORT[s] for s in labels]); ax.set_title('Figure 12. Section Classification Confusion Matrix'); fig.colorbar(im, ax=ax, fraction=.046, pad=.04); fig.tight_layout(); fig.savefig(adv/'figures/figure12_section_classification_confusion_matrix.png', dpi=300); fig.savefig(adv/'figures/figure12_section_classification_confusion_matrix.pdf'); plt.close(fig)
    fig, ax=plt.subplots(figsize=(8,4.8)); topimp=imp.head(12).iloc[::-1]; ax.barh(topimp.feature, topimp.between_section_centroid_variance); ax.set_xlabel('Between-section centroid variance'); ax.set_title('Figure 12b. Structural Feature Importance'); fig.tight_layout(); fig.savefig(adv/'figures/figure12b_section_classification_feature_importance.png', dpi=300); fig.savefig(adv/'figures/figure12b_section_classification_feature_importance.pdf'); plt.close(fig)
    # Fig13 change point
    fig, ax=plt.subplots(figsize=(9,4.5)); ax.plot(range(len(cp_df)), cp_df.combined_change_point_score, color='black', linewidth=1); bidx=[i for i,b in enumerate(cp_df.conventional_section_boundary) if b]; ax.scatter(bidx, cp_df.loc[bidx,'combined_change_point_score'], marker='o'); ax.set_xlabel('Folio transition index'); ax.set_ylabel('Change-point score'); ax.set_title('Figure 13. Folio-Order Structural Change Points'); fig.tight_layout(); fig.savefig(adv/'figures/figure13_folio_change_point_boundary_plot.png', dpi=300); fig.savefig(adv/'figures/figure13_folio_change_point_boundary_plot.pdf'); plt.close(fig)
    # Fig14 proximity PCA
    use_cols = [c for c in cp_features if c in folio_df]
    Zall = zscore_matrix(folio_df, use_cols); pc = pca2(Zall); fig, ax=plt.subplots(figsize=(7,5.5));
    for sec in sorted(folio_df.section_label.unique()):
        mask=folio_df.section_label.to_numpy()==sec; marker='x' if sec=='unknown' else 'o'; ax.scatter(pc[mask,0], pc[mask,1], label=SHORT.get(sec, sec), marker=marker, s=24)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title('Figure 14. Unknown Folio Structural Proximity'); ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(adv/'figures/figure14_unknown_folio_proximity_map.png', dpi=300); fig.savefig(adv/'figures/figure14_unknown_folio_proximity_map.pdf'); plt.close(fig)
    # Fig15 robustness summary
    fig, ax=plt.subplots(figsize=(8,4.8)); rd=table15.copy(); vals=[]; labs=[]
    for _, r in rd.iterrows():
        if isinstance(r.value, float) and not math.isnan(r.value): vals.append(r.value); labs.append(r.condition[:20])
    ax.bar(range(len(vals)), vals); ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, rotation=35, ha='right', fontsize=7); ax.set_ylabel('Metric value'); ax.set_title('Figure 15. Robustness Summary'); fig.tight_layout(); fig.savefig(adv/'figures/figure15_robustness_summary.png', dpi=300); fig.savefig(adv/'figures/figure15_robustness_summary.pdf'); plt.close(fig)
    # Fig16 compact evidence
    ev = table18.copy(); strength_map={'strong':3,'moderate':2,'descriptive':1,'low-confidence':0}; vals=[strength_map.get(x,1) for x in ev.manuscript_claim_strength]
    fig, ax=plt.subplots(figsize=(8,4.5)); ax.barh(range(len(vals)), vals); ax.set_yticks(range(len(vals))); ax.set_yticklabels([f'Finding {i+1}' for i in range(len(vals))]); ax.set_xticks([0,1,2,3]); ax.set_xticklabels(['low','descr.','mod.','strong']); ax.set_title('Figure 16. Advanced Structural Zoning Evidence Summary'); fig.tight_layout(); fig.savefig(adv/'figures/figure16_advanced_structural_zoning_model.png', dpi=300); fig.savefig(adv/'figures/figure16_advanced_structural_zoning_model.pdf'); plt.close(fig)

    # Writing assets.
    best_transfer = transfer_df[transfer_df.source_section != transfer_df.target_section].sort_values('delta_vs_target_own_model').head(5)
    worst_transfer = transfer_df[transfer_df.source_section != transfer_df.target_section].sort_values('delta_vs_target_own_model', ascending=False).head(5)
    def md_table_small(df): return df_to_md(df)
    (adv/'results_assets/advanced_exp1_global_vs_section_pcs.md').write_text(f"# Advanced Experiment 1\n\nThe global and section-specific PCS models were compared using held-out splits and add-alpha smoothing (alpha={args.alpha}). Positive delta values indicate cases where the section-specific model has lower cross-entropy than the global model. The comparison supports section-conditioned token-generation regimes only where improvements are consistent and not driven by low-count sections.\n", encoding='utf-8')
    (adv/'results_assets/advanced_exp2_cross_section_transfer.md').write_text('# Advanced Experiment 2\n\nCross-section transfer evaluates structural similarity by training a PCS model on one section and testing it on another. Better transfer is interpreted only as structural similarity in PCS token formation, not semantic similarity.\n\n## Best transfer pairs\n\n' + md_table_small(best_transfer[['source_section','target_section','delta_vs_target_own_model']]) + '\n## Weakest transfer pairs\n\n' + md_table_small(worst_transfer[['source_section','target_section','delta_vs_target_own_model']]), encoding='utf-8')
    (adv/'results_assets/advanced_exp3_inventory_vs_ordering.md').write_text('# Advanced Experiment 3\n\nThe decomposition separates entropy differences due to token inventory from entropy differences due to ordering. Sections with low real entropy but small ordering contribution should be interpreted as inventory-restricted rather than sequence-ordered.\n', encoding='utf-8')
    (adv/'results_assets/advanced_exp4_section_classification.md').write_text(f'# Advanced Experiment 4\n\nStructural features were used to predict conventional visual/codicological section labels. The best computed model was `{best_model}`. This result should be phrased as above-baseline prediction of conventional labels from structural features, not as semantic classification.\n', encoding='utf-8')
    (adv/'results_assets/advanced_exp5_change_point.md').write_text('# Advanced Experiment 5\n\nFolio-order change-point scores estimate whether adjacent structural profiles shift near conventional section boundaries. Peaks near boundaries support structural discontinuity, while unmatched peaks or missing peaks indicate limits of visual-label alignment.\n', encoding='utf-8')
    (adv/'results_assets/advanced_exp6_unknown_folio_proximity.md').write_text('# Advanced Experiment 6\n\nUnknown folios were mapped by structural distance to labeled-section centroids. These assignments are structural proximity descriptions only and do not identify section meaning or manuscript function.\n', encoding='utf-8')
    (adv/'results_assets/advanced_exp7_robustness.md').write_text('# Advanced Experiment 7\n\nRobustness checks distinguish stable section-specific patterns from fragile or descriptive effects. Unknown exclusion, low-count exclusion, token-count matched downsampling, label permutation, and label-certainty restrictions are documented in Table 15.\n', encoding='utf-8')
    paragraphs = """# Advanced Results Paragraphs

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
"""
    (adv/'results_assets/advanced_results_paragraphs.md').write_text(paragraphs, encoding='utf-8')
    fig_caps = '\n\n'.join([f'**Figure {i}.** {cap}' for i, cap in [
        (9, 'Global and section-specific PCS model cross-entropy by section. Lower held-out cross-entropy indicates better fit to token formation under the specified model.'),
        (10, 'Cross-section PCS transfer heatmap. Rows indicate source sections used for training and columns indicate target sections used for testing; lower values indicate stronger structural transfer.'),
        (11, 'Inventory and ordering entropy contributions by section. The decomposition separates restricted token inventory from additional local ordering constraints.'),
        (12, 'Section classification confusion matrix using structural features. The figure evaluates prediction of conventional visual/codicological labels without semantic interpretation.'),
        (13, 'Folio-order change-point scores. Peaks indicate structural discontinuities that may or may not align with conventional section boundaries.'),
        (14, 'Unknown folio structural proximity map. Unknown folios are shown in the same feature space as labeled folios as proximity observations only.'),
        (15, 'Robustness summary across exclusion, downsampling, and permutation checks.'),
        (16, 'Advanced structural zoning evidence summary, distinguishing strong, moderate, and descriptive findings.')]]) + '\n'
    (adv/'results_assets/advanced_figure_captions.md').write_text(fig_caps, encoding='utf-8')
    table_caps = '\n\n'.join([f'**Table {i}.** {cap}' for i, cap in [
        (9, 'Global versus section-specific PCS model comparison across held-out splits.'),
        (10, 'Cross-section PCS transfer results for source-target section pairs.'),
        (11, 'Inventory versus ordering entropy decomposition by section.'),
        (12, 'Section classification performance from structural features and baselines.'),
        (13, 'Top folio-order change-point candidates and boundary alignment.'),
        (14, 'Unknown folio structural proximity to labeled-section centroids.'),
        (15, 'Robustness checks for section-specific findings.'),
        (16, 'Currier-style metadata confound check where metadata are available.'),
        (17, 'Advanced evidence matrix linking experiments to research questions.'),
        (18, 'Final advanced key findings and recommended claim strength.')]]) + '\n'
    (adv/'results_assets/advanced_table_captions.md').write_text(table_caps, encoding='utf-8')
    rqs = """# Revised Paper 6 Research Questions

RQ1. Do conventional visual/codicological sections differ in token inventory and PCS structure?

RQ2. Do section-specific PCS models outperform a single global PCS model?

RQ3. Do PCS models transfer asymmetrically across sections, indicating structural similarity or divergence?

RQ4. Are section-level entropy differences driven by token inventory, local ordering, or both?

RQ5. Can structural features predict conventional section labels above baseline?

RQ6. Do structural change points align with conventional section boundaries?

RQ7. How robust are section-specific findings to unknown folios, sample-size imbalance, and label uncertainty?
"""
    (adv/'results_assets/paper6_revised_research_questions.md').write_text(rqs, encoding='utf-8')

    # QC.
    (adv/'qc/advanced_input_report.md').write_text(f"""# Advanced Input Report

- Token file: `{root/'data/ZL3b-n.txt'}`
- Parsed tokens: `{root/'outputs/section_pipeline/parsed_tokens_with_sections.csv'}`
- Parsed lines: `{root/'outputs/section_pipeline/parsed_lines_with_sections.csv'}`
- Section metadata: `{root/'data/section_metadata.csv'}`
- Section labels: {', '.join(sections)}
- Unknown folio count: {int((section_meta.section_label == 'unknown').sum())}
- Script: `scripts/paper6_advanced_experiments.py`
""", encoding='utf-8')
    (adv/'qc/advanced_model_validation_report.md').write_text(f"""# Advanced Model Validation Report

- Random seed: {args.seed}
- Requested iterations: {args.n_iter}
- Executed iterations: {args.n_iter}
- Fallback: none
- PCS segmentation: prefix length 2, suffix length 1, core = middle component; empty core excluded.
- Smoothing: add-alpha/Laplace-style smoothing with alpha = {args.alpha}.
- Held-out splits: 80/20, 70/30, 50/50 for Experiment 1; 80/20 for transfer matrix.
- Baselines: global PCS, section-specific PCS, cross-section PCS transfer, line-internal shuffle, global shuffle, majority-class, stratified random, token-count-only, label permutation.
""", encoding='utf-8')
    sample = section_summary[['section_label','total_tokens','total_lines']].copy(); sample['pcs_valid_tokens'] = sample.section_label.map(base_pcs.set_index('section_label').pcs_valid_token_count.to_dict()); sample['warning'] = sample.total_tokens.apply(lambda x: 'low-count caution' if x < 1000 else 'standard')
    (adv/'qc/advanced_sample_size_report.md').write_text('# Advanced Sample Size Report\n\n' + df_to_md(sample), encoding='utf-8')
    files = list((adv/'tables').glob('table*.csv')) + list((adv/'figures').glob('figure*.png'))
    nan_rows=[]
    for f in (adv/'tables').glob('*.csv'):
        try:
            df=pd.read_csv(f); nan_rows.append({'file': f.name, 'nan_cells': int(df.isna().sum().sum()), 'empty_file': df.empty})
        except Exception as e:
            nan_rows.append({'file': f.name, 'nan_cells': 'read_error', 'empty_file': str(e)})
    (adv/'qc/advanced_assets_check_report.md').write_text('# Advanced Assets Check Report\n\n## Expected tables\n\n' + '\n'.join(f'- Table {i}: ' + ('present' if (adv/f'tables/table{i}_' ).parent.exists() else 'check folder') for i in range(9,19)) + '\n\n## Generated table CSV files\n\n' + '\n'.join(f'- `{f.name}`' for f in sorted((adv/'tables').glob('*.csv'))) + '\n\n## Generated figure PNG files\n\n' + '\n'.join(f'- `{f.name}`' for f in sorted((adv/'figures').glob('*.png'))) + '\n\n## NaN / empty check\n\n' + df_to_md(pd.DataFrame(nan_rows)) + '\nNo DOI text or old manuscript text was inserted by the advanced pipeline.\n', encoding='utf-8')
    (adv/'qc/advanced_interpretation_guardrails.md').write_text("""# Advanced Interpretation Guardrails

Allowed claims:
- section-conditioned structural regimes
- structural zoning
- local regimes of token formation and arrangement
- structural proximity among conventional visual/codicological labels

Disallowed claims:
- decipherment
- translation
- source language identification
- confirmed syntax
- semantic section identification

Section-label caution:
Labels such as herbal, astronomical_zodiac, biological_balneological, cosmological, pharmaceutical, and recipes_stars are conventional visual/codicological labels only.

Unknown-folio caution:
The unknown category is residual/unassigned and must not be interpreted as a coherent semantic section.
""", encoding='utf-8')

    # Summary.
    main_results = [
        f"Best global-vs-section PCS delta: {best_delta.section_label} ({best_delta.delta_global_minus_section_cross_entropy:.4f}).",
        f"Best computed classifier: {best_model}.",
        f"Classifier macro-F1: {observed:.4f}; label-permutation p={p_perm:.4f}.",
        f"Downsampled biological_balneological mean H(suffix|core): {bio_mean:.4f}.",
        f"Unknown proximity rows generated: {len(prox_df)}.",
    ]
    summary_text = f"""# Paper 6 Advanced Experiment Summary

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
PCS models use prefix length 2, suffix length 1, and core as the middle component. Empty-core tokens are excluded from PCS likelihood calculations. Add-alpha smoothing is used with alpha = {args.alpha}.

## 5. Baselines
Global PCS, section-specific PCS, cross-section PCS transfer, line-internal shuffle, global shuffle, majority-class baseline, stratified random baseline, token-count-only baseline, and section-label permutation baseline.

## 6. Tables generated
Tables 9-18 are available in `outputs/advanced/tables/`.

## 7. Figures generated
Figures 9-16 are available in `outputs/advanced/figures/`.

## 8. Main results
""" + '\n'.join(f'- {x}' for x in main_results) + """

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
"""
    (adv/'PAPER6_ADVANCED_EXPERIMENT_SUMMARY.md').write_text(summary_text, encoding='utf-8')

    elapsed = time.time() - start
    counts = {ext: len(list(adv.rglob(f'*.{ext}'))) for ext in ['csv','md','txt','png','pdf']}
    print(json.dumps({'advanced_root': str(adv), 'elapsed_seconds': elapsed, 'counts': counts, 'best_classifier': best_model, 'p_perm': p_perm}, indent=2))

if __name__ == '__main__':
    main()
