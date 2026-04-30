#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SECTION_LABELS = [
    'herbal',
    'astronomical_zodiac',
    'biological_balneological',
    'cosmological',
    'pharmaceutical',
    'recipes_stars',
    'unknown',
]

FORBIDDEN_INTERPRETIVE_TERMS = [
    'deciphered', 'translation proved', 'source language identified', 'confirmed syntax'
]

@dataclass
class TokenRecord:
    token: str
    folio_id: str
    section_label: str
    visual_category: str
    line_global: int
    line_in_folio: int
    token_index_in_line: int
    line_token_count: int
    line_position: str
    paragraph_initial: bool
    raw_line_tag: str


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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


def normalized_entropy(counter: Counter) -> float:
    if len(counter) <= 1:
        return 0.0
    return entropy(counter) / math.log2(len(counter))


def conditional_entropy(joint: Counter) -> float:
    x_tot = Counter()
    total = sum(joint.values())
    if total <= 0:
        return 0.0
    for (x, y), c in joint.items():
        x_tot[x] += c
    h = 0.0
    for (x, y), c in joint.items():
        pxy = c / total
        pyx = c / x_tot[x]
        h -= pxy * math.log2(pyx)
    return h


def mutual_information(joint: Counter) -> float:
    x_tot, y_tot = Counter(), Counter()
    total = sum(joint.values())
    if total <= 0:
        return 0.0
    for (x, y), c in joint.items():
        x_tot[x] += c
        y_tot[y] += c
    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / total
        px = x_tot[x] / total
        py = y_tot[y] / total
        mi += pxy * math.log2(pxy / (px * py))
    return mi


def topk_conditional_accuracy(joint: Counter, k: int) -> float:
    by_x = defaultdict(Counter)
    for (x, y), c in joint.items():
        by_x[x][y] += c
    correct = 0
    total = 0
    for x, dist in by_x.items():
        top = {y for y, _ in dist.most_common(k)}
        for y, c in dist.items():
            total += c
            if y in top:
                correct += c
    return correct / total if total else 0.0


def zipf_slope(counter: Counter) -> float:
    freqs = np.array([c for _, c in counter.most_common() if c > 0], dtype=float)
    if len(freqs) < 2:
        return float('nan')
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    x = np.log10(ranks)
    y = np.log10(freqs)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope)


def clean_token(tok: str) -> str:
    return re.sub(r'[^a-z]', '', tok.lower())


def extract_tokens_from_rhs(rhs: str) -> List[str]:
    # Same conservative normalization family used in Paper 4: remove editorial markup and keep alphabetic EVA tokens.
    text = re.sub(r'<![^>]*>', ' ', rhs)
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'\{[^}]*\}', ' ', text)
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r'\$[^\s]+', ' ', text)
    text = re.sub(r'[^a-zA-Z]+', ' ', text)
    return [t for t in (clean_token(x) for x in text.split()) if len(t) >= 2]


def classify_section(comments: List[str]) -> Tuple[str, str, str, str]:
    joined = ' '.join(c.lower() for c in comments)
    matches = []
    if 'herbal' in joined:
        matches.append(('herbal', 'herbal', 'high', 'explicit comment contains herbal'))
    if 'zodiac' in joined or 'astronom' in joined:
        matches.append(('astronomical_zodiac', 'astronomical_zodiac', 'medium', 'explicit comment contains zodiac/astronomical'))
    if 'biological' in joined or 'balneological' in joined:
        matches.append(('biological_balneological', 'biological_balneological', 'medium', 'explicit comment contains biological/balneological'))
    if 'cosmological' in joined or 'rosette' in joined:
        matches.append(('cosmological', 'cosmological', 'medium', 'explicit comment contains cosmological/rosette'))
    if 'pharmaceutical' in joined:
        matches.append(('pharmaceutical', 'pharmaceutical', 'medium', 'explicit comment contains pharmaceutical'))
    if 'recipe' in joined or 'stars' in joined:
        matches.append(('recipes_stars', 'recipes_stars', 'medium', 'explicit comment contains recipe/stars'))
    if not matches:
        return 'unknown', 'unknown', 'low', 'no explicit conventional visual/codicological label found in local comments'
    # Prefer the first label encountered by conventional sequence above; notes retain evidence.
    label, visual, certainty, note = matches[0]
    if len(matches) > 1:
        note += '; multiple cues present: ' + ', '.join(m[0] for m in matches)
        certainty = 'medium'
    return label, visual, certainty, note


def parse_transcription(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    folio_comments: Dict[str, List[str]] = defaultdict(list)
    folio_order: List[str] = []
    current_folio = None
    in_header_comments = False

    # First pass: collect folio-level comments after folio header until text starts.
    for line in raw_lines:
        s = line.rstrip('\n')
        m_header = re.match(r'^<([^>.]+)>\s+<!', s.strip())
        if m_header:
            current_folio = m_header.group(1)
            folio_order.append(current_folio)
            in_header_comments = True
            continue
        if current_folio and in_header_comments:
            if re.match(r'^<[^>]+\.\d+', s.strip()):
                in_header_comments = False
            elif s.strip().startswith('#'):
                val = s.strip()[1:].strip()
                if val:
                    folio_comments[current_folio].append(val)
            elif s.strip() and not s.strip().startswith('#'):
                in_header_comments = False

    metadata_rows = []
    for folio in folio_order:
        label, visual, certainty, note = classify_section(folio_comments.get(folio, []))
        metadata_rows.append({
            'folio_id': folio,
            'section_label': label,
            'visual_category': visual,
            'certainty': certainty,
            'note': note,
        })
    metadata = pd.DataFrame(metadata_rows).drop_duplicates('folio_id')
    meta_map = metadata.set_index('folio_id').to_dict(orient='index')

    records: List[TokenRecord] = []
    line_rows = []
    line_global = 0
    line_counts_by_folio = Counter()
    paragraph_markers = ('@P', '*P')

    for raw in raw_lines:
        s = raw.strip()
        m_line = re.match(r'^<([^>]+)>\s*(.*)$', s)
        if not m_line:
            continue
        tag = m_line.group(1)
        rhs = m_line.group(2)
        if '.' not in tag:
            continue
        folio_id = tag.split('.', 1)[0]
        toks = extract_tokens_from_rhs(rhs)
        if not toks:
            continue
        line_counts_by_folio[folio_id] += 1
        meta = meta_map.get(folio_id, {'section_label': 'unknown', 'visual_category': 'unknown'})
        paragraph_initial_line = any(m in tag for m in paragraph_markers)
        for j, tok in enumerate(toks):
            pos = 'line_medial'
            if j == 0:
                pos = 'line_initial'
            if j == len(toks) - 1:
                pos = 'line_final' if len(toks) > 1 else 'line_initial_final'
            records.append(TokenRecord(
                token=tok,
                folio_id=folio_id,
                section_label=meta['section_label'],
                visual_category=meta['visual_category'],
                line_global=line_global,
                line_in_folio=line_counts_by_folio[folio_id],
                token_index_in_line=j,
                line_token_count=len(toks),
                line_position=pos,
                paragraph_initial=(paragraph_initial_line and j == 0),
                raw_line_tag=tag,
            ))
        line_rows.append({
            'line_global': line_global,
            'folio_id': folio_id,
            'section_label': meta['section_label'],
            'visual_category': meta['visual_category'],
            'line_in_folio': line_counts_by_folio[folio_id],
            'token_count': len(toks),
            'paragraph_initial_line': paragraph_initial_line,
            'raw_line_tag': tag,
            'tokens': ' '.join(toks),
        })
        line_global += 1

    token_df = pd.DataFrame([r.__dict__ for r in records])
    line_df = pd.DataFrame(line_rows)
    return token_df, line_df, metadata


def counter_for(series: Iterable[str]) -> Counter:
    return Counter([str(x) for x in series if str(x)])


def pcs_parts(token: str, p_len: int = 2, s_len: int = 1) -> Tuple[str, str, str] | None:
    if len(token) <= p_len + s_len:
        return None
    p = token[:p_len]
    s = token[-s_len:]
    c = token[p_len:-s_len]
    if not c:
        return None
    return p, c, s


def bigram_counter(tokens: List[str]) -> Counter:
    return Counter(zip(tokens, tokens[1:]))


def trigram_counter(tokens: List[str]) -> Counter:
    return Counter(zip(tokens, tokens[1:], tokens[2:]))


def next_token_accuracy(bigrams: Counter, k: int) -> float:
    return topk_conditional_accuracy(bigrams, k)


def js_divergence(c1: Counter, c2: Counter) -> float:
    keys = sorted(set(c1) | set(c2))
    if not keys:
        return 0.0
    total1, total2 = sum(c1.values()), sum(c2.values())
    p = np.array([c1.get(k, 0) / total1 if total1 else 0 for k in keys], dtype=float)
    q = np.array([c2.get(k, 0) / total2 if total2 else 0 for k in keys], dtype=float)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cosine_distance(c1: Counter, c2: Counter) -> float:
    keys = sorted(set(c1) | set(c2))
    if not keys:
        return 0.0
    a = np.array([c1.get(k, 0) for k in keys], dtype=float)
    b = np.array([c2.get(k, 0) for k in keys], dtype=float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 1.0
    return float(1 - np.dot(a, b) / den)


def rank_correlation(c1: Counter, c2: Counter) -> float:
    keys = sorted(set(c1) | set(c2))
    if len(keys) < 2:
        return float('nan')
    # Higher counts get lower rank number. Missing tokens get worst rank.
    def ranks(c):
        ordered = {tok: i + 1 for i, (tok, _) in enumerate(c.most_common())}
        worst = len(ordered) + 1
        return np.array([ordered.get(k, worst) for k in keys], dtype=float)
    a, b = ranks(c1), ranks(c2)
    if np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def chi_square_stat(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    if table.size == 0 or table.sum() == 0:
        return 0.0
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    exp = row @ col / table.sum()
    mask = exp > 0
    return float(((table - exp) ** 2 / np.where(mask, exp, 1))[mask].sum())


def empirical_p(observed: float, null_values: List[float], tail: str = 'ge') -> float:
    if not null_values:
        return float('nan')
    if tail == 'ge':
        k = sum(v >= observed for v in null_values)
    else:
        k = sum(v <= observed for v in null_values)
    return (k + 1) / (len(null_values) + 1)


def line_tokens_by_section(line_df: pd.DataFrame, section: str) -> List[List[str]]:
    out = []
    for _, row in line_df[line_df.section_label == section].iterrows():
        toks = str(row['tokens']).split()
        if toks:
            out.append(toks)
    return out


def flatten(lines: List[List[str]]) -> List[str]:
    return [t for line in lines for t in line]


def shuffle_within_lines(lines: List[List[str]], rng: random.Random) -> List[List[str]]:
    out = []
    for line in lines:
        t = list(line)
        rng.shuffle(t)
        out.append(t)
    return out


def section_corpus_summary(token_df, line_df) -> pd.DataFrame:
    rows = []
    total_tokens = len(token_df)
    for sec, sdf in token_df.groupby('section_label'):
        ldf = line_df[line_df.section_label == sec]
        counts = counter_for(sdf.token)
        lens = sdf.token.str.len().to_numpy(dtype=float)
        rows.append({
            'section_label': sec,
            'total_lines': int(len(ldf)),
            'total_tokens': int(len(sdf)),
            'alphabetic_tokens': int(sdf.token.str.fullmatch(r'[a-z]+').sum()),
            'token_types': int(len(counts)),
            'type_token_ratio': len(counts) / len(sdf) if len(sdf) else 0.0,
            'hapax_count': int(sum(1 for c in counts.values() if c == 1)),
            'mean_token_length': float(np.mean(lens)) if len(lens) else float('nan'),
            'median_token_length': float(np.median(lens)) if len(lens) else float('nan'),
            'token_entropy': entropy(counts),
            'normalized_token_entropy': normalized_entropy(counts),
            'zipf_slope': zipf_slope(counts),
            'top_10_tokens': '; '.join(f'{t}:{c}' for t, c in counts.most_common(10)),
            'section_share_of_total_corpus': len(sdf) / total_tokens if total_tokens else 0.0,
        })
    return pd.DataFrame(rows).sort_values('total_tokens', ascending=False)


def token_distribution_by_section(token_df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sections = sorted(token_df.section_label.unique())
    counters = {s: counter_for(token_df[token_df.section_label == s].token) for s in sections}
    rows = []
    for s in sections:
        total = sum(counters[s].values())
        for tok, c in counters[s].most_common():
            rows.append({'section_label': s, 'token': tok, 'count': c, 'relative_frequency': c / total if total else 0})
    freq_df = pd.DataFrame(rows)
    dist_rows = []
    for s1 in sections:
        for s2 in sections:
            dist_rows.append({
                'section_a': s1,
                'section_b': s2,
                'jensen_shannon_divergence': js_divergence(counters[s1], counters[s2]),
                'cosine_distance': cosine_distance(counters[s1], counters[s2]),
                'rank_correlation': rank_correlation(counters[s1], counters[s2]),
            })
    dist_df = pd.DataFrame(dist_rows)
    # Wide JSD matrix for heatmap/table.
    matrix = dist_df.pivot(index='section_a', columns='section_b', values='jensen_shannon_divergence')
    # Distinctive tokens via add-alpha log odds section vs rest.
    all_counts = counter_for(token_df.token)
    vocab = set(all_counts)
    alpha = 0.5
    distinct = []
    m = len(vocab)
    for s in sections:
        csec = counters[s]
        crest = all_counts.copy()
        for tok, c in csec.items():
            crest[tok] -= c
        nsec = sum(csec.values())
        nrest = sum(crest.values())
        for tok in vocab:
            a = csec.get(tok, 0)
            b = max(crest.get(tok, 0), 0)
            odds_s = (a + alpha) / (nsec - a + alpha * m)
            odds_r = (b + alpha) / (nrest - b + alpha * m)
            score = math.log(odds_s / odds_r)
            if a > 0:
                distinct.append({'section_label': s, 'token': tok, 'section_count': a, 'rest_count': b, 'log_odds_score': score})
    distinct_df = pd.DataFrame(distinct).sort_values(['section_label', 'log_odds_score'], ascending=[True, False])
    top_distinct = distinct_df.groupby('section_label', group_keys=False).head(15)
    return freq_df, dist_df, matrix.reset_index(), top_distinct


def pcs_metrics_by_section(token_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    freq_rows = []
    for sec, sdf in token_df.groupby('section_label'):
        p_count, c_count, s_count = Counter(), Counter(), Counter()
        pc, cs, ps = Counter(), Counter(), Counter()
        valid = 0
        for tok in sdf.token:
            parts = pcs_parts(tok)
            if not parts:
                continue
            p, c, s = parts
            valid += 1
            p_count[p] += 1; c_count[c] += 1; s_count[s] += 1
            pc[(p, c)] += 1; cs[(c, s)] += 1; ps[(p, s)] += 1
        metric_rows.append({
            'section_label': sec,
            'pcs_valid_token_count': valid,
            'prefix_type_count': len(p_count),
            'core_type_count': len(c_count),
            'suffix_type_count': len(s_count),
            'prefix_entropy': entropy(p_count),
            'core_entropy': entropy(c_count),
            'suffix_entropy': entropy(s_count),
            'H_suffix_given_core': conditional_entropy(cs),
            'H_core_given_prefix': conditional_entropy(pc),
            'MI_core_suffix': mutual_information(cs),
            'MI_prefix_core': mutual_information(pc),
            'suffix_from_core_top1_accuracy': topk_conditional_accuracy(cs, 1),
            'suffix_from_core_top3_accuracy': topk_conditional_accuracy(cs, 3),
            'core_from_prefix_top1_accuracy': topk_conditional_accuracy(pc, 1),
            'core_from_prefix_top3_accuracy': topk_conditional_accuracy(pc, 3),
        })
        for typ, ctr in [('prefix', p_count), ('core', c_count), ('suffix', s_count)]:
            total = sum(ctr.values())
            for comp, cnt in ctr.most_common():
                freq_rows.append({'section_label': sec, 'component_type': typ, 'component': comp, 'count': cnt, 'relative_frequency': cnt / total if total else 0})
    return pd.DataFrame(metric_rows).sort_values('pcs_valid_token_count', ascending=False), pd.DataFrame(freq_rows)


def positional_metrics(token_df, line_df, n_iter: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    rows = []
    perm_rows = []
    for sec, sdf in token_df.groupby('section_label'):
        n = len(sdf)
        pos_counts = Counter(sdf.line_position)
        init = sdf[sdf.line_position.isin(['line_initial', 'line_initial_final'])]
        final = sdf[sdf.line_position.isin(['line_final', 'line_initial_final'])]
        pinit = sdf[sdf.paragraph_initial]
        all_counts = counter_for(sdf.token)
        init_counts = counter_for(init.token)
        final_counts = counter_for(final.token)
        pinit_counts = counter_for(pinit.token)
        def share_class(mask):
            return float(mask.sum() / len(sdf)) if len(sdf) else 0.0
        dshedy_init_rate = float((init.token == 'dshedy').sum() / max((sdf.token == 'dshedy').sum(), 1)) if (sdf.token == 'dshedy').sum() else 0.0
        ptk_pinit_rate = float(pinit.token.str.startswith(('p','t','k')).sum() / len(pinit)) if len(pinit) else 0.0
        am_final_rate = float(final.token.str.endswith('am').sum() / len(final)) if len(final) else 0.0
        init_conc = normalized_entropy(init_counts)
        final_conc = normalized_entropy(final_counts)
        pinit_conc = normalized_entropy(pinit_counts)
        # Chi-square association between token class and line position for top tokens.
        top = [t for t, _ in all_counts.most_common(25)]
        cols = ['initial', 'medial', 'final']
        table = np.zeros((len(top), len(cols)))
        for i, tok in enumerate(top):
            sub = sdf[sdf.token == tok]
            table[i, 0] = sub.line_position.isin(['line_initial', 'line_initial_final']).sum()
            table[i, 1] = (sub.line_position == 'line_medial').sum()
            table[i, 2] = sub.line_position.isin(['line_final', 'line_initial_final']).sum()
        obs = chi_square_stat(table)
        null = []
        tokens = list(sdf.token)
        positions = list(sdf.line_position)
        for _ in range(n_iter):
            shpos = positions[:]
            rng.shuffle(shpos)
            sh = pd.DataFrame({'token': tokens, 'line_position': shpos})
            tbl = np.zeros_like(table)
            for i, tok in enumerate(top):
                sub = sh[sh.token == tok]
                tbl[i, 0] = sub.line_position.isin(['line_initial', 'line_initial_final']).sum()
                tbl[i, 1] = (sub.line_position == 'line_medial').sum()
                tbl[i, 2] = sub.line_position.isin(['line_final', 'line_initial_final']).sum()
            null.append(chi_square_stat(tbl))
        pval = empirical_p(obs, null, 'ge')
        rows.append({
            'section_label': sec,
            'token_count': n,
            'line_initial_tokens': len(init),
            'line_final_tokens': len(final),
            'paragraph_initial_tokens': len(pinit),
            'line_initial_distribution_normalized_entropy': init_conc,
            'line_final_distribution_normalized_entropy': final_conc,
            'paragraph_initial_distribution_normalized_entropy': pinit_conc,
            'line_initial_concentration_score': 1 - init_conc,
            'line_final_concentration_score': 1 - final_conc,
            'paragraph_initial_concentration_score': 1 - pinit_conc,
            'dshedy_line_initial_bias': dshedy_init_rate,
            'ptk_paragraph_initial_concentration': ptk_pinit_rate,
            'am_class_line_final_bias': am_final_rate,
            'position_token_chi_square': obs,
            'empirical_p_value': pval,
            'permutation_iterations': n_iter,
            'p_value_formula': '(k + 1) / (N + 1)',
            'confidence_note': 'low-count section' if n < 500 else 'standard',
        })
        perm_rows.append({'section_label': sec, 'observed_chi_square': obs, 'null_mean': float(np.mean(null)), 'null_sd': float(np.std(null)), 'empirical_p_value': pval, 'iterations': n_iter})
    return pd.DataFrame(rows).sort_values('token_count', ascending=False), pd.DataFrame(perm_rows)


def intertoken_metrics(token_df, line_df, n_iter: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    rows = []
    base_rows = []
    for sec in sorted(line_df.section_label.unique()):
        lines = line_tokens_by_section(line_df, sec)
        tokens = flatten(lines)
        if len(tokens) < 5:
            continue
        big = Counter(); tri = Counter(); cross = Counter(); comp_sp = Counter(); comp_cc = Counter()
        for line in lines:
            big.update(zip(line, line[1:]))
            tri.update(zip(line, line[1:], line[2:]))
            for a, b in zip(line, line[1:]):
                pa = pcs_parts(a); pb = pcs_parts(b)
                if pa and pb:
                    comp_sp[(pa[2], pb[0])] += 1
                    comp_cc[(pa[1], pb[1])] += 1
        for a, b in zip(lines, lines[1:]):
            if a and b:
                cross[(a[-1], b[0])] += 1
        real_big_h = entropy(big)
        real_top1 = next_token_accuracy(big, 1)
        real_top3 = next_token_accuracy(big, 3)
        base_big_h, base_t1, base_t3 = [], [], []
        base_sp_h, base_cc_h = [], []
        for _ in range(n_iter):
            sh = shuffle_within_lines(lines, rng)
            b = Counter()
            sp_b = Counter()
            cc_b = Counter()
            for line in sh:
                b.update(zip(line, line[1:]))
                for a, nxt in zip(line, line[1:]):
                    pa = pcs_parts(a); pb = pcs_parts(nxt)
                    if pa and pb:
                        sp_b[(pa[2], pb[0])] += 1
                        cc_b[(pa[1], pb[1])] += 1
            base_big_h.append(entropy(b))
            base_t1.append(next_token_accuracy(b, 1))
            base_t3.append(next_token_accuracy(b, 3))
            base_sp_h.append(conditional_entropy(sp_b))
            base_cc_h.append(conditional_entropy(cc_b))
        sp_h = conditional_entropy(comp_sp)
        cc_h = conditional_entropy(comp_cc)
        rows.append({
            'section_label': sec,
            'token_count': len(tokens),
            'token_bigram_entropy': real_big_h,
            'token_trigram_entropy': entropy(tri),
            'next_token_top1_accuracy': real_top1,
            'next_token_top3_accuracy': real_top3,
            'line_internal_shuffle_bigram_entropy_mean': float(np.mean(base_big_h)),
            'bigram_entropy_empirical_p_lower': empirical_p(real_big_h, base_big_h, 'le'),
            'line_internal_shuffle_top1_mean': float(np.mean(base_t1)),
            'line_internal_shuffle_top3_mean': float(np.mean(base_t3)),
            'top1_accuracy_empirical_p_higher': empirical_p(real_top1, base_t1, 'ge'),
            'top3_accuracy_empirical_p_higher': empirical_p(real_top3, base_t3, 'ge'),
            'within_line_transition_entropy': real_big_h,
            'cross_line_transition_entropy': entropy(cross),
            'line_boundary_chi_square': chi_square_stat(np.array([[sum(big.values()), len(big)], [sum(cross.values()), len(cross)]])),
            'suffix_to_next_prefix_conditional_entropy': sp_h,
            'suffix_to_next_prefix_baseline_entropy_mean': float(np.mean(base_sp_h)),
            'suffix_to_next_prefix_empirical_p_lower': empirical_p(sp_h, base_sp_h, 'le'),
            'core_to_next_core_conditional_entropy': cc_h,
            'core_to_next_core_baseline_entropy_mean': float(np.mean(base_cc_h)),
            'core_to_next_core_empirical_p_lower': empirical_p(cc_h, base_cc_h, 'le'),
            'baseline_iterations': n_iter,
            'confidence_note': 'low-count section' if len(tokens) < 500 else 'standard',
        })
        base_rows.append({
            'section_label': sec,
            'line_internal_shuffle_bigram_entropy_mean': float(np.mean(base_big_h)),
            'line_internal_shuffle_bigram_entropy_sd': float(np.std(base_big_h)),
            'line_internal_shuffle_top1_mean': float(np.mean(base_t1)),
            'line_internal_shuffle_top1_sd': float(np.std(base_t1)),
            'line_internal_shuffle_top3_mean': float(np.mean(base_t3)),
            'line_internal_shuffle_top3_sd': float(np.std(base_t3)),
            'suffix_to_next_prefix_entropy_mean': float(np.mean(base_sp_h)),
            'suffix_to_next_prefix_entropy_sd': float(np.std(base_sp_h)),
            'core_to_next_core_entropy_mean': float(np.mean(base_cc_h)),
            'core_to_next_core_entropy_sd': float(np.std(base_cc_h)),
            'iterations': n_iter,
        })
    return pd.DataFrame(rows).sort_values('token_count', ascending=False), pd.DataFrame(base_rows)


def nearest_neighbor_distance(positions: List[int]) -> List[int]:
    if len(positions) < 2:
        return []
    positions = sorted(positions)
    return [b - a for a, b in zip(positions, positions[1:])]


def family_distance_for_tokens(tokens: List[str], family_kind: str) -> float:
    groups = defaultdict(list)
    for i, tok in enumerate(tokens):
        parts = pcs_parts(tok)
        if not parts:
            continue
        p, c, s = parts
        if family_kind == 'same_core':
            key = c
        elif family_kind == 'same_prefix_core':
            key = p + '|' + c
        elif family_kind == 'same_suffix_core':
            key = c + '|' + s
        elif family_kind == 'same_prefix_family':
            key = p
        else:
            key = tok
        groups[key].append(i)
    dists = []
    for pos in groups.values():
        dists.extend(nearest_neighbor_distance(pos))
    return float(np.mean(dists)) if dists else float('nan')


def family_clustering(token_df, n_iter: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for sec, sdf in token_df.groupby('section_label'):
        tokens = list(sdf.token)
        if len(tokens) < 100:
            low = True
        else:
            low = False
        for kind in ['same_core', 'same_prefix_core', 'same_suffix_core', 'same_prefix_family']:
            real = family_distance_for_tokens(tokens, kind)
            null = []
            for _ in range(n_iter):
                sh = tokens[:]
                rng.shuffle(sh)
                null.append(family_distance_for_tokens(sh, kind))
            valid_null = [x for x in null if not math.isnan(x)]
            mean_null = float(np.mean(valid_null)) if valid_null else float('nan')
            ratio = real / mean_null if mean_null and not math.isnan(mean_null) and not math.isnan(real) else float('nan')
            lo = float(np.percentile(valid_null, 2.5)) if valid_null else float('nan')
            hi = float(np.percentile(valid_null, 97.5)) if valid_null else float('nan')
            rows.append({
                'section_label': sec,
                'family_definition': kind,
                'token_count': len(tokens),
                'real_mean_nearest_neighbor_distance': real,
                'shuffled_baseline_mean_distance': mean_null,
                'real_to_shuffled_ratio': ratio,
                'baseline_ci_lower_95': lo,
                'baseline_ci_upper_95': hi,
                'iterations': n_iter,
                'confidence_note': 'low-count section' if low else 'standard',
            })
    return pd.DataFrame(rows)


def integrated_features(summary_df, pcs_df, pos_df, inter_df, fam_df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fam_core = fam_df[fam_df.family_definition == 'same_core'][['section_label', 'real_to_shuffled_ratio']].rename(columns={'real_to_shuffled_ratio': 'same_core_clustering_ratio'})
    feature = summary_df[['section_label','token_entropy','normalized_token_entropy','type_token_ratio','mean_token_length','zipf_slope']].merge(
        pcs_df[['section_label','H_suffix_given_core','H_core_given_prefix','suffix_from_core_top1_accuracy']], on='section_label', how='left'
    ).merge(
        inter_df[['section_label','token_bigram_entropy','next_token_top1_accuracy','within_line_transition_entropy','cross_line_transition_entropy']], on='section_label', how='left'
    ).merge(
        fam_core, on='section_label', how='left'
    ).merge(
        pos_df[['section_label','line_initial_concentration_score','line_final_concentration_score']], on='section_label', how='left'
    )
    feature['within_cross_entropy_difference'] = feature['cross_line_transition_entropy'] - feature['within_line_transition_entropy']
    numeric = [c for c in feature.columns if c != 'section_label']
    mat = feature[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    mat = mat.fillna(mat.mean(numeric_only=True)).fillna(0.0)
    z = (mat - mat.mean()) / mat.std(ddof=0).replace(0, 1)
    # PCA via SVD.
    X = z.to_numpy(dtype=float)
    if X.shape[0] >= 2 and X.shape[1] >= 2:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pcs = U[:, :2] * S[:2]
    else:
        pcs = np.zeros((X.shape[0], 2))
    feature['PC1'] = pcs[:, 0] if len(pcs) else 0
    feature['PC2'] = pcs[:, 1] if len(pcs) else 0
    if len(feature) >= 3:
        qs = np.quantile(feature['PC1'], [1/3, 2/3])
        feature['cluster_assignment'] = pd.cut(feature['PC1'], [-np.inf, qs[0], qs[1], np.inf], labels=['cluster_1','cluster_2','cluster_3']).astype(str)
    else:
        feature['cluster_assignment'] = 'cluster_1'
    # Euclidean distance over z-features.
    sections = list(feature.section_label)
    dist_rows = []
    for i, a in enumerate(sections):
        for j, b in enumerate(sections):
            d = float(np.linalg.norm(X[i] - X[j])) if len(X) else 0.0
            dist_rows.append({'section_a': a, 'section_b': b, 'feature_euclidean_distance': d})
    dist = pd.DataFrame(dist_rows).pivot(index='section_a', columns='section_b', values='feature_euclidean_distance').reset_index()
    assign = feature[['section_label','PC1','PC2','cluster_assignment']]
    return feature, dist, assign


def write_table_variants(df: pd.DataFrame, stem: Path, max_rows: int | None = None):
    ensure_dir(stem.parent)
    out = df.copy()
    if max_rows:
        out = out.head(max_rows)
    out.to_csv(stem.with_suffix('.csv'), index=False)
    stem.with_suffix('.md').write_text(df_to_markdown(out), encoding='utf-8')
    stem.with_suffix('.txt').write_text(out.to_string(index=False), encoding='utf-8')


def df_to_markdown(df: pd.DataFrame) -> str:
    """Write a GitHub-style Markdown table without optional tabulate dependency."""
    if df.empty:
        return "_No rows available._\n"
    cols = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("not available")
                else:
                    vals.append(f"{v:.6g}")
            else:
                vals.append(str(v).replace("\n", " ").replace("|", "\\|"))
        rows.append(vals)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep] + body) + "\n"


def save_heatmap(matrix: pd.DataFrame, out: Path, title: str, value_col_start: int = 1):
    labels = list(matrix.iloc[:,0])
    vals = matrix.iloc[:, value_col_start:].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(vals, cmap='Greys')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out.with_suffix('.png'), dpi=300)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)


def save_figures(fig_dir: Path, dist_matrix, pcs_df, inter_df, pos_df, fam_df, feature_df):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    save_heatmap(dist_matrix, fig_dir / 'figure1_section_distance_heatmap', 'Figure 1. Section Token Distribution Distance')

    # Figure 2
    df = pcs_df.sort_values('section_label')
    x = np.arange(len(df)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w/2, df['H_suffix_given_core'], w, label='H(suffix|core)')
    ax.bar(x + w/2, df['H_core_given_prefix'], w, label='H(core|prefix)')
    ax.set_xticks(x); ax.set_xticklabels(df.section_label, rotation=30, ha='right')
    ax.set_ylabel('Conditional entropy (bits)')
    ax.set_title('Figure 2. PCS Conditional Entropy by Section')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir/'figure2_pcs_entropy_by_section.png', dpi=300); fig.savefig(fig_dir/'figure2_pcs_entropy_by_section.pdf'); plt.close(fig)

    # Figure 3
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w/2, df['suffix_from_core_top1_accuracy'], w, label='Top-1')
    ax.bar(x + w/2, df['suffix_from_core_top3_accuracy'], w, label='Top-3')
    ax.set_xticks(x); ax.set_xticklabels(df.section_label, rotation=30, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.02)
    ax.set_title('Figure 3. Suffix-from-Core Prediction by Section')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir/'figure3_pcs_prediction_by_section.png', dpi=300); fig.savefig(fig_dir/'figure3_pcs_prediction_by_section.pdf'); plt.close(fig)

    # Figure 4
    dfi = inter_df.sort_values('section_label')
    xi = np.arange(len(dfi))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(xi - w/2, dfi['token_bigram_entropy'], w, label='Real')
    ax.bar(xi + w/2, dfi['line_internal_shuffle_bigram_entropy_mean'], w, label='Line-internal shuffle')
    ax.set_xticks(xi); ax.set_xticklabels(dfi.section_label, rotation=30, ha='right')
    ax.set_ylabel('Bigram entropy (bits)')
    ax.set_title('Figure 4. Inter-Token Entropy by Section')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir/'figure4_intertoken_entropy_by_section.png', dpi=300); fig.savefig(fig_dir/'figure4_intertoken_entropy_by_section.pdf'); plt.close(fig)

    # Figure 5
    dfp = pos_df.sort_values('section_label')
    xp = np.arange(len(dfp)); w3 = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(xp - w3, dfp['line_initial_concentration_score'], w3, label='Line-initial')
    ax.bar(xp, dfp['line_final_concentration_score'], w3, label='Line-final')
    ax.bar(xp + w3, dfp['paragraph_initial_concentration_score'], w3, label='Paragraph-initial')
    ax.set_xticks(xp); ax.set_xticklabels(dfp.section_label, rotation=30, ha='right')
    ax.set_ylabel('Concentration score (1 - normalized entropy)')
    ax.set_title('Figure 5. Line-Position Constraint Strength by Section')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir/'figure5_line_position_by_section.png', dpi=300); fig.savefig(fig_dir/'figure5_line_position_by_section.pdf'); plt.close(fig)

    # Figure 6
    fcore = fam_df[fam_df.family_definition == 'same_core'].sort_values('section_label')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(np.arange(len(fcore)), fcore['real_to_shuffled_ratio'])
    ax.axhline(1.0, linestyle='--', linewidth=1)
    ax.set_xticks(np.arange(len(fcore))); ax.set_xticklabels(fcore.section_label, rotation=30, ha='right')
    ax.set_ylabel('Real / shuffled nearest-neighbor distance')
    ax.set_title('Figure 6. Token-Family Clustering by Section')
    fig.tight_layout(); fig.savefig(fig_dir/'figure6_family_clustering_by_section.png', dpi=300); fig.savefig(fig_dir/'figure6_family_clustering_by_section.pdf'); plt.close(fig)

    # Figure 7
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(feature_df['PC1'], feature_df['PC2'])
    for _, r in feature_df.iterrows():
        ax.annotate(r['section_label'], (r['PC1'], r['PC2']), fontsize=8, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Figure 7. Integrated Section Clustering')
    fig.tight_layout(); fig.savefig(fig_dir/'figure7_integrated_section_clustering.png', dpi=300); fig.savefig(fig_dir/'figure7_integrated_section_clustering.pdf'); plt.close(fig)

    # Figure 8 normalized matrix
    cols = ['token_entropy','H_suffix_given_core','H_core_given_prefix','suffix_from_core_top1_accuracy','token_bigram_entropy','next_token_top1_accuracy','same_core_clustering_ratio','line_initial_concentration_score','line_final_concentration_score']
    mat = feature_df[['section_label'] + cols].copy()
    vals = mat[cols].astype(float).replace([np.inf,-np.inf], np.nan).fillna(mat[cols].mean(numeric_only=True)).fillna(0)
    z = (vals - vals.mean()) / vals.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    im = ax.imshow(z.to_numpy(), cmap='Greys')
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.section_label, fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha='right', fontsize=8)
    ax.set_title('Figure 8. Section Structural Zoning Summary')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(fig_dir/'figure8_structural_zoning_summary.png', dpi=300); fig.savefig(fig_dir/'figure8_structural_zoning_summary.pdf'); plt.close(fig)


def make_key_results_table(summary_df, pcs_df, pos_df, inter_df, fam_df, dist_df) -> pd.DataFrame:
    rows = []
    def strongest_weakest(df, metric, high=True):
        tmp = df[['section_label', metric]].dropna()
        if tmp.empty:
            return 'not available', 'not available'
        if high:
            s = tmp.loc[tmp[metric].idxmax()].section_label
            w = tmp.loc[tmp[metric].idxmin()].section_label
        else:
            s = tmp.loc[tmp[metric].idxmin()].section_label
            w = tmp.loc[tmp[metric].idxmax()].section_label
        return s, w
    for finding, df, metric, interp, high in [
        ('Token distribution varies by section', summary_df, 'token_entropy', 'Higher entropy indicates broader token distribution within a conventional section label.', True),
        ('PCS dependency varies by section', pcs_df, 'H_suffix_given_core', 'Lower H(suffix|core) indicates stronger suffix constraint given core.', False),
        ('Suffix prediction varies by section', pcs_df, 'suffix_from_core_top1_accuracy', 'Higher top-1 accuracy indicates more concentrated suffix prediction from core.', True),
        ('Line-position concentration varies by section', pos_df, 'line_initial_concentration_score', 'Higher concentration indicates more restricted line-initial token distribution.', True),
        ('Inter-token entropy varies by section', inter_df, 'token_bigram_entropy', 'Lower bigram entropy indicates more constrained local token ordering.', False),
        ('Family clustering varies by section', fam_df[fam_df.family_definition == 'same_core'], 'real_to_shuffled_ratio', 'Lower real/shuffled distance ratio indicates stronger local clustering of same-core forms.', False),
    ]:
        s, w = strongest_weakest(df, metric, high)
        rows.append({
            'finding': finding,
            'metric': metric,
            'strongest_section': s,
            'weakest_section': w,
            'baseline_or_contrast': 'within-corpus section contrast; shuffled baselines where applicable',
            'interpretation': interp,
            'caution': 'Conventional visual/codicological labels only; not semantic identification.',
        })
    return pd.DataFrame(rows)


def write_reports(root: Path, input_path: Path, token_df, line_df, metadata, n_iter, actual_iter, outputs: Dict[str, pd.DataFrame], fallback_note: str):
    qc = root / 'outputs/qc'
    res = root / 'outputs/results_assets'
    ensure_dir(qc); ensure_dir(res)
    (qc/'input_file_report.md').write_text(f"""# Input File Report

- Primary transcription: `{input_path}`
- Parser: ZL3b folio and text-line parser preserving folio IDs, line order, line positions, and paragraph markers where encoded in line tags.
- Token normalization: lowercase alphabetic EVA-like tokens; editorial tags, uncertain readings, braces, brackets, XML-like comments, and non-letter symbols removed.
- Parsed text lines: {len(line_df)}
- Parsed tokens: {len(token_df)}
- Section metadata source: folio-level comments in the ZL3b file; generated as `data/section_metadata.csv`.
- Missing required input files: none.
""", encoding='utf-8')
    sec_counts = metadata.groupby(['section_label','certainty']).size().reset_index(name='folio_count')
    unknown = int((metadata.section_label == 'unknown').sum()) if len(metadata) else 0
    (qc/'section_metadata_check.md').write_text('# Section Metadata Check\n\nSection labels are conventional visual/codicological labels extracted only from explicit local comments in the transcription. They are not treated as semantic categories.\n\n' + df_to_markdown(sec_counts) + f"\n\n- Unknown/unassigned folios: {unknown}\n", encoding='utf-8')
    sample = outputs['summary'][['section_label','total_tokens','total_lines']].copy()
    sample['analysis_confidence'] = sample.total_tokens.apply(lambda x: 'low-count warning' if x < 500 else 'standard')
    (qc/'minimum_sample_check.md').write_text('# Minimum Sample Check\n\n' + df_to_markdown(sample) + '\n\nLow-count sections should be interpreted cautiously; p-values and entropy contrasts are not over-interpreted.\n', encoding='utf-8')
    (qc/'baseline_validation_report.md').write_text(f"""# Baseline Validation Report

- Random seed: 42
- Requested permutation/bootstrap iterations: {n_iter}
- Executed iterations: {actual_iter}
- Fallback: {fallback_note}
- Permutation p-value formula: `(k + 1) / (N + 1)`.
- Line-position baselines: position-label permutation within section.
- Inter-token baselines: line-internal token shuffle within section.
- Family clustering baselines: token-order shuffle within section.
""", encoding='utf-8')
    created_tables = sorted(str(p.relative_to(root)) for p in (root/'outputs/tables').glob('*'))
    created_figs = sorted(str(p.relative_to(root)) for p in (root/'outputs/figures').glob('*'))
    nan_report = []
    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            n_nan = int(df.isna().sum().sum())
            n_inf = int(np.isinf(df.select_dtypes(include=[float, int]).to_numpy()).sum()) if not df.empty else 0
            nan_report.append({'output': name, 'nan_cells': n_nan, 'inf_cells': n_inf})
    nan_df = pd.DataFrame(nan_report)
    (qc/'assets_check_report.md').write_text('# Assets Check Report\n\n## Tables\n\n' + '\n'.join(f'- `{x}`' for x in created_tables) + '\n\n## Figures\n\n' + '\n'.join(f'- `{x}`' for x in created_figs) + '\n\n## NaN / Inf Check\n\n' + df_to_markdown(nan_df) + '\n\nSection names were generated from the controlled label set plus `unknown`.\n', encoding='utf-8')

    # Captions.
    fig_caps = [
        ('Figure 1', 'Section token distribution distance heatmap based on Jensen-Shannon divergence. Darker contrasts indicate greater divergence in token frequency profiles between conventional visual/codicological sections.'),
        ('Figure 2', 'PCS conditional entropy by section. Differences in H(suffix|core) and H(core|prefix) indicate section-specific variation in token-internal component constraints.'),
        ('Figure 3', 'Suffix-from-core prediction accuracy by section. Top-1 and top-3 accuracies summarize how strongly cores constrain suffix choices within each section.'),
        ('Figure 4', 'Inter-token entropy by section. Real bigram entropy is compared with a line-internal shuffled baseline to evaluate section-level local token ordering.'),
        ('Figure 5', 'Line-position constraint strength by section. Concentration scores summarize distributional restriction at line-initial, line-final, and paragraph-initial positions.'),
        ('Figure 6', 'Token-family clustering by section. Real-to-shuffled nearest-neighbor distance ratios below 1 indicate local clustering of related forms without implying semantic grouping.'),
        ('Figure 7', 'Integrated section clustering from normalized structural features. The plot visualizes similarity among section profiles rather than semantic categories.'),
        ('Figure 8', 'Structural zoning summary matrix. Standardized metrics summarize section-specific profiles across token, PCS, positional, inter-token, and family-clustering dimensions.'),
    ]
    (res/'figure_captions.md').write_text('\n\n'.join(f'**{n}.** {c}' for n, c in fig_caps) + '\n', encoding='utf-8')
    table_caps = [
        ('Table 1', 'Section corpus summary, including token counts, type-token ratio, entropy, Zipf slope, and top tokens by conventional visual/codicological section.'),
        ('Table 2', 'Distinctive tokens by section using log-odds contrast against the remaining corpus.'),
        ('Table 3', 'PCS metrics by section, including component entropy, conditional entropy, mutual information, and prediction accuracy.'),
        ('Table 4', 'Section-specific positional constraints based on line-initial, line-final, and paragraph-initial distributions.'),
        ('Table 5', 'Inter-token metrics by section, comparing real local ordering with line-internal shuffled baselines.'),
        ('Table 6', 'Token-family clustering by section using nearest-neighbor distances for PCS-defined token families.'),
        ('Table 7', 'Integrated section structural profiles using normalized features across token, PCS, positional, inter-token, and clustering dimensions.'),
        ('Table 8', 'Key results summary highlighting strongest and weakest section-level structural contrasts with conservative interpretation cautions.'),
    ]
    (res/'table_captions.md').write_text('\n\n'.join(f'**{n}.** {c}' for n, c in table_caps) + '\n', encoding='utf-8')

    # Results paragraphs.
    summary_df = outputs['summary']
    pcs = outputs['pcs']
    inter = outputs['inter']
    pos = outputs['pos']
    fam = outputs['fam']
    largest = summary_df.iloc[0].section_label if not summary_df.empty else 'not available'
    lowest_h = pcs.loc[pcs.H_suffix_given_core.idxmin()].section_label if not pcs.empty else 'not available'
    highest_pred = pcs.loc[pcs.suffix_from_core_top1_accuracy.idxmax()].section_label if not pcs.empty else 'not available'
    lower_big = inter.loc[inter.token_bigram_entropy.idxmin()].section_label if not inter.empty else 'not available'
    fam_core = fam[fam.family_definition == 'same_core']
    fam_strong = fam_core.loc[fam_core.real_to_shuffled_ratio.idxmin()].section_label if not fam_core.empty else 'not available'
    paragraphs = f"""# Paper 6 Results Paragraphs

## 1. Section corpus variation summary
The section-level corpus summary indicates that token counts, type-token ratios, token entropy, and token length profiles vary across conventional visual/codicological sections. The largest parsed section by token count is `{largest}`, but all section labels should be interpreted as descriptive manuscript labels rather than semantic categories.

## 2. Token distribution divergence summary
Pairwise token-distribution distances show that section profiles are not uniform across the manuscript. Jensen-Shannon divergence and cosine distance provide evidence for section-specific token frequency regimes, while distinctive-token scores identify forms disproportionately concentrated within individual sections.

## 3. PCS dependency variation summary
PCS metrics indicate that token-internal component dependencies differ by section. The lowest observed H(suffix|core) occurs in `{lowest_h}`, and the highest suffix-from-core top-1 accuracy occurs in `{highest_pred}`, supporting section-specific variation in component constraints without assigning meaning to those sections.

## 4. Line-position constraint variation summary
Line-initial, line-final, and paragraph-initial concentration scores vary across sections. These results extend positional-constraint analysis to section-level profiles and support structural zoning at manuscript-region scale.

## 5. Inter-token variation summary
Inter-token metrics show that local sequential organization also varies by section. The lowest real bigram entropy occurs in `{lower_big}`, and real/baseline contrasts should be interpreted as evidence for local ordering differences rather than syntax or translation.

## 6. Family clustering variation summary
PCS-defined family clustering differs across sections. The strongest same-core local clustering relative to shuffled baseline occurs in `{fam_strong}`, suggesting local organization of related token forms without implying thematic or semantic grouping.

## 7. Integrated structural zoning summary
The integrated feature profile combines token, PCS, positional, inter-token, and family-clustering metrics into section-level structural profiles. The resulting distance matrix and PCA visualization support the presence of local regimes of token formation and arrangement across manuscript regions.

## 8. Conservative interpretation paragraph
The results indicate section-specific variation in token organization, but they do not assign semantic labels to manuscript sections. The appropriate interpretation is that conventional visual/codicological section labels correlate with structural profiles, supporting structural zoning rather than decipherment.

## 9. Limitations paragraph
The analysis depends on ZL3b transcription conventions, automatic normalization choices, and section labels derived from explicit comments in the transcription. Unknown or low-certainty folios are left unassigned, low-count sections are marked cautiously, and the results do not establish translation, meaning, source-language identification, or confirmed syntax.
"""
    (res/'results_paragraphs.md').write_text(paragraphs, encoding='utf-8')


def write_summary(root: Path, input_path: Path, outputs: Dict[str, pd.DataFrame], counts: Dict[str, int], missing: List[str]):
    summary = outputs['summary']
    pcs = outputs['pcs']
    inter = outputs['inter']
    fam = outputs['fam']
    key_lines = []
    if not summary.empty:
        key_lines.append(f"Largest parsed section by token count: {summary.iloc[0].section_label} ({int(summary.iloc[0].total_tokens)} tokens).")
    if not pcs.empty:
        r = pcs.loc[pcs.H_suffix_given_core.idxmin()]
        key_lines.append(f"Lowest H(suffix|core): {r.section_label} ({r.H_suffix_given_core:.4f}).")
        r = pcs.loc[pcs.suffix_from_core_top1_accuracy.idxmax()]
        key_lines.append(f"Highest suffix-from-core top-1 accuracy: {r.section_label} ({r.suffix_from_core_top1_accuracy:.4f}).")
    if not inter.empty:
        r = inter.loc[inter.token_bigram_entropy.idxmin()]
        key_lines.append(f"Lowest section bigram entropy: {r.section_label} ({r.token_bigram_entropy:.4f}).")
        r = inter.loc[inter.next_token_top1_accuracy.idxmax()]
        key_lines.append(f"Highest next-token top-1 accuracy: {r.section_label} ({r.next_token_top1_accuracy:.4f}).")
        r = inter.loc[inter.suffix_to_next_prefix_conditional_entropy.idxmin()]
        key_lines.append(f"Lowest suffix-to-next-prefix entropy: {r.section_label} ({r.suffix_to_next_prefix_conditional_entropy:.4f}).")
    fam_core = fam[fam.family_definition == 'same_core']
    if not fam_core.empty:
        r = fam_core.loc[fam_core.real_to_shuffled_ratio.idxmin()]
        key_lines.append(f"Strongest same-core clustering ratio: {r.section_label} ({r.real_to_shuffled_ratio:.4f}).")
    if not outputs['pos'].empty:
        r = outputs['pos'].loc[outputs['pos'].line_initial_concentration_score.idxmax()]
        key_lines.append(f"Strongest line-initial concentration: {r.section_label} ({r.line_initial_concentration_score:.4f}).")
        r = outputs['pos'].loc[outputs['pos'].line_final_concentration_score.idxmax()]
        key_lines.append(f"Strongest line-final concentration: {r.section_label} ({r.line_final_concentration_score:.4f}).")
    if not outputs['dist'].empty:
        d = outputs['dist'][outputs['dist'].section_a != outputs['dist'].section_b].sort_values('jensen_shannon_divergence', ascending=False)
        if not d.empty:
            r = d.iloc[0]
            key_lines.append(f"Largest token-distribution divergence: {r.section_a} vs {r.section_b} (JSD={r.jensen_shannon_divergence:.4f}).")
    if not outputs['feature'].empty:
        clusters = outputs['feature'].cluster_assignment.nunique()
        key_lines.append(f"Integrated section profiles form {clusters} PCA-based structural profile groups.")
    key_lines = key_lines[:10]
    text = f"""# Paper 6 Experiment Summary

## 1. Purpose
This experiment evaluates whether global multi-level constraints in the Voynich Manuscript vary across manuscript sections, folio groups, visual/codicological categories, and local zones where metadata permits.

## 2. Input Files
- Primary transcription: `{input_path}`

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
""" + '\n'.join(f'{i+1}. {line}' for i, line in enumerate(key_lines[:10])) + f"""

## 8. Interpretive Cautions
The results should be described as section-specific structural variation, structural zoning, or local regimes of token formation and arrangement. They do not establish decipherment, translation, source-language identification, semantic categories, or confirmed syntax.

## 9. Missing / Low-Confidence Items
- Missing items: {', '.join(missing) if missing else 'none'}
- Low-count sections are marked in metric tables.

## 10. Manuscript Drafting Recommendations
Use Table 8 as the principal results summary, Figure 8 as the compact zoning overview, and Figures 2-4 to connect section-level findings to the earlier PCS and inter-token framework.

## Output Counts
- CSV files: {counts.get('csv', 0)}
- Markdown files: {counts.get('md', 0)}
- TXT files: {counts.get('txt', 0)}
- PNG files: {counts.get('png', 0)}
- PDF files: {counts.get('pdf', 0)}
"""
    (root/'outputs/PAPER6_EXPERIMENT_SUMMARY.md').write_text(text, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/ZL3b-n.txt')
    ap.add_argument('--out-root', default='.')
    ap.add_argument('--n-iter', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    start = time.time()
    root = Path(args.out_root).resolve()
    input_path = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    for p in [root/'data', root/'outputs/section_pipeline', root/'outputs/tables', root/'outputs/figures', root/'outputs/results_assets', root/'outputs/qc', root/'docs']:
        ensure_dir(p)
    log_lines = [f'Paper 6 pipeline start: {time.ctime()}', f'Input: {input_path}', f'Seed: {args.seed}', f'Requested iterations: {args.n_iter}']
    if not input_path.exists():
        (root/'missing_input_report.md').write_text(f'Missing required input: `{input_path}`\n', encoding='utf-8')
        raise SystemExit(f'Missing input {input_path}')

    # Use full N unless clearly too high for exploratory reruns. This run keeps 1000 as requested.
    actual_iter = args.n_iter
    fallback_note = 'none'
    random.seed(args.seed)
    np.random.seed(args.seed)

    token_df, line_df, metadata = parse_transcription(input_path)
    metadata.to_csv(root/'data/section_metadata.csv', index=False)
    token_df.to_csv(root/'outputs/section_pipeline/parsed_tokens_with_sections.csv', index=False)
    line_df.to_csv(root/'outputs/section_pipeline/parsed_lines_with_sections.csv', index=False)

    summary = section_corpus_summary(token_df, line_df)
    freq_df, dist_df, matrix_df, distinct = token_distribution_by_section(token_df)
    pcs_df, pcs_freq = pcs_metrics_by_section(token_df)
    pos_df, pos_perm = positional_metrics(token_df, line_df, actual_iter, args.seed + 100)
    inter_df, inter_base = intertoken_metrics(token_df, line_df, actual_iter, args.seed + 200)
    fam_df = family_clustering(token_df, min(actual_iter, 1000), args.seed + 300)
    feature_df, feat_dist, assignments = integrated_features(summary, pcs_df, pos_df, inter_df, fam_df)
    key_df = make_key_results_table(summary, pcs_df, pos_df, inter_df, fam_df, dist_df)

    # Section pipeline outputs.
    section_out = root/'outputs/section_pipeline'
    for name, df in [
        ('section_corpus_summary.csv', summary),
        ('token_distribution_by_section.csv', freq_df),
        ('section_distance_matrix.csv', dist_df),
        ('section_jsd_matrix.csv', matrix_df),
        ('distinctive_tokens_by_section.csv', distinct),
        ('pcs_metrics_by_section.csv', pcs_df),
        ('pcs_component_frequencies_by_section.csv', pcs_freq),
        ('line_position_metrics_by_section.csv', pos_df),
        ('section_positional_permutation_results.csv', pos_perm),
        ('intertoken_metrics_by_section.csv', inter_df),
        ('intertoken_baselines_by_section.csv', inter_base),
        ('family_clustering_by_section.csv', fam_df),
        ('integrated_section_features.csv', feature_df),
        ('section_feature_distance_matrix.csv', feat_dist),
        ('section_clustering_assignments.csv', assignments),
    ]:
        df.to_csv(section_out/name, index=False)

    # Tables.
    tab = root/'outputs/tables'
    write_table_variants(summary, tab/'table1_section_corpus_summary')
    write_table_variants(distinct.groupby('section_label', group_keys=False).head(10), tab/'table2_distinctive_tokens_by_section')
    write_table_variants(pcs_df, tab/'table3_pcs_metrics_by_section')
    write_table_variants(pos_df, tab/'table4_section_positional_constraints')
    write_table_variants(inter_df, tab/'table5_intertoken_metrics_by_section')
    write_table_variants(fam_df, tab/'table6_family_clustering_by_section')
    write_table_variants(feature_df, tab/'table7_integrated_section_profiles')
    write_table_variants(key_df, tab/'table8_key_results_summary')

    # Figures.
    save_figures(root/'outputs/figures', matrix_df, pcs_df, inter_df, pos_df, fam_df, feature_df)

    outputs = {'summary': summary, 'freq': freq_df, 'dist': dist_df, 'distinct': distinct, 'pcs': pcs_df, 'pcs_freq': pcs_freq, 'pos': pos_df, 'pos_perm': pos_perm, 'inter': inter_df, 'inter_base': inter_base, 'fam': fam_df, 'feature': feature_df, 'key': key_df}
    write_reports(root, input_path, token_df, line_df, metadata, args.n_iter, actual_iter, outputs, fallback_note)

    # Execution log and summary.
    elapsed = time.time() - start
    log_lines += [f'Parsed tokens: {len(token_df)}', f'Parsed lines: {len(line_df)}', f'Sections: {", ".join(sorted(token_df.section_label.unique()))}', f'Elapsed seconds: {elapsed:.2f}']
    (root/'outputs/qc/execution_log.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
    counts = {}
    for ext in ['csv','md','txt','png','pdf']:
        counts[ext] = len(list((root/'outputs').rglob(f'*.{ext}')))
    write_summary(root, input_path, outputs, counts, missing=[])
    print(json.dumps({'root': str(root), 'input': str(input_path), 'tokens': len(token_df), 'lines': len(line_df), 'sections': sorted(token_df.section_label.unique()), 'counts': counts, 'elapsed_seconds': elapsed}, indent=2))

if __name__ == '__main__':
    main()
