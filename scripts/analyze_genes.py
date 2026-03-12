"""
Analyze ion-channel / receptor gene expression in excitatory neurons from
the GSE268807 processed dataset.

Requires the output of plot_umap_improved.py:
  - GSE268807_processed_full.h5ad  (normalized, annotated)
  - GSE268807_raw_counts_full.h5ad (raw UMI counts, same cell order)

For every gene in GENES:
  - Scatter plot with DESeq2 significance bracket
  - Pseudobulk dot plot and % expressing bar chart
  - Full PyDESeq2 statistical report

For the PRIMARY_GENE (KCNA1) additionally:
  - Violin, box+dot, subtype-split, all-cell UMAP

Figures are saved to figures/{GENE}/.

Usage:
  python analyze_genes.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import scanpy as sc
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

os.environ['CELLTYPIST_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', '.celltypist')
sc.settings.verbosity = 1

_ROOT = os.path.join(os.path.dirname(__file__), '..')
INPUT_FILE = os.path.join(_ROOT, "GSE268807_processed_full.h5ad")
RAW_INPUT_FILE = os.path.join(_ROOT, "GSE268807_raw_counts_full.h5ad")

GENES = [
    "KCNA1",
    "ADORA2A", "KCNT1", "GABRA5", "SCN1A", "KCNK4",
    "KCNMA1", "CACNA1G", "CACNA1E", "CACNA1I", "HCN1", "HCN2",
    "MTOR",
]
PRIMARY_GENE = "KCNA1"

COND_ORDER = ['Control', 'Disease']
COND_PALETTE = {'Control': '#2d2d2d', 'Disease': '#c0392b'}
COND_FILL = {'Control': '#4a4a4a', 'Disease': '#e74c3c'}

# Sample metadata from Kim et al. (2024) iScience Table 1 (GSE268807).
# Maps GEO sample IDs to donor, condition, histological subtype, and lobe.
# Lobe info: samples span Frontal, Temporal, Parietal, and Temporo-Occipital cortex.
SAMPLE_META = {
    'G120_F1_N':  {'donor': 'G120', 'condition': 'Control', 'fcd_subtype': 'Normal',   'lobe': 'Frontal'},
    'G133_N_FL':  {'donor': 'G133', 'condition': 'Control', 'fcd_subtype': 'Normal',   'lobe': 'Frontal'},
    'G120_D_FL':  {'donor': 'G120', 'condition': 'Disease', 'fcd_subtype': 'FCD_IIb',  'lobe': 'Frontal'},
    'G120_D_TL':  {'donor': 'G120', 'condition': 'Disease', 'fcd_subtype': 'FCD_IIb',  'lobe': 'Temporal'},
    'G129_D':     {'donor': 'G129', 'condition': 'Disease', 'fcd_subtype': 'FCD_IIb',  'lobe': 'Parietal'},
    'G133_D_FL':  {'donor': 'G133', 'condition': 'Disease', 'fcd_subtype': 'FCD_IIb',  'lobe': 'Frontal'},
    'G150_D':     {'donor': 'G150', 'condition': 'Disease', 'fcd_subtype': 'FCD_IIa',  'lobe': 'Frontal'},
    'G159_D':     {'donor': 'G159', 'condition': 'Disease', 'fcd_subtype': 'FCD_IIb',  'lobe': 'Frontal'},
    'G171_D':     {'donor': 'G171', 'condition': 'Disease', 'fcd_subtype': 'FCD_IIa',  'lobe': 'Frontal'},
    'G187_D':     {'donor': 'G187', 'condition': 'Disease', 'fcd_subtype': 'FCD_IIa',  'lobe': 'Temporal'},
    'G210_D':     {'donor': 'G210', 'condition': 'Disease', 'fcd_subtype': 'FCD_IIb',  'lobe': 'Temporo-Occipital'},
}

# Per-donor clinical metadata from Kim et al. (2024) Table 1.
# G133 clinical data not provided in the original paper table for controls.
DONOR_META = {
    'G120': {'sex': 'F', 'age_at_surgery': 2,  'age_at_onset': 1,  'lateralization': 'L'},
    'G129': {'sex': 'M', 'age_at_surgery': 50, 'age_at_onset': 10, 'lateralization': 'R'},
    'G150': {'sex': 'F', 'age_at_surgery': 12, 'age_at_onset': 1,  'lateralization': 'R'},
    'G159': {'sex': 'F', 'age_at_surgery': 18, 'age_at_onset': 11, 'lateralization': 'R'},
    'G171': {'sex': 'F', 'age_at_surgery': 32, 'age_at_onset': 6,  'lateralization': 'R'},
    'G187': {'sex': 'M', 'age_at_surgery': 31, 'age_at_onset': 18, 'lateralization': 'L'},
    'G210': {'sex': 'F', 'age_at_surgery': 31, 'age_at_onset': 20, 'lateralization': 'L'},
}

EXPR_CMAP = LinearSegmentedColormap.from_list(
    'gray_to_red', ['#d3d3d3', '#fee8c8', '#fdbb84', '#e34a33', '#b30000'], N=256
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simplify_celltypist_label(label):
    """Map detailed CellTypist labels to broad cell types."""
    lbl = label.lower()
    if 'oligo' in lbl and 'opc' not in lbl and 'cop' not in lbl:
        return 'Oligodendrocyte'
    if 'opc' in lbl or 'cop' in lbl:
        return 'OPC'
    if lbl.startswith(('l2', 'l3', 'l5', 'l6')):
        return 'Excitatory Neuron'
    if 'inn ' in lbl or lbl.startswith('inn'):
        return 'Inhibitory Neuron'
    if 'astro' in lbl:
        return 'Astrocyte'
    if 'micro' in lbl or 'macro' in lbl:
        return 'Microglia'
    if 'endo' in lbl:
        return 'Endothelial'
    if 'smc' in lbl or 'pc ' in lbl or lbl.startswith('pc'):
        return 'VLMC/Pericyte'
    if 't ' in lbl or 'b ' in lbl:
        return 'Immune (T/B)'
    return 'Other'


def get_condition(sample_id):
    sid = str(sample_id)
    if sid in SAMPLE_META:
        return SAMPLE_META[sid]['condition']
    return 'Control' if ('_N' in sid) else 'Disease'


def get_donor(sample_id):
    sid = str(sample_id)
    if sid in SAMPLE_META:
        return SAMPLE_META[sid]['donor']
    return sid.split('_')[0]


def get_lobe(sample_id):
    sid = str(sample_id)
    if sid in SAMPLE_META:
        return SAMPLE_META[sid]['lobe']
    return 'Unknown'


def p_to_stars(p):
    try:
        if np.isnan(p):
            return 'n/a'
    except (TypeError, ValueError):
        pass
    if p < 0.0001:
        return '****'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def to_dense(x):
    if scipy.sparse.issparse(x):
        return x.toarray().flatten()
    return np.asarray(x).flatten()


def shorten_subtype(name):
    for prefix in ('CUX2 ', 'FEZF2 ', 'RORB ', 'OPRK1 '):
        name = name.replace(prefix, '')
    return name


def expression_vmax(values, percentile=95):
    expr = values[values > 0]
    return np.percentile(expr, percentile) if len(expr) > 0 else 1.0


def set_plot_theme():
    sns.set_theme(
        style='whitegrid', font_scale=1.1,
        rc={'axes.facecolor': '#fafafa', 'grid.color': '#e8e8e8',
            'grid.linewidth': 0.6, 'axes.edgecolor': '#cccccc'}
    )


def _annotate_sample_sizes(ax, df, df_all, groups, positions):
    trans = ax.get_xaxis_transform()
    for i, grp in zip(positions, groups):
        col = 'condition' if grp in COND_ORDER else 'subtype'
        n = (df[col] == grp).sum()
        total = (df_all[col] == grp).sum()
        pct = n / total * 100 if total > 0 else 0
        ax.text(i, -0.07, f'n={n}/{total} ({pct:.0f}% expr.)',
                ha='center', va='top', fontsize=8, color='#666666',
                transform=trans)


def _fisher_p(ctrl, dis):
    """Return two-sided Fisher's exact p-value for % expressing."""
    ct = np.array([[np.sum(ctrl > 0), np.sum(ctrl == 0)],
                    [np.sum(dis > 0), np.sum(dis == 0)]])
    _, p = scipy.stats.fisher_exact(ct, alternative='two-sided')
    return p


# ---------------------------------------------------------------------------
# Plotting — shared across all genes
# ---------------------------------------------------------------------------

def plot_scatter_condition(df, df_all, expr_values, gene, outdir, p_val=np.nan):
    """Scatter plot with significance bracket."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(5, 6))

    rng = np.random.default_rng(42)
    for i, cond in enumerate(COND_ORDER):
        vals = df.loc[df['condition'] == cond, 'expression'].values
        ax.scatter(i + rng.uniform(-0.25, 0.25, size=len(vals)), vals,
                   s=14, alpha=0.5, color=COND_PALETTE[cond],
                   edgecolors='white', linewidths=0.3, zorder=3)

    _annotate_sample_sizes(ax, df, df_all, COND_ORDER, range(2))

    y_max = df['expression'].max() if len(df) > 0 else 1.0
    bar_y = y_max * 1.05
    tip = y_max * 0.02
    ax.plot([0, 0, 1, 1], [bar_y - tip, bar_y, bar_y, bar_y - tip],
            color='#333333', linewidth=1.2, clip_on=False)
    ax.text(0.5, bar_y + tip * 0.5, p_to_stars(p_val),
            ha='center', va='bottom', fontsize=14, fontweight='bold', color='#333333')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(COND_ORDER, fontsize=12, fontweight='bold')
    ax.set_title(f'{gene} in Excitatory Neurons\nExpressing cells only',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel(f'{gene} expression (log-normalized)', fontsize=11)
    ax.set_xlabel('')
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(top=bar_y + y_max * 0.18)
    sns.despine(ax=ax, left=False, bottom=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(os.path.join(outdir, f'scatter_{gene}_exc_condition.png'),
                dpi=200, facecolor='white')
    plt.close(fig)
    sns.reset_defaults()



def plot_pseudobulk_dot(sample_agg, gene, outdir, pb_mean_p=np.nan, method="pseudobulk"):
    """Dot plot of per-donor pseudobulk mean expression, one point per donor."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(4.5, 5.5))

    for i, cond in enumerate(COND_ORDER):
        vals = sample_agg.loc[sample_agg['condition'] == cond, 'mean_expr'].values
        labels = sample_agg.loc[sample_agg['condition'] == cond, 'donor'].values
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(i + jitter, vals, s=70, alpha=0.85,
                   color=COND_PALETTE[cond], edgecolors='white',
                   linewidths=0.8, zorder=4)
        for x, y, lbl in zip(i + jitter, vals, labels):
            ax.annotate(lbl, (x, y), fontsize=6, color='#555555',
                        xytext=(6, 0), textcoords='offset points',
                        va='center')

    for i, cond in enumerate(COND_ORDER):
        vals = sample_agg.loc[sample_agg['condition'] == cond, 'mean_expr'].values
        if len(vals) > 0:
            mean_v = np.mean(vals)
            ax.hlines(mean_v, i - 0.22, i + 0.22, colors=COND_PALETTE[cond],
                      linewidths=2.5, zorder=5)

    y_max = sample_agg['mean_expr'].max()
    bar_y = y_max * 1.12
    tip = y_max * 0.02
    ax.plot([0, 0, 1, 1], [bar_y - tip, bar_y, bar_y, bar_y - tip],
            color='#333333', linewidth=1.2, clip_on=False)
    stars = p_to_stars(pb_mean_p)
    stat_label = "DESeq2 padj" if method == "pseudobulk" else "Wilcoxon padj"
    bracket_label = f'{stars}\n{stat_label}' if stars != 'n/a' else stars
    ax.text(0.5, bar_y + tip * 0.5, bracket_label,
            ha='center', va='bottom', fontsize=10,
            fontweight='bold', color='#333333')

    n_ctrl = (sample_agg['condition'] == 'Control').sum()
    n_dis = (sample_agg['condition'] == 'Disease').sum()
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Control\n(n={n_ctrl})', f'Disease\n(n={n_dis})'],
                       fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{gene} mean expression (log-normalized)', fontsize=11)
    ax.set_title(f'{gene} — Donor-Level Expression\nExcitatory Neurons',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(bottom=0, top=bar_y + y_max * 0.2)
    sns.despine(ax=ax, left=False, bottom=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'pseudobulk_dot_{gene}.png'),
                dpi=200, facecolor='white')
    plt.close(fig)
    sns.reset_defaults()


def plot_pct_expressing_by_donor(sample_agg, gene, outdir, pb_pct_p=np.nan, method="pseudobulk"):
    """Bar chart: % excitatory neurons expressing gene, one bar per donor."""
    set_plot_theme()
    ordered = sample_agg.sort_values(['condition', 'pct_expressing'],
                                     ascending=[True, False]).reset_index(drop=True)
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 5.5))

    colors = [COND_PALETTE[c] for c in ordered['condition']]
    bars = ax.bar(range(n), ordered['pct_expressing'], width=0.6,
                  color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)

    for i, (_, row) in enumerate(ordered.iterrows()):
        ax.text(i, row['pct_expressing'] + 0.8, f"{row['pct_expressing']:.1f}%",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#333333')

    ax.set_xticks(range(n))
    ax.set_xticklabels(ordered['donor'], rotation=45, ha='right',
                       fontsize=9)

    for i, (_, row) in enumerate(ordered.iterrows()):
        ax.text(i, -0.02, f"n={int(row['n_cells'])}",
                ha='center', va='top', fontsize=7, color='#888888',
                transform=ax.get_xaxis_transform())

    legend_els = [Line2D([0], [0], marker='s', color='w',
                         markerfacecolor=COND_PALETTE[c],
                         markersize=10, label=c) for c in COND_ORDER]
    ax.legend(handles=legend_els, loc='upper right', framealpha=0.9,
              edgecolor='#cccccc', fontsize=10)

    ax.set_ylabel(f'% excitatory neurons expressing {gene}', fontsize=11)
    stat_label = "DESeq2 padj" if method == "pseudobulk" else "Wilcoxon padj"
    title = (f'{gene} — % Expressing by Donor\n{stat_label} = {pb_pct_p:.3f}'
             if not np.isnan(pb_pct_p)
             else f'{gene} — % Expressing by Donor')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(bottom=0, top=ordered['pct_expressing'].max() * 1.25)
    sns.despine(ax=ax, left=False, bottom=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(os.path.join(outdir, f'pct_expressing_by_donor_{gene}.png'),
                dpi=200, facecolor='white')
    plt.close(fig)
    sns.reset_defaults()


# ---------------------------------------------------------------------------
# Plotting — PRIMARY_GENE extras
# ---------------------------------------------------------------------------

def plot_violin_condition(df, df_all, gene, outdir):
    """Violin + dot plot: Control vs Disease (expressing cells only)."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(5, 6))

    vp = ax.violinplot(
        [df.loc[df['condition'] == c, 'expression'].values for c in COND_ORDER],
        positions=[0, 1], widths=0.6, showmedians=False, showextrema=False)
    for body, cond in zip(vp['bodies'], COND_ORDER):
        body.set_facecolor(COND_FILL[cond])
        body.set_edgecolor(COND_PALETTE[cond])
        body.set_alpha(0.3)
        body.set_linewidth(1.5)

    for i, cond in enumerate(COND_ORDER):
        vals = df.loc[df['condition'] == cond, 'expression'].values
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.vlines(i, q1, q3, color=COND_PALETTE[cond], linewidth=3, zorder=4)
        ax.scatter([i], [med], color='white', s=30, zorder=5,
                   edgecolors=COND_PALETTE[cond], linewidths=1.2)

    rng = np.random.default_rng(42)
    for i, cond in enumerate(COND_ORDER):
        vals = df.loc[df['condition'] == cond, 'expression'].values
        ax.scatter(i + rng.uniform(-0.12, 0.12, size=len(vals)), vals,
                   s=5, alpha=0.45, color=COND_PALETTE[cond], linewidths=0, zorder=3)

    _annotate_sample_sizes(ax, df, df_all, COND_ORDER, range(2))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(COND_ORDER, fontsize=12, fontweight='bold')
    ax.set_title(f'{gene} in Excitatory Neurons\nExpressing cells only',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel(f'{gene} expression (log-normalized)', fontsize=11)
    ax.set_xlabel('')
    ax.set_xlim(-0.6, 1.6)
    sns.despine(ax=ax, left=False, bottom=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'violin_{gene}_exc_condition.png'),
                dpi=200, facecolor='white')
    plt.close(fig)
    sns.reset_defaults()


def plot_violin_condition_with_zeros(df_all, gene, outdir):
    """Violin + dot plot: Control vs Disease (all cells, including zeros)."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(5, 6))

    data_by_cond = [df_all.loc[df_all['condition'] == c, 'expression'].values
                    for c in COND_ORDER]
    vp = ax.violinplot(data_by_cond, positions=[0, 1], widths=0.6,
                       showmedians=False, showextrema=False)
    for body, cond in zip(vp['bodies'], COND_ORDER):
        body.set_facecolor(COND_FILL[cond])
        body.set_edgecolor(COND_PALETTE[cond])
        body.set_alpha(0.3)
        body.set_linewidth(1.5)

    for i, cond in enumerate(COND_ORDER):
        vals = df_all.loc[df_all['condition'] == cond, 'expression'].values
        # med = np.median(vals)
        # ax.hlines(med, i - 0.15, i + 0.15, color=COND_PALETTE[cond],
        #           linewidth=2.5, zorder=5)
        mean_v = np.mean(vals)
        ax.scatter([i], [mean_v], color=COND_PALETTE[cond], s=60, zorder=6,
                   marker='o', edgecolors='white', linewidths=1.0)
        ax.annotate(f'mean={mean_v:.3f}', (i, mean_v), fontsize=7,
                    xytext=(18, 0), textcoords='offset points', va='center',
                    color=COND_PALETTE[cond], fontweight='bold')

    trans = ax.get_xaxis_transform()
    for i, cond in enumerate(COND_ORDER):
        n = (df_all['condition'] == cond).sum()
        n_expr = ((df_all['condition'] == cond) & (df_all['expression'] > 0)).sum()
        pct = n_expr / n * 100 if n > 0 else 0
        ax.text(i, -0.07, f'n={n} ({pct:.1f}% expr.)',
                ha='center', va='top', fontsize=8, color='#666666',
                transform=trans)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(COND_ORDER, fontsize=12, fontweight='bold')
    ax.set_title(f'{gene} in Excitatory Neurons\nAll cells (including zeros)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel(f'{gene} expression (log-normalized)', fontsize=11)
    ax.set_xlabel('')
    ax.set_xlim(-0.6, 1.6)
    sns.despine(ax=ax, left=False, bottom=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(os.path.join(outdir, f'violin_{gene}_exc_condition_with_zeros.png'),
                dpi=200, facecolor='white')
    plt.close(fig)
    sns.reset_defaults()


def plot_combined_expression_panel(df, df_all, gene, outdir, p_val=np.nan,
                                   method="pseudobulk"):
    """Two-panel figure: % expressing (bar) + expression level (violin, expr only).

    Left panel shows the fraction of cells expressing the gene per condition,
    which is the key metric for zero-inflated genes. Right panel shows the
    per-cell expression distribution among expressing cells only.
    """
    set_plot_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.5),
                                    gridspec_kw={'width_ratios': [1, 1.3]})

    # --- Left panel: % expressing bar chart ---
    pcts = []
    ns = []
    for cond in COND_ORDER:
        n = (df_all['condition'] == cond).sum()
        n_expr = ((df_all['condition'] == cond) & (df_all['expression'] > 0)).sum()
        pcts.append(n_expr / n * 100 if n > 0 else 0)
        ns.append(n)

    bars = ax1.bar(range(2), pcts, width=0.55,
                   color=[COND_FILL[c] for c in COND_ORDER],
                   edgecolor=[COND_PALETTE[c] for c in COND_ORDER],
                   linewidth=1.2, alpha=0.85)

    for i, (pct, n) in enumerate(zip(pcts, ns)):
        ax1.text(i, pct + 1.2, f'{pct:.1f}%', ha='center', va='bottom',
                 fontsize=11, fontweight='bold', color='#333333')
        ax1.text(i, -0.08, f'n={n}', ha='center', va='top', fontsize=8,
                 color='#888888', transform=ax1.get_xaxis_transform())

    y_top = max(pcts) * 1.35
    bar_y = max(pcts) * 1.15
    tip = max(pcts) * 0.02
    stars = p_to_stars(p_val)
    if stars not in ('n/a', 'ns'):
        ax1.text(0.5, 0.97, stars, ha='center', va='top',
                 transform=ax1.transAxes, fontsize=13, fontweight='bold',
                 color='#333333')

    ax1.set_xticks(range(2))
    ax1.set_xticklabels(COND_ORDER, fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'% cells expressing {gene}', fontsize=11)
    ax1.set_title('% Expressing', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylim(0, y_top)
    sns.despine(ax=ax1, left=False, bottom=True)

    # --- Right panel: violin, all cells including zeros ---
    all_data = [df_all.loc[df_all['condition'] == c, 'expression'].values
                for c in COND_ORDER]

    vp2 = ax2.violinplot(all_data, positions=[0, 1], widths=0.6,
                         showmedians=False, showextrema=False)
    for body, cond in zip(vp2['bodies'], COND_ORDER):
        body.set_facecolor(COND_FILL[cond])
        body.set_edgecolor(COND_PALETTE[cond])
        body.set_alpha(0.3)
        body.set_linewidth(1.5)

    for i, cond in enumerate(COND_ORDER):
        vals = df_all.loc[df_all['condition'] == cond, 'expression'].values
        mean_v = np.mean(vals)
        ax2.scatter([i], [mean_v], color=COND_PALETTE[cond], s=60, zorder=6,
                    marker='o', edgecolors='white', linewidths=1.0)
        ax2.annotate(f'mean={mean_v:.3f}', (i, mean_v), fontsize=7,
                     xytext=(18, 0), textcoords='offset points', va='center',
                     color=COND_PALETTE[cond], fontweight='bold')

    stars2 = p_to_stars(p_val)
    if stars2 not in ('n/a', 'ns'):
        ax2.text(0.5, 0.97, stars2, ha='center', va='top',
                 transform=ax2.transAxes, fontsize=13, fontweight='bold',
                 color='#333333')

    for i, cond in enumerate(COND_ORDER):
        n_total = (df_all['condition'] == cond).sum()
        ax2.text(i, -0.08, f'n={n_total}', ha='center', va='top',
                 fontsize=8, color='#888888',
                 transform=ax2.get_xaxis_transform())

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(COND_ORDER, fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'{gene} expression (log-normalized)', fontsize=11)
    ax2.set_title('Expression Level (all cells)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylim(bottom=0)
    sns.despine(ax=ax2, left=False, bottom=True)

    fig.suptitle(f'{gene} in Excitatory Neurons', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20, top=0.90)
    fig.savefig(os.path.join(outdir, f'combined_panel_{gene}.png'),
                dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    sns.reset_defaults()


def plot_boxdot_condition(df, df_all, gene, outdir):
    """Box + dot plot: Control vs Disease (expressing cells only)."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(5, 6))

    bp = ax.boxplot(
        [df.loc[df['condition'] == c, 'expression'].values for c in COND_ORDER],
        positions=[0, 1], widths=0.45, patch_artist=True, showfliers=False,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
    for patch, cond in zip(bp['boxes'], COND_ORDER):
        patch.set_facecolor(COND_FILL[cond])
        patch.set_alpha(0.25)
        patch.set_edgecolor(COND_PALETTE[cond])
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(42)
    for i, cond in enumerate(COND_ORDER):
        vals = df.loc[df['condition'] == cond, 'expression'].values
        ax.scatter(i + rng.uniform(-0.15, 0.15, size=len(vals)), vals,
                   s=5, alpha=0.55, color=COND_PALETTE[cond], linewidths=0, zorder=3)

    _annotate_sample_sizes(ax, df, df_all, COND_ORDER, range(2))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(COND_ORDER, fontsize=12, fontweight='bold')
    ax.set_title(f'{gene} in Excitatory Neurons\nExpressing cells only',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel(f'{gene} expression (log-normalized)', fontsize=11)
    ax.set_xlabel('')
    ax.set_xlim(-0.6, 1.6)
    sns.despine(ax=ax, left=False, bottom=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'boxdot_{gene}_exc_condition.png'),
                dpi=200, facecolor='white')
    plt.close(fig)
    sns.reset_defaults()


def plot_boxdot_subtype_split(df, subtype_order, short_labels, gene, outdir):
    """Grouped box + dot by subtype and condition (expressing cells only)."""
    set_plot_theme()
    n_st = len(subtype_order)
    fig, ax = plt.subplots(figsize=(max(16, n_st * 1.1), 6.5))
    box_w = 0.35
    pos_ctrl = np.arange(n_st) - box_w / 2 - 0.02
    pos_dis = np.arange(n_st) + box_w / 2 + 0.02

    for pos_arr, cond in [(pos_ctrl, 'Control'), (pos_dis, 'Disease')]:
        data_list = [df.loc[(df['subtype'] == st) & (df['condition'] == cond), 'expression'].values
                     for st in subtype_order]
        bp = ax.boxplot(
            data_list, positions=pos_arr, widths=box_w, patch_artist=True,
            showfliers=False, medianprops=dict(color='white', linewidth=1.8),
            whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0))
        for patch in bp['boxes']:
            patch.set_facecolor(COND_FILL[cond])
            patch.set_alpha(0.25)
            patch.set_edgecolor(COND_PALETTE[cond])
            patch.set_linewidth(1.3)

    rng = np.random.default_rng(42)
    for i, st in enumerate(subtype_order):
        for cond, xc in [('Control', pos_ctrl[i]), ('Disease', pos_dis[i])]:
            vals = df.loc[(df['subtype'] == st) & (df['condition'] == cond), 'expression'].values
            if len(vals) == 0:
                continue
            ax.scatter(xc + rng.uniform(-box_w * 0.35, box_w * 0.35, size=len(vals)), vals,
                       s=10, alpha=0.55, color=COND_PALETTE[cond],
                       edgecolors='white', linewidths=0.2, zorder=3)

    legend_els = [Line2D([0], [0], marker='s', color='w', markerfacecolor=COND_FILL[c],
                         markeredgecolor=COND_PALETTE[c], markersize=10, label=c)
                  for c in COND_ORDER]
    ax.legend(handles=legend_els, title='Condition', loc='upper right',
              framealpha=0.9, edgecolor='#cccccc', fontsize=10)

    ax.set_xticks(range(n_st))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
    ax.set_title(f'{gene} — Control vs Disease by Excitatory Subtype\nExpressing cells only',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel(f'{gene} expression (log-normalized)', fontsize=11)
    ax.set_xlabel('')
    sns.despine(ax=ax, left=False, bottom=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'boxdot_{gene}_exc_subtype_split.png'),
                dpi=200, facecolor='white')
    plt.close(fig)
    sns.reset_defaults()


def plot_umap_allcells(adata_full, gene, outdir):
    """UMAP of all cells: cell type + gene expression."""
    if 'X_umap' not in adata_full.obsm:
        return
    full_expr = to_dense(adata_full[:, gene].X)
    vmax_full = expression_vmax(full_expr)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sc.pl.umap(adata_full, color='cell_type', ax=axes[0], show=False,
               title='Cell Type', frameon=False, size=8)
    sc.pl.umap(adata_full, color=gene, ax=axes[1], show=False,
               title=f'{gene} Expression', color_map=EXPR_CMAP,
               vmin=0, vmax=vmax_full, frameon=False, size=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'umap_{gene}_feature.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical tests (console output)
# ---------------------------------------------------------------------------

def _sig_label(p):
    try:
        if np.isnan(p):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return "YES" if p < 0.05 else "NO"


def _fmt_p(p):
    """Format a p-value, handling NaN."""
    try:
        if np.isnan(p):
            return f"{'N/A':>10s}"
    except (TypeError, ValueError):
        pass
    return f"{p:10.2e}"


def run_pseudoreplication_pipeline(adata_exc, genes):
    """Cell-level Wilcoxon rank-sum test per gene (pseudoreplication).

    Treats every cell as an independent observation — this inflates
    statistical power and is labelled 'pseudoreplication' deliberately.
    Useful as an exploratory consistency check on donor subsets.

    Returns a DataFrame with the same schema as the DESeq2 pipeline
    (log2FoldChange, pvalue, padj) so downstream code is agnostic.
    """
    from scipy.stats import mannwhitneyu, false_discovery_control

    conditions = adata_exc.obs['condition'].values
    ctrl_mask = conditions == 'Control'
    dis_mask = conditions == 'Disease'
    n_ctrl, n_dis = ctrl_mask.sum(), dis_mask.sum()

    print(f"Running cell-level Wilcoxon rank-sum (pseudoreplication)...", flush=True)
    print(f"  {n_ctrl} Control cells, {n_dis} Disease cells", flush=True)

    rows = []
    for gene in genes:
        expr = to_dense(adata_exc[:, gene].X)
        ctrl_vals = expr[ctrl_mask]
        dis_vals = expr[dis_mask]

        mean_c = np.mean(ctrl_vals)
        mean_d = np.mean(dis_vals)
        log2fc = np.log2(mean_d + 1) - np.log2(mean_c + 1)

        try:
            _, pval = mannwhitneyu(dis_vals, ctrl_vals, alternative='two-sided')
        except ValueError:
            pval = np.nan

        rows.append({'gene': gene, 'log2FoldChange': log2fc,
                     'pvalue': pval, 'padj': np.nan})

    res = pd.DataFrame(rows).set_index('gene')

    valid = ~res['pvalue'].isna()
    if valid.sum() > 0:
        pvals = res.loc[valid, 'pvalue'].values
        adjusted = false_discovery_control(pvals, method='bh')
        res.loc[valid, 'padj'] = adjusted

    n_sig = (res['padj'] < 0.05).sum()
    print(f"  {len(genes)} genes tested, {n_sig} significant (padj < 0.05)", flush=True)
    return res


def run_pydeseq2_pipeline(raw_adata_exc, min_umi=0):
    """Run PyDESeq2 on all genes using pseudobulk aggregation per donor.

    Aggregates UMI counts by (donor, condition) to avoid pseudoreplication
    (Squair et al., 2021). Uses a paired design ``~ donor + condition`` so
    that donors with both Control and Disease tissue (G120, G133) contribute
    paired information.

    Parameters
    ----------
    min_umi : int
        Minimum total UMI count per cell. Cells below this threshold are
        excluded before pseudobulk aggregation. Default 0 (no filter).
    """
    print("Aggregating raw counts to pseudobulk (per donor) for PyDESeq2...", flush=True)

    raw_adata_exc.obs['donor'] = raw_adata_exc.obs['sample_id'].map(get_donor)

    if min_umi > 0:
        umi_per_cell = np.asarray(raw_adata_exc.X.sum(axis=1)).flatten() \
            if scipy.sparse.issparse(raw_adata_exc.X) \
            else np.asarray(raw_adata_exc.X.sum(axis=1)).flatten()
        keep = umi_per_cell >= min_umi
        n_removed = (~keep).sum()
        raw_adata_exc = raw_adata_exc[keep].copy()
        print(f"  UMI filter (>= {min_umi}): removed {n_removed} cells, "
              f"{raw_adata_exc.n_obs} remaining.", flush=True)

    # Build metadata: one row per (donor, condition) combination
    donor_cond = (
        raw_adata_exc.obs[['donor', 'condition']]
        .drop_duplicates()
        .sort_values(['condition', 'donor'])
    )
    donor_cond_keys = list(zip(donor_cond['donor'], donor_cond['condition']))
    pb_ids = [f"{d}_{c}" for d, c in donor_cond_keys]

    df_meta = pd.DataFrame({
        'donor': [d for d, _ in donor_cond_keys],
        'condition': pd.Categorical(
            [c for _, c in donor_cond_keys],
            categories=['Control', 'Disease'],
            ordered=True,
        ),
    }, index=pb_ids)

    # Sum raw counts per (donor, condition) — must be integers for DESeq2
    counts = np.zeros((len(pb_ids), raw_adata_exc.n_vars), dtype=np.int64)
    for i, (donor, cond) in enumerate(donor_cond_keys):
        mask = ((raw_adata_exc.obs['donor'] == donor) &
                (raw_adata_exc.obs['condition'] == cond)).values
        if mask.sum() > 0:
            row_sum = np.asarray(raw_adata_exc.X[mask, :].sum(axis=0)).flatten()
            counts[i, :] = np.rint(row_sum).astype(np.int64)

    counts_df = pd.DataFrame(counts, index=pb_ids, columns=raw_adata_exc.var_names)

    keep_genes = counts_df.sum(axis=0) >= 10
    counts_df = counts_df.loc[:, keep_genes]
    n_pseudobulks = len(pb_ids)
    print(f"  {n_pseudobulks} pseudobulks (donors × condition), "
          f"{counts_df.shape[1]} genes retained.", flush=True)

    # Paired design accounts for donor baseline differences
    # Model matrix columns: 1 intercept + (n_donors-1) donor dummies + 1 condition dummy
    n_donors = df_meta['donor'].nunique()
    n_params = 1 + (n_donors - 1) + 1
    residual_df = n_pseudobulks - n_params
    print(f"  Design: ~ donor + condition  ({n_donors} donors, "
          f"{n_pseudobulks} pseudobulks, {residual_df} residual df)", flush=True)
    if residual_df < 3:
        print(f"  WARNING: Only {residual_df} residual degrees of freedom. "
              f"Statistical power is very limited.", flush=True)

    use_parametric = n_pseudobulks >= 15

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=df_meta,
        design="~ donor + condition",
        refit_cooks=(n_pseudobulks >= 15),
        n_cpus=1,
    )
    dds.deseq2(fit_type="parametric" if use_parametric else "mean")

    if not use_parametric:
        print(f"  Note: Using mean-based dispersion (only {n_pseudobulks} pseudobulks; "
              f"parametric fit requires ~15+).", flush=True)

    stat_res = DeseqStats(dds, contrast=["condition", "Disease", "Control"])
    stat_res.summary()
    return stat_res.results_df


def format_stat_report(gene, res_df, ctrl, dis, ctrl_pct, dis_pct,
                       sample_agg, outdir, method="pseudobulk", file_suffix=""):
    """Format and save the stats report for a single gene."""
    n_ctrl_s = (sample_agg['condition'] == 'Control').sum()
    n_dis_s = (sample_agg['condition'] == 'Disease').sum()

    if method == "pseudobulk":
        header_tag = "PyDESeq2 Pseudobulk — paired design"
        test_line = "Test: PyDESeq2 on Donor-level Summed Raw Counts (design: ~ donor + condition)"
    else:
        header_tag = "Cell-level Wilcoxon (pseudoreplication — exploratory)"
        test_line = "Test: Wilcoxon rank-sum on individual cells (includes zeros)"

    lines = []
    lines.append(f"Statistical Tests [{header_tag}]: {gene}")
    lines.append("=" * 72)
    lines.append(f"\nDonors:  Control = {n_ctrl_s}  ({len(ctrl)} cells)"
                 f"   Disease = {n_dis_s}  ({len(dis)} cells)")
    lines.append(f"Cell-level mean:  Control = {np.mean(ctrl):.4f}"
                 f"   Disease = {np.mean(dis):.4f}")
    lines.append(f"Cell-level % expr:  Control = {ctrl_pct:.1f}%"
                 f"   Disease = {dis_pct:.1f}%")

    lines.append(f"\n  Per-donor breakdown:")
    lines.append(f"  {'Donor':<20s}  {'Cond':<10s}  {'Cells':>6s}"
                 f"  {'Mean':>8s}  {'%Expr':>7s}")
    for _, row in sample_agg.sort_values(['condition', 'donor']).iterrows():
        lines.append(f"  {row['donor']:<20s}  {row['condition']:<10s}"
                     f"  {int(row['n_cells']):6d}  {row['mean_expr']:8.4f}"
                     f"  {row['pct_expressing']:6.1f}%")

    lines.append(f"\n{'─' * 72}")
    lines.append(test_line)

    if gene in res_df.index:
        row = res_df.loc[gene]
        log2fc = row['log2FoldChange']
        pvalue = row['pvalue']
        padj = row['padj']

        lines.append(f"  log2FoldChange (Disease vs Control) = {log2fc:.3f}")
        lines.append(f"  p-value = {_fmt_p(pvalue)}")
        lines.append(f"  Adjusted p-value (BH-FDR) = {_fmt_p(padj)}")
        lines.append(f"  Significant (p < 0.05): {_sig_label(pvalue)}")
        lines.append(f"  Significant (padj < 0.05): {_sig_label(padj)}")
    else:
        lines.append("  Gene not in results (filtered out or too low expression).")
        pvalue, padj, log2fc = np.nan, np.nan, np.nan

    if method == "pseudoreplication":
        lines.append(f"\n  NOTE: This test treats cells as independent observations.")
        lines.append(f"  It is an exploratory consistency check, not a generalizable")
        lines.append(f"  significance claim. See Squair et al. (2021).")

    report = "\n".join(lines)
    for line in lines:
        print(f"  {line}", flush=True)

    fname = f"stats_{gene}{file_suffix}.txt"
    with open(os.path.join(outdir, fname), 'w') as f:
        f.write(report + "\n")

    return {
        'gene': gene,
        'n_ctrl': len(ctrl), 'n_dis': len(dis),
        'pct_ctrl': ctrl_pct, 'pct_dis': dis_pct,
        'n_ctrl_samples': n_ctrl_s, 'n_dis_samples': n_dis_s,
        'log2fc': log2fc, 'pvalue': pvalue, 'padj': padj
    }


def write_stats_overview(summaries, outpath, method="pseudobulk"):
    """Write a summary table of all gene results."""
    s0 = summaries[0]
    lines = []

    if method == "pseudobulk":
        tag = "PyDESeq2 PSEUDOBULK — paired design"
    else:
        tag = "Cell-level Wilcoxon (PSEUDOREPLICATION — exploratory)"

    lines.append(f"Statistical Tests Overview [{tag}] — Disease vs Control (Excitatory Neurons)")
    lines.append("=" * 110)
    lines.append(f"Donors:  Control = {s0['n_ctrl_samples']}   Disease = {s0['n_dis_samples']}")
    lines.append(f"Cells:    Control = {s0['n_ctrl']}   Disease = {s0['n_dis']}")
    lines.append("")

    header = (f"{'Gene':<10s}  {'%Ctrl':>6s}  {'%Dis':>6s}  "
              f"{'log2FC':>8s}  {'p-value':>10s} {'Sig':>3s}  "
              f"{'padj (FDR)':>10s} {'Sig':>3s}")
    lines.append(header)
    lines.append("─" * len(header))

    for s in summaries:
        row = (f"{s['gene']:<10s}  {s['pct_ctrl']:5.1f}%  {s['pct_dis']:5.1f}%  "
               f"{s['log2fc']:8.3f}  "
               f"{_fmt_p(s['pvalue'])} {_sig_label(s['pvalue']):>3s}  "
               f"{_fmt_p(s['padj'])} {_sig_label(s['padj']):>3s}")
        lines.append(row)

    lines.append("─" * len(header))

    sig = [s['gene'] for s in summaries if not np.isnan(s['pvalue']) and s['pvalue'] < 0.05]
    sig_adj = [s['gene'] for s in summaries if not np.isnan(s['padj']) and s['padj'] < 0.05]

    lines.append(f"\nSignificant (raw p < 0.05): {', '.join(sig) or 'None'}")
    lines.append(f"Significant (padj < 0.05):  {', '.join(sig_adj) or 'None'}")

    report = "\n".join(lines)
    print(f"\n{report}", flush=True)

    with open(outpath, 'w') as f:
        f.write(report + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze gene expression in excitatory neurons."
    )
    parser.add_argument(
        "--method", choices=["pseudoreplication", "pseudobulk"],
        default="pseudoreplication",
        help="Statistical method. 'pseudoreplication' = cell-level Wilcoxon "
             "(treats cells as independent). 'pseudobulk' = DESeq2 on "
             "donor-aggregated counts. Default: pseudoreplication."
    )
    parser.add_argument(
        "--donors", type=str, default=None, metavar="D1,D2,...",
        help="Comma-separated donor IDs to include (e.g. G120,G133). "
             "Default: all donors."
    )
    parser.add_argument(
        "--min-umi", type=int, default=0, metavar="N",
        help="Minimum total UMI count per cell. Cells below this threshold "
             "are excluded before pseudobulk aggregation. Default: 0 (no filter)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    method = args.method
    donor_set = None
    donor_suffix = ""
    if args.donors:
        donor_set = set(d.strip() for d in args.donors.split(','))
        donor_suffix = "_" + "_".join(sorted(donor_set))
        print(f"Donor subsample: {sorted(donor_set)}", flush=True)

    fig_root = os.path.join(_ROOT, 'subsample_figures' if donor_set else 'figures')
    file_suffix = f"_{method}{donor_suffix}" if donor_set else f"_{method}"

    if args.min_umi > 0:
        print(f"UMI count filter enabled: cells with < {args.min_umi} total UMIs "
              f"will be excluded.", flush=True)
    print(f"Method: {method}", flush=True)
    print("Loading data...", flush=True)
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run scripts/preprocess.py first.")
        return

    adata = sc.read_h5ad(INPUT_FILE)

    if adata.raw is not None:
        adata_full = adata.raw.to_adata()
        adata_full.obs = adata.obs.copy()
    else:
        adata_full = adata.copy()
        sc.pp.normalize_total(adata_full, target_sum=1e4)
        sc.pp.log1p(adata_full)

    if 'X_umap' in adata.obsm:
        adata_full.obsm['X_umap'] = adata.obsm['X_umap']

    adata_full.obs['condition'] = adata_full.obs['sample_id'].apply(get_condition)
    adata_full.obs['donor'] = adata_full.obs['sample_id'].apply(get_donor)
    adata_full.obs['lobe'] = adata_full.obs['sample_id'].apply(get_lobe)
    if 'celltypist_label' in adata_full.obs.columns:
        adata_full.obs['cell_type'] = adata_full.obs['celltypist_label'].apply(simplify_celltypist_label)
        adata_full.obs['cell_type_detailed'] = adata_full.obs['celltypist_label']
    else:
        adata_full.obs['cell_type'] = adata_full.obs.get('leiden', 'Unknown')

    exc_mask = adata_full.obs['cell_type'] == 'Excitatory Neuron'
    print(f"Excitatory neurons (all donors): {exc_mask.sum()} cells", flush=True)
    if exc_mask.sum() == 0:
        print("No excitatory neurons found.")
        return

    adata_exc = adata_full[exc_mask].copy()
    adata_exc.obs['subtype'] = adata_exc.obs['cell_type_detailed'].astype(str)

    raw_adata = sc.read_h5ad(RAW_INPUT_FILE)
    if len(adata_full) != len(raw_adata) or not np.array_equal(
        adata_full.obs_names.values, raw_adata.obs_names.values
    ):
        raise ValueError(
            f"Cell barcodes differ between {INPUT_FILE} ({len(adata_full)} cells) "
            f"and {RAW_INPUT_FILE} ({len(raw_adata)} cells). "
            "Regenerate both files by re-running plot_umap_improved.py."
        )
    raw_adata.obs['condition'] = raw_adata.obs['sample_id'].apply(get_condition)
    raw_adata.obs['donor'] = raw_adata.obs['sample_id'].apply(get_donor)
    raw_adata.obs['lobe'] = raw_adata.obs['sample_id'].apply(get_lobe)
    raw_adata_exc = raw_adata[exc_mask].copy()

    # --- Apply donor subsample ---
    if donor_set:
        keep_exc = adata_exc.obs['donor'].isin(donor_set)
        adata_exc = adata_exc[keep_exc].copy()
        raw_adata_exc = raw_adata_exc[keep_exc].copy()
        print(f"After donor filter: {adata_exc.n_obs} excitatory neurons", flush=True)
        if adata_exc.n_obs == 0:
            print("No cells remain after donor filter.")
            return

    available = [g for g in GENES if g in adata_exc.var_names]
    missing = [g for g in GENES if g not in adata_exc.var_names]
    if missing:
        print(f"WARNING: genes not in dataset: {missing}", flush=True)

    donor_ids = adata_exc.obs['donor'].values

    # --- Run statistical pipeline ---
    if method == "pseudobulk":
        res_df = run_pydeseq2_pipeline(raw_adata_exc, min_umi=args.min_umi)
    else:
        res_df = run_pseudoreplication_pipeline(adata_exc, available)

    summaries = []
    for gene in available:
        gene_dir = os.path.join(fig_root, gene)
        os.makedirs(gene_dir, exist_ok=True)

        expr_values = to_dense(adata_exc[:, gene].X)
        conditions = adata_exc.obs['condition'].values

        df_all = pd.DataFrame({
            'expression': expr_values,
            'subtype': adata_exc.obs['subtype'].values,
            'condition': conditions,
        })
        df = df_all[df_all['expression'] > 0].copy()
        n_expr = len(df)
        pct = n_expr / len(df_all) * 100 if len(df_all) > 0 else 0

        print(f"\n{'═' * 56}")
        print(f"  {gene}  —  expressing {n_expr}/{len(df_all)} ({pct:.1f}%)", flush=True)

        pb_agg = pd.DataFrame({
            'expression': expr_values,
            'condition': np.asarray(conditions, dtype=str),
            'donor': np.asarray(donor_ids, dtype=str),
        }).groupby(['donor', 'condition']).agg(
            mean_expr=('expression', 'mean'),
            pct_expressing=('expression', lambda x: (x > 0).mean() * 100),
            n_cells=('expression', 'size'),
        ).reset_index()
        pb_agg = pb_agg[pb_agg['n_cells'] > 0]

        ctrl = expr_values[conditions == 'Control']
        dis = expr_values[conditions == 'Disease']
        ctrl_pct = np.sum(ctrl > 0) / len(ctrl) * 100 if len(ctrl) > 0 else 0
        dis_pct = np.sum(dis > 0) / len(dis) * 100 if len(dis) > 0 else 0

        stat_res = format_stat_report(
            gene, res_df, ctrl, dis, ctrl_pct, dis_pct, pb_agg, gene_dir,
            method=method, file_suffix=file_suffix,
        )
        summaries.append(stat_res)
        padj = stat_res['padj']

        # --- plots generated for every gene ---
        if n_expr >= 2:
            plot_scatter_condition(df, df_all, expr_values, gene, gene_dir, p_val=padj)
            print(f"  -> scatter_{gene}_exc_condition.png", flush=True)
        else:
            print(f"  Skipping scatter — <2 expressing cells.", flush=True)

        plot_umap_allcells(adata_full, gene, gene_dir)
        print(f"  -> umap_{gene}_feature.png", flush=True)

        # --- extra plots for primary gene only ---
        if gene == PRIMARY_GENE:
            plot_violin_condition(df, df_all, gene, gene_dir)
            print(f"  -> violin_{gene}_exc_condition.png", flush=True)

            plot_violin_condition_with_zeros(df_all, gene, gene_dir)
            print(f"  -> violin_{gene}_exc_condition_with_zeros.png", flush=True)

            plot_combined_expression_panel(df, df_all, gene, gene_dir,
                                           p_val=padj, method=method)
            print(f"  -> combined_panel_{gene}.png", flush=True)

            plot_boxdot_condition(df, df_all, gene, gene_dir)
            print(f"  -> boxdot_{gene}_exc_condition.png", flush=True)

            subtype_order = (df.groupby('subtype')['expression']
                             .mean().sort_values(ascending=False).index.tolist())
            short_labels = [shorten_subtype(s) for s in subtype_order]
            plot_boxdot_subtype_split(df, subtype_order, short_labels, gene, gene_dir)
            print(f"  -> boxdot_{gene}_exc_subtype_split.png", flush=True)

        plot_pseudobulk_dot(pb_agg, gene, gene_dir, pb_mean_p=padj, method=method)
        print(f"  -> pseudobulk_dot_{gene}.png", flush=True)
        plot_pct_expressing_by_donor(pb_agg, gene, gene_dir, pb_pct_p=padj, method=method)
        print(f"  -> pct_expressing_by_donor_{gene}.png", flush=True)

    if summaries:
        overview_name = f"stats_overview{file_suffix}.txt"
        write_stats_overview(summaries, os.path.join(fig_root, overview_name),
                             method=method)

    print(f"\n{'═' * 56}")
    print(f"Done — {len(available)} genes processed. Output in {fig_root}/", flush=True)


if __name__ == "__main__":
    main()
