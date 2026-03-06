"""
Single-cell RNA-seq processing pipeline for GSE268807 (Focal Cortical Dysplasia).

Reproduces the analysis from:
  Kim et al. (2024) iScience. DOI: 10.1016/j.isci.2024.111337

Pipeline steps:
  1. Load 10x Genomics samples (Gene Expression features only)
  2. Per-sample quality control (min genes, mitochondrial %)
  3. Concatenate, normalize, select batch-aware HVGs
  4. PCA followed by Harmony batch correction
  5. UMAP embedding and Leiden clustering
  6. CellTypist reference-based cell type annotation
  7. Save processed AnnData and UMAP plot

Usage:
  python plot_umap_improved.py
"""

import os
import glob

os.environ['CELLTYPIST_FOLDER'] = os.path.join(os.path.dirname(__file__), '.celltypist')

import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import celltypist
from celltypist import models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sc.settings.verbosity = 3
sc.settings.n_jobs = 1

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
SUBSAMPLE = None
CELLTYPIST_MODEL = 'Adult_Human_PrefrontalCortex.pkl'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample(prefix, all_files):
    """Load a single 10x sample (matrix + features + barcodes), keeping only
    Gene Expression features."""
    matrix_file = next((f for f in all_files if os.path.basename(f) == f"{prefix}_matrix.mtx.gz"), None)
    features_file = next((f for f in all_files if os.path.basename(f) == f"{prefix}_features.tsv.gz"), None)
    barcodes_file = next((f for f in all_files if os.path.basename(f) == f"{prefix}_barcodes.tsv.gz"), None)

    if not all([matrix_file, features_file, barcodes_file]):
        print(f"  Skipping {prefix}: missing files")
        return None

    print(f"  Loading {prefix}...")
    adata = sc.read_mtx(matrix_file).T

    features = pd.read_csv(features_file, sep='\t', header=None, compression='gzip')
    adata.var['gene_ids'] = features[0].values
    adata.var_names = features[1].values
    adata.var['feature_types'] = features[2].values if features.shape[1] > 2 else "Gene Expression"
    adata.var_names_make_unique()

    barcodes = pd.read_csv(barcodes_file, sep='\t', header=None, compression='gzip')
    adata.obs_names = barcodes[0].values

    sample_name = prefix.replace("GSE268807_", "")
    adata.obs['sample_id'] = sample_name

    is_gene_expr = adata.var['feature_types'] == 'Gene Expression'
    n_before = adata.shape[1]
    adata = adata[:, is_gene_expr].copy()
    if n_before != adata.shape[1]:
        print(f"    Filtered {n_before} -> {adata.shape[1]} features (Gene Expression only)")

    return adata


def load_all_samples():
    """Discover and load all 10x samples from DATA_DIR."""
    all_files = glob.glob(os.path.join(DATA_DIR, "*"))
    matrix_files = sorted(
        f for f in all_files
        if 'matrix.mtx.gz' in os.path.basename(f) and ' (1)' not in f
    )

    adatas = []
    for m_file in matrix_files:
        prefix = os.path.basename(m_file).replace("_matrix.mtx.gz", "")
        adata = load_sample(prefix, all_files)
        if adata is not None:
            adatas.append(adata)

    print(f"\nLoaded {len(adatas)} samples.")
    return adatas


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def per_sample_qc(adatas, min_genes=200, max_mt_pct=15):
    """Filter cells per sample: minimum gene count and mitochondrial %."""
    filtered = []
    for adata in adatas:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, log1p=False)

        n_before = adata.n_obs
        sc.pp.filter_cells(adata, min_genes=min_genes)
        adata = adata[adata.obs['pct_counts_mt'] < max_mt_pct, :].copy()
        print(f"  {adata.obs['sample_id'].iloc[0]}: {n_before} -> {adata.n_obs} cells after QC")
        filtered.append(adata)

    return filtered


def detect_doublets(adatas, expected_doublet_rate=0.06):
    """Detect and remove predicted doublets via scanpy's Scrublet wrapper (per sample)."""
    filtered = []
    total_before, total_removed = 0, 0
    for adata in adatas:
        n_before = adata.n_obs
        total_before += n_before
        sid = adata.obs['sample_id'].iloc[0]
        sc.pp.scrublet(adata, expected_doublet_rate=expected_doublet_rate, verbose=False)
        predicted = adata.obs['predicted_doublet'].values
        n_doublets = int(predicted.sum())
        total_removed += n_doublets
        pct = n_doublets / n_before * 100
        print(f"  {sid}: {n_doublets}/{n_before} doublets removed ({pct:.1f}%)")
        adata = adata[~predicted].copy()
        filtered.append(adata)
    print(f"  Total doublets removed: {total_removed}/{total_before} "
          f"({total_removed / total_before * 100:.1f}%)")
    return filtered


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # Step 1 — Load
    print("=" * 60)
    print("STEP 1: Loading samples (Gene Expression only)")
    print("=" * 60)
    adatas = load_all_samples()

    # Step 2 — QC
    print("\n" + "=" * 60)
    print("STEP 2: Per-sample QC")
    print("=" * 60)
    adatas = per_sample_qc(adatas)

    # Step 2b — Doublet detection
    print("\n" + "=" * 60)
    print("STEP 2b: Doublet detection (Scrublet)")
    print("=" * 60)
    adatas = detect_doublets(adatas)

    # Step 3 — Merge and normalize
    print("\n" + "=" * 60)
    print("STEP 3: Concatenate and preprocess")
    print("=" * 60)
    adata = sc.concat(adatas, join='inner')
    print(f"Combined: {adata.shape}")

    if SUBSAMPLE and adata.n_obs > SUBSAMPLE:
        print(f"Subsampling to {SUBSAMPLE} cells...")
        sc.pp.subsample(adata, n_obs=SUBSAMPLE)

    sc.pp.filter_genes(adata, min_cells=3)
    print(f"After gene filtering: {adata.shape}")

    # SAVE RAW COUNTS FOR PYDESEQ2 BEFORE ANY NORMALIZATION
    raw_adata = adata.copy()

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['log1p_norm'] = adata.X.copy()

    sc.pp.highly_variable_genes(adata, batch_key='sample_id', n_top_genes=3000)
    n_hvg = sum(adata.var['highly_variable'])
    print(f"Selected {n_hvg} highly variable genes (batch-aware)")

    adata.raw = adata
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata, max_value=10)

    # Step 4 — PCA
    print("\n" + "=" * 60)
    print("STEP 4: PCA")
    print("=" * 60)
    sc.tl.pca(adata, n_comps=50, svd_solver='randomized')

    # Step 5 — Harmony
    print("\n" + "=" * 60)
    print("STEP 5: Harmony batch correction")
    print("=" * 60)
    ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'sample_id', max_iter_harmony=20)
    Z = np.array(ho.Z_corr)
    adata.obsm['X_pca_harmony'] = Z if Z.shape[0] == adata.n_obs else Z.T
    print("Harmony correction complete.")

    # Step 6 — Neighbors, UMAP, Leiden
    print("\n" + "=" * 60)
    print("STEP 6: Neighbors + UMAP + Leiden")
    print("=" * 60)
    sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=20, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0)

    # Step 7 — CellTypist annotation
    print("\n" + "=" * 60)
    print("STEP 7: CellTypist annotation")
    print("=" * 60)
    adata_ct = sc.AnnData(
        X=adata.raw.X.copy(),
        obs=adata.obs.copy(),
        var=adata.raw.var.copy()
    )
    adata_ct.obs_names_make_unique()

    try:
        model = models.Model.load(model=CELLTYPIST_MODEL)
    except (FileNotFoundError, Exception):
        print(f"Downloading CellTypist model: {CELLTYPIST_MODEL}...")
        models.download_models(model=CELLTYPIST_MODEL)
        model = models.Model.load(model=CELLTYPIST_MODEL)

    predictions = celltypist.annotate(adata_ct, model=model, majority_voting=True)
    adata.obs['celltypist_label'] = predictions.predicted_labels['majority_voting'].values
    adata.obs['celltypist_raw'] = predictions.predicted_labels['predicted_labels'].values

    print("\nCellTypist predicted cell types:")
    print(adata.obs['celltypist_label'].value_counts())

    # Step 8 — Save
    print("\n" + "=" * 60)
    print("STEP 8: Save results and plot")
    print("=" * 60)
    os.makedirs('figures', exist_ok=True)

    suffix = f"_{SUBSAMPLE // 1000}k" if SUBSAMPLE else "_full"

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sc.pl.umap(adata, color='sample_id', ax=axes[0], show=False, title='Sample ID')
    sc.pl.umap(adata, color='celltypist_label', ax=axes[1], show=False, title='Cell Type (CellTypist)')
    fig.tight_layout()
    root = os.path.join(os.path.dirname(__file__), '..')
    figures_dir = os.path.join(root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    outpath = os.path.join(figures_dir, f"umap_harmony_celltypist{suffix}.png")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {outpath}")

    h5ad_path = os.path.join(root, f"GSE268807_processed{suffix}.h5ad")
    adata.X = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    adata.write(h5ad_path)
    print(f"Saved processed data to {h5ad_path}")

    # Also save the pure unnormalized raw data for ALL genes (needed by PyDESeq2)
    raw_adata.obs = adata.obs.copy() # transfer celltypist labels
    raw_h5ad_path = os.path.join(root, f"GSE268807_raw_counts{suffix}.h5ad")
    raw_adata.write(raw_h5ad_path)
    print(f"Saved raw counts to {raw_h5ad_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
