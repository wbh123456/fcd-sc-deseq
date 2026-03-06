# fcd-sc-deseq

Single-cell RNA-seq analysis of ion-channel and receptor gene expression in excitatory neurons from focal cortical dysplasia (FCD) patient tissue.

**Dataset**: [GSE268807](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268807) — 11 samples from 8 FCD donors (Kim et al., 2024 *iScience*).

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data

Download the raw 10x Genomics files from [GEO GSE268807](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268807) and place them in `data/raw/`.

Expected filenames follow the pattern:
```
GSE268807_G120_D_FL_barcodes.tsv.gz
GSE268807_G120_D_FL_features.tsv.gz
GSE268807_G120_D_FL_matrix.mtx.gz
...
```

---

## Usage

### Step 1 — Preprocess raw data (~20 min, run once)

```bash
python scripts/preprocess.py
```

Outputs: `GSE268807_processed_full.h5ad`, `GSE268807_raw_counts_full.h5ad`, `figures/umap_harmony_celltypist_full.png`

### Step 2 — Gene expression analysis

**Full cohort, pseudoreplication (default):**
```bash
python scripts/analyze_genes.py
```

**Paired donors only (G120 + G133), pseudoreplication:**
```bash
python scripts/analyze_genes.py --method pseudoreplication --donors G120,G133
```

**Full cohort, pseudobulk DESeq2:**
```bash
python scripts/analyze_genes.py --method pseudobulk
```

**With minimum UMI count filter:**
```bash
python scripts/analyze_genes.py --min-umi 500
```

## Outputs

| Directory | Description |
|---|---|
| `figures/{GENE}/` | Per-gene plots for the full cohort (10 donors) |
| `subsample_figures/{GENE}/` | Per-gene plots for paired donors G120 + G133 |

---

## References

1. **Kim J, et al.** (2024). Single-nucleus multiomics reveals the disrupted-in-epilepsy gene regulatory programs in focal cortical dysplasia. *iScience*, 27(12), 111337. [DOI: 10.1016/j.isci.2024.111337](https://doi.org/10.1016/j.isci.2024.111337)

2. **Squair JW, et al.** (2021). Confronting false discoveries in single-cell differential expression. *Nature Communications*, 12, 5692. [DOI: 10.1038/s41467-021-25960-2](https://doi.org/10.1038/s41467-021-25960-2)
