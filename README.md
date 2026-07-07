# Public single-cell control compatibility analysis

This repository contains the analysis code and reproducible output tables for the manuscript:

**The illusion of the universal baseline in public single-cell controls**

The repository supports a Critical Comment on reuse of public single-cell control samples. The central point is that a shared `healthy`, `normal` or `control` label does not by itself prove that public samples are exchangeable. Integration can align representations, but study compatibility must be evaluated with source-study metadata, pre/post integration diagnostics and donor-level sensitivity analyses.

## Repository contents

- `code/`: Python analysis pipeline.
- `result/tables/`: CSV outputs used to build the manuscript and supplementary tables.
- `result/figures/`: generated analysis figures.
- `result/report/`: run summary and machine-readable parameter/input inventories.
- `paper_revision2_assets/`: final revision-2 manuscript figures, supplementary figures, supplementary workbook and manuscript Markdown used for submission.
- `DATA_AVAILABILITY.md`: input data layout and GEO sample manifest.
- `UPLOAD_INSTRUCTIONS.md`: short checklist for the code author before uploading.

Raw `.h5` input files are intentionally **not** included in this upload package. They should be downloaded from GEO and placed in the dataset folders listed in `DATA_AVAILABILITY.md` if the full analysis is rerun.

## Analysis design

The analysis uses 28 public lung control samples from five GEO series:

- `GSE132771`: diagnostic-only, lineage-negative/collagen-cell enriched scRNA-seq.
- `GSE159354`: diagnostic-only, FACS-enriched scRNA-seq.
- `GSE171524`: sensitivity-only, frozen rapid-autopsy snRNA-seq.
- `GSE173896`: controlled-primary, whole-tissue scRNA-seq.
- `GSE227691`: controlled-primary, whole-tissue scRNA-seq.

The controlled primary subset is `GSE173896 + GSE227691`. Other studies are retained as diagnostic or sensitivity datasets because their protocol classes are not interchangeable with whole-tissue composition samples.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Check whether input `.h5` files and required dependencies are visible:

```bash
python code/run.py --dry-run
```

Run the full pipeline:

```bash
python code/run.py --resume
```

Run without optional scVI:

```bash
python code/run.py --resume --skip-scvi
```

Run a small smoke test:

```bash
python code/run.py --smoke-test --max-samples 2 --max-cells 3000
```

## Pipeline steps

1. Metadata scan.
2. QC and preprocessing.
3. Marker-based cell type annotation.
4. Harmony integration.
5. Integration metrics: ASW, iLISI, cLISI and graph connectivity.
6. UMAP figures.
7. Donor/sample-level composition analysis.
8. Donor/sample-level pseudobulk state analysis.
9. Pseudobulk differential-expression screen for the controlled subset.
10. Protocol-restricted and leave-one-study-out sensitivity analysis.
11. Optional scVI integration.
12. Exploratory network analysis, retained only as non-main exploratory output.
13. All-study pairwise pseudobulk differential-expression diagnostic.
14. Revision figure assembly.
15. Final report generation.

## Important interpretation boundary

The code and outputs should not be described as proving a universal biological mechanism. They support a bounded case study showing that public control reuse requires compatibility checks before pooling. Residual study-conditioned heterogeneity is interpreted as a compatibility and transparency issue, not as a single biological selection-bias mechanism.
