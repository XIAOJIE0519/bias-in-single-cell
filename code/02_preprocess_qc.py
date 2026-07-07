from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc

import config
from common_io import load_all_10x, step_arg_parser, write_adata, write_table
from progress import StepTimer


def _qc_summary(adata, label: str) -> pd.DataFrame:
    rows = []
    for study, obs in adata.obs.groupby("study", observed=False):
        rows.append(
            {
                "stage": label,
                "study": study,
                "cells": int(len(obs)),
                "median_counts": float(obs["total_counts"].median()),
                "median_genes": float(obs["n_genes_by_counts"].median()),
                "median_pct_mt": float(obs["pct_counts_mt"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = step_arg_parser("Load, QC, normalize, HVG and PCA").parse_args()
    config.ensure_result_dirs()
    timer = StepTimer("02_preprocess_qc")
    adata = load_all_10x(
        max_samples=args.max_samples if args.smoke_test else None,
        max_cells=args.max_cells if args.smoke_test else None,
    )
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    write_adata(adata, config.CACHE_DIR / "adata_raw.h5ad")

    timer.update("calculating QC metrics")
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    before = _qc_summary(adata, "before_filter")
    q = config.QC_PARAMS
    sc.pp.filter_genes(adata, min_cells=q["min_cells_per_gene"])
    adata = adata[
        (adata.obs["n_genes_by_counts"] >= q["min_genes"])
        & (adata.obs["n_genes_by_counts"] <= q["max_genes"])
        & (adata.obs["total_counts"] >= q["min_counts"])
        & (adata.obs["pct_counts_mt"] <= q["max_mito_pct"])
    ].copy()
    after = _qc_summary(adata, "after_filter")
    write_table(pd.concat([before, after], ignore_index=True), config.TABLE_DIR / "qc_summary.csv")

    timer.update("normalizing and selecting highly variable genes")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=config.PROCESSING_PARAMS["n_top_genes"],
            batch_key="study",
            flavor="seurat_v3",
            layer="counts",
            subset=False,
        )
    except Exception:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=config.PROCESSING_PARAMS["n_top_genes"],
            batch_key="study",
            flavor="cell_ranger",
            subset=False,
        )
    timer.update("running unintegrated PCA")
    sc.tl.pca(
        adata,
        n_comps=min(config.PROCESSING_PARAMS["n_pcs"], adata.n_obs - 1, adata.n_vars - 1),
        use_highly_variable=True,
        random_state=config.PROCESSING_PARAMS["random_seed"],
    )
    write_table(
        pd.DataFrame(
            {
                "metric": ["cells_after_qc", "genes_after_qc", "highly_variable_genes"],
                "value": [adata.n_obs, adata.n_vars, int(adata.var["highly_variable"].sum())],
            }
        ),
        config.TABLE_DIR / "preprocess_summary.csv",
    )
    write_adata(adata, config.CACHE_DIR / "adata_qc.h5ad")
    timer.done()


if __name__ == "__main__":
    main()

