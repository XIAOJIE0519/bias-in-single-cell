from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans

import config
from common_io import read_adata, step_arg_parser, write_adata, write_table
from markers import filter_marker_sets, load_markers
from progress import StepTimer


def _mean_marker_score(adata, genes: list[str]) -> np.ndarray:
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return np.zeros(adata.n_obs)
    idx = [adata.var_names.get_loc(g) for g in present]
    x = adata.layers["log1p"][:, idx] if "log1p" in adata.layers else adata.X[:, idx]
    score = x.mean(axis=1)
    return np.asarray(score).ravel()


def main() -> None:
    args = step_arg_parser("Leiden clustering and marker-based annotation").parse_args()
    config.ensure_result_dirs()
    timer = StepTimer("03_annotation")
    adata = read_adata(config.CACHE_DIR / "adata_qc.h5ad")
    raw_markers = load_markers()
    markers, marker_filter_report = filter_marker_sets(raw_markers, adata.var_names)
    write_table(marker_filter_report, config.TABLE_DIR / "marker_filter_report.csv")
    if not markers:
        raise RuntimeError("No marker sets passed filtering; check marker-lung.xlsx or fallback marker definitions.")
    timer.update("computing neighbor graph")
    sc.pp.neighbors(adata, n_neighbors=config.PROCESSING_PARAMS["n_neighbors"], use_rep="X_pca")
    timer.update("running Leiden clustering")
    try:
        sc.tl.leiden(
            adata,
            resolution=config.PROCESSING_PARAMS["leiden_resolution"],
            key_added="leiden",
            random_state=config.PROCESSING_PARAMS["random_seed"],
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except Exception:
        labels = KMeans(n_clusters=min(30, max(2, adata.n_obs // 3000)), random_state=config.PROCESSING_PARAMS["random_seed"]).fit_predict(
            adata.obsm["X_pca"][:, : min(30, adata.obsm["X_pca"].shape[1])]
        )
        adata.obs["leiden"] = labels.astype(str)

    timer.update("scoring marker genes")
    score_cols = []
    score_info = []
    for celltype, genes in markers.items():
        col = f"score__{celltype}"
        adata.obs[col] = _mean_marker_score(adata, genes)
        score_cols.append(col)
        score_info.append({"celltype": celltype, "markers_total": len(genes), "markers_present": sum(g in adata.var_names for g in genes)})
    write_table(pd.DataFrame(score_info), config.TABLE_DIR / "marker_gene_coverage.csv")

    timer.update("assigning cell types to clusters")
    cluster_rows = []
    cluster_to_celltype = {}
    for cluster, obs in adata.obs.groupby("leiden", observed=False):
        means = obs[score_cols].mean().sort_values(ascending=False)
        best_col = means.index[0]
        second = float(means.iloc[1]) if len(means) > 1 else 0.0
        best = float(means.iloc[0])
        celltype = best_col.replace("score__", "")
        confidence = best - second
        if best <= 0:
            celltype = "Unknown"
        cluster_to_celltype[str(cluster)] = celltype
        cluster_rows.append(
            {
                "cluster": cluster,
                "cells": int(len(obs)),
                "celltype": celltype,
                "best_mean_marker_score": best,
                "second_mean_marker_score": second,
                "confidence_delta": confidence,
            }
        )
    adata.obs["celltype"] = adata.obs["leiden"].astype(str).map(cluster_to_celltype).fillna("Unknown")
    write_table(pd.DataFrame(cluster_rows), config.TABLE_DIR / "cluster_annotation.csv")
    write_table(
        adata.obs.groupby(["study", "sample_id", "celltype"], observed=False).size().reset_index(name="cells"),
        config.TABLE_DIR / "celltype_counts_by_sample.csv",
    )
    write_adata(adata, config.CACHE_DIR / "adata_annotated.h5ad")
    timer.done()


if __name__ == "__main__":
    main()
