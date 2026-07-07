from __future__ import annotations

import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

import config
from common_io import read_adata, step_arg_parser, write_adata, write_table
from common_plot import savefig, setup_plotting
from progress import StepTimer, log


def _lisi(x: np.ndarray, labels: np.ndarray, k: int = 30) -> float:
    k = min(k, len(labels) - 1)
    if k < 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    vals = []
    for row in idx:
        counts = pd.Series(labels[row]).value_counts(normalize=True).values
        vals.append(1.0 / np.sum(counts**2))
    return float(np.mean(vals))


def _graph_connectivity(x: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    k = min(k, len(labels) - 1)
    if k < 2:
        return float("nan")
    graph = kneighbors_graph(x, n_neighbors=k, mode="connectivity", include_self=False)
    scores = []
    for label in pd.Series(labels).dropna().unique():
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            continue
        sub = graph[idx][:, idx]
        _, comps = connected_components(sub, directed=False)
        largest = pd.Series(comps).value_counts().max()
        scores.append(largest / len(idx))
    return float(np.mean(scores)) if scores else float("nan")


def _sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(config.PROCESSING_PARAMS["random_seed"])
    return np.sort(rng.choice(n, size=max_n, replace=False))


def main() -> None:
    args = step_arg_parser("Optional scVI integration").parse_args()
    config.ensure_result_dirs()
    timer = StepTimer("11_scvi_optional")
    if args.skip_scvi:
        write_table(__import__("pandas").DataFrame([{"status": "skipped", "reason": "--skip-scvi"}]), config.TABLE_DIR / "scvi_status.csv")
        timer.done()
        return
    try:
        import torch
        import scvi
    except Exception as exc:
        write_table(__import__("pandas").DataFrame([{"status": "skipped", "reason": f"missing dependency: {exc}"}]), config.TABLE_DIR / "scvi_status.csv")
        timer.done()
        return
    if not torch.cuda.is_available():
        write_table(__import__("pandas").DataFrame([{"status": "skipped", "reason": "CUDA unavailable"}]), config.TABLE_DIR / "scvi_status.csv")
        timer.done()
        return
    setup_plotting()
    adata = read_adata(config.CACHE_DIR / "adata_qc.h5ad")
    annotated_path = config.CACHE_DIR / "adata_integrated.h5ad"
    if annotated_path.exists():
        annotated = read_adata(annotated_path)
        for col in ["celltype", "leiden"]:
            if col in annotated.obs:
                vals = annotated.obs[col].astype("object").reindex(adata.obs_names)
                adata.obs[col] = vals.where(vals.notna(), "Unknown").astype(str).values
    if "celltype" not in adata.obs:
        adata.obs["celltype"] = "Unknown"
    timer.update("training scVI on counts layer")
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="study")
    model = scvi.model.SCVI(adata, n_latent=30, gene_likelihood="nb")
    epochs = 5 if args.smoke_test else 100
    model.train(max_epochs=epochs, accelerator="gpu", devices=1)
    adata.obsm["X_scVI"] = model.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata, random_state=config.PROCESSING_PARAMS["random_seed"])
    adata.obsm["X_umap_scvi"] = adata.obsm["X_umap"].copy()
    for color in ["study", "celltype", "sample_id", "protocol_class"]:
        if color in adata.obs:
            import matplotlib.pyplot as plt

            adata.obsm["X_umap_plot"] = adata.obsm["X_umap_scvi"].copy()
            fig, ax = plt.subplots(figsize=(8, 6))
            sc.pl.embedding(adata, basis="umap_plot", color=color, ax=ax, show=False, frameon=False, title=f"scVI: {color}", s=5)
            savefig(config.FIGURE_DIR / f"umap_scvi_by_{color}.png")
    idx = _sample_indices(adata.n_obs, config.PROCESSING_PARAMS["metrics_max_cells"])
    x = adata.obsm["X_scVI"][idx]
    obs = adata.obs.iloc[idx]
    study = obs["study"].astype(str).values
    celltype = obs["celltype"].astype(str).values
    metrics = pd.DataFrame(
        [
            {
                "method": "scvi",
                "n_cells_used": len(idx),
                "batch_asw_lower_is_better": silhouette_score(x, study) if len(set(study)) > 1 else float("nan"),
                "celltype_asw_higher_is_better": silhouette_score(x, celltype) if len(set(celltype)) > 1 else float("nan"),
                "iLISI_study_higher_is_better": _lisi(x, study),
                "cLISI_celltype_lower_is_better": _lisi(x, celltype),
                "graph_connectivity_celltype_higher_is_better": _graph_connectivity(x, celltype),
            }
        ]
    )
    write_table(metrics, config.TABLE_DIR / "scvi_integration_metrics.csv")
    write_adata(adata, config.CACHE_DIR / "adata_scvi.h5ad")
    write_table(pd.DataFrame([{"status": "completed", "max_epochs": epochs, "device": torch.cuda.get_device_name(0)}]), config.TABLE_DIR / "scvi_status.csv")
    timer.done()


if __name__ == "__main__":
    main()
