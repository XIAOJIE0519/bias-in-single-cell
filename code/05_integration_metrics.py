from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

import config
from common_io import read_adata, step_arg_parser, write_table
from progress import StepTimer


def _sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(config.PROCESSING_PARAMS["random_seed"])
    return np.sort(rng.choice(n, size=max_n, replace=False))


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


def main() -> None:
    args = step_arg_parser("Integration quality metrics").parse_args()
    config.ensure_result_dirs()
    timer = StepTimer("05_integration_metrics")
    adata = read_adata(config.CACHE_DIR / "adata_integrated.h5ad")
    idx = _sample_indices(adata.n_obs, config.PROCESSING_PARAMS["metrics_max_cells"])
    obs = adata.obs.iloc[idx]
    rows = []
    for method, key in [("unintegrated", "X_pca"), ("harmony", "X_pca_harmony")]:
        timer.update(f"metrics for {method}")
        x = adata.obsm[key][idx]
        study = obs["study"].astype(str).values
        celltype = obs["celltype"].astype(str).values
        batch_asw = silhouette_score(x, study) if len(set(study)) > 1 else float("nan")
        cell_asw = silhouette_score(x, celltype) if len(set(celltype)) > 1 else float("nan")
        rows.append(
            {
                "method": method,
                "n_cells_used": len(idx),
                "batch_asw_lower_is_better": batch_asw,
                "celltype_asw_higher_is_better": cell_asw,
                "iLISI_study_higher_is_better": _lisi(x, study),
                "cLISI_celltype_lower_is_better": _lisi(x, celltype),
                "graph_connectivity_celltype_higher_is_better": _graph_connectivity(x, celltype),
            }
        )
    write_table(pd.DataFrame(rows), config.TABLE_DIR / "integration_metrics.csv")
    timer.done()


if __name__ == "__main__":
    main()

