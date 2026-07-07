from __future__ import annotations

import scanpy as sc
import numpy as np
import pandas as pd

import config
from common_io import read_adata, step_arg_parser, write_adata
from progress import StepTimer, log


def _run_harmony(adata) -> tuple[bool, str]:
    try:
        import harmonypy as hm
    except Exception as exc:
        return False, f"harmonypy import failed: {exc}"

    if "X_pca" not in adata.obsm:
        return False, "X_pca missing"
    try:
        meta = pd.DataFrame({"study": adata.obs["study"].astype(str).values}, index=adata.obs_names)
        ho = hm.run_harmony(
            np.asarray(adata.obsm["X_pca"], dtype=np.float64),
            meta,
            vars_use=["study"],
            random_state=config.PROCESSING_PARAMS["random_seed"],
        )
        corrected = np.asarray(ho.Z_corr)
        if corrected.shape == adata.obsm["X_pca"].shape:
            adata.obsm["X_pca_harmony"] = corrected
        elif corrected.T.shape == adata.obsm["X_pca"].shape:
            adata.obsm["X_pca_harmony"] = corrected.T
        else:
            return False, f"unexpected Harmony shape {corrected.shape}; expected {adata.obsm['X_pca'].shape}"
        return True, "harmonypy completed"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    args = step_arg_parser("Compute unintegrated and Harmony embeddings").parse_args()
    config.ensure_result_dirs()
    timer = StepTimer("04_integration_harmony")
    adata = read_adata(config.CACHE_DIR / "adata_annotated.h5ad")
    timer.update("computing unintegrated UMAP")
    sc.pp.neighbors(adata, n_neighbors=config.PROCESSING_PARAMS["n_neighbors"], use_rep="X_pca")
    sc.tl.umap(adata, random_state=config.PROCESSING_PARAMS["random_seed"])
    adata.obsm["X_umap_unintegrated"] = adata.obsm["X_umap"].copy()
    timer.update("running Harmony")
    ok, message = _run_harmony(adata)
    if ok:
        log(f"Harmony completed: {message}; shape={adata.obsm['X_pca_harmony'].shape}")
    else:
        log(f"Harmony failed; using unintegrated PCA as fallback: {message}")
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"].copy()
    timer.update("computing Harmony UMAP")
    sc.pp.neighbors(adata, n_neighbors=config.PROCESSING_PARAMS["n_neighbors"], use_rep="X_pca_harmony")
    sc.tl.umap(adata, random_state=config.PROCESSING_PARAMS["random_seed"])
    adata.obsm["X_umap_harmony"] = adata.obsm["X_umap"].copy()
    write_adata(adata, config.CACHE_DIR / "adata_integrated.h5ad")
    timer.done()


if __name__ == "__main__":
    main()
