from __future__ import annotations

import matplotlib.pyplot as plt
import scanpy as sc

import config
from common_io import read_adata, step_arg_parser
from common_plot import savefig, setup_plotting
from progress import StepTimer


def _plot_embedding(adata, basis_key: str, color: str, out_name: str, title: str) -> None:
    adata.obsm["X_umap_plot"] = adata.obsm[basis_key].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.embedding(adata, basis="umap_plot", color=color, ax=ax, show=False, frameon=False, title=title, s=5)
    savefig(config.FIGURE_DIR / out_name)


def main() -> None:
    args = step_arg_parser("UMAP figure generation").parse_args()
    config.ensure_result_dirs()
    setup_plotting()
    timer = StepTimer("06_umap_figures")
    adata = read_adata(config.CACHE_DIR / "adata_integrated.h5ad")
    for basis, label in [("X_umap_unintegrated", "unintegrated"), ("X_umap_harmony", "harmony")]:
        for color in ["study", "celltype", "sample_id", "protocol_class"]:
            timer.update(f"plotting {label} by {color}")
            _plot_embedding(adata, basis, color, f"umap_{label}_by_{color}.png", f"{label}: {color}")
    timer.done()


if __name__ == "__main__":
    main()

