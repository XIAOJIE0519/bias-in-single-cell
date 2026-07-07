from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from common_io import controlled_obs_mask, get_layer, read_adata, step_arg_parser, write_table
from common_plot import savefig, setup_plotting
from progress import StepTimer


def _aggregate(adata, mask) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    x = get_layer(adata, "counts")
    groups = adata.obs.loc[mask, ["study", "sample_id", "celltype"]].copy()
    rows = []
    matrices = {}
    for celltype, idx in groups.groupby("celltype", observed=False).groups.items():
        obs_idx = np.array(list(idx))
        local = adata.obs_names.get_indexer(obs_idx)
        sub_groups = adata.obs.iloc[local].groupby(["study", "sample_id"], observed=False).indices
        mat_rows, meta_rows = [], []
        for (study, sample), pos in sub_groups.items():
            summed = x[local[list(pos)]].sum(axis=0)
            mat_rows.append(np.asarray(summed).ravel())
            meta_rows.append({"study": study, "sample_id": sample, "celltype": celltype, "n_cells": len(pos)})
        if mat_rows:
            matrices[celltype] = pd.DataFrame(np.vstack(mat_rows), index=pd.MultiIndex.from_frame(pd.DataFrame(meta_rows)[["study", "sample_id", "celltype"]]), columns=adata.var_names)
            rows.extend(meta_rows)
    return matrices, pd.DataFrame(rows)


def _logcpm(df: pd.DataFrame) -> pd.DataFrame:
    lib = df.sum(axis=1).replace(0, np.nan)
    return np.log2(df.div(lib, axis=0) * 1e6 + 1).fillna(0)


def main() -> None:
    args = step_arg_parser("Donor-level pseudobulk state analysis").parse_args()
    config.ensure_result_dirs()
    setup_plotting()
    timer = StepTimer("08_pseudobulk_state")
    for path in config.FIGURE_DIR.glob("pseudobulk_pca_*.png"):
        path.unlink(missing_ok=True)
    adata = read_adata(config.CACHE_DIR / "adata_integrated.h5ad")
    mask = controlled_obs_mask(adata.obs).values
    matrices, pb_meta = _aggregate(adata, mask)
    write_table(pb_meta, config.TABLE_DIR / "pseudobulk_sample_celltype_metadata.csv")
    rows = []
    for celltype, counts in matrices.items():
        if counts.shape[0] < 4 or counts.index.get_level_values("study").nunique() < 2:
            continue
        timer.update(f"pseudobulk PCA: {celltype}")
        expr = _logcpm(counts)
        top = expr.var(axis=0).sort_values(ascending=False).head(config.PROCESSING_PARAMS["pseudobulk_top_genes"]).index
        x = StandardScaler().fit_transform(expr[top].values)
        n_comp = min(5, x.shape[0] - 1, x.shape[1])
        if n_comp < 2:
            continue
        pcs = PCA(n_components=n_comp, random_state=config.PROCESSING_PARAMS["random_seed"]).fit_transform(x)
        studies = counts.index.get_level_values("study").astype(str)
        total_var = np.var(pcs[:, 0])
        grand = np.mean(pcs[:, 0])
        between = 0.0
        for study in sorted(set(studies)):
            vals = pcs[studies == study, 0]
            between += len(vals) * (np.mean(vals) - grand) ** 2
        eta2_pc1 = between / max(len(studies) * total_var, 1e-12)
        try:
            groups = [pcs[studies == s, 0] for s in sorted(set(studies))]
            h, p = stats.kruskal(*groups)
        except Exception:
            h, p = np.nan, np.nan
        rows.append(
            {
                "celltype": celltype,
                "n_pseudobulk_samples": counts.shape[0],
                "n_genes": counts.shape[1],
                "pc1_study_eta_squared": eta2_pc1,
                "kruskal_h_pc1": h,
                "kruskal_p_pc1": p,
                "note": "controlled subset; donor/sample is statistical unit",
            }
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        for study in sorted(set(studies)):
            pts = pcs[studies == study]
            ax.scatter(pts[:, 0], pts[:, 1], label=study, s=40)
        ax.set_title(f"Pseudobulk PCA: {celltype}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        savefig(config.FIGURE_DIR / f"pseudobulk_pca_{celltype.replace('/', '_')}.png")
    write_table(pd.DataFrame(rows), config.TABLE_DIR / "pseudobulk_state_summary.csv")
    timer.done()


if __name__ == "__main__":
    main()
