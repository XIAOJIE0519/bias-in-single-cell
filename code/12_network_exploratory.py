from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config
from common_io import controlled_obs_mask, get_layer, read_adata, step_arg_parser, write_table
from progress import StepTimer


def _aggregate(adata, celltype: str) -> tuple[pd.DataFrame, pd.Series]:
    mask = controlled_obs_mask(adata.obs) & (adata.obs["celltype"] == celltype)
    x = get_layer(adata, "counts")
    obs = adata.obs.loc[mask, ["study", "sample_id"]]
    local = np.where(mask.values)[0]
    rows, studies = [], []
    for (study, sample), pos in obs.groupby(["study", "sample_id"], observed=False).indices.items():
        idx = local[list(pos)]
        rows.append(np.asarray(x[idx].sum(axis=0)).ravel())
        studies.append(study)
    return pd.DataFrame(rows, columns=adata.var_names), pd.Series(studies)


def main() -> None:
    args = step_arg_parser("Exploratory lightweight network analysis").parse_args()
    config.ensure_result_dirs()
    timer = StepTimer("12_network_exploratory")
    adata = read_adata(config.CACHE_DIR / "adata_integrated.h5ad")
    ct_counts = adata.obs.loc[controlled_obs_mask(adata.obs), "celltype"].value_counts().head(12)
    rows = []
    for celltype in ct_counts.index:
        timer.update(f"network similarity: {celltype}")
        counts, studies = _aggregate(adata, celltype)
        if counts.shape[0] < 6 or studies.nunique() < 2:
            continue
        lib = counts.sum(axis=1).replace(0, np.nan)
        expr = np.log2(counts.div(lib, axis=0) * 1e6 + 1).fillna(0)
        top = expr.var(axis=0).sort_values(ascending=False).head(config.PROCESSING_PARAMS["network_top_genes"]).index
        pair_rows = []
        study_corr = {}
        for study in sorted(studies.unique()):
            mat = expr.loc[studies.values == study, top]
            if mat.shape[0] < 3:
                continue
            corr = np.corrcoef(mat.values, rowvar=False)
            iu = np.triu_indices_from(corr, k=1)
            study_corr[study] = corr[iu]
        for a, b in itertools.combinations(study_corr, 2):
            if len(study_corr[a]) == len(study_corr[b]):
                sim = float(np.corrcoef(study_corr[a], study_corr[b])[0, 1])
                pair_rows.append(sim)
                rows.append(
                    {
                        "celltype": celltype,
                        "study1": a,
                        "study2": b,
                        "network_similarity_correlation": sim,
                        "n_top_genes": len(top),
                        "note": "Exploratory pseudobulk network similarity; not a primary claim.",
                    }
                )
    write_table(pd.DataFrame(rows), config.TABLE_DIR / "network_exploratory_summary.csv")
    timer.done()


if __name__ == "__main__":
    main()

