from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from common_io import bh_fdr, controlled_obs_mask, get_layer, read_adata, step_arg_parser, write_table
from common_plot import savefig, setup_plotting
from progress import StepTimer, log


def _aggregate_counts(adata, celltype: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = controlled_obs_mask(adata.obs) & (adata.obs["celltype"] == celltype)
    x = get_layer(adata, "counts")
    obs = adata.obs.loc[mask, ["study", "sample_id"]]
    local = np.where(mask.values)[0]
    rows, meta = [], []
    for (study, sample), pos in obs.groupby(["study", "sample_id"], observed=False).indices.items():
        idx = local[list(pos)]
        rows.append(np.asarray(x[idx].sum(axis=0)).ravel())
        meta.append({"study": study, "sample_id": sample, "n_cells": len(idx)})
    return pd.DataFrame(rows, columns=adata.var_names), pd.DataFrame(meta)


def _fallback_de(counts: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    studies = sorted(meta["study"].unique())
    if len(studies) != 2:
        return pd.DataFrame()
    keep = (counts.sum(axis=0) >= 10) & ((counts > 0).sum(axis=0) >= 2)
    counts = counts.loc[:, keep]
    lib = counts.sum(axis=1).replace(0, np.nan)
    logcpm = np.log2(counts.div(lib, axis=0) * 1e6 + 1).fillna(0)
    g1 = meta["study"] == studies[0]
    g2 = meta["study"] == studies[1]
    rows = []
    for gene in logcpm.columns:
        a = logcpm.loc[g1, gene].values
        b = logcpm.loc[g2, gene].values
        if len(a) < 2 or len(b) < 2:
            p = np.nan
            stat = np.nan
        else:
            stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        rows.append({"gene": gene, "log2fc": float(np.mean(a) - np.mean(b)), "statistic": stat, "pvalue": p})
    out = pd.DataFrame(rows)
    out["fdr"] = bh_fdr(out["pvalue"].fillna(1.0).values)
    out["method"] = "logCPM_Welch_ttest_fallback"
    out["group1"] = studies[0]
    out["group2"] = studies[1]
    return out.sort_values(["fdr", "pvalue", "gene"])


def _pydeseq2_de(counts: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    studies = sorted(meta["study"].unique())
    if len(studies) != 2:
        return pd.DataFrame()
    keep = (counts.sum(axis=0) >= 10) & ((counts > 0).sum(axis=0) >= 2)
    counts = counts.loc[:, keep].round().astype(int)
    if counts.shape[1] == 0:
        return pd.DataFrame()
    counts.index = [f"sample_{i}" for i in range(len(counts))]
    metadata = meta[["study"]].copy()
    metadata.index = counts.index
    try:
        dds = DeseqDataSet(counts=counts, metadata=metadata, design_factors="study", refit_cooks=True, quiet=True)
    except TypeError:
        dds = DeseqDataSet(counts=counts, clinical=metadata, design_factors="study", refit_cooks=True, quiet=True)
    dds.deseq2()
    try:
        stat_res = DeseqStats(dds, contrast=["study", studies[0], studies[1]], quiet=True)
    except TypeError:
        stat_res = DeseqStats(dds, contrast=["study", studies[0], studies[1]])
    stat_res.summary()
    res = stat_res.results_df.reset_index()
    gene_col = "index" if "index" in res.columns else res.columns[0]
    out = pd.DataFrame(
        {
            "gene": res[gene_col].astype(str),
            "log2fc": res.get("log2FoldChange", pd.Series(np.nan, index=res.index)),
            "statistic": res.get("stat", pd.Series(np.nan, index=res.index)),
            "pvalue": res.get("pvalue", pd.Series(np.nan, index=res.index)),
            "fdr": res.get("padj", pd.Series(np.nan, index=res.index)),
        }
    )
    out["method"] = "PyDESeq2"
    out["group1"] = studies[0]
    out["group2"] = studies[1]
    return out.sort_values(["fdr", "pvalue", "gene"])


def main() -> None:
    args = step_arg_parser("Donor-level pseudobulk differential expression").parse_args()
    config.ensure_result_dirs()
    setup_plotting()
    timer = StepTimer("09_pseudobulk_de")
    for path in config.TABLE_DIR.glob("pseudobulk_de_*.csv"):
        if path.name != "pseudobulk_de_summary.csv":
            path.unlink(missing_ok=True)
    for path in config.FIGURE_DIR.glob("pseudobulk_de_volcano_*.png"):
        path.unlink(missing_ok=True)
    try:
        import pydeseq2  # noqa: F401

        has_pydeseq2 = True
        log("PyDESeq2 detected; using PyDESeq2 first, fallback only on per-celltype failure.")
    except Exception:
        has_pydeseq2 = False
        log("PyDESeq2 not detected; using logCPM Welch t-test fallback.")
    adata = read_adata(config.CACHE_DIR / "adata_integrated.h5ad")
    summary = []
    for celltype in sorted(adata.obs.loc[controlled_obs_mask(adata.obs), "celltype"].dropna().unique()):
        counts, meta = _aggregate_counts(adata, celltype)
        if counts.empty or meta["study"].nunique() != 2 or (meta.groupby("study").size() < 2).any():
            continue
        timer.update(f"DE: {celltype}")
        if has_pydeseq2:
            try:
                de = _pydeseq2_de(counts, meta)
            except Exception as exc:
                log(f"PyDESeq2 failed for {celltype}; using fallback: {exc}")
                de = _fallback_de(counts, meta)
        else:
            de = _fallback_de(counts, meta)
        if de.empty:
            continue
        sig = de[(de["fdr"] < 0.05) & (de["log2fc"].abs() >= 0.5)]
        safe = celltype.replace("/", "_")
        write_table(de, config.TABLE_DIR / f"pseudobulk_de_{safe}.csv")
        summary.append(
            {
                "celltype": celltype,
                "n_pseudobulk_samples": len(meta),
                "group_counts": "; ".join(f"{k}:{v}" for k, v in meta.groupby("study").size().items()),
                "genes_tested": len(de),
                "significant_genes_fdr_0_05_abs_log2fc_0_5": len(sig),
                "method": de["method"].iloc[0],
            }
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(de["log2fc"], -np.log10(de["pvalue"].clip(lower=1e-300)), s=4, alpha=0.4)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(-0.5, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(f"Pseudobulk DE: {celltype}")
        ax.set_xlabel("log2FC")
        ax.set_ylabel("-log10(p)")
        savefig(config.FIGURE_DIR / f"pseudobulk_de_volcano_{safe}.png")
    write_table(pd.DataFrame(summary), config.TABLE_DIR / "pseudobulk_de_summary.csv")
    timer.done()


if __name__ == "__main__":
    main()
