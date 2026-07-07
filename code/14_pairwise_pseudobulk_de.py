from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config
from common_io import bh_fdr, get_layer, read_adata, step_arg_parser, write_table
from common_plot import setup_plotting
from progress import StepTimer, log


MIN_SAMPLES_PER_STUDY = 2
MIN_TOTAL_COUNTS_PER_GENE = 10
MIN_EXPRESSED_SAMPLES_PER_GENE = 2
SIG_FDR = 0.05
SIG_ABS_LOG2FC = 0.5


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def _aggregate_counts_all_studies(adata, celltype: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = adata.obs["celltype"].astype(str) == celltype
    x = get_layer(adata, "counts")
    obs = adata.obs.loc[mask, ["study", "sample_id"]].copy()
    local = np.where(mask.values)[0]
    rows, meta = [], []
    for (study, sample), pos in obs.groupby(["study", "sample_id"], observed=False).indices.items():
        idx = local[list(pos)]
        rows.append(np.asarray(x[idx].sum(axis=0)).ravel())
        meta.append({"study": str(study), "sample_id": str(sample), "celltype": celltype, "n_cells": len(idx)})
    if not rows:
        return pd.DataFrame(columns=adata.var_names), pd.DataFrame()
    counts = pd.DataFrame(rows, columns=adata.var_names)
    return counts, pd.DataFrame(meta)


def _logcpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=1).replace(0, np.nan)
    return np.log2(counts.div(lib, axis=0) * 1e6 + 1).fillna(0.0)


def _welch_de(counts: pd.DataFrame, meta: pd.DataFrame, study1: str, study2: str) -> pd.DataFrame:
    pair_mask = meta["study"].isin([study1, study2]).values
    pair_counts = counts.loc[pair_mask].reset_index(drop=True)
    pair_meta = meta.loc[pair_mask].reset_index(drop=True)
    if pair_counts.empty:
        return pd.DataFrame()

    keep = (pair_counts.sum(axis=0) >= MIN_TOTAL_COUNTS_PER_GENE) & (
        (pair_counts > 0).sum(axis=0) >= MIN_EXPRESSED_SAMPLES_PER_GENE
    )
    pair_counts = pair_counts.loc[:, keep]
    if pair_counts.shape[1] == 0:
        return pd.DataFrame()

    expr = _logcpm(pair_counts)
    g1 = pair_meta["study"].values == study1
    g2 = pair_meta["study"].values == study2
    a = expr.loc[g1].to_numpy(dtype=float)
    b = expr.loc[g2].to_numpy(dtype=float)
    stat, pvalue = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit")
    mean_a = np.nanmean(a, axis=0)
    mean_b = np.nanmean(b, axis=0)

    out = pd.DataFrame(
        {
            "gene": expr.columns.astype(str),
            "log2fc": mean_a - mean_b,
            "statistic": stat,
            "pvalue": pvalue,
        }
    )
    out["fdr"] = bh_fdr(out["pvalue"].fillna(1.0).values)
    out["method"] = "logCPM_Welch_ttest"
    out["group1"] = study1
    out["group2"] = study2
    return out.sort_values(["fdr", "pvalue", "gene"])


def _pydeseq2_de(counts: pd.DataFrame, meta: pd.DataFrame, study1: str, study2: str) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    pair_mask = meta["study"].isin([study1, study2]).values
    pair_counts = counts.loc[pair_mask].reset_index(drop=True)
    pair_meta = meta.loc[pair_mask, ["study"]].reset_index(drop=True)
    keep = (pair_counts.sum(axis=0) >= MIN_TOTAL_COUNTS_PER_GENE) & (
        (pair_counts > 0).sum(axis=0) >= MIN_EXPRESSED_SAMPLES_PER_GENE
    )
    pair_counts = pair_counts.loc[:, keep].round().astype(int)
    if pair_counts.shape[1] == 0:
        return pd.DataFrame()

    pair_counts.index = [f"sample_{i}" for i in range(len(pair_counts))]
    pair_meta.index = pair_counts.index
    try:
        dds = DeseqDataSet(counts=pair_counts, metadata=pair_meta, design_factors="study", refit_cooks=True, quiet=True)
    except TypeError:
        dds = DeseqDataSet(counts=pair_counts, clinical=pair_meta, design_factors="study", refit_cooks=True, quiet=True)
    dds.deseq2()
    try:
        stat_res = DeseqStats(dds, contrast=["study", study1, study2], quiet=True)
    except TypeError:
        stat_res = DeseqStats(dds, contrast=["study", study1, study2])
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
    out["group1"] = study1
    out["group2"] = study2
    return out.sort_values(["fdr", "pvalue", "gene"])


def _run_de(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    study1: str,
    study2: str,
    method: str,
    has_pydeseq2: bool,
) -> pd.DataFrame:
    if method in {"auto", "pydeseq2"} and has_pydeseq2:
        try:
            return _pydeseq2_de(counts, meta, study1, study2)
        except Exception as exc:
            if method == "pydeseq2":
                raise
            log(f"PyDESeq2 failed for {study1} vs {study2}; using Welch fallback: {exc}")
    return _welch_de(counts, meta, study1, study2)


def _plot_summary_heatmap(summary: pd.DataFrame, out_prefix: Path) -> None:
    ok = summary[summary["status"] == "tested"].copy()
    if ok.empty:
        log("No pairwise pseudobulk DE comparisons passed sample-count filters; heatmap skipped.")
        return
    ok["comparison"] = ok["study1"] + " vs " + ok["study2"]
    counts = ok.pivot(index="celltype", columns="comparison", values="significant_genes")
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    log_counts = np.log10(counts.astype(float) + 1.0)
    annot = counts.fillna(0).astype(int).astype(str)

    width = max(9.0, 0.65 * counts.shape[1] + 3.2)
    height = max(6.5, 0.34 * counts.shape[0] + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        log_counts,
        ax=ax,
        cmap="Blues",
        linewidths=0.35,
        linecolor="white",
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 6},
        cbar_kws={"label": "log10(significant genes + 1)"},
    )
    ax.set_title("All-study pairwise pseudobulk DE by cell type")
    ax.set_xlabel("Study pair")
    ax.set_ylabel("Cell type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    for suffix in [".png", ".pdf", ".eps"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight", dpi=600 if suffix == ".png" else None)
    plt.close(fig)
    log(f"wrote figure: {out_prefix.with_suffix('.png')}")


def main() -> None:
    parser = step_arg_parser("All-study pairwise donor-level pseudobulk DE")
    parser.add_argument(
        "--method",
        choices=["welch", "auto", "pydeseq2"],
        default="welch",
        help="welch is fast and default; auto tries PyDESeq2 first then falls back; pydeseq2 fails if any model fails.",
    )
    args = parser.parse_args()
    config.ensure_result_dirs()
    setup_plotting()
    timer = StepTimer("14_pairwise_pseudobulk_de")

    out_dir = config.TABLE_DIR / "pairwise_pseudobulk_de"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*.csv"):
        path.unlink(missing_ok=True)
    for path in config.FIGURE_DIR.glob("pairwise_pseudobulk_de_summary_heatmap.*"):
        path.unlink(missing_ok=True)

    try:
        import pydeseq2  # noqa: F401

        has_pydeseq2 = True
    except Exception:
        has_pydeseq2 = False
    if args.method in {"auto", "pydeseq2"}:
        log(f"Pairwise DE method={args.method}; PyDESeq2 available={has_pydeseq2}.")
    else:
        log("Pairwise DE method=welch; using logCPM + Welch t-test for all comparisons.")

    adata = read_adata(config.CACHE_DIR / "adata_integrated.h5ad")
    studies = sorted(adata.obs["study"].dropna().astype(str).unique())
    pairs = list(combinations(studies, 2))
    celltypes = sorted(adata.obs["celltype"].dropna().astype(str).unique())

    metadata_rows = []
    summary_rows = []
    for celltype in celltypes:
        timer.update(f"aggregate pseudobulk: {celltype}")
        counts, meta = _aggregate_counts_all_studies(adata, celltype)
        if counts.empty:
            continue
        metadata_rows.extend(meta.to_dict("records"))
        sample_counts = meta.groupby("study").size().to_dict()
        for study1, study2 in pairs:
            n1 = int(sample_counts.get(study1, 0))
            n2 = int(sample_counts.get(study2, 0))
            base = {
                "celltype": celltype,
                "study1": study1,
                "study2": study2,
                "n_study1": n1,
                "n_study2": n2,
                "min_samples_per_study": MIN_SAMPLES_PER_STUDY,
            }
            if n1 < MIN_SAMPLES_PER_STUDY or n2 < MIN_SAMPLES_PER_STUDY:
                summary_rows.append({**base, "status": "skipped_low_samples", "genes_tested": 0, "significant_genes": 0, "up_in_study1": 0, "up_in_study2": 0, "method": ""})
                continue

            timer.update(f"DE: {celltype} | {study1} vs {study2}")
            de = _run_de(counts, meta, study1, study2, args.method, has_pydeseq2)
            if de.empty:
                summary_rows.append({**base, "status": "skipped_no_genes", "genes_tested": 0, "significant_genes": 0, "up_in_study1": 0, "up_in_study2": 0, "method": ""})
                continue

            sig = de[(de["fdr"] < SIG_FDR) & (de["log2fc"].abs() >= SIG_ABS_LOG2FC)]
            write_table(
                de,
                out_dir / f"pairwise_de_{_safe_name(celltype)}__{study1}_vs_{study2}.csv",
            )
            summary_rows.append(
                {
                    **base,
                    "status": "tested",
                    "genes_tested": len(de),
                    "significant_genes": len(sig),
                    "up_in_study1": int((sig["log2fc"] > 0).sum()),
                    "up_in_study2": int((sig["log2fc"] < 0).sum()),
                    "method": de["method"].iloc[0],
                    "fdr_cutoff": SIG_FDR,
                    "abs_log2fc_cutoff": SIG_ABS_LOG2FC,
                }
            )

    write_table(pd.DataFrame(metadata_rows), config.TABLE_DIR / "pairwise_pseudobulk_sample_celltype_metadata.csv")
    summary = pd.DataFrame(summary_rows)
    write_table(summary, config.TABLE_DIR / "pairwise_pseudobulk_de_summary.csv")
    _plot_summary_heatmap(summary, config.FIGURE_DIR / "pairwise_pseudobulk_de_summary_heatmap")
    timer.done()


if __name__ == "__main__":
    main()
