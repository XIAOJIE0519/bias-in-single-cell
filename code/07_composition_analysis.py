from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from common_io import bh_fdr, controlled_obs_mask, read_adata, step_arg_parser, write_table
from common_plot import savefig, setup_plotting
from progress import StepTimer


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = (p + q) / 2
    return float(0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m))


def _composition_tables(obs: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab([obs["study"], obs["sample_id"]], obs["celltype"])
    props = counts.div(counts.sum(axis=1), axis=0).reset_index()
    props.insert(0, "analysis_set", label)
    stats_rows = []
    for celltype in counts.columns:
        groups = []
        group_labels = []
        for study, group in props.groupby("study", observed=True):
            vals = group[celltype].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                group_labels.append(study)
        p = np.nan
        h = np.nan
        test_name = "not_tested"
        if len(groups) == 2:
            test_name = "mannwhitneyu_two_sided"
            try:
                h, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            except Exception:
                pass
        elif len(groups) > 2:
            test_name = "kruskal"
            try:
                h, p = stats.kruskal(*groups)
            except Exception:
                pass
        stats_rows.append(
            {
                "analysis_set": label,
                "celltype": celltype,
                "test": test_name,
                "statistic": h,
                "pvalue": p,
                "groups": "; ".join(f"{name}:n={len(vals)}" for name, vals in zip(group_labels, groups)),
            }
        )
    stat_df = pd.DataFrame(stats_rows)
    stat_df["fdr"] = bh_fdr(stat_df["pvalue"].fillna(1.0).values) if len(stat_df) else []
    by_study = pd.crosstab(obs["study"], obs["celltype"], normalize="index")
    js_rows = []
    for s1, s2 in itertools.combinations(by_study.index, 2):
        js_rows.append({"analysis_set": label, "study1": s1, "study2": s2, "js_divergence": _js_divergence(by_study.loc[s1].values, by_study.loc[s2].values)})
    return props, stat_df, pd.DataFrame(js_rows)


def _plot_top(props: pd.DataFrame, name: str) -> None:
    meta_cols = {"analysis_set", "study", "sample_id"}
    value_cols = [c for c in props.columns if c not in meta_cols]
    if not value_cols:
        return
    top = props[value_cols].mean().sort_values(ascending=False).head(12).index.tolist()
    plot = props.melt(id_vars=["study", "sample_id"], value_vars=top, var_name="celltype", value_name="proportion")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, celltype in enumerate(top):
        sub = plot[plot["celltype"] == celltype]
        ax.scatter([i] * len(sub), sub["proportion"], c=pd.factorize(sub["study"])[0], s=18, alpha=0.75)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top, rotation=45, ha="right")
    ax.set_ylabel("Donor/sample-level proportion")
    ax.set_title(name)
    savefig(config.FIGURE_DIR / f"composition_{name.replace(' ', '_').lower()}.png")


def main() -> None:
    args = step_arg_parser("Donor-level composition analysis").parse_args()
    config.ensure_result_dirs()
    setup_plotting()
    timer = StepTimer("07_composition_analysis")
    adata = read_adata(config.CACHE_DIR / "adata_integrated.h5ad")
    analyses = {
        "all_study_diagnostic": adata.obs,
        "controlled_primary": adata.obs[controlled_obs_mask(adata.obs)],
    }
    all_props, all_stats, all_js = [], [], []
    for label, obs in analyses.items():
        timer.update(f"composition: {label}")
        props, stat_df, js_df = _composition_tables(obs, label)
        all_props.append(props)
        all_stats.append(stat_df)
        all_js.append(js_df)
        write_table(props, config.TABLE_DIR / f"composition_{label}.csv")
        if label == "controlled_primary":
            write_table(props, config.TABLE_DIR / "composition_controlled.csv")
        _plot_top(props, label)
    write_table(pd.concat(all_props, ignore_index=True), config.TABLE_DIR / "composition_by_sample.csv")
    write_table(pd.concat(all_stats, ignore_index=True), config.TABLE_DIR / "composition_celltype_statistics.csv")
    write_table(pd.concat(all_js, ignore_index=True), config.TABLE_DIR / "composition_js_divergence.csv")
    timer.done()


if __name__ == "__main__":
    main()
