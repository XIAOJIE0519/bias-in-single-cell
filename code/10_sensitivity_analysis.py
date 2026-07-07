from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from common_io import read_adata, step_arg_parser, write_table
from common_plot import savefig, setup_plotting
from progress import StepTimer


def _js(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = (p + q) / 2
    return float(0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m))


def _scenario_summary(obs: pd.DataFrame, name: str, rule: str) -> dict:
    comp = pd.crosstab(obs["study"], obs["celltype"], normalize="index")
    js_vals = [_js(comp.loc[a].values, comp.loc[b].values) for a, b in itertools.combinations(comp.index, 2)] if comp.shape[0] > 1 else []
    return {
        "scenario": name,
        "rule": rule,
        "n_cells": int(len(obs)),
        "n_samples": int(obs["sample_id"].nunique()),
        "n_studies": int(obs["study"].nunique()),
        "n_celltypes": int(obs["celltype"].nunique()),
        "mean_pairwise_js_divergence": float(np.mean(js_vals)) if js_vals else np.nan,
        "max_pairwise_js_divergence": float(np.max(js_vals)) if js_vals else np.nan,
    }


def main() -> None:
    args = step_arg_parser("Sensitivity analyses").parse_args()
    config.ensure_result_dirs()
    setup_plotting()
    timer = StepTimer("10_sensitivity_analysis")
    adata = read_adata(config.CACHE_DIR / "adata_integrated.h5ad")
    obs = adata.obs.copy()
    scenarios = [
        ("all_study_diagnostic", obs, "all local control samples"),
        ("controlled_only", obs[obs["study"].isin(config.CONTROLLED_DATASETS)], "GSE173896 + GSE227691"),
        ("exclude_snRNA", obs[~obs["study"].isin(config.EXCLUDE_SNRNA_DATASETS)], "exclude GSE171524"),
        ("exclude_enriched", obs[~obs["study"].isin(config.EXCLUDE_ENRICHED_DATASETS)], "exclude GSE159354 and GSE132771"),
    ]
    for study in sorted(obs["study"].unique()):
        scenarios.append((f"leave_one_out_{study}", obs[obs["study"] != study], f"exclude {study}"))
    rows = []
    for name, sub, rule in scenarios:
        timer.update(f"sensitivity: {name}")
        if len(sub) == 0:
            continue
        rows.append(_scenario_summary(sub, name, rule))
    out = pd.DataFrame(rows)
    write_table(out, config.TABLE_DIR / "sensitivity_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    plot = out.dropna(subset=["mean_pairwise_js_divergence"])
    ax.bar(plot["scenario"], plot["mean_pairwise_js_divergence"], color="steelblue")
    ax.set_ylabel("Mean pairwise JS divergence")
    ax.set_title("Composition sensitivity summaries")
    ax.tick_params(axis="x", rotation=45)
    savefig(config.FIGURE_DIR / "sensitivity_js_summary.png")
    timer.done()


if __name__ == "__main__":
    main()

