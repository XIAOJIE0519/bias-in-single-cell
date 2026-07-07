from __future__ import annotations

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from common_io import step_arg_parser
from common_plot import setup_plotting
from progress import StepTimer, log


PANEL_TITLE = "Protocol-aware reanalysis of public lung control samples"
STUDY_ORDER = ["GSE132771", "GSE159354", "GSE171524", "GSE173896", "GSE227691"]


def _set_pub_style() -> None:
    setup_plotting()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str, dpi: int = 600) -> None:
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".eps"]:
        fig.savefig(config.FIGURE_DIR / f"{stem}{suffix}", bbox_inches="tight", dpi=dpi if suffix == ".png" else None)
    plt.close(fig)
    log(f"wrote figure: {config.FIGURE_DIR / f'{stem}.png'}")


def _panel_label(ax, label: str, x: float = -0.08, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top", ha="left")


def _show_png(ax, filename: str, label: str, title: str) -> None:
    img = mpimg.imread(config.FIGURE_DIR / filename)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=4)
    _panel_label(ax, label)


def _metrics_panel(ax) -> None:
    metrics = pd.read_csv(config.TABLE_DIR / "integration_metrics.csv")
    scvi_path = config.TABLE_DIR / "scvi_integration_metrics.csv"
    if scvi_path.exists():
        metrics = pd.concat([metrics, pd.read_csv(scvi_path)], ignore_index=True)
    methods = [m for m in ["unintegrated", "harmony", "scvi"] if m in set(metrics["method"])]
    metrics = metrics.set_index("method").loc[methods]
    cols = [
        ("batch_asw_lower_is_better", "batch ASW\nlower", "#4C78A8"),
        ("iLISI_study_higher_is_better", "iLISI\nhigher", "#54A24B"),
        ("celltype_asw_higher_is_better", "cell-type ASW\nhigher", "#F58518"),
    ]
    x = np.arange(len(metrics.index))
    width = 0.24
    for offset, (col, label, color) in zip([-width, 0, width], cols):
        ax.bar(x + offset, metrics[col].astype(float).values, width=width, label=label, color=color)
    ax.axhline(0, color="0.25", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Unintegrated", "Harmony", "scVI"][: len(x)])
    ax.set_ylabel("Metric value")
    ax.set_title("Integration metrics", fontsize=11, pad=6)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _panel_label(ax, "D")


def _short_celltype(label: str) -> str:
    mapping = {
        "Alveolar macrophages": "AM",
        "Alveolar type I cells": "AT1",
        "Alveolar type II cells": "AT2",
        "Activated T cells": "Act. T",
        "CD4+ memory T cells": "CD4 mem. T",
        "CD8+ effector T cells": "CD8 eff. T",
        "Classical monocytes": "Class. mono",
        "Non-classical monocytes": "Non-class. mono",
        "Endothelial tip cells": "Endo tip",
        "Vascular endothelial cells": "Vasc. endo",
        "Smooth muscle cells": "SMC",
        "Follicular B cells": "Foll. B",
    }
    return mapping.get(label, label)


def _composition_means() -> pd.DataFrame:
    comp = pd.read_csv(config.TABLE_DIR / "composition_all_study_diagnostic.csv")
    value_cols = [c for c in comp.columns if c not in {"analysis_set", "study", "sample_id"}]
    means = comp.groupby("study", observed=True)[value_cols].mean()
    means = means.reindex([s for s in STUDY_ORDER if s in means.index])
    means = means.div(means.sum(axis=1), axis=0).fillna(0.0) * 100.0
    order = means.mean(axis=0).sort_values(ascending=False).index.tolist()
    return means[order]


def _composition_panel(ax, legend: bool) -> None:
    means = _composition_means()
    colors = list(mpl.colormaps["tab20"].colors) + list(mpl.colormaps["tab20b"].colors)
    y = np.arange(len(means.index))
    left = np.zeros(len(means.index))
    for i, celltype in enumerate(means.columns):
        vals = means[celltype].values
        ax.barh(
            y,
            vals,
            left=left,
            height=0.62,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.45,
            label=_short_celltype(celltype),
        )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(means.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean donor-level proportion (%)")
    ax.set_title("All-study composition by data source", fontsize=11, pad=6)
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    if legend:
        ax.legend(
            frameon=False,
            fontsize=6,
            ncol=4,
            bbox_to_anchor=(0.0, -0.26, 1.0, 0.1),
            loc="upper left",
            mode="expand",
            title="Cell type",
            title_fontsize=7,
            borderaxespad=0,
        )
    _panel_label(ax, "E")


def _pairwise_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(config.TABLE_DIR / "pairwise_pseudobulk_de_summary.csv")
    tested = summary[summary["status"] == "tested"].copy()
    tested["comparison"] = tested["study1"] + " vs " + tested["study2"]
    counts = tested.pivot(index="celltype", columns="comparison", values="significant_genes")
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    log_counts = np.log10(counts.astype(float) + 1.0)
    return counts, log_counts


def _pairwise_heatmap(ax, annotate: bool, label: str | None = "F") -> None:
    counts, log_counts = _pairwise_matrix()
    annot = counts.fillna(0).astype(int).astype(str) if annotate else False
    sns.heatmap(
        log_counts,
        ax=ax,
        cmap="Blues",
        linewidths=0.3,
        linecolor="white",
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 5.2},
        cbar_kws={"label": "log10(significant genes + 1)", "shrink": 0.82},
    )
    ax.set_title("All-study pairwise pseudobulk DE by cell type", fontsize=11, pad=7)
    ax.set_xlabel("Study pair")
    ax.set_ylabel("Cell type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=42, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    if label is not None:
        _panel_label(ax, label, x=-0.055, y=1.04)


def make_individual_figures() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    _composition_panel(ax, legend=True)
    fig.tight_layout()
    _save(fig, "Figure1E_all_study_composition_unmerged")

    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    _pairwise_heatmap(ax, annotate=True, label="F")
    fig.tight_layout()
    _save(fig, "Figure1F_pairwise_pseudobulk_de_heatmap")


def make_combined_figure() -> None:
    fig = plt.figure(figsize=(13.6, 13.4))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.85, 1.55], width_ratios=[1, 1.12, 1.25], hspace=0.45, wspace=0.34)
    fig.suptitle(PANEL_TITLE, fontsize=17, fontweight="bold", y=0.985)
    _show_png(fig.add_subplot(grid[0, 0]), "umap_unintegrated_by_study.png", "A", "Unintegrated UMAP by study")
    _show_png(fig.add_subplot(grid[0, 1]), "umap_harmony_by_study.png", "B", "Harmony UMAP by study")
    _show_png(fig.add_subplot(grid[0, 2]), "umap_scvi_by_study.png", "C", "scVI UMAP by study")
    _metrics_panel(fig.add_subplot(grid[1, 0]))
    _composition_panel(fig.add_subplot(grid[1, 1:]), legend=False)
    _pairwise_heatmap(fig.add_subplot(grid[2, :]), annotate=True, label="F")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.95)
    _save(fig, "Figure1")


def main() -> None:
    step_arg_parser("Build revision Figure 1 from completed result tables and figures").parse_args()
    _set_pub_style()
    timer = StepTimer("15_make_revision_figures")
    make_individual_figures()
    make_combined_figure()
    timer.done()


if __name__ == "__main__":
    main()
