from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def setup_plotting() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(path) -> None:
    path = Path(path)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
