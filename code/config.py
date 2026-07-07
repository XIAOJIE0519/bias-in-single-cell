from __future__ import annotations

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
WORKSPACE_DIR = ROOT_DIR.parent

RESULT_DIR = ROOT_DIR / "result"
CACHE_DIR = RESULT_DIR / "cache"
TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"
LOG_DIR = RESULT_DIR / "logs"
REPORT_DIR = RESULT_DIR / "report"

DATASETS = {
    "GSE159354": {
        "disease_context": "scleroderma/idiopathic pulmonary fibrosis",
        "analysis_set": "diagnostic_only",
        "modality": "scRNA-seq",
        "protocol_class": "FACS_enriched",
        "controlled_primary": False,
        "composition_primary": False,
        "exclusion_reason": "FACS-sorted/enriched fractions; not unbiased whole-lung composition.",
    },
    "GSE132771": {
        "disease_context": "scleroderma/idiopathic pulmonary fibrosis",
        "analysis_set": "diagnostic_only",
        "modality": "scRNA-seq",
        "protocol_class": "lineage_negative_enriched",
        "controlled_primary": False,
        "composition_primary": False,
        "exclusion_reason": "Lineage-negative/collagen-cell enrichment affects composition.",
    },
    "GSE171524": {
        "disease_context": "COVID-19",
        "analysis_set": "sensitivity_only",
        "modality": "snRNA-seq",
        "protocol_class": "frozen_autopsy_single_nucleus",
        "controlled_primary": False,
        "composition_primary": False,
        "exclusion_reason": "Frozen rapid-autopsy single-nucleus data; not comparable with whole-cell scRNA composition.",
    },
    "GSE173896": {
        "disease_context": "COPD",
        "analysis_set": "controlled_primary",
        "modality": "scRNA-seq",
        "protocol_class": "whole_tissue_scRNA",
        "controlled_primary": True,
        "composition_primary": True,
        "exclusion_reason": "",
    },
    "GSE227691": {
        "disease_context": "COPD",
        "analysis_set": "controlled_primary",
        "modality": "scRNA-seq",
        "protocol_class": "whole_tissue_scRNA",
        "controlled_primary": True,
        "composition_primary": True,
        "exclusion_reason": "",
    },
}

CONTROLLED_DATASETS = tuple(k for k, v in DATASETS.items() if v["controlled_primary"])
EXCLUDE_SNRNA_DATASETS = ("GSE171524",)
EXCLUDE_ENRICHED_DATASETS = ("GSE159354", "GSE132771")

QC_PARAMS = {
    "min_genes": 200,
    "max_genes": 6000,
    "min_counts": 500,
    "max_mito_pct": 20.0,
    "min_cells_per_gene": 3,
}

PROCESSING_PARAMS = {
    "n_top_genes": 3000,
    "n_pcs": 50,
    "n_neighbors": 15,
    "leiden_resolution": 0.8,
    "random_seed": 42,
    "metrics_max_cells": 30000,
    "pseudobulk_top_genes": 2000,
    "network_top_genes": 300,
}

STEPS = [
    ("01_prepare_metadata", "01_prepare_metadata.py", "metadata scan", False),
    ("02_preprocess_qc", "02_preprocess_qc.py", "QC + preprocessing", False),
    ("03_annotation", "03_annotation.py", "cell type annotation", False),
    ("04_integration_harmony", "04_integration_harmony.py", "Harmony integration", False),
    ("05_integration_metrics", "05_integration_metrics.py", "integration metrics", False),
    ("06_umap_figures", "06_umap_figures.py", "UMAP figures", False),
    ("07_composition_analysis", "07_composition_analysis.py", "donor-level composition", False),
    ("08_pseudobulk_state", "08_pseudobulk_state.py", "pseudobulk state", False),
    ("09_pseudobulk_de", "09_pseudobulk_de.py", "pseudobulk DE", False),
    ("10_sensitivity_analysis", "10_sensitivity_analysis.py", "sensitivity analysis", False),
    ("11_scvi_optional", "11_scvi_optional.py", "optional scVI", True),
    ("12_network_exploratory", "12_network_exploratory.py", "exploratory network", True),
    ("14_pairwise_pseudobulk_de", "14_pairwise_pseudobulk_de.py", "all-study pairwise pseudobulk DE", False),
    ("15_make_revision_figures", "15_make_revision_figures.py", "revision figure assembly", False),
    ("13_make_report", "13_make_report.py", "final report", False),
]


def ensure_result_dirs() -> None:
    for path in [RESULT_DIR, CACHE_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
