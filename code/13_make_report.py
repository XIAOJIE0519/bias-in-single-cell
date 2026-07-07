from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

import config
from common_io import step_arg_parser, write_json
from progress import StepTimer


def _table_preview(path, max_rows=8) -> str:
    try:
        df = pd.read_csv(path)
        preview = df.head(max_rows)
        try:
            return preview.to_markdown(index=False)
        except ImportError:
            return "```csv\n" + preview.to_csv(index=False, lineterminator="\n").strip() + "\n```"
    except Exception as exc:
        return f"Could not preview {path.name}: {exc}"


def _summary_celltypes(path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if "celltype" not in df.columns:
        return set()
    return set(df["celltype"].dropna().astype(str))


def _current_tables() -> list:
    de_celltypes = _summary_celltypes(config.TABLE_DIR / "pseudobulk_de_summary.csv")
    tables = []
    for table in sorted(config.TABLE_DIR.glob("*.csv")):
        if table.name.startswith("pseudobulk_de_") and table.name != "pseudobulk_de_summary.csv":
            celltype = table.stem.replace("pseudobulk_de_", "")
            if de_celltypes and celltype not in de_celltypes:
                continue
        tables.append(table)
    return tables


def _current_figures() -> list:
    de_celltypes = _summary_celltypes(config.TABLE_DIR / "pseudobulk_de_summary.csv")
    state_celltypes = _summary_celltypes(config.TABLE_DIR / "pseudobulk_state_summary.csv")
    figures = []
    for fig in sorted(config.FIGURE_DIR.glob("*.png")):
        name = fig.stem
        if name.startswith("pseudobulk_de_volcano_"):
            celltype = name.replace("pseudobulk_de_volcano_", "")
            if de_celltypes and celltype not in de_celltypes:
                continue
        if name.startswith("pseudobulk_pca_"):
            celltype = name.replace("pseudobulk_pca_", "")
            if state_celltypes and celltype not in state_celltypes:
                continue
        figures.append(fig)
    return figures


def main() -> None:
    args = step_arg_parser("Build final markdown report").parse_args()
    config.ensure_result_dirs()
    timer = StepTimer("13_make_report")
    params = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(config.ROOT_DIR),
        "result_dir": str(config.RESULT_DIR),
        "controlled_primary": list(config.CONTROLLED_DATASETS),
        "qc_params": config.QC_PARAMS,
        "processing_params": config.PROCESSING_PARAMS,
        "metadata_limitation": "No local age/sex/smoking covariate table was available; models are unadjusted unless explicitly stated.",
    }
    write_json(params, config.REPORT_DIR / "methods_parameters.json")
    lines = [
        "# Analysis Summary",
        "",
        f"Generated: {params['generated_at']}",
        "",
        "## Key Design Choices",
        "",
        "- Main controlled subset: GSE173896 + GSE227691.",
        "- GSE159354, GSE132771 and GSE171524 are diagnostic/sensitivity datasets, not primary composition datasets.",
        "- Donor/sample is the statistical unit for composition, pseudobulk state and pseudobulk DE.",
        "- Missing age/sex/smoking metadata were not imputed or fabricated.",
        "",
        "## Tables",
        "",
    ]
    for table in _current_tables():
        lines += [f"### {table.name}", "", _table_preview(table), ""]
    figures = _current_figures()
    lines += ["## Figures", ""]
    for fig in figures:
        lines.append(f"- {fig.name}")
    report = "\n".join(lines) + "\n"
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "analysis_summary.md").write_text(report, encoding="utf-8")
    timer.done()


if __name__ == "__main__":
    main()
