from __future__ import annotations

import config
from common_io import scan_h5_files, step_arg_parser, write_json, write_table
from progress import StepTimer


def main() -> None:
    args = step_arg_parser("Scan input h5 files and write metadata tables").parse_args()
    config.ensure_result_dirs()
    timer = StepTimer("01_prepare_metadata")
    meta = scan_h5_files(max_samples=args.max_samples if args.smoke_test else None)
    if meta.empty:
        raise FileNotFoundError(f"No GSE*/GSM*.h5 files found under {config.ROOT_DIR}")
    write_table(meta, config.TABLE_DIR / "sample_metadata.csv")
    summary = (
        meta.groupby("study", dropna=False)
        .agg(
            samples=("sample_id", "count"),
            cells=("cells", "sum"),
            median_cells=("cells", "median"),
            max_cells=("cells", "max"),
            nnz=("nnz", "sum"),
            file_mb=("file_mb", "sum"),
            analysis_set=("analysis_set", "first"),
            modality=("modality", "first"),
            protocol_class=("protocol_class", "first"),
            controlled_primary=("controlled_primary", "first"),
            exclusion_reason=("exclusion_reason", "first"),
        )
        .reset_index()
    )
    write_table(summary, config.TABLE_DIR / "dataset_summary.csv")
    write_table(
        meta[["study", "sample_id", "controlled_primary", "composition_primary", "analysis_set", "exclusion_reason"]],
        config.TABLE_DIR / "controlled_subset_inclusion.csv",
    )
    write_json(
        {
            "n_samples": int(len(meta)),
            "n_cells_raw": int(meta["cells"].sum()),
            "n_genes_unique": sorted(int(x) for x in meta["genes"].unique()),
            "controlled_datasets": list(config.CONTROLLED_DATASETS),
            "note": "Age, sex, smoking and donor-level clinical covariates were not present in local input files.",
        },
        config.REPORT_DIR / "input_data_inventory.json",
    )
    timer.done()


if __name__ == "__main__":
    main()

