from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

import config
from progress import log


def step_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-scvi", action="store_true")
    return parser


def scan_h5_files(max_samples: int | None = None) -> pd.DataFrame:
    rows = []
    for study_dir in sorted(config.ROOT_DIR.glob("GSE*")):
        if not study_dir.is_dir():
            continue
        study = study_dir.name
        ds_meta = config.DATASETS.get(study, {})
        for h5_path in sorted(study_dir.glob("*.h5")):
            with h5py.File(h5_path, "r") as f:
                shape = tuple(int(x) for x in f["matrix/shape"][()])
                barcodes = int(len(f["matrix/barcodes"]))
                nnz = int(len(f["matrix/data"]))
            genes, cells = shape
            if barcodes == shape[0] and barcodes != shape[1]:
                cells, genes = shape
            else:
                genes, cells = shape
            rows.append(
                {
                    "study": study,
                    "sample_id": h5_path.stem,
                    "path": str(h5_path),
                    "file_mb": h5_path.stat().st_size / 1024**2,
                    "genes": genes,
                    "cells": cells,
                    "nnz": nnz,
                    "nnz_per_cell": nnz / max(cells, 1),
                    "density_pct": 100 * nnz / max(genes * cells, 1),
                    **ds_meta,
                }
            )
    df = pd.DataFrame(rows)
    if max_samples is not None and len(df) > max_samples:
        ordered = []
        for study in config.CONTROLLED_DATASETS:
            sub = df[df["study"] == study].sort_values("sample_id")
            if not sub.empty:
                ordered.append(sub.head(1))
        remaining = df[~df["sample_id"].isin(pd.concat(ordered)["sample_id"] if ordered else [])].sort_values(["study", "sample_id"])
        if ordered:
            selected = pd.concat(ordered + [remaining], ignore_index=True).head(max_samples)
        else:
            selected = remaining.head(max_samples)
        df = selected.copy()
    return df


def write_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    log(f"wrote table: {path}")


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote json: {path}")


def load_all_10x(max_samples: int | None = None, max_cells: int | None = None):
    import scanpy as sc
    import anndata as ad

    meta = scan_h5_files(max_samples=max_samples)
    if meta.empty:
        raise FileNotFoundError(f"No .h5 files found under {config.ROOT_DIR}")
    adatas = []
    rng = np.random.default_rng(config.PROCESSING_PARAMS["random_seed"])
    for _, row in meta.iterrows():
        path = Path(row["path"])
        log(f"reading {row['study']} / {row['sample_id']} ({row['cells']} cells)")
        a = sc.read_10x_h5(path)
        a.var_names_make_unique()
        a.obs_names = [f"{row['sample_id']}_{bc}" for bc in a.obs_names]
        for col in [
            "study",
            "sample_id",
            "disease_context",
            "analysis_set",
            "modality",
            "protocol_class",
            "controlled_primary",
            "composition_primary",
            "exclusion_reason",
        ]:
            a.obs[col] = row.get(col, "")
        if max_cells is not None and a.n_obs > max_cells:
            idx = np.sort(rng.choice(a.n_obs, size=max_cells, replace=False))
            a = a[idx].copy()
        adatas.append(a)
    combined = ad.concat(adatas, join="outer", label="loaded_sample", keys=[x.obs["sample_id"].iloc[0] for x in adatas], index_unique=None)
    combined.obs["dataset_id"] = combined.obs["study"].astype(str)
    combined.obs["condition"] = "control"
    return combined


def read_adata(path: Path):
    import scanpy as sc

    if not path.exists():
        raise FileNotFoundError(path)
    log(f"reading AnnData: {path}")
    return sc.read_h5ad(path)


def write_adata(adata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.obs.index = pd.Index([str(x) for x in adata.obs_names], dtype=object)
    adata.var.index = pd.Index([str(x) for x in adata.var_names], dtype=object)
    for df in [adata.obs, adata.var]:
        for col in df.columns:
            if (
                pd.api.types.is_categorical_dtype(df[col])
                or str(df[col].dtype).lower().startswith("string")
                or "arrow" in str(type(df[col].array)).lower()
            ):
                df[col] = pd.Series([str(x) for x in df[col].tolist()], index=df.index, dtype=object)
    adata.write(path, compression="gzip")
    log(f"wrote AnnData: {path}")


def controlled_obs_mask(obs: pd.DataFrame) -> pd.Series:
    return obs["study"].isin(config.CONTROLLED_DATASETS)


def get_layer(adata, layer: str = "counts"):
    if layer in adata.layers:
        return adata.layers[layer]
    return adata.X


def matrix_to_dense_small(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def bh_fdr(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float).copy()
    p[np.isnan(p)] = 1.0
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q_sorted, 0, 1)
    return out


def subset_for_smoke(adata, max_cells: int | None):
    if max_cells is None or adata.n_obs <= max_cells:
        return adata
    rng = np.random.default_rng(config.PROCESSING_PARAMS["random_seed"])
    idx = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    return adata[idx].copy()
