from __future__ import annotations

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

import config
from progress import log


FALLBACK_LUNG_MARKERS = {
    "Alveolar type II cells": ["SFTPC", "SFTPA1", "SFTPA2", "ABCA3", "LAMP3"],
    "Alveolar type I cells": ["AGER", "PDPN", "AQP5", "CAV1"],
    "Club cells": ["SCGB1A1", "SCGB3A2", "CYP2F1"],
    "Ciliated cells": ["FOXJ1", "TPPP3", "PIFO", "DNAH5"],
    "Goblet cells": ["MUC5AC", "MUC5B", "SPDEF"],
    "Basal cells": ["KRT5", "KRT17", "TP63"],
    "Endothelial cells": ["PECAM1", "VWF", "KDR", "CLDN5"],
    "Lymphatic endothelial cells": ["PROX1", "PDPN", "LYVE1"],
    "Fibroblasts": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "Smooth muscle cells": ["ACTA2", "TAGLN", "MYH11"],
    "Pericytes": ["RGS5", "PDGFRB", "CSPG4"],
    "Alveolar macrophages": ["MARCO", "FABP4", "PPARG", "MRC1"],
    "Monocytes": ["LYZ", "S100A8", "S100A9", "FCN1"],
    "Classical monocytes": ["FCN1", "S100A8", "S100A9", "VCAN"],
    "M2 macrophages": ["MRC1", "CD163", "MSR1"],
    "Neutrophils": ["S100A8", "S100A9", "CSF3R", "FCGR3B"],
    "Mast cells": ["TPSAB1", "TPSB2", "CPA3", "KIT"],
    "CD4-positive, alpha-beta T cells": ["CD3D", "CD3E", "CD4", "IL7R"],
    "CD8+ effector T cells": ["CD3D", "CD8A", "CD8B", "GZMB", "NKG7"],
    "Effector memory T cells": ["CD3D", "IL7R", "GZMK", "CCL5"],
    "Gamma delta T cells": ["TRDC", "TRGC1", "TRGC2", "CD3D"],
    "T-helper 17 cells": ["CCR6", "IL7R", "RORC", "IL23R"],
    "Naive B cells": ["MS4A1", "CD79A", "CD79B", "TCL1A"],
    "Mature B cells": ["MS4A1", "CD79A", "BANK1"],
    "Plasma cells": ["MZB1", "JCHAIN", "XBP1", "SDC1"],
    "Langerhans cells": ["CD207", "CD1A", "LAMP3"],
    "Tuft cells": ["POU2F3", "TRPM5", "AVIL"],
}

NONSPECIFIC_MARKERS = {
    "ACTB",
    "GAPDH",
    "MALAT1",
    "B2M",
    "TMSB4X",
    "TMSB10",
    "FOS",
    "JUN",
    "JUNB",
    "DUSP1",
    "HSP90AA1",
    "HSP90AB1",
    "HSPA1A",
    "HSPA1B",
}


def _is_nonspecific_gene(gene: str) -> bool:
    symbol = str(gene).strip().upper()
    return (
        symbol in NONSPECIFIC_MARKERS
        or symbol.startswith("RPL")
        or symbol.startswith("RPS")
        or symbol.startswith("MT-")
    )


def _read_marker_excel(path: Path) -> dict[str, list[str]]:
    try:
        df = pd.read_excel(path)
    except ImportError:
        rows = _read_xlsx_without_openpyxl(path)
        if not rows:
            raise
        width = max(len(row) for row in rows)
        df = pd.DataFrame([row + [""] * (width - len(row)) for row in rows])
    marker_dict = {}
    for _, row in df.iterrows():
        cell_type = str(row.iloc[0]).strip()
        markers = [str(x).strip() for x in row.iloc[1:] if pd.notna(x) and str(x).strip()]
        if cell_type and markers:
            marker_dict[cell_type] = markers
    return marker_dict


def _read_xlsx_without_openpyxl(path: Path) -> list[list[str]]:
    ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f".//{ns_main}si"):
                text = "".join(t.text or "" for t in si.findall(f".//{ns_main}t"))
                shared.append(text)

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = wb.find(f".//{ns_main}sheet")
        if first_sheet is None:
            return []
        rid = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall(f"{ns_rel}Relationship"):
            if rel.attrib.get("Id") == rid:
                target = rel.attrib.get("Target")
                break
        if not target:
            return []
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        root = ET.fromstring(zf.read(sheet_path))
        rows = []
        for row in root.findall(f".//{ns_main}sheetData/{ns_main}row"):
            vals = []
            for cell in row.findall(f"{ns_main}c"):
                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(f".//{ns_main}t"))
                else:
                    v = cell.find(f"{ns_main}v")
                    if v is not None and v.text is not None:
                        value = v.text
                        if cell_type == "s":
                            value = shared[int(value)]
                vals.append(value)
            if any(str(v).strip() for v in vals):
                rows.append(vals)
        return rows


def load_markers() -> dict[str, list[str]]:
    candidates = [
        config.CODE_DIR / "marker-lung.xlsx",
        config.ROOT_DIR / "marker-lung.xlsx",
        config.WORKSPACE_DIR / "2" / "marker-lung.xlsx",
    ]
    for path in candidates:
        if path.exists():
            try:
                markers = _read_marker_excel(path)
                if markers:
                    log(f"loaded marker table: {path} ({len(markers)} cell types)")
                    return markers
            except Exception as exc:
                log(f"marker table failed, using fallback if needed: {path}: {exc}")
    log(f"using built-in fallback lung markers ({len(FALLBACK_LUNG_MARKERS)} cell types)")
    return FALLBACK_LUNG_MARKERS.copy()


def filter_marker_sets(
    markers: dict[str, list[str]],
    var_names,
    min_present_markers: int = 3,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    var_set = set(map(str, var_names))
    filtered = {}
    rows = []
    for celltype, genes in markers.items():
        unique_genes = list(dict.fromkeys(str(g).strip() for g in genes if str(g).strip()))
        present = [g for g in unique_genes if g in var_set]
        nonspecific = [g for g in present if _is_nonspecific_gene(g)]
        used = [g for g in present if not _is_nonspecific_gene(g)]
        used_for_annotation = len(used) >= min_present_markers
        reason = ""
        if not used_for_annotation:
            reason = f"fewer than {min_present_markers} usable present markers after nonspecific-gene filtering"
        else:
            filtered[celltype] = used
        rows.append(
            {
                "celltype": celltype,
                "markers_total": len(unique_genes),
                "markers_present": len(present),
                "nonspecific_present_removed": len(nonspecific),
                "markers_used": len(used),
                "used_for_annotation": used_for_annotation,
                "excluded_reason": reason,
                "removed_nonspecific_markers": ";".join(nonspecific),
            }
        )
    log(f"marker filtering retained {len(filtered)}/{len(markers)} cell types for annotation")
    return filtered, pd.DataFrame(rows)
