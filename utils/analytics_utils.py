"""
Utility functions for the analytics dashboard.
Handles session loading, denormalization, and labeling.
"""

import json
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd


# ── Session loading ────────────────────────────────────────────────────────────

def list_sessions(exports_dir: str) -> list[str]:
    """Return available session folders sorted newest first."""
    if not os.path.exists(exports_dir):
        return []
    folders = [
        f for f in os.listdir(exports_dir)
        if os.path.isdir(os.path.join(exports_dir, f))
        and os.path.exists(os.path.join(exports_dir, f, "tft_config.json"))
    ]
    return sorted(folders, reverse=True)


def load_session(session_dir: str) -> dict:
    """
    Load all artifacts from a session folder.
    Returns dict with keys: tft, split, inv, merged, processed, session, groups.
    split, inv, merged, processed are optional (None if missing).
    groups is loaded from project root column_groups.json (optional).
    """
    def _load_json(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _load_pkl(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _load_csv(path):
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    tft      = _load_json(os.path.join(session_dir, "tft_config.json"))
    split    = _load_json(os.path.join(session_dir, "split_config.json"))
    session  = _load_json(os.path.join(session_dir, "session_config.json"))
    prep_cfg = _load_json(os.path.join(session_dir, "prep_config.json"))
    inv      = _load_pkl(os.path.join(session_dir, "inverse_transforms.pkl"))
    merged   = _load_csv(os.path.join(session_dir, "merged_data.csv"))
    proc     = _load_csv(os.path.join(session_dir, "processed_data.csv"))

    # column_groups.json — project root (two levels up from session_dir)
    # session_dir: .../exports/ActualData_xxx  → project root: .../dashboard_project
    project_root = os.path.dirname(os.path.dirname(session_dir))
    groups_path  = os.path.join(project_root, "column_groups.json")
    groups = _load_json(groups_path) or {}

    # Attach date column to merged if available
    if merged is not None and split and split.get("date_col") in merged.columns:
        merged[split["date_col"]] = pd.to_datetime(merged[split["date_col"]], errors="coerce")

    return {
        "tft":      tft,
        "split":    split,
        "session":  session,
        "prep_cfg": prep_cfg,
        "inv":      inv,
        "merged":   merged,
        "proc":     proc,
        "groups":   groups,
    }


# ── Labeling ───────────────────────────────────────────────────────────────────

def label(col: str, groups: dict) -> str:
    """Return human-readable label for a column, falling back to column name."""
    return groups.get("display_names", {}).get(col, col)


def decode_value(col: str, value, inv: dict) -> str:
    """
    Decode an encoded categorical value back to its label.
    Uses value_labels from inverse_transforms.pkl.
    Falls back to str(value) if no label found.
    """
    if inv is None:
        return str(value)
    meta = inv.get(col, {})
    vl   = meta.get("value_labels", {})
    return vl.get(str(int(value)) if isinstance(value, float) and value == int(value)
                  else str(value), str(value))


# ── Denormalization ────────────────────────────────────────────────────────────

def denormalize(series: pd.Series, col: str, inv: dict) -> pd.Series:
    """
    Reverse preprocessing transform for a column.
    Supports: zscore, minmax, robust, log1p. Others returned as-is.
    """
    if inv is None or col not in inv:
        return series
    meta   = inv[col]
    method = meta.get("method", "none")
    params = meta.get("params", {})

    if method == "zscore":
        mean = params.get("mean", 0)
        std  = params.get("std",  1)
        return series * std + mean

    if method == "minmax":
        mn = params.get("min", 0)
        mx = params.get("max", 1)
        return series * (mx - mn) + mn

    if method == "robust":
        median = params.get("median", 0)
        iqr    = params.get("iqr",    1)
        return series * iqr + median

    if method == "log1p":
        return np.expm1(series)

    return series


def denormalize_df(df: pd.DataFrame, cols: list[str], inv: dict) -> pd.DataFrame:
    """Denormalize multiple columns in a copy of df."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = denormalize(out[col], col, inv)
    return out


# ── Split masks ────────────────────────────────────────────────────────────────

def get_split_masks(df: pd.DataFrame, split: Optional[dict]) -> dict:
    """
    Return boolean masks for train / val / test splits.
    Uses date_col from split config applied to df.
    Returns {'train': mask, 'val': mask, 'test': mask} or empty dict if no split.
    """
    if split is None or df is None:
        return {}
    date_col  = split.get("date_col")
    if date_col not in df.columns:
        return {}

    dates     = pd.to_datetime(df[date_col], errors="coerce")
    train_end = pd.Timestamp(split["train_end"])
    val_end   = pd.Timestamp(split["val_end"])

    return {
        "train": dates <= train_end,
        "val":   (dates > train_end) & (dates <= val_end),
        "test":  dates > val_end,
    }


def split_label(split: Optional[dict]) -> str:
    """Short human-readable description of the split."""
    if not split:
        return "Сплит не задан"
    return (
        f"Train до {split.get('train_end', '?')} · "
        f"Val до {split.get('val_end', '?')} · "
        f"Test после"
    )


# ── Date axis helpers ──────────────────────────────────────────────────────────

def get_date_col(tft: dict, session: Optional[dict]) -> Optional[str]:
    """
    Return the original date column name.
    Tries session tidx_config first, then tft time_col.
    """
    if session:
        tidx = session.get("tidx_config", {})
        if tidx.get("src_col"):
            return tidx["src_col"]
    return tft.get("time_col") if tft else None


def add_date_to_proc(proc: pd.DataFrame, merged: pd.DataFrame,
                     tft: dict, session: Optional[dict]) -> pd.DataFrame:
    """
    If proc uses integer time_idx, join original date from merged.
    Returns proc with date column added (if possible).
    """
    if proc is None or merged is None:
        return proc

    src_date = get_date_col(tft, session)
    time_col = tft.get("time_col") if tft else None
    group_col = tft.get("group_col") if tft else None

    if src_date and src_date not in proc.columns and time_col in proc.columns:
        key_cols = [c for c in [time_col, group_col] if c and c in merged.columns]
        if src_date in merged.columns and key_cols:
            date_map = merged[key_cols + [src_date]].drop_duplicates()
            proc = proc.merge(date_map, on=key_cols, how="left")
            proc[src_date] = pd.to_datetime(proc[src_date], errors="coerce")

    return proc


# ── Group helpers ──────────────────────────────────────────────────────────────

def available_fuel_types(groups: dict, columns: list[str]) -> dict:
    """Return fuel_types entries where at least one column exists in data."""
    result = {}
    for name, entry in groups.get("fuel_types", {}).items():
        filtered = {k: v for k, v in entry.items() if v in columns}
        if filtered:
            result[name] = filtered
    return result


def available_traffic_lanes(groups: dict, columns: list[str]) -> dict:
    """Return traffic_lanes entries where column exists in data."""
    return {
        name: col
        for name, col in groups.get("traffic_lanes", {}).items()
        if col in columns
    }


def available_shop_categories(groups: dict, columns: list[str]) -> dict:
    """Return shop_categories entries where column exists in data."""
    return {
        name: col
        for name, col in groups.get("shop_categories", {}).items()
        if col in columns
    }


def filter_merged_duplicates(df: pd.DataFrame, session: Optional[dict]) -> pd.DataFrame:
    """
    Drop columns that pandas created as suffixed duplicates during a multi-file merge.
    E.g. col_gas_stations_temporal_2023_2025 is dropped when col also exists.
    """
    if df is None or session is None:
        return df
    merge_configs = session.get("merge_configs", [])
    if not merge_configs:
        return df
    cols = set(df.columns)
    to_drop = []
    for mc in merge_configs:
        stem = os.path.splitext(mc.get("right_name", ""))[0]
        if not stem:
            continue
        suffix = "_" + stem
        for col in df.columns:
            if col.endswith(suffix) and col not in to_drop:
                base = col[:-len(suffix)]
                if base in cols:
                    to_drop.append(col)
    return df.drop(columns=to_drop) if to_drop else df


def col_value_labels(col: str, inv: Optional[dict]) -> dict:
    """Return {str_key: label} mapping for a column from inverse_transforms."""
    if inv is None or col not in inv:
        return {}
    return inv[col].get("value_labels", {})


def decode_col_series(series: pd.Series, col: str, inv: Optional[dict]) -> pd.Series:
    """Map a column's numeric codes to human-readable labels via inverse_transforms."""
    vl = col_value_labels(col, inv)
    if not vl:
        return series.astype(str)

    def _map(v):
        if pd.isna(v):
            return v
        try:
            key = str(int(float(v)))
        except (ValueError, TypeError):
            key = str(v)
        return vl.get(key, str(v))

    return series.apply(_map)
