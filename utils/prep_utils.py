"""Preprocessing utility functions (no Streamlit dependency)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Type detection ────────────────────────────────────────────────────────────

def detect_col_type(series: pd.Series) -> str:
    """Return one of: numerical / categorical / binary / datetime / text."""
    n = len(series)

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null = series.dropna()
    if len(non_null) == 0:
        return "numerical"

    nunique = int(non_null.nunique())

    # Binary: at most 2 distinct values, all in {0, 1, True, False}
    if nunique <= 2:
        binary_set: set = {0, 1, True, False, 0.0, 1.0}
        if set(non_null.unique()).issubset(binary_set):
            return "binary"

    if pd.api.types.is_numeric_dtype(series):
        if nunique <= 15 and n > 0 and nunique / n < 0.05:
            return "categorical"
        return "numerical"

    is_cat_dtype = str(series.dtype) == "category"
    if pd.api.types.is_object_dtype(series) or is_cat_dtype:
        if n > 0 and (nunique / n < 0.5 or nunique <= 50):
            return "categorical"
        return "text"

    # Catch pandas StringDtype / ArrowDtype strings that is_object_dtype misses
    if pd.api.types.is_string_dtype(series) and not pd.api.types.is_numeric_dtype(series):
        if n > 0 and (nunique / n < 0.5 or nunique <= 50):
            return "categorical"
        return "text"

    return "numerical"


def detect_col_types(df: pd.DataFrame) -> Dict[str, str]:
    """Apply detect_col_type to every column; return {col: type}."""
    return {col: detect_col_type(df[col]) for col in df.columns}


# ─── Preprocessing ─────────────────────────────────────────────────────────────

def apply_preprocessing(
    df: pd.DataFrame,
    prep_config: Dict[str, Dict[str, Any]],
    train_mask: Optional["pd.Series"] = None,  # type: ignore[type-arg]
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Apply preprocessing methods defined in prep_config.

    prep_config keys per column:
      method, params, skip, pre_fillna, pre_fillna_value,
      fit_on ("train"|"full"), value_labels ({value: label})

    train_mask: boolean Series aligned to df.index; when provided and a column
                has fit_on=="train", scaling parameters are estimated only on
                df[train_mask] — avoids data leakage.

    Returns: (transformed_df, log_messages, inverse_params)
      inverse_params: {col: {method, fit_on, params, value_labels}} for
                      later inverse-transform or reporting.
    """
    from sklearn.preprocessing import LabelEncoder  # lazy import

    new_df = df.copy()
    log: List[str] = []
    inverse_params: Dict[str, Any] = {}

    # ── Pre-fillna pass ────────────────────────────────────────────────────────
    for col, config in prep_config.items():
        if config.get("pre_fillna") and col in new_df.columns:
            fill_val = config.get("pre_fillna_value", "")
            n_filled = int(new_df[col].isna().sum())
            if n_filled > 0:
                new_df[col] = new_df[col].fillna(fill_val)
                log.append(f"[pre_fillna] {col}: заполнено {n_filled} пропусков → {fill_val!r}")

    _extra_cols: Dict[str, Any] = {}

    for col, config in prep_config.items():
        if config.get("skip", False):
            continue

        method = config.get("method", "none")
        params = config.get("params", {})
        value_labels = config.get("value_labels", {})

        fit_on_train = (
            config.get("fit_on") == "train"
            and train_mask is not None
            and bool(train_mask.sum()) > 0
        )

        if method == "none" or col not in new_df.columns:
            if value_labels:
                inverse_params[col] = {"method": "none", "value_labels": value_labels}
            continue

        try:
            # ── Numerical transforms ───────────────────────────────────────────
            if method == "zscore":
                fit_s = new_df.loc[train_mask, col] if fit_on_train else new_df[col]
                mean = float(fit_s.mean())
                std  = float(fit_s.std())
                if std != 0:
                    new_df[col] = (new_df[col] - mean) / std
                    log.append(
                        f"[zscore] {col}: μ={mean:.4f}, σ={std:.4f}"
                        + (" (по train)" if fit_on_train else "")
                    )
                else:
                    log.append(f"[zscore] {col}: пропущено (std=0)")
                inverse_params[col] = {
                    "method": "zscore",
                    "fit_on": "train" if fit_on_train else "full",
                    "params": {"mean": mean, "std": std},
                    **({"skipped": True, "skip_reason": "std=0"} if std == 0 else {}),
                }

            elif method == "minmax":
                fit_s = new_df.loc[train_mask, col] if fit_on_train else new_df[col]
                mn = float(fit_s.min())
                mx = float(fit_s.max())
                if mx != mn:
                    new_df[col] = (new_df[col] - mn) / (mx - mn)
                    log.append(
                        f"[minmax] {col}: min={mn:.4f}, max={mx:.4f}"
                        + (" (по train)" if fit_on_train else "")
                    )
                else:
                    log.append(f"[minmax] {col}: пропущено (min==max)")
                inverse_params[col] = {
                    "method": "minmax",
                    "fit_on": "train" if fit_on_train else "full",
                    "params": {"min": mn, "max": mx},
                    **({"skipped": True, "skip_reason": "min==max"} if mx == mn else {}),
                }

            elif method == "robust":
                fit_s = new_df.loc[train_mask, col] if fit_on_train else new_df[col]
                q25 = float(fit_s.quantile(0.25))
                q75 = float(fit_s.quantile(0.75))
                iqr = q75 - q25
                if iqr != 0:
                    new_df[col] = (new_df[col] - q25) / iqr
                    log.append(
                        f"[robust] {col}: q25={q25:.4f}, q75={q75:.4f}, IQR={iqr:.4f}"
                        + (" (по train)" if fit_on_train else "")
                    )
                else:
                    log.append(f"[robust] {col}: пропущено (IQR=0)")
                inverse_params[col] = {
                    "method": "robust",
                    "fit_on": "train" if fit_on_train else "full",
                    "params": {"q25": q25, "iqr": iqr},
                    **({"skipped": True, "skip_reason": "IQR=0"} if iqr == 0 else {}),
                }

            elif method == "log1p":
                new_df[col] = np.log1p(new_df[col].clip(lower=0))
                log.append(f"[log1p] {col}: применено")
                inverse_params[col] = {"method": "log1p", "params": {}}

            elif method == "fillna":
                fill_val = params.get("value", 0)
                count = int(new_df[col].isna().sum())
                new_df[col] = new_df[col].fillna(fill_val)
                log.append(f"[fillna] {col}: заполнено {count} пропусков → {fill_val!r}")
                inverse_params[col] = {"method": "fillna", "params": {"value": fill_val}}

            elif method == "cyclical":
                period = float(params.get("period", 24))
                _extra_cols[f"{col}_sin"] = np.sin(2 * np.pi * new_df[col] / period)
                _extra_cols[f"{col}_cos"] = np.cos(2 * np.pi * new_df[col] / period)
                new_df.drop(columns=[col], inplace=True)
                log.append(f"[cyclical] {col}: период={period} → {col}_sin, {col}_cos")
                inverse_params[col] = {
                    "method": "cyclical",
                    "params": {
                        "period": period,
                        "sin_col": f"{col}_sin",
                        "cos_col": f"{col}_cos",
                    },
                }

            # ── Encoding ──────────────────────────────────────────────────────
            elif method == "label_enc":
                le = LabelEncoder()
                _extra_cols[f"{col}_enc"] = pd.Series(
                    le.fit_transform(new_df[col].astype(str)).astype("int64"),
                    index=new_df.index,
                )
                log.append(f"[label_enc] {col}: → {col}_enc ({len(le.classes_)} классов)")
                inverse_params[col] = {
                    "method": "label_enc",
                    "params": {"classes": le.classes_.tolist(), "enc_col": f"{col}_enc"},
                }

            elif method == "onehot":
                nunique = int(new_df[col].nunique())
                dummies = pd.get_dummies(new_df[col], prefix=col)
                new_df.drop(columns=[col], inplace=True)
                for _dc in dummies.columns:
                    _extra_cols[_dc] = dummies[_dc]
                log.append(f"[onehot] {col}: → {len(dummies.columns)} колонок ({nunique} значений)")
                inverse_params[col] = {
                    "method": "onehot",
                    "params": {"categories": [str(c) for c in dummies.columns]},
                }

            # ── Datetime ──────────────────────────────────────────────────────
            elif method == "dt_extract":
                components = params.get("components", ["year", "month", "day"])
                dt_series = pd.to_datetime(new_df[col], errors="coerce")
                comp_extractors = _DT_COMP_EXTRACTORS
                added: List[str] = []
                for comp in components:
                    if comp in comp_extractors:
                        _extra_cols[f"{col}_{comp}"] = comp_extractors[comp](dt_series)
                        added.append(f"{col}_{comp}")
                log.append(f"[dt_extract] {col}: → {', '.join(added)}")
                inverse_params[col] = {
                    "method": "dt_extract",
                    "params": {"components": components},
                }

            elif method == "drop":
                new_df.drop(columns=[col], inplace=True)
                log.append(f"[drop] {col}: удалена")
                inverse_params[col] = {"method": "drop", "params": {}}

        except Exception as exc:
            raise RuntimeError(
                f"Колонка «{col}» (метод: {method}): {type(exc).__name__}: {exc}"
            ) from exc

        if value_labels and method != "drop":
            inverse_params.setdefault(col, {})["value_labels"] = value_labels

    if _extra_cols:
        new_df = pd.concat(
            [new_df, pd.DataFrame(_extra_cols, index=new_df.index)],
            axis=1,
        )

    return new_df.copy(), log, inverse_params


# ─── Merge ─────────────────────────────────────────────────────────────────────

def merge_dataframes(
    files_dict: Dict[str, pd.DataFrame],
    merge_configs: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Execute a chain of JOINs.

    merge_configs: [{right_name, on, how}]
    The left side always accumulates from the first file.
    """
    names = list(files_dict.keys())
    result = files_dict[names[0]].copy()

    for config in merge_configs:
        right_name = config["right_name"]
        right = files_dict[right_name]
        on = config["on"]
        how = config.get("how", "left")
        safe_suffix = "".join(
            c if c.isalnum() or c == "_" else "_"
            for c in right_name.replace(".csv", "")
        )
        result = result.merge(right, on=on, how=how, suffixes=("", f"_{safe_suffix}"))

    return result


# ─── Column summary ────────────────────────────────────────────────────────────

def get_column_summary(
    df: pd.DataFrame,
    col_types: Dict[str, str],
) -> pd.DataFrame:
    """Build a per-column summary table: name, type, missing, unique, info."""
    n = len(df)
    rows = []

    for col in df.columns:
        ctype = col_types.get(col, "numerical")
        n_miss = int(df[col].isna().sum())
        pct_miss = n_miss / n * 100 if n > 0 else 0.0
        n_uniq = int(df[col].nunique())

        if ctype == "numerical":
            try:
                num = pd.to_numeric(df[col], errors="coerce")
                mean = num.mean()
                std = num.std()
                mn = num.min()
                mx = num.max()
                info = f"μ={mean:.2f}, σ={std:.2f}, [{mn:.2f}…{mx:.2f}]"
            except Exception:
                info = "—"

        elif ctype in ("categorical", "text"):
            vc = df[col].value_counts()
            top = str(vc.index[0]) if len(vc) > 0 else "—"
            info = f"топ: {top}"

        elif ctype == "binary":
            vc = df[col].value_counts()
            top = str(vc.index[0]) if len(vc) > 0 else "—"
            try:
                n_ones = int(df[col].isin([1, True, 1.0]).sum())
                pct_ones = n_ones / n * 100 if n > 0 else 0.0
                info = f"топ: {top} | 1: {n_ones} ({pct_ones:.1f}%)"
            except Exception:
                info = f"топ: {top}"

        elif ctype == "datetime":
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                info = f"{dt.min().date()} … {dt.max().date()}"
            except Exception:
                info = "—"

        else:
            info = "—"

        rows.append({
            "Колонка": col,
            "Тип": ctype,
            "Пропуски": f"{n_miss} ({pct_miss:.1f}%)",
            "Уникальных": n_uniq,
            "Инфо": info,
        })

    return pd.DataFrame(rows)


# ─── TFT config builder ────────────────────────────────────────────────────────

def build_tft_config(roles: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the TFT JSON config from role assignments."""
    return {
        "time_col": roles.get("time_col"),
        "group_col": roles.get("group_col"),
        "target": roles.get("target", []),
        "static_cats": roles.get("static_cat", []),
        "static_reals": roles.get("static_real", []),
        "time_varying_known_categoricals": roles.get("known_cat", []),
        "time_varying_known_reals": roles.get("known_real", []),
        "time_varying_unknown_reals": roles.get("unknown_real", []),
        "dropped": roles.get("dropped", []),
    }


# ─── Report name validation ────────────────────────────────────────────────────

_WINDOWS_INVALID_CHARS = r'\/:*?"<>|'


def validate_report_name(name: str) -> Tuple[bool, str]:
    """Check that *name* is safe to use as a Windows filename stem.

    Returns (is_valid, error_message).  error_message is empty when valid.
    """
    stripped = name.strip()
    if not stripped:
        return False, "Название не может быть пустым."
    found = [c for c in _WINDOWS_INVALID_CHARS if c in stripped]
    if found:
        chars = " ".join(f'«{c}»' for c in found)
        return False, f"Недопустимые символы: {chars}"
    if stripped.endswith("."):
        return False, "Название не должно заканчиваться точкой."
    return True, ""


# ─── Formula evaluation ────────────────────────────────────────────────────────

def eval_formula(
    df: pd.DataFrame,
    formula: str,
) -> Tuple[Any, Optional[str], int]:
    """Evaluate a pandas formula string against df.

    Returns (result, error_message_or_None, n_inf).
    """
    try:
        result = df.eval(formula, engine="python")
        if not hasattr(result, "__len__") or len(result) != len(df):
            return None, "Формула должна возвращать одно значение на каждую строку.", 0
        n_inf = int(np.isinf(pd.to_numeric(result, errors="coerce").fillna(0)).sum())
        return result, None, n_inf
    except ZeroDivisionError:
        return None, "Деление на ноль.", 0
    except Exception as exc:
        return None, str(exc), 0


# ─── Date component extraction ─────────────────────────────────────────────────

_DT_COMP_EXTRACTORS: Dict[str, Any] = {
    "year":       lambda s: s.dt.year,
    "month":      lambda s: s.dt.month,
    "day":        lambda s: s.dt.day,
    "hour":       lambda s: s.dt.hour,
    "minute":     lambda s: s.dt.minute,
    "dayofweek":  lambda s: s.dt.dayofweek,
    "dayofyear":  lambda s: s.dt.dayofyear,
    "weekofyear": lambda s: s.dt.isocalendar().week.astype(int),
}


def apply_dt_extractions(
    df: pd.DataFrame,
    dtx_configs: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply date-component extractions defined in dtx_configs.

    Each entry: {src_col: str, components: list[str]}.
    Returns (new_df, list_of_added_column_names).
    """
    new_df = df.copy()
    added: List[str] = []
    for cfg in dtx_configs:
        src_col = cfg.get("src_col")
        components = cfg.get("components", [])
        if not src_col or src_col not in new_df.columns:
            continue
        dt_series = pd.to_datetime(new_df[src_col], format="mixed", errors="coerce")
        if dt_series.isna().all():
            continue
        for comp in components:
            if comp in _DT_COMP_EXTRACTORS:
                col_name = f"{src_col}_{comp}"
                new_df[col_name] = _DT_COMP_EXTRACTORS[comp](dt_series)
                added.append(col_name)
    return new_df, added


# ─── Session config builder ────────────────────────────────────────────────────

def build_session_config(
    source_file_names: List[str],
    merge_configs: List[Dict[str, Any]],
    added_col_formulas: Dict[str, str],
    tidx_config: Dict[str, Any],
    dtx_configs: List[Dict[str, Any]],
    col_types: Dict[str, str],
    preprocessing_applied: bool,
    file_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the session_config dict used for save/download.

    Temporal split is stored separately in split_config.json — not here.
    """
    return {
        "source_files_order": source_file_names,
        "merge_configs": merge_configs,
        "added_col_formulas": added_col_formulas,
        "tidx_config": tidx_config,
        "dtx_configs": dtx_configs,
        "col_types": col_types,
        "preprocessing_applied": preprocessing_applied,
        "file_settings": file_settings,
    }


# ─── Temporal split ────────────────────────────────────────────────────────────

def compute_split_masks(
    df: pd.DataFrame,
    split_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Recompute train/val/test boolean masks from split_config.

    Returns the split_masks dict (same structure as session_state["split_masks"]),
    or None if the date column is missing or parsing fails.
    """
    try:
        date_col = split_config.get("date_col")
        if not date_col or date_col not in df.columns:
            return None
        if pd.api.types.is_numeric_dtype(df[date_col]):
            return None

        dt_s = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
        train_end = pd.Timestamp(split_config["train_end"])
        val_end   = pd.Timestamp(split_config["val_end"])

        train_m = (dt_s <= train_end) & dt_s.notna()
        val_m   = (dt_s > train_end) & (dt_s <= val_end) & dt_s.notna()
        test_m  = (dt_s > val_end) & dt_s.notna()

        n_tr = int(train_m.sum())
        n_vl = int(val_m.sum())
        n_ts = int(test_m.sum())

        if n_tr == 0 or n_vl == 0 or n_ts == 0:
            return None

        return {
            "train": train_m.values.tolist(),
            "val":   val_m.values.tolist(),
            "test":  test_m.values.tolist(),
            "stats": {
                "n_train": n_tr, "n_val": n_vl, "n_test": n_ts,
                "date_col":  date_col,
                "train_end": split_config["train_end"],
                "val_end":   split_config["val_end"],
            },
        }
    except Exception:
        return None


# ─── Markdown report builder ───────────────────────────────────────────────────

def build_report(
    uploaded: Dict[str, pd.DataFrame],
    merged: Optional[pd.DataFrame],
    prep_df: Optional[pd.DataFrame],
    col_types: Dict[str, str],
    prep_config: Dict[str, Dict[str, Any]],
    tft_roles: Dict[str, Any],
    method_labels: Dict[str, str],
    split_config: Optional[Dict[str, Any]] = None,
    inverse_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a Markdown data-preparation report from processed data."""
    from datetime import datetime

    L: List[str] = [
        "# Отчёт о подготовке данных",
        f"*Сформирован: {datetime.now().strftime('%d.%m.%Y %H:%M')}*",
        "",
    ]

    # ── 1. Source files ────────────────────────────────────────────────────────
    L += ["## 1. Исходные данные", ""]
    if uploaded:
        L += [
            f"Загружено файлов: **{len(uploaded)}**", "",
            "| Файл | Строк | Колонок |",
            "|------|------:|--------:|",
        ]
        for fname, df_u in uploaded.items():
            L.append(f"| {fname} | {len(df_u):,} | {len(df_u.columns)} |")
        L.append("")
    if merged is not None:
        L.append(f"**Объединённый датасет:** {len(merged):,} строк × {len(merged.columns)} колонок")
        L.append("")

    # ── 2. Column types ────────────────────────────────────────────────────────
    if col_types and merged is not None:
        L += [
            "## 2. Типы колонок", "",
            "| Колонка | Тип | Пропусков | Уникальных |",
            "|---------|-----|----------:|-----------:|",
        ]
        for col, ctype in col_types.items():
            if col not in merged.columns:
                continue
            n_miss = int(merged[col].isna().sum())
            pct    = f"{n_miss / len(merged) * 100:.1f}%" if len(merged) > 0 else "—"
            n_uniq = int(merged[col].nunique())
            miss_s = f"{n_miss} ({pct})" if n_miss > 0 else "—"
            L.append(f"| {col} | {ctype} | {miss_s} | {n_uniq} |")
        L.append("")

    # ── 3. Preprocessing ───────────────────────────────────────────────────────
    L += ["## 3. Препроцессинг", ""]
    active = {
        col: cfg for col, cfg in prep_config.items()
        if cfg.get("method", "none") not in (None, "none") or cfg.get("pre_fillna")
    }
    if active:
        L += [
            "| Колонка | Метод | Пре-fillna | Значение заполнения |",
            "|---------|-------|:----------:|---------------------|",
        ]
        for col, cfg in active.items():
            method = method_labels.get(cfg.get("method", "none"), cfg.get("method", "—"))
            pf     = "✓" if cfg.get("pre_fillna") else "—"
            pf_val = str(cfg.get("pre_fillna_value", "—")) if cfg.get("pre_fillna") else "—"
            L.append(f"| {col} | {method} | {pf} | {pf_val} |")
        L.append("")
    else:
        L += ["*Методы не назначены.*", ""]
    if prep_df is not None and merged is not None:
        diff = len(prep_df.columns) - len(merged.columns)
        sign = f"+{diff}" if diff > 0 else str(diff)
        L += [
            f"**Датасет после препроцессинга:** {len(prep_df):,} строк × "
            f"{len(prep_df.columns)} колонок ({sign} колонок)",
            "",
        ]

    # ── 4. TFT roles ───────────────────────────────────────────────────────────
    L += [
        "## 4. Распределение ролей TFT", "",
        f"- **Временная колонка:** {tft_roles.get('time_col') or '—'}",
        f"- **Группирующая колонка:** {tft_roles.get('group_col') or '—'}",
        "",
        "| Роль | Колонки |",
        "|------|---------|",
    ]
    role_names = {
        "target":       "Целевые переменные",
        "static_cat":   "Статические категориальные",
        "static_real":  "Статические вещественные",
        "known_cat":    "Известные будущие категориальные",
        "known_real":   "Известные будущие вещественные",
        "unknown_real": "Наблюдаемые прошлые (энкодер)",
        "dropped":      "Исключены из модели",
    }
    for rk, rname in role_names.items():
        cols_r = tft_roles.get(rk, [])
        L.append(f"| {rname} | {', '.join(cols_r) if cols_r else '—'} |")
    L.append("")

    # ── 4b. Temporal split ─────────────────────────────────────────────────────
    if split_config:
        sc = split_config
        L += ["## 4b. Временной сплит", ""]
        L.append(f"Колонка: `{sc.get('date_col', '—')}`  ")
        L.append(f"Режим: {'по процентам' if sc.get('mode') == 'pct' else 'по датам'}  ")
        if sc.get("mode") == "pct":
            test_pct = 100 - sc.get("train_pct", 0) - sc.get("val_pct", 0)
            L.append(f"Train {sc.get('train_pct')}% · Val {sc.get('val_pct')}% · Test {test_pct}%  ")
        L.append(f"Train до: `{sc.get('train_end', '—')}` · Val до: `{sc.get('val_end', '—')}`")
        L.append("")

    # ── 4c. Value labels ───────────────────────────────────────────────────────
    _vl_rows: List[str] = []
    for col, cfg in prep_config.items():
        vl = cfg.get("value_labels", {})
        if vl:
            for val, lbl in vl.items():
                if str(val) != str(lbl):
                    _vl_rows.append(f"| `{col}` | {val} | {lbl} |")
    if _vl_rows:
        L += [
            "## 4c. Расшифровки значений", "",
            "| Колонка | Значение | Расшифровка |",
            "|---------|----------|-------------|",
        ] + _vl_rows + [""]

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    L += ["## 5. Итог", ""]
    total_assigned = (
        sum(len(tft_roles.get(rk, [])) for rk in role_names)
        + (1 if tft_roles.get("time_col") else 0)
        + (1 if tft_roles.get("group_col") else 0)
    )
    final = prep_df if prep_df is not None else merged
    if final is not None:
        L += [
            "| | |",
            "|---|---|",
            f"| Строк в итоговом датасете | {len(final):,} |",
            f"| Колонок в итоговом датасете | {len(final.columns)} |",
            f"| Колонок с назначенной ролью TFT | {total_assigned} |",
            "",
        ]

    return "\n".join(L)
