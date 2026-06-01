"""Training utility functions (no Streamlit dependency)."""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPORTS_DIR  = os.path.join(ROOT, "exports")
TRAINING_DIR = os.path.join(ROOT, "training")
TRAIN_PY     = os.path.join(ROOT, "tft", "train.py")
PREDICT_PY   = os.path.join(ROOT, "tft", "predict.py")

CPU_DEFAULTS: Dict[str, Any] = {
    "batch_size":             32,
    "max_epochs":             50,
    "hidden_size":            64,
    "attention_head_size":    2,
    "hidden_continuous_size": 32,
    "learning_rate":          3e-4,
    "dropout":                0.15,
    "gradient_clip":          1.0,
    "patience":               12,
    "encoder_length":         30,
    "prediction_length":      7,
}

GPU_DEFAULTS: Dict[str, Any] = {
    "batch_size":             64,
    "max_epochs":             80,
    "hidden_size":            128,
    "attention_head_size":    4,
    "hidden_continuous_size": 64,
    "learning_rate":          3e-4,
    "dropout":                0.15,
    "gradient_clip":          1.0,
    "patience":               12,
    "encoder_length":         60,
    "prediction_length":      14,
}

# Required to run training
REQUIRED_FOR_TRAINING  = ["processed_data.csv", "tft_config.json"]
# Required to build predictions (on top of training files)
REQUIRED_FOR_PREDICTION = ["inverse_transforms.pkl", "merged_data.csv"]
# Optional: split boundaries (can be entered manually in the dashboard)
OPTIONAL_FILES         = ["split_config.json"]
# Present in export but not consumed by train/predict pipelines
UNUSED_FILES           = ["prep_config.json", "session_config.json"]

# Legacy aliases kept for check_session_files
REQUIRED_EXPORT_FILES = REQUIRED_FOR_TRAINING
KNOWN_OPTIONAL_FILES  = REQUIRED_FOR_PREDICTION + OPTIONAL_FILES + UNUSED_FILES


# ─── Export sessions ───────────────────────────────────────────────────────────

def list_export_sessions() -> List[str]:
    """Return export session names that have the mandatory training files, newest first."""
    if not os.path.exists(EXPORTS_DIR):
        return []
    entries = [
        e for e in os.listdir(EXPORTS_DIR)
        if os.path.isdir(os.path.join(EXPORTS_DIR, e))
        and os.path.exists(os.path.join(EXPORTS_DIR, e, "processed_data.csv"))
        and os.path.exists(os.path.join(EXPORTS_DIR, e, "tft_config.json"))
    ]
    return sorted(entries, reverse=True)


def get_export_path(session_name: str) -> str:
    return os.path.join(EXPORTS_DIR, session_name)


def get_data_info(session_name: str) -> Optional[Dict[str, Any]]:
    """Return {n_rows, n_cols, columns} without fully loading the CSV."""
    path = os.path.join(get_export_path(session_name), "processed_data.csv")
    if not os.path.exists(path):
        return None
    try:
        header = pd.read_csv(path, encoding="utf-8", nrows=0)
        with open(path, encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1
        return {"n_rows": n_rows, "n_cols": len(header.columns),
                "columns": header.columns.tolist()}
    except Exception:
        return None


def load_tft_config(session_name: str) -> Dict[str, Any]:
    path = os.path.join(get_export_path(session_name), "tft_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_split_config(session_name: str) -> Dict[str, Any]:
    path = os.path.join(get_export_path(session_name), "split_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_session_config(session_name: str) -> Dict[str, Any]:
    path = os.path.join(get_export_path(session_name), "session_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_group_count(session_name: str) -> Optional[int]:
    """Count unique groups by reading only the group column of the CSV."""
    tft_cfg   = load_tft_config(session_name)
    group_col = (tft_cfg or {}).get("group_col")
    if not group_col:
        return None
    path = os.path.join(get_export_path(session_name), "processed_data.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=[group_col], encoding="utf-8")
        return int(df[group_col].nunique())
    except Exception:
        return None


def check_session_files(session_name: str) -> Dict[str, Any]:
    """Return presence/size info and content validation for the export session folder."""
    import pickle as _pickle

    export_path = get_export_path(session_name)
    present: Dict[str, int] = {}
    if os.path.isdir(export_path):
        for fname in os.listdir(export_path):
            fpath = os.path.join(export_path, fname)
            if os.path.isfile(fpath):
                present[fname] = os.path.getsize(fpath)

    all_known = (REQUIRED_FOR_TRAINING + REQUIRED_FOR_PREDICTION
                 + OPTIONAL_FILES + UNUSED_FILES)

    def _entry(name, category):
        return {"name": name, "exists": name in present,
                "size": present.get(name), "category": category}

    train_files  = [_entry(f, "train")      for f in REQUIRED_FOR_TRAINING]
    pred_files   = [_entry(f, "predict")    for f in REQUIRED_FOR_PREDICTION]
    opt_files    = [_entry(f, "optional")   for f in OPTIONAL_FILES if f in present]
    unused_files = [_entry(f, "unused")     for f in UNUSED_FILES   if f in present]
    other_files  = [{"name": k, "exists": True, "size": v, "category": "other"}
                    for k, v in present.items() if k not in all_known]

    # ── Content validation ────────────────────────────────────────────────────
    issues:   List[str] = []  # blockers (❌)
    warnings: List[str] = []  # non-critical (⚠️)

    tft_cfg = load_tft_config(session_name)

    # tft_config.json — required fields
    if tft_cfg:
        for field, label in (("time_col", "time_col"), ("group_col", "group_col")):
            if not tft_cfg.get(field):
                issues.append(f"tft_config.json: не задан «{label}»")
        if not tft_cfg.get("target"):
            issues.append("tft_config.json: не заданы целевые переменные (target)")
    elif "tft_config.json" in present:
        issues.append("tft_config.json: не удалось прочитать файл")

    # processed_data.csv — columns match tft_config
    data_path = os.path.join(export_path, "processed_data.csv")
    if tft_cfg and os.path.exists(data_path):
        try:
            _header = pd.read_csv(data_path, nrows=0, encoding="utf-8")
            _cols   = set(_header.columns)
            _missing = []
            for _f in ("time_col", "group_col"):
                _v = tft_cfg.get(_f)
                if _v and _v not in _cols:
                    _missing.append(_v)
            for _t in tft_cfg.get("target", []):
                if _t not in _cols:
                    _missing.append(_t)
            if _missing:
                issues.append(
                    f"processed_data.csv: отсутствуют колонки из tft_config: "
                    f"{', '.join(_missing)}"
                )
            if len(_header.columns) == 0:
                issues.append("processed_data.csv: файл пустой (нет колонок)")
        except Exception as _e:
            issues.append(f"processed_data.csv: ошибка чтения ({_e})")

    # split_config.json — train_end and val_end
    split_cfg = load_split_config(session_name)
    if split_cfg:
        for _k in ("train_end", "val_end"):
            if not split_cfg.get(_k):
                issues.append(f"split_config.json: не задан «{_k}»")
    elif "split_config.json" in present:
        issues.append("split_config.json: не удалось прочитать файл")

    # inverse_transforms.pkl — loadable, covers targets
    inv_path = os.path.join(export_path, "inverse_transforms.pkl")
    if os.path.exists(inv_path):
        try:
            with open(inv_path, "rb") as _f:
                _inv = _pickle.load(_f)
            if tft_cfg:
                _no_inv = [t for t in tft_cfg.get("target", []) if t not in _inv]
                if _no_inv:
                    warnings.append(
                        f"inverse_transforms.pkl: нет параметров для таргетов: "
                        f"{', '.join(_no_inv)} — обратное преобразование не будет применено"
                    )
        except Exception as _e:
            issues.append(f"inverse_transforms.pkl: не удалось загрузить ({_e})")

    # merged_data.csv — has a date-like column for time_idx mapping
    merged_path = os.path.join(export_path, "merged_data.csv")
    if os.path.exists(merged_path):
        try:
            _mh = pd.read_csv(merged_path, nrows=0, encoding="utf-8")
            if not any("date" in c.lower() for c in _mh.columns):
                warnings.append(
                    "merged_data.csv: не найдена колонка с датой — "
                    "маппинг time_idx→date для прогнозирования может не работать"
                )
        except Exception as _e:
            warnings.append(f"merged_data.csv: ошибка чтения ({_e})")

    return {
        "path":             export_path,
        "train":            train_files,
        "predict":          pred_files,
        "optional":         opt_files,
        "unused":           unused_files,
        "other":            other_files,
        "content_issues":   issues,
        "content_warnings": warnings,
        "content_ok":       len(issues) == 0,
        # legacy keys
        "required":         train_files,
        "all_required_present": all(f["exists"] for f in train_files),
    }


# ─── Training artifact paths ───────────────────────────────────────────────────

def get_training_dir(session_name: str) -> str:
    return os.path.join(TRAINING_DIR, session_name)


def get_train_config_path(session_name: str) -> str:
    return os.path.join(get_training_dir(session_name), "train_config.json")


def get_log_dir(session_name: str) -> str:
    return os.path.join(get_training_dir(session_name), "logs")


def get_ckpt_dir(session_name: str) -> str:
    return os.path.join(get_training_dir(session_name), "checkpoints")


def get_model_path(session_name: str) -> str:
    return os.path.join(get_training_dir(session_name), "model.ckpt")


def get_predictions_dir(session_name: str) -> str:
    return os.path.join(get_training_dir(session_name), "predictions")


def list_ckpt_files(session_name: str) -> List[Dict[str, Any]]:
    """Return available checkpoints for a session, best model first."""
    result: List[Dict[str, Any]] = []
    model_path = get_model_path(session_name)
    if os.path.exists(model_path):
        result.append({
            "label": "Лучшая модель  (model.ckpt)",
            "path":  model_path,
            "is_best": True,
        })
    ckpt_dir = get_ckpt_dir(session_name)
    if os.path.isdir(ckpt_dir):
        for fname in sorted(os.listdir(ckpt_dir), reverse=True):
            if fname.endswith(".ckpt"):
                result.append({
                    "label": fname,
                    "path":  os.path.join(ckpt_dir, fname),
                    "is_best": False,
                })
    return result


# ─── Training config ───────────────────────────────────────────────────────────

def load_train_config(session_name: str) -> Dict[str, Any]:
    cfg = dict(CPU_DEFAULTS)
    path = get_train_config_path(session_name)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save_train_config(session_name: str, cfg: Dict[str, Any]) -> None:
    training_dir = get_training_dir(session_name)
    os.makedirs(training_dir, exist_ok=True)
    with open(get_train_config_path(session_name), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def delete_train_config(session_name: str) -> None:
    path = get_train_config_path(session_name)
    if os.path.exists(path):
        os.remove(path)


def save_split_config_to_training(session_name: str, train_end: str, val_end: str) -> None:
    """Save manually entered temporal split dates to the training session directory."""
    training_dir = get_training_dir(session_name)
    os.makedirs(training_dir, exist_ok=True)
    path = os.path.join(training_dir, "split_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"train_end": train_end.strip(), "val_end": val_end.strip()},
            f, indent=2, ensure_ascii=False,
        )


# ─── TensorBoard ───────────────────────────────────────────────────────────────

def list_tb_versions(session_name: str) -> List[str]:
    log_dir = get_log_dir(session_name)
    return sorted(glob.glob(os.path.join(log_dir, "tft_model", "version_*")))


def read_tb_losses(session_name: str) -> List[Tuple[str, Any, Any]]:
    """Read loss curves from TensorBoard logs for *session_name*.

    Returns list of (label, train_df, val_df).  train_df / val_df may be None.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return []

    versions = list_tb_versions(session_name)
    if not versions:
        return []

    result = []
    for v_path in versions:
        v_num = os.path.basename(v_path).replace("version_", "v")
        ea = EventAccumulator(v_path, size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            result.append((v_num, None, None))
            continue

        tags = ea.Tags().get("scalars", [])

        def _to_df(tag):
            if tag not in tags:
                return None
            evs = ea.Scalars(tag)
            if not evs:
                return None
            return pd.DataFrame({
                "step":      [e.step      for e in evs],
                "value":     [e.value     for e in evs],
                "wall_time": [e.wall_time for e in evs],
            })

        train_df = next(
            (df for tag in ("train_loss_epoch", "train_loss_step", "train_loss")
             if (df := _to_df(tag)) is not None),
            None,
        )
        val_df = _to_df("val_loss")
        result.append((v_num, train_df, val_df))

    if result:
        label, td, vd = result[-1]
        result[-1] = (f"{label} (текущий)", td, vd)

    return result


def fmt_duration(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return "—"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}ч {m:02d}мин"
    if m > 0:
        return f"{m}мин {s:02d}с"
    return f"{s}с"
