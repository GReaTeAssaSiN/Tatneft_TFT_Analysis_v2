"""
TFT Prediction Script.

Environment variables:
    TFT_SESSION         — export session name (folder in exports/)
    TFT_CHECKPOINT      — full path to .ckpt file
    TFT_OUTPUT_FILE     — full path for output CSV
    TFT_PREDICT_SPLIT   — "test" (default) or "val"

Run from project root:
    python tft/predict.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Patch torch.load for PyTorch 2.6+ (weights_only default changed)
import torch as _torch
_orig_load = _torch.load
def _patched_load(f, *args, **kwargs):
    kwargs["weights_only"] = False  # force: PL passes weights_only=True explicitly
    return _orig_load(f, *args, **kwargs)
_torch.load = _patched_load

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer

from utils.train_utils import (
    get_export_path,
    get_training_dir,
    load_split_config,
    load_tft_config,
    load_train_config,
)


def _inverse_transform(values: np.ndarray, inv_params: dict) -> np.ndarray:
    """Apply inverse of the preprocessing transform used during data preparation."""
    method = inv_params.get("method", "none")
    params = inv_params.get("params", {})
    if method == "log1p":
        return np.expm1(np.maximum(values, 0.0))
    if method == "zscore":
        mean = params.get("mean", 0.0)
        std  = params.get("std",  1.0)
        return values * std + mean if std != 0 else values + mean
    if method == "minmax":
        mn = params.get("min", 0.0)
        mx = params.get("max", 1.0)
        return values * (mx - mn) + mn if mx != mn else values + mn
    if method == "robust":
        median = params.get("median", 0.0)
        iqr    = params.get("iqr",    1.0)
        return values * iqr + median if iqr != 0 else values + median
    return values  # none / fillna / unsupported — return as-is

# ─── Environment variables ────────────────────────────────────────────────────
SESSION   = os.environ.get("TFT_SESSION")
CKPT_PATH = os.environ.get("TFT_CHECKPOINT")
OUT_FILE  = os.environ.get("TFT_OUTPUT_FILE")
SPLIT     = os.environ.get("TFT_PREDICT_SPLIT", "test")

if not SESSION or not CKPT_PATH or not OUT_FILE:
    print("❌ Ошибка: необходимо задать переменные окружения:")
    print("   TFT_SESSION, TFT_CHECKPOINT, TFT_OUTPUT_FILE")
    sys.exit(1)

print(f"📂 Сессия:         {SESSION}")
print(f"🔖 Чекпоинт:      {os.path.basename(CKPT_PATH)}")
print(f"💾 Выходной файл:  {OUT_FILE}")
print(f"📅 Разбивка:       {SPLIT}")
print("-" * 60)

# ─── Load configs ─────────────────────────────────────────────────────────────
tft_cfg   = load_tft_config(SESSION)
train_cfg = load_train_config(SESSION)
split_cfg = load_split_config(SESSION)
if not split_cfg:
    _tr_split = os.path.join(get_training_dir(SESSION), "split_config.json")
    if os.path.exists(_tr_split):
        with open(_tr_split, encoding="utf-8") as f:
            split_cfg = json.load(f)

if not tft_cfg:
    print("❌ tft_config.json не найден в сессии экспорта"); sys.exit(1)
if not split_cfg:
    print("❌ split_config.json не найден"); sys.exit(1)

# Load inverse transform params from data prep pipeline
_inv_pkl = os.path.join(get_export_path(SESSION), "inverse_transforms.pkl")
if os.path.exists(_inv_pkl):
    with open(_inv_pkl, "rb") as _f:
        inverse_transforms: dict = pickle.load(_f)
    print(f"   inverse_transforms.pkl загружен ({len(inverse_transforms)} колонок)")
else:
    inverse_transforms = {}
    print("   ⚠️  inverse_transforms.pkl не найден — будет применён expm1 для всех таргетов")

# ─── TFT structure (must mirror train.py exactly) ─────────────────────────────
TIME_COL      = tft_cfg["time_col"]
GROUP_COL     = tft_cfg["group_col"]
TARGETS       = tft_cfg["target"]
STATIC_CATS   = tft_cfg.get("static_cats", [])
STATIC_REALS  = tft_cfg.get("static_reals", [])
KNOWN_CATS    = tft_cfg.get("time_varying_known_categoricals", [])
KNOWN_REALS   = [c for c in tft_cfg.get("time_varying_known_reals", []) if c != TIME_COL]
UNKNOWN_REALS = [c for c in tft_cfg.get("time_varying_unknown_reals", []) if c not in TARGETS]

ENCODER_LENGTH    = int(train_cfg.get("encoder_length", 30))
PREDICTION_LENGTH = int(train_cfg.get("prediction_length", 7))
BATCH_SIZE        = int(train_cfg["batch_size"])

TRAIN_END = pd.Timestamp(split_cfg["train_end"])
VAL_END   = pd.Timestamp(split_cfg["val_end"])

# ─── Load data ────────────────────────────────────────────────────────────────
export_path = get_export_path(SESSION)
data_path   = os.path.join(export_path, "processed_data.csv")
print(f"⏳ Загрузка данных: {data_path}")
df = pd.read_csv(data_path, encoding="utf-8")
print(f"   Строк: {len(df):,}  Колонок: {len(df.columns)}")

df[TIME_COL]  = df[TIME_COL].astype(int)
df[GROUP_COL] = df[GROUP_COL].astype(str)
for col in STATIC_CATS + KNOWN_CATS:
    if col in df.columns:
        df[col] = df[col].astype(str)

# ─── Build time_idx → date mapping ────────────────────────────────────────────
_tidx_to_date: pd.Series | None = None

if "date" in df.columns:
    _parsed = pd.to_datetime(df["date"], errors="coerce")
    if _parsed.notna().mean() > 0.9:
        df["date"] = _parsed
        _tidx_to_date = df.drop_duplicates(subset=[TIME_COL]).set_index(TIME_COL)["date"]

if _tidx_to_date is None:
    merged_path = os.path.join(export_path, "merged_data.csv")
    if os.path.exists(merged_path):
        print("   date не найдена в processed_data — читаю из merged_data.csv")
        _dmap = pd.read_csv(merged_path, usecols=["date", TIME_COL],
                            parse_dates=["date"], encoding="utf-8")
        _dmap = _dmap.drop_duplicates(subset=[TIME_COL])
        _tidx_to_date = _dmap.set_index(TIME_COL)["date"]

if _tidx_to_date is None:
    print("❌ Не удалось получить маппинг time_idx→date")
    sys.exit(1)

# ─── Compute split boundaries (time_idx) ──────────────────────────────────────
_train_max_idx = int(_tidx_to_date[_tidx_to_date <= TRAIN_END].index.max())
_val_max_idx   = int(_tidx_to_date[_tidx_to_date <= VAL_END].index.max())
_all_max_idx   = int(df[TIME_COL].max())

print(f"   train_end : {TRAIN_END.date()}  (time_idx ≤ {_train_max_idx})")
print(f"   val_end   : {VAL_END.date()}  (time_idx ≤ {_val_max_idx})")

if SPLIT == "test":
    pred_min_idx = _val_max_idx + 1
    pred_max_idx = _all_max_idx
    print(f"\n🎯 Прогноз на тестовый период  (time_idx {pred_min_idx}–{pred_max_idx})")
else:
    pred_min_idx = _train_max_idx + 1
    pred_max_idx = _val_max_idx
    print(f"\n🎯 Прогноз на валидационный период  (time_idx {pred_min_idx}–{pred_max_idx})")

if pred_min_idx > pred_max_idx:
    print("❌ Нет данных для прогноза в выбранном периоде.")
    sys.exit(1)

# ─── Recreate training TimeSeriesDataSet (mirrors train.py) ───────────────────
print("\n⏳ Воссоздание TimeSeriesDataSet...")
train_df = df[df[TIME_COL] <= _train_max_idx].copy()
print(f"   Train строк: {len(train_df):,}")

training = TimeSeriesDataSet(
    train_df,
    time_idx=TIME_COL,
    target=TARGETS,
    group_ids=[GROUP_COL],
    min_encoder_length=ENCODER_LENGTH // 2,
    max_encoder_length=ENCODER_LENGTH,
    min_prediction_length=1,
    max_prediction_length=PREDICTION_LENGTH,
    static_categoricals=STATIC_CATS,
    static_reals=STATIC_REALS,
    time_varying_known_categoricals=KNOWN_CATS,
    time_varying_known_reals=KNOWN_REALS,
    time_varying_unknown_reals=UNKNOWN_REALS,
    target_normalizer=MultiNormalizer([EncoderNormalizer() for _ in TARGETS]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# Prediction data includes encoder context window + target prediction period
context_start = max(0, pred_min_idx - ENCODER_LENGTH)
pred_data = df[df[TIME_COL] >= context_start].copy()

prediction_ds = TimeSeriesDataSet.from_dataset(
    training, pred_data, stop_randomization=True, predict=False,
)
pred_loader = prediction_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=0,
)
print(f"   Образцов для прогноза: {len(prediction_ds)}")

# ─── Load model ───────────────────────────────────────────────────────────────
print(f"\n⏳ Загрузка модели: {os.path.basename(CKPT_PATH)}")
if not os.path.exists(CKPT_PATH):
    print(f"❌ Файл чекпоинта не найден: {CKPT_PATH}")
    sys.exit(1)

model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
model.eval()
print(f"   Параметров: {sum(p.numel() for p in model.parameters()):,}")

# ─── Predict ──────────────────────────────────────────────────────────────────
print("\n⚡ Вычисление предсказаний...")

# mode="quantiles": EncoderNormalizer inverse-transform applied automatically
# result is in log1p space → apply expm1 to recover original scale
result = model.predict(
    pred_loader,
    mode="quantiles",
    return_index=True,
    trainer_kwargs={"logger": False, "enable_progress_bar": True},
)

if hasattr(result, "output"):
    preds  = result.output
    idx_df = result.index
else:
    preds, idx_df = result

n_samples = len(idx_df)
n_q       = preds[0].shape[-1]
print(f"   Образцов: {n_samples}  Горизонт: {PREDICTION_LENGTH}  Квантилей: {n_q}")

# Decode categorical group_ids back to original string values if needed
encoders = training.categorical_encoders  # None in some pf versions
sid_encoder = encoders.get(GROUP_COL) if encoders is not None else None
if sid_encoder is not None:
    idx_df = idx_df.copy()
    idx_df[GROUP_COL] = sid_encoder.inverse_transform(
        pd.Series(idx_df[GROUP_COL].astype(int))
    )
elif GROUP_COL in idx_df.columns and not pd.api.types.is_string_dtype(idx_df[GROUP_COL]):
    # fallback: map integer codes back via training dataset's label encoder
    try:
        le = training.get_parameters()["categorical_encoders"][GROUP_COL]
        idx_df = idx_df.copy()
        idx_df[GROUP_COL] = le.inverse_transform(pd.Series(idx_df[GROUP_COL].astype(int)))
    except Exception:
        pass  # values may already be original strings in newer pf versions

# ─── Build output DataFrame ───────────────────────────────────────────────────
print("\n📊 Сборка таблицы предсказаний...")

pred_np = [p.detach().cpu().numpy() for p in preds]  # list[(n, pred_len, n_q)]

# Expand each sample to PREDICTION_LENGTH rows
idx_exp     = idx_df.loc[idx_df.index.repeat(PREDICTION_LENGTH)].reset_index(drop=True)
horizon_arr = np.tile(np.arange(PREDICTION_LENGTH), n_samples)
sample_arr  = np.repeat(np.arange(n_samples), PREDICTION_LENGTH)

# time_idx in idx_df = first decoder step → add horizon to get each predicted step
idx_exp["_pred_tidx"] = idx_exp[TIME_COL].values + horizon_arr
idx_exp["_horizon"]   = horizon_arr + 1

# Keep only rows within the target prediction window
mask    = (idx_exp["_pred_tidx"] >= pred_min_idx) & (idx_exp["_pred_tidx"] <= pred_max_idx)
idx_exp = idx_exp[mask].reset_index(drop=True)
s_filt  = sample_arr[mask.values]
h_filt  = horizon_arr[mask.values]

if idx_exp.empty:
    print("❌ Нет предсказаний в целевом периоде после фильтрации.")
    sys.exit(1)

# Quantile indices (symmetric layout: 0.02 0.1 0.25 0.5 0.75 0.9 0.98)
Q_MED = n_q // 2
Q_LO  = max(0, n_q // 2 - 2)
Q_HI  = min(n_q - 1, n_q // 2 + 2)

for t_i, col in enumerate(TARGETS):
    arr   = pred_np[t_i]
    inv_p = inverse_transforms.get(col, {"method": "log1p", "params": {}})
    idx_exp[f"_pred_{col}"]     = _inverse_transform(arr[s_filt, h_filt, Q_MED], inv_p)
    if n_q >= 5:
        idx_exp[f"_pred_{col}_q10"] = _inverse_transform(arr[s_filt, h_filt, Q_LO], inv_p)
        idx_exp[f"_pred_{col}_q90"] = _inverse_transform(arr[s_filt, h_filt, Q_HI], inv_p)

# For each (group, time_idx) keep prediction with smallest horizon (most accurate)
pred_val_cols = [f"_pred_{col}" for col in TARGETS]
pred_ci_cols  = (
    [f"_pred_{col}_q10" for col in TARGETS] + [f"_pred_{col}_q90" for col in TARGETS]
    if n_q >= 5 else []
)
best_preds = (
    idx_exp[[GROUP_COL, "_pred_tidx", "_horizon"] + pred_val_cols + pred_ci_cols]
    .sort_values([GROUP_COL, "_pred_tidx", "_horizon"])
    .groupby([GROUP_COL, "_pred_tidx"], as_index=False)
    .first()
    .drop(columns=["_horizon"])
    .rename(columns={"_pred_tidx": TIME_COL})
)

# Base: original processed_data rows for the prediction period
base_df = df[(df[TIME_COL] >= pred_min_idx) & (df[TIME_COL] <= pred_max_idx)].copy()
base_df = base_df.merge(best_preds, on=[GROUP_COL, TIME_COL], how="left")

# Replace target columns with inverse-transformed predictions
for col in TARGETS:
    if f"_pred_{col}" in base_df.columns:
        base_df[col] = base_df[f"_pred_{col}"]
        base_df = base_df.drop(columns=[f"_pred_{col}"])

# Rename CI columns to clean names and keep them
for col in TARGETS:
    for suffix in ("_q10", "_q90"):
        tmp = f"_pred_{col}{suffix}"
        if tmp in base_df.columns:
            base_df = base_df.rename(columns={tmp: f"{col}{suffix}"})

# Add/update date column from mapping
_dates = base_df[TIME_COL].map(_tidx_to_date)
if "date" in base_df.columns:
    base_df["date"] = _dates
else:
    base_df.insert(0, "date", _dates)

out_df = base_df

n_na = out_df["date"].isna().sum()
if n_na:
    print(f"   ⚠️  {n_na} строк без даты — удалены")
    out_df = out_df.dropna(subset=["date"])

print(f"   Строк: {len(out_df):,}")
print(f"   Групп ({GROUP_COL}): {out_df[GROUP_COL].nunique()}")
print(f"   Период: {out_df['date'].min()} → {out_df['date'].max()}")

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
out_df.to_csv(OUT_FILE, index=False, encoding="utf-8")

print(f"\n✅ Прогноз сохранён: {OUT_FILE}")
print(f"   Строк: {len(out_df):,}  Колонок: {len(out_df.columns)}")
print("=" * 60)
print("Готово.")
print("=" * 60)
