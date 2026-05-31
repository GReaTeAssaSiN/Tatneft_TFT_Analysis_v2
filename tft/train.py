"""
TFT Training Script.
Run from project root: python tft/train.py

Reads TFT_SESSION env var to locate the export session.
Hyperparameters are loaded from training/{session}/train_config.json.
"""

import glob
import json
import os
import sys

os.environ["LIGHTNING_DISABLE_REMOTE_TIPS"] = "1"

import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Patch torch.load for PyTorch 2.6+ (weights_only default changed)
import torch as _torch
_orig_torch_load = _torch.load
def _patched_load(f, *args, **kwargs):
    kwargs["weights_only"] = False  # force: PL passes weights_only=True explicitly
    return _orig_torch_load(f, *args, **kwargs)
_torch.load = _patched_load

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from utils.train_utils import (
    get_export_path,
    get_training_dir,
    load_split_config,
    load_tft_config,
    load_train_config,
)

# ─── Session ──────────────────────────────────────────────────────────────────
SESSION = os.environ.get("TFT_SESSION")
if not SESSION:
    print("❌ TFT_SESSION не задан. Запустите через дашборд или задайте вручную:")
    print("   CMD:        set TFT_SESSION=<имя_папки> && python tft/train.py")
    print("   PowerShell: $env:TFT_SESSION = \"<имя_папки>\"; python tft/train.py")
    sys.exit(1)

print("=" * 60)
print("ПАРАМЕТРЫ ОБУЧЕНИЯ TFT")
print("=" * 60)
print(f"  Сессия             : {SESSION}")

# ─── Configs ──────────────────────────────────────────────────────────────────
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

# ─── TFT structure ────────────────────────────────────────────────────────────
TIME_COL      = tft_cfg["time_col"]           # "time_idx"
GROUP_COL     = tft_cfg["group_col"]          # "station_id"
TARGETS       = tft_cfg["target"]
STATIC_CATS   = tft_cfg.get("static_cats", [])
STATIC_REALS  = tft_cfg.get("static_reals", [])
KNOWN_CATS    = tft_cfg.get("time_varying_known_categoricals", [])
# Exclude the time index itself from known reals — it's the dataset index, not a feature
KNOWN_REALS   = [c for c in tft_cfg.get("time_varying_known_reals", []) if c != TIME_COL]
# Unknown reals = explicitly unknown columns minus targets (targets handled separately)
UNKNOWN_REALS = [c for c in tft_cfg.get("time_varying_unknown_reals", []) if c not in TARGETS]

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE        = int(train_cfg["batch_size"])
EPOCHS            = int(train_cfg["max_epochs"])
HIDDEN_SIZE       = int(train_cfg["hidden_size"])
ATTN_HEADS        = int(train_cfg["attention_head_size"])
HIDDEN_CONTINUOUS = int(train_cfg["hidden_continuous_size"])
LEARNING_RATE     = float(train_cfg["learning_rate"])
DROPOUT           = float(train_cfg["dropout"])
GRADIENT_CLIP     = float(train_cfg["gradient_clip"])
PATIENCE          = int(train_cfg["patience"])
ENCODER_LENGTH    = int(train_cfg.get("encoder_length", 30))
PREDICTION_LENGTH = int(train_cfg.get("prediction_length", 7))

# ─── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    ACCELERATOR = "gpu"
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    ACCELERATOR = "cpu"
    DEVICE_NAME = "CPU"

print(f"  Устройство         : {ACCELERATOR.upper()} ({DEVICE_NAME})")
print(f"  Эпох               : {EPOCHS}")
print(f"  Batch size         : {BATCH_SIZE}")
print(f"  Learning rate      : {LEARNING_RATE}")
print(f"  Hidden size        : {HIDDEN_SIZE}")
print(f"  Attention heads    : {ATTN_HEADS}")
print(f"  Hidden continuous  : {HIDDEN_CONTINUOUS}")
print(f"  Dropout            : {DROPOUT}")
print(f"  Gradient clip      : {GRADIENT_CLIP}")
print(f"  Encoder length     : {ENCODER_LENGTH}")
print(f"  Prediction length  : {PREDICTION_LENGTH}")
print("=" * 60)

# ─── Load data ────────────────────────────────────────────────────────────────
print("\nЗагрузка данных...")
data_path = os.path.join(get_export_path(SESSION), "processed_data.csv")
df = pd.read_csv(data_path, encoding="utf-8")
print(f"  Строк: {len(df):,}  Колонок: {len(df.columns)}")

# Ensure correct dtypes
df[TIME_COL]  = df[TIME_COL].astype(int)
df[GROUP_COL] = df[GROUP_COL].astype(str)
for col in STATIC_CATS + KNOWN_CATS:
    if col in df.columns:
        df[col] = df[col].astype(str)

# ─── Temporal split ───────────────────────────────────────────────────────────
TRAIN_END = split_cfg["train_end"]
VAL_END   = split_cfg["val_end"]
print(f"  train_end: {TRAIN_END}")
print(f"  val_end:   {VAL_END}")

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    _train_max_idx = int(df.loc[df["date"] <= pd.Timestamp(TRAIN_END), TIME_COL].max())
    _val_max_idx   = int(df.loc[df["date"] <= pd.Timestamp(VAL_END),   TIME_COL].max())
else:
    # date column was dropped — fall back to merged_data.csv for the mapping
    merged_path = os.path.join(get_export_path(SESSION), "merged_data.csv")
    if os.path.exists(merged_path):
        print("  ⚠️  date не найдена в processed_data — читаю из merged_data.csv")
        _dmap = pd.read_csv(merged_path, usecols=["date", TIME_COL],
                            parse_dates=["date"], encoding="utf-8")
        _dmap = _dmap.drop_duplicates(subset=[TIME_COL])
        _train_max_idx = int(_dmap.loc[_dmap["date"] <= pd.Timestamp(TRAIN_END), TIME_COL].max())
        _val_max_idx   = int(_dmap.loc[_dmap["date"] <= pd.Timestamp(VAL_END),   TIME_COL].max())
    else:
        _all_idx = sorted(df[TIME_COL].unique())
        n = len(_all_idx)
        _train_max_idx = _all_idx[int(n * 0.70)]
        _val_max_idx   = _all_idx[int(n * 0.85)]
        print("  ⚠️  Нет колонки date — пропорциональный сплит (70/85/100%)")

train_df = df[df[TIME_COL] <= _train_max_idx].copy()
val_df   = df[df[TIME_COL] <= _val_max_idx].copy()
print(f"  Train строк: {len(train_df):,}  (time_idx ≤ {_train_max_idx})")
print(f"  Val строк:   {len(val_df):,}  (time_idx ≤ {_val_max_idx})")

# ─── TimeSeriesDataSet ────────────────────────────────────────────────────────
print("\nСоздание TimeSeriesDataSet...")

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
    # Data is already preprocessed (log1p / zscore). EncoderNormalizer does a
    # lightweight per-sequence normalisation on top, which helps stability.
    target_normalizer=MultiNormalizer([EncoderNormalizer() for _ in TARGETS]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training, val_df, stop_randomization=True, predict=False
)

train_loader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
)
val_loader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=0
)

print(f"  Train батчей: {len(train_loader)}")
print(f"  Val батчей:   {len(val_loader)}")

# ─── Model ────────────────────────────────────────────────────────────────────
print("\nСоздание модели TFT...")

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LEARNING_RATE,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTN_HEADS,
    dropout=DROPOUT,
    hidden_continuous_size=HIDDEN_CONTINUOUS,
    loss=QuantileLoss(),
    log_interval=20,
    reduce_on_plateau_patience=5,
)

total_params = sum(p.numel() for p in tft.parameters())
print(f"  Параметров: {total_params:,}")

# ─── Paths ────────────────────────────────────────────────────────────────────
TRAINING_DIR = get_training_dir(SESSION)
CKPT_DIR     = os.path.join(TRAINING_DIR, "checkpoints")
LOG_DIR      = os.path.join(TRAINING_DIR, "logs")
MODEL_PATH   = os.path.join(TRAINING_DIR, "model.ckpt")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ─── Callbacks ────────────────────────────────────────────────────────────────
checkpoint_cb = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename="tft-epoch={epoch:02d}-val_loss={val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

import shutil

class BestModelSync(pl.Callback):
    """Copies best checkpoint to model.ckpt after each validation end.

    Uses on_validation_end (fires after ModelCheckpoint.on_validation_epoch_end)
    to guarantee the checkpoint file is already on disk.
    """
    def __init__(self):
        self._last_best: float | None = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        src = checkpoint_cb.best_model_path
        if not src or not os.path.exists(src):
            candidates = sorted(glob.glob(os.path.join(CKPT_DIR, "*.ckpt")))
            src = candidates[-1] if candidates else None
        if not src or not os.path.exists(src):
            return

        score = checkpoint_cb.best_model_score
        score_f = float(score) if score is not None else None

        if not os.path.exists(MODEL_PATH):
            shutil.copy(src, MODEL_PATH)
            if score_f is not None:
                self._last_best = score_f
            print(
                f"\n✅ BEST  val_loss={score_f:.4f}  →  model.ckpt"
                if score_f is not None else f"\n✅ Сохранено  →  model.ckpt",
                flush=True,
            )
        elif score_f is not None and (self._last_best is None or score_f < self._last_best):
            shutil.copy(src, MODEL_PATH)
            self._last_best = score_f
            print(
                f"\n✅ BEST  val_loss={score_f:.4f}  →  model.ckpt",
                flush=True,
            )

early_stop_cb = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=True)
lr_monitor    = LearningRateMonitor(logging_interval="epoch")
tb_logger     = TensorBoardLogger(LOG_DIR, name="tft_model")

# ─── Resume from checkpoint if available ──────────────────────────────────────
ckpt_files  = sorted(glob.glob(os.path.join(CKPT_DIR, "*.ckpt")))
resume_ckpt = ckpt_files[-1] if ckpt_files else None
if resume_ckpt:
    print(f"\nПродолжение с чекпоинта: {os.path.basename(resume_ckpt)}")
else:
    print("\nЧекпоинт не найден — обучение с нуля.")

# ─── Training ─────────────────────────────────────────────────────────────────
print(f"\nЗапуск обучения ({EPOCHS} эпох, {ACCELERATOR.upper()})...")
print("-" * 60)
print(f"  Чекпоинты : {CKPT_DIR}")
print(f"  Модель    : {MODEL_PATH}")
print(f"  Логи      : {LOG_DIR}")
print("-" * 60)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator=ACCELERATOR,
    devices=1,
    gradient_clip_val=GRADIENT_CLIP,
    callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, BestModelSync()],
    enable_model_summary=True,
    log_every_n_steps=50,
    num_sanity_val_steps=1,
    precision="32-true",
    logger=tb_logger,
)

trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    ckpt_path=resume_ckpt,
)

# ─── Finalize ─────────────────────────────────────────────────────────────────
best_path = checkpoint_cb.best_model_path
if best_path:
    print(f"\nЛучший чекпоинт : {os.path.basename(best_path)}")
if checkpoint_cb.best_model_score is not None:
    print(f"Лучший val_loss : {checkpoint_cb.best_model_score:.4f}")
print(f"Модель          : {MODEL_PATH} {'✓' if os.path.exists(MODEL_PATH) else '✗'}")

print("\n" + "=" * 60)
print("Готово.")
print(f"  Модель : {MODEL_PATH}")
print(f"  Логи   : tensorboard --logdir {LOG_DIR}")
print("=" * 60)
