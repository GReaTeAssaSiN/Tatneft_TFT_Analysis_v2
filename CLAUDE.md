# Tatneft TFT Analysis — контекст для Claude Code

## Что это

Три Streamlit-дашборда + два скрипта для прогнозирования продаж топлива и магазина на АЗС Татнефть с помощью **Temporal Fusion Transformer** (pytorch-forecasting). Учебный проект (магистратура КАИ, ТАБД).

Виртуальное окружение: `venv/` (Python 3.14, Windows).

---

## Запуск

```bash
# из корня проекта, с активированным venv/
streamlit run dashboard/data_prep_dashboard.py   # шаг 1: предобработка
streamlit run dashboard/train_dashboard.py       # шаг 2: обучение и прогнозы
streamlit run dashboard/analytics_dashboard.py   # EDA-аналитика (независимо)

# обучение/прогноз можно запустить напрямую:
set TFT_SESSION=ActualData_2026-05-30_15-55-39
python tft/train.py
python tft/predict.py
```

---

## Структура проекта

```
dashboard/
  data_prep_dashboard.py   — UI предобработки (5 вкладок)
  train_dashboard.py       — UI обучения и прогнозирования (4 вкладки)
  analytics_dashboard.py   — EDA-аналитика (4 вкладки)
utils/
  prep_utils.py            — логика предобработки (без st.*)
  train_utils.py           — хелперы обучения, валидация сессии (без st.*)
  analytics_utils.py       — хелперы аналитики (без st.*)
tft/
  train.py                 — скрипт обучения, читает TFT_SESSION env var
  predict.py               — скрипт прогнозирования, читает TFT_SESSION + TFT_CHECKPOINT
exports/                   — экспортированные сессии предобработки
training/                  — артефакты обучения: checkpoints/, logs/, predictions/
column_groups.json         — группировки колонок для аналитики (опциональный)
```

**Главное правило**: `utils/*.py` не импортирует `streamlit`. `dashboard/*.py` не содержит тяжёлых вычислений — только UI и session_state.

---

## Поток данных

```
SourceData/*.csv
    ↓  data_prep_dashboard (merge → preprocessing → split → TFT roles)
exports/<session>/
    ↓  train_dashboard → tft/train.py  (subprocess, TFT_SESSION env var)
training/<session>/model.ckpt
    ↓  train_dashboard → tft/predict.py  (subprocess, TFT_SESSION + TFT_CHECKPOINT)
training/<session>/predictions/*.csv
    ↓  analytics_dashboard (читает напрямую из exports/ + training/)
```

---

## Сессия предобработки (`exports/<name>/`)

| Файл | Статус для обучения | Назначение |
|------|---------------------|-----------|
| `processed_data.csv` | обязательный | данные после препроцессинга |
| `tft_config.json` | обязательный | роли колонок TFT (`time_col`, `group_col`, `target`, `static_cats`, `static_reals`, `time_varying_known_*`, `time_varying_unknown_reals`, `dropped`) |
| `prep_config.json` | обязательный при загрузке в prep-дашборд | `{col: {method, params, value_labels, ...}}` — строго dict of dicts |
| `session_config.json` | обязательный при загрузке в prep-дашборд | `merge_configs`, `col_types`, `tidx_config`, `source_files_order`, `preprocessing_applied` |
| `inverse_transforms.pkl` | обязательный для прогнозов | `{col: {method, params}}` — параметры обратного преобразования таргетов |
| `split_config.json` | опциональный | `{train_end, val_end, date_col, mode, train_pct, val_pct}` |
| `merged_data.csv` | опциональный | данные до препроцессинга; нужен аналитике и маппингу `time_idx→date` |
| `source_files/` | опциональный | исходные CSV |

Валидация содержимого (не только наличие файлов): `utils/train_utils.py::check_session_files()` — возвращает `content_issues` (блокеры) и `content_warnings`.

Тестовые сессии: `exports/TestSession_VALID_2026-06-01` (всё ок), `exports/TestSession_BROKEN_2026-06-01` (3 блокера + 2 предупреждения).

---

## dashboard/data_prep_dashboard.py

Пять вкладок:

| # | Вкладка | Что делает |
|---|---------|-----------|
| 1 | 📂 Файлы | Загрузка CSV или восстановление сессии из `exports/` |
| 2 | 🔗 Объединение | Цепочка JOIN нескольких CSV |
| 3 | ⚙️ Препроцессинг | Методы нормализации/кодирования по колонкам, формульные колонки, datetime-экстракции |
| 4 | ✂️ Сплит | Временной train/val/test сплит по дате |
| 5 | 🏷️ TFT + Экспорт | Назначение ролей TFT, экспорт сессии, отчёт в Markdown, настройка `column_groups.json` |

### Ключевые session_state переменные

```
merged_df          — датафрейм после JOIN (до препроцессинга)
prep_df            — датафрейм после препроцессинга
col_types          — {col: "numerical"|"categorical"|"binary"|"datetime"|"text"}
prep_config        — {col: {method, params, skip, pre_fillna, ...}}
tft_roles          — {time_col, group_col, target, static_cat, static_real,
                      known_cat, known_real, unknown_real, dropped}
split_config       — {date_col, mode, train_end, val_end, ...}
split_masks        — {train: list[bool], val: list[bool], test: list[bool], stats} | None
inverse_params     — {col: {method, fit_on, params, value_labels}}
applied_prep_config  — снимок prep_config на момент последнего "Применить"
applied_split_config — снимок split_config на момент последнего "Применить"
load_mode          — "csv_upload" | "saved_files"
uploaded_files     — {filename: pd.DataFrame}
```

### Неочевидные паттерны

**Версионные ключи виджетов** — Streamlit кэширует значение по ключу. Чтобы сбросить виджет в дефолт, ключ инкрементируется (НИКОГДА не сбрасывается в 0, иначе Streamlit вспомнит старое значение):
- `tft_reset_v` — все multiselect на вкладке TFT
- `split_col_sel_v` — selectbox «Колонка с датой» на вкладке Сплит

**`_pending_*` ключи** — при загрузке сессии виджеты ещё не отрендерены, поэтому нельзя писать напрямую в widget-ключи. Используются временные ключи (`_pending_tidx_src_col`, `_pending_tidx_gran`, `_pending_join_col_N` и т.д.), которые читаются непосредственно перед созданием виджета.

**`reset_downstream()`** — сбрасывает merged_df, prep, split, TFT-роли + инкрементирует оба счётчика. Вызывать при смене набора файлов.

**`reset_tft_roles()`** — только TFT-роли и `tft_reset_v`. Вызывать при смене данных без полного сброса.

**`_uploader_was_populated`** — `True` только если файлы пришли через виджет (не из восстановленной сессии). Нужен чтобы событие «пользователь убрал файл через ×» не перепутать с «файлов нет, потому что сессия restored».

---

## dashboard/train_dashboard.py

Четыре вкладки: ⚙️ Конфигурация → ▶ Обучение → 📊 Результаты → 🔮 Прогнозы.

### Поток работы пользователя

1. Выбирает экспортную сессию из `exports/` — показывается экран проверки файлов (`check_session_files`).
2. Нажимает «Приступить» → `_session_loaded = True`, переход к вкладкам.
3. На вкладке Конфигурация задаёт гиперпараметры → сохраняет в `training/<session>/train_config.json` → параметры **блокируются** (`params_locked = True`).
4. На вкладке Обучение запускает `tft/train.py` как subprocess, вывод стримится в реальном времени через `queue.Queue` + фоновый поток.
5. На вкладке Результаты смотрит loss-кривые (читаются из TensorBoard-логов через `read_tb_losses`), метрики.
6. На вкладке Прогнозы выбирает чекпоинт, запускает `tft/predict.py` как subprocess, скачивает CSV результатов.

### Ключевые session_state переменные

```
selected_session   — имя папки в exports/
train_proc         — subprocess.Popen | None
train_output       — список строк вывода
output_queue       — queue.Queue для стриминга stdout
reader_thread      — threading.Thread (фоновый reader)
_session_loaded    — False = экран выбора, True = вкладки
_params_locked     — None=auto (по наличию train_config.json), True, False
_locked_for        — имя сессии, для которой применён lock
_training_done     — True когда последнее обучение завершилось с exit_code 0
pred_proc          — subprocess.Popen прогнозирования | None
_last_pred_file    — путь к последнему успешному prediction CSV
```

### Обучение как subprocess

`tft/train.py` запускается через `subprocess.Popen` с `TFT_SESSION` в env. Stdout читается фоновым threading.Thread в `queue.Queue`; дашборд вычитывает очередь при каждом rerun и дописывает строки в `train_output`. Остановка через `proc.terminate()`.

`training/<session>/training_complete.flag` — файл-маркер успешного завершения, создаётся скриптом в конце, проверяется дашбордом.

---

## dashboard/analytics_dashboard.py

Четыре вкладки: 📊 EDA-анализ → 📈 Статистика → 🔮 Прогнозы → 💡 Рекомендации.

- Данные читаются из `exports/<session>/` через `utils/analytics_utils.load_session()`, кэшируются `@st.cache_data`.
- `column_groups.json` (корень проекта) — опциональный конфиг с группировками: `fuel_types`, `shop_categories`, `special_cols`, `value_remaps`. Если файл есть — берётся оттуда, иначе из автодетекта в сессии.
- Глобальные фильтры (станции, диапазон дат, дорога, направление) — ключи виджетов содержат суффикс имени сессии, поэтому при смене датасета сбрасываются к дефолту.
- Прогнозы читаются из `training/<session>/predictions/*.csv` если есть.

---

## tft/train.py

Читает конфиги из `exports/<TFT_SESSION>/` и `training/<TFT_SESSION>/train_config.json`.  
Строит `TimeSeriesDataSet` → `TemporalFusionTransformer` → обучает с `EarlyStopping` + `ModelCheckpoint` + `TensorBoardLogger`.  
Сохраняет лучшую модель в `training/<session>/model.ckpt`.  
В конце создаёт `training/<session>/training_complete.flag`.

Патч `torch.load` нужен для PyTorch 2.6+ (изменился дефолт `weights_only`).

## tft/predict.py

env vars: `TFT_SESSION`, `TFT_CHECKPOINT` (путь к .ckpt), `TFT_OUTPUT_FILE` (путь к выходному CSV), `TFT_PREDICT_SPLIT` ("test" или "val", дефолт "test").

Загружает модель, строит датасет, применяет обратное преобразование из `inverse_transforms.pkl`, сохраняет CSV с колонками: group, time_idx, date (если есть маппинг из merged_data), target_actual, target_pred (квантили q10/q50/q90).

---

## utils/train_utils.py — публичный API

| Функция | Назначение |
|---------|-----------|
| `list_export_sessions()` | сессии в exports/ у которых есть processed_data + tft_config |
| `check_session_files(session)` | наличие файлов + валидация содержимого; возвращает `content_issues`, `content_warnings`, `content_ok` |
| `load_tft_config / load_split_config / load_train_config / load_session_config` | загрузка конфигов |
| `save_train_config / delete_train_config` | сохранение/удаление train_config.json |
| `save_split_config_to_training` | сохраняет split в папку training/ (если split не был в exports/) |
| `get_data_info / get_group_count` | информация о датасете без полной загрузки |
| `list_ckpt_files / list_tb_versions / read_tb_losses` | артефакты обучения |
| `fmt_duration` | форматирование секунд → "2ч 03м 15с" |
| `CPU_DEFAULTS / GPU_DEFAULTS` | дефолтные гиперпараметры |

## utils/prep_utils.py — публичный API

`apply_preprocessing`, `merge_dataframes`, `detect_col_types`, `detect_col_type`, `compute_split_masks`, `build_tft_config`, `build_session_config`, `build_report`, `eval_formula`, `get_column_summary`, `apply_dt_extractions`, `validate_report_name`

## utils/analytics_utils.py — публичный API

`list_sessions`, `load_session`, `available_fuel_types`, `available_shop_categories`, `decode_col_series`, `denormalize`, `label`, `filter_merged_duplicates`

---

## Артефакты обучения (`training/<session>/`)

```
train_config.json         — гиперпараметры (batch_size, epochs, hidden_size, ...)
split_config.json         — копия split если введён вручную в дашборде
checkpoints/              — PyTorch Lightning .ckpt файлы
logs/                     — TensorBoard логи
model.ckpt                — лучшая модель (symlink или копия из checkpoints/)
predictions/              — CSV результатов прогноза
training_complete.flag    — маркер успешного завершения
```

---

## Соглашения по коду

- Комментарии только там, где неочевидно **почему** (не что).
- Без лишних абстракций: три похожих блока лучше чем преждевременная функция.
- Новые вычисления — в `utils/`, новый UI — в `dashboard/`. Не смешивать.
- Кодировка: UTF-8 везде (`encoding="utf-8"` явно).

**Цветовая схема UI** (одинакова во всех трёх дашбордах):
```python
GOLD    = "#c8a84b"
GREEN   = "#4ECB71"
RED     = "#E24B4A"
BLUE    = "#2E75B6"
TEAL    = "#1ABC9C"
GRAY    = "#8B949E"
CARD_BG = "#13161f"
GRID    = "#1e2235"
```
