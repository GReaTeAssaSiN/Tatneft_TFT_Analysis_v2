"""
Universal data-preprocessing dashboard.
Run: streamlit run dashboard/data_prep_dashboard.py  (from project root)
"""

import copy
import io
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


class _NpEncoder(json.JSONEncoder):
    """Convert numpy scalar types so json.dumps doesn't raise TypeError."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Ensure the project root (parent of dashboard/) is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.prep_utils import (  # noqa: E402
    apply_dt_extractions,
    apply_preprocessing,
    build_report,
    build_session_config,
    build_tft_config,
    compute_split_masks,
    detect_col_type,
    detect_col_types,
    eval_formula,
    get_column_summary,
    merge_dataframes,
    validate_report_name,
)
from utils.analytics_utils import filter_merged_duplicates  # noqa: E402

# ─── Color scheme ──────────────────────────────────────────────────────────────
GOLD    = "#c8a84b"
GREEN   = "#4ECB71"
RED     = "#E24B4A"
BLUE    = "#2E75B6"
TEAL    = "#1ABC9C"
GRAY    = "#8B949E"
CARD_BG = "#13161f"
GRID    = "#1e2235"

# ─── Preprocessing constants ───────────────────────────────────────────────────
DEFAULT_METHOD = {
    "numerical":   "zscore",
    "categorical": "label_enc",
    "binary":      "none",
    "datetime":    "none",
    "text":        "label_enc",
}

METHOD_OPTIONS = {
    "numerical":   ["none", "zscore", "minmax", "robust", "log1p", "cyclical", "drop"],
    "categorical": ["none", "label_enc", "onehot", "drop"],
    "binary":      ["none", "drop"],
    "datetime":    ["none", "drop"],
    "text":        ["none", "label_enc", "drop"],
}

METHOD_LABELS = {
    "none":      "Без изменений",
    "zscore":    "Z-score нормализация",
    "minmax":    "Min-Max нормализация [0, 1]",
    "robust":    "Robust (IQR) нормализация",
    "log1p":     "log1p преобразование",
    "fillna":    "Заполнить пропуски",
    "cyclical":  "Циклическое sin/cos кодирование",
    "label_enc": "Label Encoding → _enc",
    "onehot":    "One-Hot Encoding",
    "drop":      "Удалить колонку",
}

TYPE_COLORS = {
    "numerical":   BLUE,
    "categorical": "#9B59B6",
    "binary":      GREEN,
    "datetime":    GOLD,
    "text":        GRAY,
}

# ─── Session-state schema ──────────────────────────────────────────────────────
DEFAULTS: dict = {
    "upload_key": 0,
    "uploaded_files": {},
    "added_cols": [],
    "added_col_formulas": {},         # {col_name: formula | "__time_idx__" | "__dtx_{src}__"}
    "tidx_config": {},                # {src_col, gran, name} when time_idx was created
    "dtx_configs": [],                # [{src_col, components}] — applied date extractions
    "tidx_open": False,
    "dtx_open": False,
    "drop_open": False,
    "agg_open": False,
    "type_override_v": 0,
    "merged_df": None,
    "col_types": {},
    "prep_config": {},
    "prep_df": None,
    "load_mode": "csv_upload",        # "csv_upload" | "saved_files"
    "load_choice_used": "full",       # always "full" with folder-based restore
    "saved_files_upload_key": 0,
    "_uploader_was_populated": False, # True only when files came from Tab-1 widget
    "file_sep":  ",",                 # column separator widget key
    "file_enc":  "auto",              # encoding widget key
    "file_dec":  ".",                 # decimal separator widget key
    "_detected_encs": {},             # per-file encoding {fname: enc_str}
    "tidx_name":  "time_idx",         # text_input key for time_idx column name
    "tft_reset_v": 0,                 # incremented to force-recreate all TFT widgets
    "_t5_load_exp_open": None,        # None = use default; True/False = explicit override
    "split_config": {},               # temporal split parameters
    "split_masks": None,              # {train/val/test: list[bool], stats: dict} or None
    "split_col_sel_v": 0,             # incremented to force-recreate the date-column selectbox
    "inverse_params": {},             # {col: {method, fit_on, params, value_labels}}
    "applied_prep_config": None,      # snapshot of prep_config at last "Apply preprocessing"
    "applied_split_config": None,     # snapshot of split_config at last "Apply preprocessing"
    "tft_roles": {
        "time_col": None,
        "group_col": None,
        "target": [],
        "static_cat": [],
        "static_real": [],
        "known_cat": [],
        "known_real": [],
        "unknown_real": [],
        "dropped": [],
    },
}


def init_state() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = copy.deepcopy(v)


def reset_downstream() -> None:
    """Reset everything except uploaded_files when the file set changes."""
    for k in ("merged_df", "col_types", "prep_config", "prep_df", "tft_roles",
              "added_cols", "added_col_formulas", "tidx_config", "dtx_configs",
              "tidx_open", "dtx_open", "drop_open", "agg_open", "load_mode", "load_choice_used",
              "_uploader_was_populated", "_detected_encs",
              "split_config", "split_masks", "inverse_params",
              "applied_prep_config", "applied_split_config"):
        st.session_state[k] = copy.deepcopy(DEFAULTS[k])
    # Increment version to force-recreate the date-column selectbox (never reset to 0)
    st.session_state["split_col_sel_v"] = st.session_state.get("split_col_sel_v", 0) + 1
    _clear_vl_data()
    reset_tft_roles()  # also pops widget keys so multiselects reinitialise


def _clear_vl_data(cols: "list | None" = None) -> None:
    """Delete value-label text_input keys from session_state.

    cols=None clears all; cols=[...] clears only the specified columns.
    """
    if cols is None:
        for _k in [k for k in st.session_state if k.startswith("vl_inp_")]:
            del st.session_state[_k]
    else:
        for _c in cols:
            for _k in [k for k in st.session_state if k.startswith(f"vl_inp_{_c}_")]:
                del st.session_state[_k]


def _trim_tft_roles(valid_cols: set) -> None:
    """Remove columns that no longer exist from TFT roles; force widget recreation.

    Unlike reset_tft_roles(), preserves assignments for surviving columns.
    """
    roles = st.session_state["tft_roles"]
    for key in ("target", "static_cat", "static_real", "known_cat",
                "known_real", "unknown_real", "dropped"):
        roles[key] = [c for c in roles.get(key, []) if c in valid_cols]
    if roles.get("time_col") and roles["time_col"] not in valid_cols:
        roles["time_col"] = None
    if roles.get("group_col") and roles["group_col"] not in valid_cols:
        roles["group_col"] = None
    st.session_state["tft_reset_v"] = st.session_state.get("tft_reset_v", 0) + 1


def reset_tft_roles() -> None:
    """Clear TFT role assignments by bumping tft_reset_v.

    Incrementing the version forces Streamlit to create all TFT widgets under
    new keys, bypassing any cached _widget_state from the previous render.
    This is more reliable than popping keys, which only clears user-facing
    session state while the internal widget cache may still hold old values.
    """
    st.session_state["tft_roles"] = copy.deepcopy(DEFAULTS["tft_roles"])
    st.session_state["tft_reset_v"] = st.session_state.get("tft_reset_v", 0) + 1


# ─── CSS & badges ──────────────────────────────────────────────────────────────
CSS = f"""
<style>
    .badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 4px;
    }}
    .badge-num  {{ background:{BLUE}33;  color:{BLUE};  border:1px solid {BLUE}66; }}
    .badge-cat  {{ background:#9B59B633; color:#9B59B6; border:1px solid #9B59B666; }}
    .badge-bin  {{ background:{GREEN}33; color:{GREEN}; border:1px solid {GREEN}66; }}
    .badge-dt   {{ background:{GOLD}33;  color:{GOLD};  border:1px solid {GOLD}66; }}
    .badge-text {{ background:{GRAY}33;  color:{GRAY};  border:1px solid {GRAY}66; }}
    .section-title {{ color:{TEAL}; font-size:18px; font-weight:700; margin-bottom:8px; }}
    .gold-title    {{ color:{GOLD}; font-size:24px; font-weight:700; }}
</style>
"""

TYPE_BADGE = {
    "numerical":   f'<span class="badge badge-num">числовая</span>',
    "categorical": f'<span class="badge badge-cat">категориальная</span>',
    "binary":      f'<span class="badge badge-bin">бинарная</span>',
    "datetime":    f'<span class="badge badge-dt">дата/время</span>',
    "text":        f'<span class="badge badge-text">текст</span>',
}

# ─── Plotly helpers ────────────────────────────────────────────────────────────

def _base_layout(**kw) -> dict:
    return dict(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=GRAY),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        margin=dict(l=40, r=20, t=40, b=40),
        **kw,
    )


def plot_histogram(series: pd.Series, title: str = "", color: str = BLUE) -> go.Figure:
    data = series.dropna().head(500)
    fig = go.Figure(go.Histogram(x=data, marker_color=color, opacity=0.8))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(color=GOLD)), height=280,
    ))
    return fig


def plot_bar(series: pd.Series, title: str = "", color: str = BLUE, top_n: int = 20) -> go.Figure:
    vc = series.value_counts().head(top_n)
    fig = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values, marker_color=color))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(color=GOLD)), height=280,
    ))
    return fig


def plot_before_after(before: pd.Series, after: pd.Series, col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=before.dropna().head(200), name="До", opacity=0.6, marker_color=BLUE,
    ))
    fig.add_trace(go.Histogram(
        x=after.dropna().head(200), name="После", opacity=0.6, marker_color=GREEN,
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=f"{col}: до / после", font=dict(color=GOLD)),
        barmode="overlay", height=260,
    ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Data Prep Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)
init_state()

st.markdown('<p class="gold-title">📊 Универсальный дашборд предобработки данных</p>', unsafe_allow_html=True)
st.markdown(
    f'<span style="color:{GRAY}">Загрузите CSV-файлы, настройте препроцессинг '
    f'и экспортируйте датасет + конфиг для TFT.</span>',
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Файлы",
    "🔗 Объединение",
    "📊 Анализ колонок",
    "⚙️ Препроцессинг",
    "🏷️ TFT + Экспорт",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Files
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">📂 Загрузка файлов</p>', unsafe_allow_html=True)

    # Apply pending file-settings from session restore (must run before widget instantiation)
    for _pk, _wk in (("_pending_file_sep", "file_sep"),
                     ("_pending_file_enc", "file_enc"),
                     ("_pending_file_dec", "file_dec")):
        _pv = st.session_state.pop(_pk, None)
        if _pv is not None:
            st.session_state[_wk] = _pv

    # Computed before widgets so we can pass disabled= to selectboxes
    _in_saved_with_files = (
        st.session_state.get("load_mode") == "saved_files"
        and bool(st.session_state.get("uploaded_files"))
    )
    _files_loaded = bool(st.session_state.get("uploaded_files"))
    _lock_settings = _in_saved_with_files or _files_loaded
    _lock_help = (
        "Зафиксировано — файлы загружены из сохранённой сессии."
        if _in_saved_with_files
        else "Зафиксировано — файлы уже загружены. Очистите файлы чтобы изменить настройки."
        if _files_loaded
        else None
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        sep = st.selectbox(
            "Разделитель столбцов",
            [",", ";", "\t", "|", " "],
            format_func=lambda x: {
                "\t": r"Tab (\t) — TSV",
                " ":  "Пробел",
                ",":  ", (запятая) — стандартный CSV",
                ";":  "; (точка с запятой) — европейский CSV",
                "|":  "| (pipe)",
            }.get(x, x),
            key="file_sep",
            disabled=_lock_settings,
            help=_lock_help,
        )
    with p2:
        enc = st.selectbox(
            "Кодировка",
            ["auto", "utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1", "koi8-r"],
            format_func=lambda x: {
                "auto":      "🔍 Авто (определить для каждого файла)",
                "utf-8":     "utf-8 — универсальная",
                "utf-8-sig": "utf-8-sig — Excel UTF-8 (с BOM)",
                "cp1251":    "cp1251 — Excel русская Windows",
                "cp1252":    "cp1252 — Excel западная Windows",
                "latin-1":   "latin-1 — западноевропейская",
                "koi8-r":    "koi8-r — старые русские файлы",
            }.get(x, x),
            key="file_enc",
            disabled=_lock_settings,
            help=_lock_help,
        )
    with p3:
        dec = st.selectbox(
            "Десятичный разделитель",
            [".", ","],
            format_func=lambda x: ". (точка) — международный" if x == "." else ", (запятая) — европейский",
            key="file_dec",
            disabled=_lock_settings,
            help=_lock_help,
        )

    if sep == dec and dec == ",":
        st.error(
            "❌ Разделитель столбцов и десятичный разделитель совпадают — оба «,». "
            "pandas не сможет корректно разобрать файл: числа вроде «1,5» будут "
            "разбиты на два отдельных столбца. "
            "Используйте «; (точка с запятой)» в качестве разделителя столбцов "
            "или «. (точка)» как десятичный разделитель."
        )

    if _in_saved_with_files:
        # File uploader can't be pre-populated — show a visual chip list instead
        _sf_fnames = list(st.session_state["uploaded_files"].keys())
        _chips_html = " ".join(
            f'<span style="background:{BLUE}22; color:{BLUE}; border:1px solid {BLUE}55; '
            f'padding:3px 12px; border-radius:14px; font-size:13px; margin:2px 4px 2px 0; '
            f'display:inline-block">📄 {fn}</span>'
            for fn in _sf_fnames
        )
        _sep_lbl = {"\t": "Tab", " ": "Пробел"}.get(sep, sep)
        st.markdown(
            f'<div style="border:1px dashed {GRAY}55; border-radius:6px; padding:12px 16px; '
            f'background:{CARD_BG}; margin-bottom:2px">'
            f'<div style="color:{GRAY}; font-size:11px; margin-bottom:8px">'
            f'📁 Файлы загружены из сохранённой сессии · '
            f'sep=<code>{_sep_lbl}</code> · enc=<code>{enc}</code> · dec=<code>{dec}</code></div>'
            f'{_chips_html}</div>',
            unsafe_allow_html=True,
        )
        uploaded = []
    else:
        uploaded = st.file_uploader(
            "Загрузите CSV-файлы",
            accept_multiple_files=True,
            type=["csv"],
            label_visibility="collapsed",
            key=f"file_uploader_{st.session_state['upload_key']}",
        )

    if uploaded or st.session_state["uploaded_files"]:
        _, btn_col = st.columns([5, 1])
        with btn_col:
            if st.button("🗑️ Очистить всё", width="stretch"):
                st.session_state["upload_key"] += 1
                st.session_state["uploaded_files"] = {}
                reset_downstream()
                st.rerun()
    st.caption("Принимаются только файлы .csv. Excel (.xlsx) → сохраните через «Файл → Сохранить как → CSV UTF-8».")

    detected_encs: dict = st.session_state.get("_detected_encs", {})
    if (len(uploaded) == 0
            and st.session_state.get("_uploader_was_populated")
            and st.session_state["uploaded_files"]):
        # All files removed via × in the widget — same effect as Clear All.
        # Only fires when files previously came from the Tab-1 widget (not from restore).
        st.session_state["uploaded_files"] = {}
        reset_downstream()
        st.rerun()
    elif uploaded and sep == dec:
        st.warning("⚠️ Файлы не загружены — исправьте конфликт разделителей выше.")
    elif uploaded:
        new_files: dict = {}
        _seen_names: set = set()
        for f in uploaded:
            if f.name in _seen_names:
                st.warning(
                    f"Файл **{f.name}** уже загружен — повторная загрузка пропущена."
                )
                continue
            _seen_names.add(f.name)
            if enc == "auto":
                from charset_normalizer import from_bytes
                raw = f.read()
                f.seek(0)
                result = from_bytes(raw).best()
                file_enc = result.encoding if result is not None else "utf-8"
            else:
                file_enc = enc
            detected_encs[f.name] = file_enc
            try:
                df_f = pd.read_csv(f, sep=sep, encoding=file_enc, decimal=dec)
                new_files[f.name] = df_f
            except Exception as exc:
                st.error(f"Ошибка чтения **{f.name}** (кодировка `{file_enc}`): {exc}")

        _was_empty = not bool(st.session_state["uploaded_files"])
        old_names = set(st.session_state["uploaded_files"].keys())
        new_names = set(new_files.keys())
        st.session_state["uploaded_files"] = new_files
        st.session_state["_uploader_was_populated"] = True  # Files came from Tab-1 widget
        st.session_state["_detected_encs"] = detected_encs  # Persist auto-detected encodings
        if new_names != old_names:
            reset_downstream()
            st.session_state["_uploader_was_populated"] = True  # Restore after reset
            st.session_state["_detected_encs"] = detected_encs  # Restore after reset
        if _was_empty:
            st.rerun()  # lock selectboxes on first file load

    files = st.session_state["uploaded_files"]

    if files:
        for fname, df_show in files.items():
            n_miss_total = int(df_show.isna().sum().sum())
            ct_file = detect_col_types(df_show)
            type_counts: dict = {}
            for t in ct_file.values():
                type_counts[t] = type_counts.get(t, 0) + 1

            used_enc = detected_encs.get(fname, enc)
            enc_badge = f" · `{used_enc}`" if enc == "auto" else ""
            with st.expander(
                f"📄 **{fname}**{enc_badge} — {len(df_show):,} строк × {len(df_show.columns)} колонок",
                expanded=True,
            ):
                m1, m2, m3 = st.columns(3)
                m1.metric("Строк", f"{len(df_show):,}")
                m2.metric("Колонок", len(df_show.columns))
                m3.metric("Пропусков", f"{n_miss_total:,}")

                badge_html = " ".join(
                    f"{TYPE_BADGE.get(t, t)} ×{c}"
                    for t, c in sorted(type_counts.items())
                )
                st.markdown(badge_html, unsafe_allow_html=True)
                st.dataframe(df_show.head(5), width='stretch')

                with st.expander("Подробная статистика", expanded=False):
                    stat_rows = []
                    for col in df_show.columns:
                        ctype = ct_file.get(col, "numerical")
                        n_col_miss = int(df_show[col].isna().sum())
                        n_col_uniq = int(df_show[col].nunique())
                        examples = df_show[col].dropna().head(3).tolist()
                        stat_rows.append({
                            "Колонка": col,
                            "Тип": ctype,
                            "Пропуски": n_col_miss,
                            "Уникальных": n_col_uniq,
                            "Пример": ", ".join(str(v) for v in examples),
                        })
                    st.dataframe(pd.DataFrame(stat_rows), width='stretch', hide_index=True)
    else:
        if st.session_state.get("load_mode") == "saved_files":
            _t1_df = st.session_state.get("merged_df")
            if _t1_df is not None:
                st.caption(
                    "Данные загружены из сохранённых файлов. "
                    "Исходные CSV можно восстановить через source_files/ на вкладке «🏷️ TFT + Экспорт»."
                )
                _t1_miss = int(_t1_df.isna().sum().sum())
                _t1_ct = st.session_state.get("col_types", {})
                _t1_tcounts: dict = {}
                for _t in _t1_ct.values():
                    _t1_tcounts[_t] = _t1_tcounts.get(_t, 0) + 1
                with st.expander(
                    f"📄 **processed_data.csv** — {len(_t1_df):,} строк × {len(_t1_df.columns)} колонок",
                    expanded=True,
                ):
                    _m1, _m2, _m3 = st.columns(3)
                    _m1.metric("Строк", f"{len(_t1_df):,}")
                    _m2.metric("Колонок", len(_t1_df.columns))
                    _m3.metric("Пропусков", f"{_t1_miss:,}")
                    st.markdown(
                        " ".join(f"{TYPE_BADGE.get(t, t)} ×{c}" for t, c in sorted(_t1_tcounts.items())),
                        unsafe_allow_html=True,
                    )
                    st.dataframe(_t1_df.head(5), width='stretch')
            elif st.session_state.get("load_choice_used") == "model_only":
                st.info("Данные загружены в режиме «Только конфиг TFT» — исходные файлы недоступны.")
            else:
                st.info("Загрузите один или несколько CSV-файлов выше.")
        else:
            st.info("Загрузите один или несколько CSV-файлов выше.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Merge
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">🔗 Объединение датафреймов</p>', unsafe_allow_html=True)

    # Flash message from merge / single-file operations (shown once, then cleared)
    if "_merge_flash" in st.session_state:
        st.success(st.session_state.pop("_merge_flash"))

    files = st.session_state["uploaded_files"]

    if not files:
        if st.session_state.get("load_mode") == "saved_files":
            st.info(
                "Исходные CSV не найдены в сохранённой сессии — "
                "вкладка объединения недоступна."
            )
        else:
            st.info("Сначала загрузите файлы на вкладке «📂 Файлы».")

    elif len(files) == 1:
        fname_single = list(files.keys())[0]
        st.write(f"Загружен один файл: **{fname_single}**")
        if st.session_state.get("merged_df") is not None:
            st.success("Используется как основной датафрейм.")
        elif st.button("Использовать как основной датафрейм"):
            df_single = list(files.values())[0]
            st.session_state["merged_df"]           = df_single
            st.session_state["col_types"]           = detect_col_types(df_single)
            st.session_state["prep_config"]         = {}
            st.session_state["prep_df"]             = None
            st.session_state["added_cols"]          = []
            st.session_state["tidx_open"]           = False
            st.session_state["drop_open"]           = False
            st.session_state["agg_open"]            = False
            st.session_state["split_config"]        = {}
            st.session_state["split_masks"]         = None
            st.session_state["inverse_params"]      = {}
            st.session_state["applied_prep_config"] = None
            st.session_state["applied_split_config"]= None
            st.session_state["split_col_sel_v"] = st.session_state.get("split_col_sel_v", 0) + 1
            reset_tft_roles()
            for k in ["tidx_src_col", "tidx_gran", "tidx_name", "drop_cols_select"]:
                st.session_state.pop(k, None)
            st.session_state["_merge_flash"] = "Датафрейм установлен как основной. Роли TFT сброшены."
            st.rerun()

    else:
        names = list(files.keys())
        merge_configs: list = []
        _merge_blocked: list = []  # pairs with no common columns

        if st.session_state.get("merged_df") is not None and st.session_state.get("load_mode") == "saved_files":
            st.success(
                "Объединённый датасет восстановлен из сессии. "
                "Параметры объединения отображены ниже — при необходимости измените и повторно нажмите «▶ Выполнить объединение»."
            )

        for i in range(len(names) - 1):
            left_name = names[i]
            right_name = names[i + 1]
            left_df = files[left_name]
            right_df = files[right_name]

            common_cols = sorted(set(left_df.columns) & set(right_df.columns))
            preview_str = ", ".join(common_cols[:5]) + ("…" if len(common_cols) > 5 else "")

            with st.expander(f"**{left_name}** JOIN **{right_name}**", expanded=True):
                if not common_cols:
                    _merge_blocked.append((left_name, right_name))
                    st.warning(
                        f"⚠️ Нет общих столбцов между **{left_name}** и **{right_name}**. "
                        "Объединение по ключу невозможно — выберите колонку из левого файла "
                        "или загрузите файлы с общим идентификатором (например, `station_id`)."
                    )
                else:
                    st.markdown(
                        f'<span style="color:{GRAY}">Общих колонок: {len(common_cols)} → {preview_str}</span>',
                        unsafe_allow_html=True,
                    )
                c1, c2 = st.columns(2)
                with c1:
                    col_options = common_cols if common_cols else left_df.columns.tolist()
                    # Apply pending value from session restore (before widget is created)
                    _pcol = st.session_state.pop(f"_pending_join_col_{i}", None)
                    if _pcol and _pcol in col_options:
                        st.session_state[f"join_col_{i}"] = _pcol
                    on_col = st.selectbox(
                        "Колонка объединения",
                        options=col_options,
                        key=f"join_col_{i}",
                    )
                with c2:
                    _phow = st.session_state.pop(f"_pending_join_how_{i}", None)
                    if _phow:
                        st.session_state[f"join_how_{i}"] = _phow
                    how = st.selectbox(
                        "Тип JOIN",
                        options=["left", "inner", "outer", "right"],
                        key=f"join_how_{i}",
                    )
                merge_configs.append({"right_name": right_name, "on": on_col, "how": how})

        # ── Use a single file without merging ────────────────────────────────
        with st.expander("📌 Использовать один файл без объединения", expanded=False):
            st.caption(
                "Если объединение не требуется — выберите один из загруженных файлов "
                "и используйте его как основной датафрейм."
            )
            _sc1, _sc2 = st.columns([3, 1])
            with _sc1:
                _solo_choice = st.selectbox(
                    "Файл",
                    options=names,
                    key="solo_file_choice",
                    label_visibility="collapsed",
                )
            with _sc2:
                if st.button("✅ Использовать", key="use_solo_file", width="stretch"):
                    _solo_df = files[_solo_choice]
                    st.session_state["merged_df"]           = _solo_df
                    st.session_state["col_types"]           = detect_col_types(_solo_df)
                    st.session_state["prep_config"]         = {}
                    st.session_state["prep_df"]             = None
                    st.session_state["added_cols"]          = []
                    st.session_state["added_col_formulas"]  = {}
                    st.session_state["tidx_config"]         = {}
                    st.session_state["dtx_configs"]         = []
                    st.session_state["tidx_open"]           = False
                    st.session_state["dtx_open"]            = False
                    st.session_state["drop_open"]           = False
                    st.session_state["load_mode"]           = "csv_upload"
                    st.session_state["split_config"]         = {}
                    st.session_state["split_masks"]          = None
                    st.session_state["inverse_params"]       = {}
                    st.session_state["applied_prep_config"]  = None
                    st.session_state["applied_split_config"] = None
                    st.session_state["split_col_sel_v"] = st.session_state.get("split_col_sel_v", 0) + 1
                    reset_tft_roles()
                    for k in ["tidx_src_col", "tidx_gran", "tidx_name", "drop_cols_select"]:
                        st.session_state.pop(k, None)
                    st.session_state["_merge_flash"] = (
                        f"Файл «{_solo_choice}» ({len(_solo_df):,} строк × "
                        f"{len(_solo_df.columns)} колонок) установлен как основной. "
                        "Роли TFT сброшены."
                    )
                    st.rerun()

        st.divider()
        if _merge_blocked:
            st.error(
                "❌ Объединение невозможно — следующие пары файлов не имеют общих колонок: "
                + ", ".join(f"**{a}** + **{b}**" for a, b in _merge_blocked)
                + ". Добавьте общий идентификатор или используйте «Один файл без объединения»."
            )
        if st.button("▶ Выполнить объединение", type="primary",
                     disabled=bool(_merge_blocked)):
            try:
                merged = merge_dataframes(files, merge_configs)
                st.session_state["merged_df"] = merged
                st.session_state["col_types"] = detect_col_types(merged)
                st.session_state["prep_config"]         = {}
                st.session_state["prep_df"]             = None
                st.session_state["added_cols"]          = []
                st.session_state["added_col_formulas"]  = {}
                st.session_state["tidx_config"]         = {}
                st.session_state["dtx_configs"]         = []
                st.session_state["tidx_open"]           = False
                st.session_state["dtx_open"]            = False
                st.session_state["drop_open"]           = False
                st.session_state["load_mode"]           = "csv_upload"
                st.session_state["split_config"]         = {}
                st.session_state["split_masks"]          = None
                st.session_state["inverse_params"]       = {}
                st.session_state["applied_prep_config"]  = None
                st.session_state["applied_split_config"] = None
                st.session_state["split_col_sel_v"] = st.session_state.get("split_col_sel_v", 0) + 1
                reset_tft_roles()
                for k in ["tidx_src_col", "tidx_gran", "tidx_name", "drop_cols_select"]:
                    st.session_state.pop(k, None)
                st.session_state["_merge_flash"] = (
                    f"Объединение выполнено: {len(merged):,} строк × {len(merged.columns)} колонок. "
                    "Роли TFT сброшены — распределите колонки заново на вкладке «🏷️ TFT + Экспорт»."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Ошибка объединения: {exc}")

    merged_df = st.session_state.get("merged_df")
    if merged_df is not None:
        st.divider()
        st.markdown('<p class="section-title">Предпросмотр результата</p>', unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        mc1.metric("Строк", f"{len(merged_df):,}")
        mc2.metric("Колонок", len(merged_df.columns))

        miss_cols = [c for c in merged_df.columns if merged_df[c].isna().any()]
        if miss_cols:
            st.warning(f"Найдены пропуски в {len(miss_cols)} колонках.")
            with st.expander("Колонки с пропусками", expanded=False):
                miss_data = pd.DataFrame({
                    "Колонка": miss_cols,
                    "Пропусков": [int(merged_df[c].isna().sum()) for c in miss_cols],
                })
                st.dataframe(miss_data, width='stretch', hide_index=True)

        st.dataframe(merged_df.head(10), width='stretch')

        # ── Create time_idx ───────────────────────────────────────────────────
        def _mark_tidx_open() -> None:
            st.session_state["tidx_open"] = True

        # Apply pending tidx settings from session restore (before widget instantiation)
        _pt_src = st.session_state.pop("_pending_tidx_src_col", None)
        if _pt_src is not None and _pt_src in merged_df.columns:
            st.session_state["tidx_src_col"] = _pt_src
        _pt_gran = st.session_state.pop("_pending_tidx_gran", None)
        if _pt_gran is not None:
            st.session_state["tidx_gran"] = _pt_gran
        _pt_name = st.session_state.pop("_pending_tidx_name", None)
        if _pt_name is not None:
            st.session_state["tidx_name"] = _pt_name

        st.divider()
        with st.expander("⏱️ Создать колонку time_idx", expanded=st.session_state.get("tidx_open", False)):
            st.caption(
                "time_idx — целое число от 0, равномерно растущее по времени. "
                "Обязателен для TFT. Создаётся из существующей колонки с датой."
            )
            ti1, ti2, ti3 = st.columns(3)
            with ti1:
                tidx_src = st.selectbox(
                    "Исходная колонка (дата)",
                    options=merged_df.columns.tolist(),
                    key="tidx_src_col",
                    on_change=_mark_tidx_open,
                )
            with ti2:
                tidx_gran = st.selectbox(
                    "Гранулярность",
                    options=["день", "час", "неделя", "месяц"],
                    key="tidx_gran",
                    on_change=_mark_tidx_open,
                )
            with ti3:
                tidx_name = st.text_input(
                    "Название колонки", key="tidx_name",
                    on_change=_mark_tidx_open,
                )

            if st.button("➕ Добавить time_idx в датафрейм"):
                try:
                    dt = pd.to_datetime(merged_df[tidx_src], errors="coerce")
                    n_nat = int(dt.isna().sum())
                    dt_min = dt.min()
                    if pd.isna(dt_min):
                        st.error(
                            f"❌ Колонка «{tidx_src}» не содержит ни одной распознаваемой даты. "
                            "Выберите другую колонку."
                        )
                    else:
                        if n_nat > 0:
                            st.warning(
                                f"⚠️ {n_nat} строк не удалось распознать как дату — "
                                "они получат NaN в time_idx. Проверьте исходную колонку."
                            )
                        if tidx_gran == "день":
                            idx_col = (dt - dt_min).dt.days.astype("Int64")
                        elif tidx_gran == "час":
                            idx_col = ((dt - dt_min).dt.total_seconds() // 3600).astype("Int64")
                        elif tidx_gran == "неделя":
                            idx_col = ((dt - dt_min).dt.days // 7).astype("Int64")
                        else:  # месяц
                            idx_col = (
                                (dt.dt.year - dt_min.year) * 12 + (dt.dt.month - dt_min.month)
                            ).astype("Int64")

                        new_df = merged_df.copy()
                        new_df.insert(0, tidx_name, idx_col)
                        st.session_state["merged_df"] = new_df
                        ct = detect_col_types(new_df)
                        ct[tidx_name] = "numerical"
                        st.session_state["col_types"] = ct
                        st.session_state["prep_df"] = None
                        if tidx_name not in st.session_state["added_cols"]:
                            st.session_state["added_cols"].append(tidx_name)
                        st.session_state["added_col_formulas"][tidx_name] = "__time_idx__"
                        st.session_state["tidx_config"] = {
                            "src_col": tidx_src, "gran": tidx_gran, "name": tidx_name,
                        }
                        st.session_state["applied_prep_config"] = None
                        _trim_tft_roles(set(new_df.columns))
                        st.toast(
                            f"Колонка «{tidx_name}» добавлена: "
                            f"диапазон {idx_col.min()} … {idx_col.max()}",
                            icon="✅",
                        )
                        st.rerun()
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")

        # ── Extract date components ───────────────────────────────────────────
        def _mark_dtx_open() -> None:
            st.session_state["dtx_open"] = True

        _pending_dtx_src = st.session_state.pop("_pending_dtx_src_col", None)
        if _pending_dtx_src is not None and _pending_dtx_src in merged_df.columns:
            st.session_state["dtx_src_col"] = _pending_dtx_src

        st.divider()
        with st.expander("📅 Извлечь компоненты даты", expanded=st.session_state.get("dtx_open", False)):
            st.caption(
                "Разложить колонку с датой на числовые компоненты: год, месяц, день, час, "
                "день недели, день года, номер недели. Компоненты добавляются как отдельные "
                "числовые колонки в датафрейм."
            )
            _dtx_cfgs = st.session_state.get("dtx_configs", [])
            if _dtx_cfgs:
                for _dxc in _dtx_cfgs:
                    _comps_str = ", ".join(_dxc.get("components", []))
                    st.markdown(
                        f'<span style="background:{BLUE}22; color:{BLUE}; border:1px solid {BLUE}55; '
                        f'border-radius:4px; padding:2px 8px; font-size:0.85em; margin-right:4px;">'
                        f'<code>{_dxc["src_col"]}</code> → {_comps_str}</span>',
                        unsafe_allow_html=True,
                    )
                st.divider()

            _dx1, _dx2 = st.columns([1, 2])
            with _dx1:
                _dtx_added_set = set(st.session_state.get("added_cols", []))
                _dtx_src_options = [c for c in merged_df.columns if c not in _dtx_added_set]
                _dtx_src = st.selectbox(
                    "Колонка с датой",
                    options=_dtx_src_options or merged_df.columns.tolist(),
                    key="dtx_src_col",
                    on_change=_mark_dtx_open,
                )
            with _dx2:
                _dtx_comps = st.multiselect(
                    "Компоненты",
                    options=["year", "month", "day", "hour", "minute",
                             "dayofweek", "dayofyear", "weekofyear"],
                    key="dtx_comps",
                    on_change=_mark_dtx_open,
                )

            if st.button("➕ Добавить компоненты в датафрейм", key="dtx_apply_btn"):
                if not _dtx_comps:
                    st.error("Выберите хотя бы один компонент.")
                else:
                    try:
                        _new_merged, _dtx_added = apply_dt_extractions(
                            merged_df, [{"src_col": _dtx_src, "components": _dtx_comps}]
                        )
                        if not _dtx_added:
                            st.error(
                                f"❌ Не удалось извлечь компоненты из «{_dtx_src}». "
                                "Убедитесь, что колонка содержит даты."
                            )
                        else:
                            _ct = detect_col_types(_new_merged)
                            for _c in _dtx_added:
                                _ct[_c] = "numerical"
                            st.session_state["merged_df"] = _new_merged
                            st.session_state["col_types"] = _ct
                            st.session_state["prep_df"]   = None
                            for _c in _dtx_added:
                                if _c not in st.session_state["added_cols"]:
                                    st.session_state["added_cols"].append(_c)
                                st.session_state["added_col_formulas"][_c] = f"__dtx_{_dtx_src}__"
                            _ex_cfg = next(
                                (x for x in st.session_state["dtx_configs"]
                                 if x["src_col"] == _dtx_src),
                                None,
                            )
                            if _ex_cfg:
                                _ex_cfg["components"] = list(
                                    dict.fromkeys(_ex_cfg["components"] + _dtx_comps)
                                )
                            else:
                                st.session_state["dtx_configs"].append(
                                    {"src_col": _dtx_src, "components": _dtx_comps}
                                )
                            st.session_state["applied_prep_config"] = None
                            _trim_tft_roles(set(_new_merged.columns))
                            st.toast(f"Добавлено: {', '.join(_dtx_added)}", icon="✅")
                            st.rerun()
                    except Exception as _dtx_exc:
                        st.error(f"Ошибка: {_dtx_exc}")

        # ── Drop added columns ────────────────────────────────────────────────
        def _mark_drop_open() -> None:
            st.session_state["drop_open"] = True

        added_cols_present = [
            c for c in st.session_state["added_cols"]
            if c in merged_df.columns
        ]
        st.divider()
        with st.expander("🗑️ Удалить добавленные колонки", expanded=st.session_state.get("drop_open", False)):
            if not added_cols_present:
                st.caption("Нет добавленных колонок. Создайте time_idx, извлеките компоненты даты или добавьте вычисляемую колонку выше.")
            else:
                st.caption("Колонки, добавленные вручную в этом сеансе: time_idx, компоненты даты и вычисляемые.")
                cols_to_drop = st.multiselect(
                    "Выберите колонки для удаления",
                    options=added_cols_present,
                    key="drop_cols_select",
                    on_change=_mark_drop_open,
                )
                if cols_to_drop and st.button("🗑️ Удалить выбранные"):
                    _drop_set = set(cols_to_drop)
                    new_df = merged_df.drop(columns=cols_to_drop)
                    st.session_state["merged_df"] = new_df
                    st.session_state["col_types"] = detect_col_types(new_df)
                    st.session_state["prep_df"] = None
                    st.session_state["drop_open"] = False
                    st.session_state["added_cols"] = [
                        c for c in st.session_state["added_cols"] if c not in _drop_set
                    ]
                    st.session_state["prep_config"] = {
                        k: v for k, v in st.session_state["prep_config"].items()
                        if k not in _drop_set
                    }
                    _remaining = set(st.session_state["added_cols"])
                    st.session_state["dtx_configs"] = [
                        {**_dc, "components": [
                            c for c in _dc["components"]
                            if f"{_dc['src_col']}_{c}" in _remaining
                        ]}
                        for _dc in st.session_state.get("dtx_configs", [])
                        if any(f"{_dc['src_col']}_{c}" in _remaining for c in _dc["components"])
                    ]
                    st.session_state["applied_prep_config"] = None
                    _trim_tft_roles(set(new_df.columns))
                    st.toast(f"Удалено: {', '.join(cols_to_drop)}", icon="🗑️")
                    st.rerun()

        # ── Computed column ───────────────────────────────────────────────────
        def _mark_agg_open() -> None:
            st.session_state["agg_open"] = True

        st.divider()
        with st.expander("📐 Создать вычисляемую колонку", expanded=st.session_state.get("agg_open", False)):
            st.caption(
                "Введите формулу, используя названия колонок и математические операции: "
                "+ − * / ** (степень) // (целое деление) % (остаток). "
                "Для названий с пробелами используйте обратные кавычки: `` `название колонки` ``."
            )

            agg_name = st.text_input(
                "Название новой колонки",
                key="agg_col_name",
                placeholder="my_feature",
                on_change=_mark_agg_open,
            )

            num_cols_agg = [c for c in merged_df.columns if pd.api.types.is_numeric_dtype(merged_df[c])]
            other_cols_agg = [c for c in merged_df.columns if c not in num_cols_agg]

            ref_parts = []
            if num_cols_agg:
                ref_parts.append(
                    "**Числовые:** "
                    + "  ".join(f"`{c}`" if " " not in c else f"`` `{c}` ``" for c in num_cols_agg)
                )
            if other_cols_agg:
                ref_parts.append(
                    f'<span style="color:{GRAY}">Нечисловые (с осторожностью): </span>'
                    + "  ".join(f"`{c}`" for c in other_cols_agg)
                )
            if ref_parts:
                st.markdown("  \n".join(ref_parts), unsafe_allow_html=True)

            agg_formula = st.text_area(
                "Формула",
                key="agg_formula",
                placeholder="(col1 + col2) / col3\ncol1 ** 2 - col2 * 0.5",
                height=80,
                on_change=_mark_agg_open,
            )

            btn_check, btn_apply = st.columns(2)
            with btn_check:
                check_clicked = st.button("✅ Проверить формулу", width="stretch")
            with btn_apply:
                apply_clicked = st.button("➕ Добавить колонку", type="primary", width="stretch")

            # Messages always outside column containers — full width, cleared on next rerun
            if check_clicked:
                if not agg_formula.strip():
                    st.error("Введите формулу.")
                elif not agg_name.strip():
                    st.error("Введите название новой колонки.")
                else:
                    if agg_name in merged_df.columns and agg_name not in st.session_state["added_cols"]:
                        st.warning(f"⚠️ Колонка «{agg_name}» уже существует в данных — будет перезаписана.")
                    res, err, n_inf = eval_formula(merged_df, agg_formula)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        sample = pd.to_numeric(res, errors="coerce").dropna().head(5).tolist()
                        st.success(
                            f"✅ Формула корректна · тип `{res.dtype}` · "
                            f"примеры: {[round(v, 4) if isinstance(v, float) else v for v in sample]}"
                        )
                        if n_inf:
                            st.warning(f"⚠️ {n_inf} значений равны ±inf (деление на ноль или переполнение).")
                        non_num = int(pd.to_numeric(res, errors="coerce").isna().sum()) - int(res.isna().sum())
                        if non_num > 0:
                            st.warning(f"⚠️ {non_num} значений не удалось привести к числу.")

            if apply_clicked:
                if not agg_formula.strip():
                    st.error("Введите формулу.")
                elif not agg_name.strip():
                    st.error("Введите название новой колонки.")
                else:
                    res, err, n_inf = eval_formula(merged_df, agg_formula)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        new_df = merged_df.copy()
                        new_df[agg_name] = res
                        st.session_state["merged_df"] = new_df
                        st.session_state["col_types"] = detect_col_types(new_df)
                        st.session_state["prep_df"] = None
                        if agg_name not in st.session_state["added_cols"]:
                            st.session_state["added_cols"].append(agg_name)
                        st.session_state["added_col_formulas"][agg_name] = agg_formula
                        st.session_state["applied_prep_config"] = None
                        _trim_tft_roles(set(new_df.columns))
                        inf_note = f" Содержит {n_inf} значений ±inf." if n_inf else ""
                        st.toast(f"Колонка «{agg_name}» добавлена.{inf_note}", icon="✅")
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Column analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">📊 Анализ колонок</p>', unsafe_allow_html=True)

    merged_df = st.session_state.get("merged_df")
    col_types = st.session_state.get("col_types", {})

    if merged_df is None:
        if st.session_state.get("load_mode") == "saved_files":
            if st.session_state.get("load_choice_used") == "model_only":
                st.info(
                    "Вкладка недоступна в режиме «Только конфиг TFT». "
                    "Для анализа колонок используйте режим «Восстановить весь проект» на вкладке «🏷️ TFT + Экспорт»."
                )
            else:
                st.info(
                    "Исходные файлы загружены, но датафрейм ещё не объединён. "
                    "Перейдите на вкладку «🔗 Объединение»."
                )
        else:
            st.info("Сначала загрузите и объедините файлы.")
    else:
        # Top metrics per type
        type_counts_all: dict = {}
        for t in col_types.values():
            type_counts_all[t] = type_counts_all.get(t, 0) + 1

        t_cols = st.columns(6)
        t_cols[0].metric("Всего колонок", len(merged_df.columns))
        for idx, t_name in enumerate(["numerical", "categorical", "binary", "datetime", "text"]):
            t_cols[idx + 1].metric(t_name.capitalize(), type_counts_all.get(t_name, 0))

        st.divider()

        # ── Manual type override ──────────────────────────────────────────────
        with st.expander("✏️ Скорректировать типы колонок вручную", expanded=False):
            st.caption(
                "Авто-определение иногда ошибается. Измените тип — "
                "это сразу повлияет на доступные методы в «⚙️ Препроцессинг»."
            )
            type_search = st.text_input(
                "Поиск колонки",
                key="type_override_search",
                placeholder="Введите название...",
                label_visibility="collapsed",
            )
            type_rows = [
                {
                    "Колонка": col,
                    "Авто": detect_col_type(merged_df[col]),
                    "Тип": col_types.get(col, "numerical"),
                }
                for col in merged_df.columns
                if not type_search or type_search.lower() in col.lower()
            ]
            if not type_rows:
                st.caption("Нет совпадений по запросу.")
            else:
                editor_key = f"type_override_editor_{st.session_state.get('type_override_v', 0)}"
                edited_types = st.data_editor(
                    pd.DataFrame(type_rows),
                    column_config={
                        "Колонка": st.column_config.TextColumn(disabled=True),
                        "Авто":    st.column_config.TextColumn("Авто-определение", disabled=True),
                        "Тип":     st.column_config.SelectboxColumn(
                            "Итоговый тип",
                            options=["numerical", "categorical", "binary", "datetime", "text"],
                            required=True,
                        ),
                    },
                    hide_index=True,
                    key=editor_key,
                    width='stretch',
                )
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("✅ Применить изменения типов", width="stretch"):
                        _old_ctypes = dict(st.session_state["col_types"])
                        for _, row in edited_types.iterrows():
                            st.session_state["col_types"][row["Колонка"]] = row["Тип"]
                        _changed_cols = [
                            _col for _col, _new_ct in st.session_state["col_types"].items()
                            if _old_ctypes.get(_col) != _new_ct
                        ]
                        for _col in _changed_cols:
                            st.session_state["prep_config"].pop(_col, None)
                        _clear_vl_data(_changed_cols)
                        st.session_state["prep_df"] = None
                        st.session_state["applied_prep_config"] = None
                        st.toast("Типы обновлены. Настройки препроцессинга изменённых колонок сброшены.")
                        st.rerun()
                with btn2:
                    if st.button("🔄 Вернуть к авто-определению", width="stretch"):
                        st.session_state["col_types"] = detect_col_types(merged_df)
                        st.session_state["prep_config"] = {}
                        st.session_state["prep_df"] = None
                        st.session_state["applied_prep_config"] = None
                        st.session_state["type_override_v"] = st.session_state.get("type_override_v", 0) + 1
                        _clear_vl_data()
                        st.toast("Типы сброшены к авто-определению.")
                        st.rerun()

        st.divider()

        # Filters
        f1, f2 = st.columns([2, 3])
        with f1:
            type_filter = st.multiselect(
                "Фильтр по типу",
                options=["numerical", "categorical", "binary", "datetime", "text"],
                default=["numerical", "categorical", "binary", "datetime", "text"],
                key="tab3_type_filter",
            )
        with f2:
            search_text = st.text_input("Поиск по названию колонки", key="tab3_search")

        if merged_df.empty or len(merged_df.columns) == 0:
            st.warning("Датафрейм не содержит колонок. Вернитесь на вкладку «🔗 Объединение».")
            st.stop()
        summary_df = get_column_summary(merged_df, col_types)
        filtered = summary_df[summary_df["Тип"].isin(type_filter)]
        if search_text:
            filtered = filtered[
                filtered["Колонка"].str.lower().str.contains(search_text.lower(), na=False)
            ]

        st.dataframe(
            filtered,
            width='stretch',
            hide_index=True,
            column_config={"Уникальных": st.column_config.TextColumn("Уникальных")},
        )

        st.divider()
        st.markdown('<p class="section-title">Детальный просмотр</p>', unsafe_allow_html=True)

        sel_col = st.selectbox(
            "Выберите колонку для детального просмотра",
            options=merged_df.columns.tolist(),
            key="tab3_detail_col",
        )

        if sel_col:
            ctype = col_types.get(sel_col, "numerical")
            series = merged_df[sel_col]
            n_miss = int(series.isna().sum())
            n_uniq = int(series.nunique())

            dc_left, dc_right = st.columns([1, 2])

            with dc_left:
                st.markdown(
                    f"**{sel_col}** {TYPE_BADGE.get(ctype, ctype)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span style="color:{GRAY}">Пропуски: <b>{n_miss}</b> '
                    f'({n_miss / len(merged_df) * 100:.1f}%)</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span style="color:{GRAY}">Уникальных: <b>{n_uniq}</b></span>',
                    unsafe_allow_html=True,
                )

                if ctype == "numerical":
                    st.dataframe(series.describe().to_frame(), width='stretch')
                else:
                    vc = series.value_counts().head(20).reset_index()
                    vc.columns = ["Значение", "Количество"]
                    st.dataframe(vc, width='stretch', hide_index=True)

            with dc_right:
                color = TYPE_COLORS.get(ctype, BLUE)
                if ctype == "numerical":
                    st.plotly_chart(plot_histogram(series, title=sel_col, color=color), width='stretch')
                else:
                    st.plotly_chart(plot_bar(series, title=sel_col, color=color), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">⚙️ Препроцессинг</p>', unsafe_allow_html=True)

    merged_df = st.session_state.get("merged_df")
    col_types = st.session_state.get("col_types", {})

    if merged_df is None:
        if st.session_state.get("load_mode") == "saved_files":
            if st.session_state.get("load_choice_used") == "model_only":
                st.info(
                    "Вкладка недоступна в режиме «Только конфиг TFT». "
                    "Для настройки препроцессинга используйте режим «Восстановить весь проект» на вкладке «🏷️ TFT + Экспорт»."
                )
            else:
                st.info(
                    "Исходные файлы загружены, но датафрейм ещё не объединён. "
                    "Перейдите на вкладку «🔗 Объединение»."
                )
        else:
            st.info("Сначала загрузите и объедините файлы.")
    else:
        # ── Presets ──────────────────────────────────────────────────────────
        def _set_prep(col: str, method: str, ctype: str) -> None:
            st.session_state["prep_config"].setdefault(col, {})
            st.session_state["prep_config"][col]["method"] = method
            st.session_state["prep_config"][col].setdefault("params", {})

        # Top bar: Auto (1/3) | empty (1/3) | Reset (1/3)
        _pa, _pb, _pc = st.columns(3)
        with _pa:
            if st.button("🤖 Auto", width="stretch"):
                for col, ctype in col_types.items():
                    _set_prep(col, DEFAULT_METHOD.get(ctype, "none"), ctype)
                st.rerun()
        with _pc:
            if st.button("🔄 Сбросить всё", width="stretch"):
                st.session_state["prep_config"] = {}
                _clear_vl_data()
                st.rerun()

        # Per-type preset rows — equal-width buttons, right-aligned
        # "fillna" is NOT a method; Fill NA button sets pre_fillna checkboxes instead.
        PRESET_ROWS = {
            "numerical":   ["zscore", "minmax", "robust", "log1p", "cyclical", "fillna", "drop"],
            "categorical": ["label_enc", "onehot", "fillna", "drop"],
            "binary":      ["fillna", "drop"],
            "datetime":    ["drop"],
            "text":        ["label_enc", "drop"],
        }
        PRESET_SHORT = {
            "zscore":     "Z-score",
            "minmax":     "Min-Max",
            "robust":     "Robust",
            "log1p":      "log1p",
            "cyclical":   "sin/cos",
            "fillna":    "Fill NA",
            "label_enc": "Label Enc",
            "onehot":    "One-Hot",
            "drop":      "Удалить",
        }
        _MAX_BTNS = max(len(v) for v in PRESET_ROWS.values())  # 7
        for _ctype, _methods in PRESET_ROWS.items():
            _n = sum(1 for _ct in col_types.values() if _ct == _ctype)
            if _n == 0:
                continue
            _has_miss = any(merged_df[c].isna().any() for c, ct in col_types.items() if ct == _ctype and c in merged_df.columns)
            _n_pad = _MAX_BTNS - len(_methods)
            # Label col + _MAX_BTNS equal button slots (pad on left to right-align buttons)
            _all_cols = st.columns([2] + [1] * _MAX_BTNS)
            with _all_cols[0]:
                st.markdown(
                    f'{TYPE_BADGE.get(_ctype, _ctype)}'
                    f'<span style="color:{GRAY}; font-size:12px"> &nbsp;→ все {_n} кол.</span>',
                    unsafe_allow_html=True,
                )
            for _i, _m in enumerate(_methods):
                with _all_cols[1 + _n_pad + _i]:
                    if st.button(
                        PRESET_SHORT[_m],
                        key=f"preset_{_ctype}_{_m}",
                        width="stretch",
                        disabled=(_m == "fillna" and not _has_miss),
                    ):
                        for col, ct in col_types.items():
                            if ct != _ctype or col not in merged_df.columns:
                                continue
                            if _m == "fillna":
                                # Fill NA checks the pre_fillna checkbox for missing cols
                                if merged_df[col].isna().any():
                                    st.session_state["prep_config"].setdefault(col, {})
                                    st.session_state["prep_config"][col]["pre_fillna"] = True
                                    st.session_state[f"pf_chk_{col}"] = True
                            else:
                                _set_prep(col, _m, ct)
                        st.rerun()

        st.divider()

        # ── Temporal split ────────────────────────────────────────────────────
        _cur_split_masks = st.session_state.get("split_masks")
        with st.expander(
            "⏳ Временной сплит (train / val / test)"
            + (f" — ✅ активен: Train {_cur_split_masks['stats']['n_train']:,} · "
               f"Val {_cur_split_masks['stats']['n_val']:,} · "
               f"Test {_cur_split_masks['stats']['n_test']:,}"
               if _cur_split_masks else ""),
            expanded=False,
        ):
            st.caption(
                "Разбивает датасет по времени. "
                "Параметры нормировки (Z-score, Min-Max, Robust) можно вычислять "
                "только по train, чтобы исключить утечку данных в val/test."
            )

            _sp_c1, _sp_c2 = st.columns([3, 1])
            with _sp_c1:
                _scv = st.session_state.get("split_col_sel_v", 0)
                _split_date_col = st.selectbox(
                    "Колонка с датой",
                    options=["— не выбрано —"] + merged_df.columns.tolist(),
                    index=0 if not st.session_state.get("split_config", {}).get("date_col")
                          else (["— не выбрано —"] + merged_df.columns.tolist()).index(
                              st.session_state["split_config"]["date_col"]
                          ) if st.session_state["split_config"]["date_col"] in merged_df.columns else 0,
                    key=f"split_date_col_sel_{_scv}",
                )

            if _split_date_col != "— не выбрано —":
                try:
                    # Numeric columns cannot be date columns — integers parse as
                    # nanoseconds since epoch (0 → 1970-01-01 00:00:00.000000000)
                    # which looks like a valid date but is meaningless for splitting.
                    if pd.api.types.is_numeric_dtype(merged_df[_split_date_col]):
                        st.error(
                            f"❌ Колонка «{_split_date_col}» содержит числа, а не даты. "
                            "Числовые значения (0, 1, 2, …) ошибочно распознаются как "
                            "1970-01-01 (Unix-эпоха). "
                            "Выберите колонку с датами в текстовом формате (например, YYYY-MM-DD)."
                        )
                    else:
                        # format="mixed" suppresses the pandas 2+ per-element fallback warning
                        _dt_s = pd.to_datetime(
                            merged_df[_split_date_col], format="mixed", errors="coerce"
                        )
                        _n_nat = int(_dt_s.isna().sum())
                        if _n_nat == len(merged_df):
                            st.error(f"❌ Колонка «{_split_date_col}» не содержит ни одного распознаваемого значения даты.")
                        else:
                            if _n_nat > 0:
                                st.warning(f"⚠️ {_n_nat} строк не распознаны как дата — они не войдут ни в один сплит.")
                            _dt_min = _dt_s.min()
                            _dt_max = _dt_s.max()
                            with _sp_c2:
                                st.caption("Диапазон дат")
                                st.markdown(f"`{_dt_min.date()}` → `{_dt_max.date()}`")

                            # Compare at day level: nanosecond-precision ints (0,1,2…)
                            # all map to 1970-01-01 but differ by nanoseconds.
                            _n_unique_days = _dt_s.dt.normalize().nunique()
                            if _dt_min.date() == _dt_max.date():
                                st.error(
                                    f"❌ Колонка «{_split_date_col}» содержит только одну уникальную дату "
                                    f"(`{_dt_min.date()}`). "
                                    "Возможно, числовые значения были ошибочно распознаны как дата "
                                    "(например, 0 → 1970-01-01). "
                                    "Выберите колонку с реальным диапазоном дат."
                                )
                            elif _n_unique_days < 3:
                                st.error(
                                    f"❌ Колонка «{_split_date_col}» содержит только {_n_unique_days} уникальных дат — "
                                    "недостаточно для разбивки на train / val / test (нужно минимум 3)."
                                )
                            else:
                                _split_mode = st.radio(
                                    "Режим",
                                    ["По процентам", "По датам"],
                                    horizontal=True,
                                    key="split_mode_radio",
                                )

                                if _split_mode == "По процентам":
                                    _spc1, _spc2, _spc3 = st.columns(3)
                                    with _spc1:
                                        _tr_pct = st.slider("Train %", 1, 97,
                                            st.session_state.get("split_config", {}).get("train_pct", 70),
                                            key="split_train_pct_sl")
                                    with _spc2:
                                        _vl_pct = st.slider("Val %", 1, 99 - _tr_pct,
                                            min(st.session_state.get("split_config", {}).get("val_pct", 15),
                                                99 - _tr_pct),
                                            key="split_val_pct_sl")
                                    with _spc3:
                                        st.metric("Test %", 100 - _tr_pct - _vl_pct)

                                    _sorted_u = sorted(_dt_s.dropna().unique())
                                    _nd = len(_sorted_u)
                                    _tr_cut = _sorted_u[max(0, int(_nd * _tr_pct / 100) - 1)]
                                    _vl_cut = _sorted_u[max(0, int(_nd * (_tr_pct + _vl_pct) / 100) - 1)]
                                    _train_m = (_dt_s <= _tr_cut) & _dt_s.notna()
                                    _val_m   = (_dt_s > _tr_cut) & (_dt_s <= _vl_cut) & _dt_s.notna()
                                    _test_m  = (_dt_s > _vl_cut) & _dt_s.notna()
                                    _new_cfg = {
                                        "date_col": _split_date_col, "mode": "pct",
                                        "train_pct": _tr_pct, "val_pct": _vl_pct,
                                        "train_end": str(_tr_cut.date()),
                                        "val_end":   str(_vl_cut.date()),
                                    }
                                    st.caption(
                                        f"Граница train: **{_tr_cut.date()}** · "
                                        f"граница val: **{_vl_cut.date()}**"
                                    )
                                else:
                                    _prev = st.session_state.get("split_config", {})
                                    _spc1, _spc2 = st.columns(2)
                                    with _spc1:
                                        _tr_end = st.date_input(
                                            "Train до (включительно)",
                                            value=(pd.to_datetime(_prev["train_end"]).date()
                                                   if _prev.get("train_end") else _dt_min.date()),
                                            min_value=_dt_min.date(), max_value=_dt_max.date(),
                                            key="split_train_end_di",
                                        )
                                    with _spc2:
                                        _vl_end = st.date_input(
                                            "Val до (включительно)",
                                            value=(pd.to_datetime(_prev["val_end"]).date()
                                                   if _prev.get("val_end") else _dt_min.date()),
                                            min_value=_dt_min.date(), max_value=_dt_max.date(),
                                            key="split_val_end_di",
                                        )
                                    _tr_end_ts = pd.Timestamp(_tr_end)
                                    _vl_end_ts = pd.Timestamp(_vl_end)
                                    _train_m = (_dt_s <= _tr_end_ts) & _dt_s.notna()
                                    _val_m   = (_dt_s > _tr_end_ts) & (_dt_s <= _vl_end_ts) & _dt_s.notna()
                                    _test_m  = (_dt_s > _vl_end_ts) & _dt_s.notna()
                                    _new_cfg = {
                                        "date_col": _split_date_col, "mode": "date",
                                        "train_end": str(_tr_end), "val_end": str(_vl_end),
                                    }
                                    _n_valid = int(_dt_s.notna().sum())
                                    if _n_valid > 0:
                                        _tr_pct_disp = int(round(_train_m.sum() / _n_valid * 100))
                                        _vl_pct_disp = int(round(_val_m.sum()   / _n_valid * 100))
                                        _ts_pct_disp = int(round(_test_m.sum()  / _n_valid * 100))
                                        st.caption(
                                            f"Примерное соотношение: "
                                            f"Train **{_tr_pct_disp}%** · "
                                            f"Val **{_vl_pct_disp}%** · "
                                            f"Test **{_ts_pct_disp}%**"
                                        )

                                _n_tr = int(_train_m.sum())
                                _n_vl = int(_val_m.sum())
                                _n_ts = int(_test_m.sum())
                                _sm1, _sm2, _sm3 = st.columns(3)
                                _sm1.metric("Train", f"{_n_tr:,} строк")
                                _sm2.metric("Val",   f"{_n_vl:,} строк")
                                _sm3.metric("Test",  f"{_n_ts:,} строк")

                                _sp_errs = []
                                if _n_tr == 0: _sp_errs.append("Train пустой")
                                if _n_vl == 0: _sp_errs.append("Val пустой")
                                if _n_ts == 0: _sp_errs.append("Test пустой")
                                if _split_mode == "По датам" and pd.Timestamp(_tr_end) >= pd.Timestamp(_vl_end):
                                    _sp_errs.append("Дата Train должна быть раньше даты Val")

                                if _sp_errs:
                                    st.error("❌ " + "; ".join(_sp_errs) + ". Скорректируйте параметры.")
                                else:
                                    if st.button("✅ Применить сплит", type="primary", key="apply_split_btn"):
                                        st.session_state["split_config"] = _new_cfg
                                        st.session_state["split_masks"] = {
                                            "train": _train_m.values.tolist(),
                                            "val":   _val_m.values.tolist(),
                                            "test":  _test_m.values.tolist(),
                                            "stats": {
                                                "n_train": _n_tr, "n_val": _n_vl, "n_test": _n_ts,
                                                "date_col": _split_date_col,
                                                "train_end": _new_cfg["train_end"],
                                                "val_end":   _new_cfg["val_end"],
                                            },
                                        }
                                        st.rerun()

                except Exception as _sp_exc:
                    st.error(f"Ошибка при обработке дат: {_sp_exc}")

            if _cur_split_masks:
                _ss = _cur_split_masks["stats"]
                st.info(
                    f"✅ Сплит активен по `{_ss['date_col']}`: "
                    f"Train **{_ss['n_train']:,}** · Val **{_ss['n_val']:,}** · "
                    f"Test **{_ss['n_test']:,}** строк "
                    f"(train до `{_ss['train_end']}`, val до `{_ss['val_end']}`)"
                )
                if st.button("🗑️ Сбросить сплит", key="reset_split_btn"):
                    st.session_state["split_masks"] = None
                    st.session_state["split_config"] = {}
                    st.rerun()

        st.divider()

        # ── Cheat sheet ───────────────────────────────────────────────────────
        with st.expander("📖 Шпаргалка: что выбрать?", expanded=False):
            st.markdown(f"""
<span style="color:{TEAL}">**Числовые (numerical)**</span>

| Метод | Когда использовать |
|---|---|
| **Z-score** ✅ | Универсальный выбор для нейросетей и TFT. Центрирует данные вокруг 0 |
| **Min-Max** | Когда нужен строго ограниченный диапазон [0, 1] |
| **Robust (IQR)** | Когда много выбросов: цены, трафик, объёмы |
| **log1p** | Сильно скошенное распределение (продажи, счётчики) |
| **sin/cos** | Цикличные признаки: час суток, день недели, месяц |
| **Удалить** | Колонка не нужна модели |

<span style="color:{TEAL}">**Категориальные (categorical) и текст (text)**</span>

| Метод | Когда использовать |
|---|---|
| **Label Enc** ✅ | Рекомендуется для TFT. Каждое уникальное значение → целое число |
| **One-Hot** | Для линейных моделей. Осторожно: при >30 уникальных значений добавляет много колонок |
| **Удалить** | Колонка не нужна модели |

<span style="color:{TEAL}">**Дата/время (datetime)**</span>

| Метод | Когда использовать |
|---|---|
| **Без изменений** ✅ | Дата уже разложена на компоненты на вкладке «🔗 Объединение» |
| **Удалить** | Исходная колонка с датой больше не нужна модели |

<span style="color:{TEAL}">**Бинарные (binary)**</span>

| Метод | Когда использовать |
|---|---|
| **Без изменений** ✅ | 0/1 уже в правильном формате — ничего делать не надо |
| **Удалить** | Колонка не нужна модели |

<span style="color:{TEAL}">**Заполнение пропусков (пре-fillna)**</span>

Заполнение пропусков — это **не метод**, а предшествующий шаг.
Галочка появляется только у колонок, которые имеют пропуски.
Если колонка имеет пропуски и выбран любой метод (Z-score, Label Enc и т.д.) —
сначала нужно заполнить пропуски, иначе метод может упасть с ошибкой.

| Тип колонки | Чем заполнять |
|---|---|
| Числовая | 0, среднее или медиана |
| Категориальная / текст | «Неизвестно», «N/A», пустая строка |
| Бинарная | 0 |

Кнопка **Fill NA** (пресет) ставит галочки всем колонкам с пропусками в группе сразу.
После простановки галочек укажите значение заполнения в секции ниже.
""", unsafe_allow_html=True)

        # ── Quick-edit table (grouped by type) ────────────────────────────────
        st.markdown(
            f'<p class="section-title">Быстрое редактирование методов</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Выберите метод в таблице и нажмите **✅ Применить** в нижней части каждой группы — "
            "без этого изменения не сохранятся в конфиг. "
            "Для колонок с пропусками отметьте галочку пре-fillna и укажите значение заполнения ниже."
        )

        prep_config = st.session_state["prep_config"]
        label_to_method = {v: k for k, v in METHOD_LABELS.items()}

        from collections import defaultdict as _dd
        type_groups: dict = _dd(list)
        for col in merged_df.columns:
            type_groups[col_types.get(col, "numerical")].append(col)

        for ctype, cols_in_group in type_groups.items():
            type_labels = [METHOD_LABELS.get(m, m) for m in METHOD_OPTIONS.get(ctype, ["none"])]
            badge = TYPE_BADGE.get(ctype, ctype)
            with st.expander(
                f"{ctype.capitalize()} — {len(cols_in_group)} колонок",
                expanded=True,
            ):
                st.markdown(badge, unsafe_allow_html=True)
                # Search is outside the form so it filters live
                grp_search = st.text_input(
                    "Поиск",
                    key=f"search_{ctype}",
                    placeholder="Фильтр по названию колонки...",
                    label_visibility="collapsed",
                )
                rows = []
                missing_in_view: list = []
                for col in cols_in_group:
                    if grp_search and grp_search.lower() not in col.lower():
                        continue
                    cfg = prep_config.get(col, {})
                    cur_label = METHOD_LABELS.get(cfg.get("method", "none"), METHOD_LABELS["none"])
                    if cur_label not in type_labels:
                        cur_label = type_labels[0]
                    n_miss = int(merged_df[col].isna().sum())
                    rows.append({
                        "Колонка": col,
                        "Метод": cur_label,
                        "Пропуски": f"{n_miss} ({n_miss / len(merged_df) * 100:.1f}%)" if n_miss else "—",
                    })
                    if n_miss > 0:
                        missing_in_view.append((col, n_miss))

                if not rows:
                    st.caption("Нет совпадений по запросу.")
                else:
                    # Wrap table + checkboxes in a form — no reruns until "Применить"
                    with st.form(f"form_{ctype}", border=False):
                        editor_key = f"editor_{ctype}" if not grp_search else f"editor_{ctype}_{grp_search}"
                        edited = st.data_editor(
                            pd.DataFrame(rows),
                            column_config={
                                "Колонка":  st.column_config.TextColumn("Колонка", disabled=True),
                                "Метод":    st.column_config.SelectboxColumn("Метод", options=type_labels, required=True),
                                "Пропуски": st.column_config.TextColumn("Пропуски", disabled=True),
                            },
                            hide_index=True,
                            key=editor_key,
                            width='stretch',
                        )

                        # Pre-fillna checkboxes — only for columns WITH missing values
                        if missing_in_view:
                            st.caption("Пре-fillna — отметьте колонки, где нужно заполнить пропуски перед методом:")
                            for col, n_miss in missing_in_view:
                                _chk_key = f"pf_chk_{col}"
                                # Initialize from prep_config only if key not yet in session_state
                                # (avoids Streamlit warning about conflicting value= and session_state)
                                if _chk_key not in st.session_state:
                                    st.session_state[_chk_key] = bool(
                                        prep_config.get(col, {}).get("pre_fillna", False)
                                    )
                                st.checkbox(
                                    f"**{col}** — {n_miss} пропусков",
                                    key=_chk_key,
                                )

                        if st.form_submit_button("✅ Применить", type="primary", use_container_width=True):
                            # Sync methods → prep_config only on submit
                            for _, row in edited.iterrows():
                                col = row["Колонка"]
                                method = label_to_method.get(row["Метод"], "none")
                                prep_config.setdefault(col, {"method": method, "params": {}})
                                prep_config[col]["method"] = method
                                prep_config[col].setdefault("params", {})
                            # Sync checkboxes (session_state committed on submit)
                            for col, n_miss in missing_in_view:
                                chk = bool(st.session_state.get(f"pf_chk_{col}", False))
                                prep_config.setdefault(col, {})
                                prep_config[col]["pre_fillna"] = chk
                            st.rerun()

        # ── Fill values for checked pre-fillna columns (auto-sync, no submit button) ──
        pf_checked = [
            col for col in merged_df.columns
            if prep_config.get(col, {}).get("pre_fillna", False)
            and merged_df[col].isna().any()
        ]
        if pf_checked:
            st.divider()
            st.markdown(
                f'<p class="section-title">Значения для заполнения пропусков</p>',
                unsafe_allow_html=True,
            )
            st.caption("Укажите значение, которым будут заполнены пропуски перед применением метода.")
            for col in pf_checked:
                ctype_pf = col_types.get(col, "numerical")
                n_miss_pf = int(merged_df[col].isna().sum())
                method_pf = prep_config.get(col, {}).get("method", "none")
                cfg_pf = prep_config.get(col, {})

                inp_left, inp_right = st.columns([4, 1])
                with inp_left:
                    if ctype_pf == "numerical":
                        val = st.number_input(
                            f"{col} — {n_miss_pf} пропусков",
                            value=float(cfg_pf.get("pre_fillna_value", 0.0)),
                            key=f"pf_val_{col}",
                        )
                    else:
                        val = st.text_input(
                            f"{col} — {n_miss_pf} пропусков",
                            value=str(cfg_pf.get("pre_fillna_value", "")),
                            placeholder="Неизвестно",
                            key=f"pf_val_{col}",
                        )
                        if str(val).strip() == "":
                            st.warning("Введите значение заполнения")
                with inp_right:
                    st.markdown(
                        f'<span style="color:{GRAY}; font-size:12px">'
                        f'→ {METHOD_LABELS.get(method_pf, method_pf)}'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                # Auto-sync: value saved to prep_config on every render
                prep_config.setdefault(col, {})
                prep_config[col]["pre_fillna_value"] = val

        # ── Extra params ──────────────────────────────────────────────────────
        NEEDS_PARAMS = {"cyclical", "onehot", "zscore", "minmax", "robust"}
        extra_cols = [
            col for col in merged_df.columns
            if prep_config.get(col, {}).get("method") in NEEDS_PARAMS
        ]

        if extra_cols:
            st.divider()
            st.markdown(
                f'<p class="section-title">Параметры выбранных методов</p>',
                unsafe_allow_html=True,
            )
            for col in extra_cols:
                ctype = col_types.get(col, "numerical")
                sel_method = prep_config[col]["method"]
                params = prep_config[col].setdefault("params", {})
                n_miss = int(merged_df[col].isna().sum())

                with st.expander(f"**{col}** [{ctype}] — {METHOD_LABELS[sel_method]}", expanded=False):
                    p_left, p_right = st.columns([1, 2])

                    with p_left:
                        if sel_method in ("zscore", "minmax", "robust"):
                            _active_masks = st.session_state.get("split_masks")
                            if _active_masks and len(_active_masks.get("train", [])) == len(merged_df):
                                _cur_fit = prep_config[col].get("fit_on", "train")
                                _fit_choice = st.radio(
                                    "Параметры нормировки вычислять по:",
                                    ["Только train (рекомендуется)", "Весь датасет"],
                                    index=0 if _cur_fit == "train" else 1,
                                    key=f"fit_on_{col}",
                                )
                                prep_config[col]["fit_on"] = (
                                    "train" if _fit_choice.startswith("Только") else "full"
                                )
                            else:
                                prep_config[col]["fit_on"] = "full"
                                if not _active_masks:
                                    st.caption("💡 Задайте временной сплит для вычисления параметров только по train.")

                        elif sel_method == "cyclical":
                            period_v = st.number_input(
                                "Период (напр. 24 для часов, 7 для дней, 52 для недель)",
                                value=float(params.get("period", 24.0)),
                                min_value=1.0,
                                key=f"cyclical_period_{col}",
                            )
                            prep_config[col]["params"]["period"] = period_v

                        elif sel_method == "onehot":
                            nu = merged_df[col].nunique()
                            if nu > 30:
                                st.warning(f"⚠️ {nu} уникальных значений — добавится {nu} новых колонок!")
                            else:
                                st.info(f"Добавится {nu} новых колонок.")

                    with p_right:
                        if ctype == "numerical" and sel_method in ("zscore", "minmax", "robust", "log1p"):
                            try:
                                tmp_df = merged_df[[col]].copy()
                                tmp_result, _, _ = apply_preprocessing(tmp_df, {col: {"method": sel_method, "params": {}}})
                                after_col = tmp_result[col] if col in tmp_result.columns else tmp_df[col]
                                st.plotly_chart(
                                    plot_before_after(merged_df[col], after_col, col),
                                    width='stretch',
                                )
                            except Exception:
                                pass

        # ── Value labels for numeric categorical / binary ──────────────────────
        def _collect_numeric_label_cols(target_type: str) -> list:
            result = []
            for _c in merged_df.columns:
                if col_types.get(_c) != target_type:
                    continue
                if prep_config.get(_c, {}).get("method") == "drop":
                    continue
                _u = merged_df[_c].dropna().unique()
                if len(_u) == 0:
                    continue
                try:
                    pd.to_numeric(pd.Series(_u))
                    result.append(_c)
                except Exception:
                    pass
            return result

        def _fmt_val(v: Any) -> str:
            try:
                fi = float(v)
                return str(int(fi)) if fi == int(fi) else str(fi)
            except Exception:
                return str(v)

        def _render_value_labels(cols: list, section_caption: str) -> None:
            st.caption(section_caption)
            for _vlc in cols:
                _vlc_ct = col_types.get(_vlc, "categorical")
                try:
                    _vlc_uniq_sorted = sorted(
                        merged_df[_vlc].dropna().unique(), key=lambda x: float(x)
                    )
                except Exception:
                    _vlc_uniq_sorted = list(merged_df[_vlc].dropna().unique())

                _cur_vl = prep_config.get(_vlc, {}).get("value_labels", {})

                # Pre-populate text_input keys from prep_config on first render
                for _v in _vlc_uniq_sorted:
                    _vk = _fmt_val(_v)
                    _inp_key = f"vl_inp_{_vlc}_{_vk}"
                    if _inp_key not in st.session_state:
                        st.session_state[_inp_key] = _cur_vl.get(_vk, _vk)

                _saved = bool(_cur_vl and any(
                    _cur_vl.get(_fmt_val(_v), _fmt_val(_v)) != _fmt_val(_v)
                    for _v in _vlc_uniq_sorted
                ))
                _vlc_method = prep_config.get(_vlc, {}).get("method", "none")
                _show_enc = _vlc_method == "label_enc"
                # LabelEncoder sorts by str representation — precompute the correct mapping
                if _show_enc:
                    _le_sorted = sorted(_vlc_uniq_sorted, key=lambda x: str(_fmt_val(x)))
                    _le_map = {_fmt_val(v): i for i, v in enumerate(_le_sorted)}

                with st.expander(
                    f"**{_vlc}** [{_vlc_ct}] — {len(_vlc_uniq_sorted)} значений"
                    + (" ✅" if _saved else ""),
                    expanded=False,
                ):
                    if _show_enc:
                        st.caption("Числовой код — результат Label Encoding.")
                    for _v in _vlc_uniq_sorted:
                        _vk = _fmt_val(_v)
                        if _show_enc:
                            _lc1, _lc2, _lc3 = st.columns([2, 1, 3])
                        else:
                            _lc1, _lc3 = st.columns([2, 3])
                        with _lc1:
                            st.markdown(f"`{_vk}`")
                        if _show_enc:
                            with _lc2:
                                st.markdown(f"→ **{_le_map[_vk]}**")
                        with _lc3:
                            st.text_input(
                                _vk,
                                key=f"vl_inp_{_vlc}_{_vk}",
                                label_visibility="collapsed",
                            )

                    if st.button("✅ Сохранить расшифровки", key=f"vl_save_{_vlc}",
                                 type="primary", width="stretch"):
                        prep_config.setdefault(_vlc, {})
                        prep_config[_vlc]["value_labels"] = {
                            _fmt_val(_v): st.session_state.get(
                                f"vl_inp_{_vlc}_{_fmt_val(_v)}", _fmt_val(_v)
                            )
                            for _v in _vlc_uniq_sorted
                        }
                        st.toast(f"Расшифровки для «{_vlc}» сохранены.", icon="✅")

        _vl_cat_cols = _collect_numeric_label_cols("categorical")
        _vl_bin_cols = _collect_numeric_label_cols("binary")

        if _vl_cat_cols or _vl_bin_cols:
            st.divider()

        if _vl_cat_cols:
            st.markdown(
                '<p class="section-title">Расшифровки категориальных значений</p>',
                unsafe_allow_html=True,
            )
            _render_value_labels(
                _vl_cat_cols,
                "Числовые коды категорий — задайте текстовые расшифровки. "
                "Используются в отчёте и файле обратных преобразований.",
            )

        if _vl_bin_cols:
            if _vl_cat_cols:
                st.divider()
            st.markdown(
                '<p class="section-title">Расшифровки бинарных значений</p>',
                unsafe_allow_html=True,
            )
            _render_value_labels(
                _vl_bin_cols,
                "Задайте расшифровку для 0 и 1 (или других бинарных значений). "
                "Используются в отчёте и файле обратных преобразований.",
            )

        st.divider()

        if st.button("▶ Применить препроцессинг", type="primary"):
            # Validate: non-numerical pre-fillna columns must have a non-empty fill value
            _pf_empty = [
                col for col in merged_df.columns
                if prep_config.get(col, {}).get("pre_fillna", False)
                and merged_df[col].isna().any()
                and col_types.get(col, "numerical") != "numerical"
                and str(prep_config.get(col, {}).get("pre_fillna_value", "")).strip() == ""
            ]
            if _pf_empty:
                st.error(
                    f"Укажите значение заполнения для: **{', '.join(_pf_empty)}**. "
                    f"Пустое значение недопустимо."
                )
            else:
                try:
                    _masks = st.session_state.get("split_masks")
                    _train_mask_ap = None
                    if _masks and len(_masks.get("train", [])) == len(merged_df):
                        _train_mask_ap = pd.Series(
                            _masks["train"], index=merged_df.index, dtype=bool
                        )
                    elif _masks:
                        st.warning(
                            "⚠️ Размер сплита не совпадает с датасетом — "
                            "параметры нормировки вычислены по всему датасету. "
                            "Переопределите сплит."
                        )
                    prep_df, log, _inv_params = apply_preprocessing(
                        merged_df, prep_config, train_mask=_train_mask_ap
                    )
                    st.session_state["prep_df"] = prep_df
                    st.session_state["inverse_params"] = _inv_params
                    # Freeze configs as they were at the moment preprocessing ran
                    st.session_state["applied_prep_config"]  = copy.deepcopy(prep_config)
                    st.session_state["applied_split_config"] = copy.deepcopy(
                        st.session_state.get("split_config", {})
                    )
                    reset_tft_roles()
                    st.success(
                        f"Готово: {len(prep_df):,} строк × {len(prep_df.columns)} колонок."
                    )

                    rm1, rm2 = st.columns(2)
                    rm1.metric("Строк", f"{len(prep_df):,}")
                    rm2.metric("Колонок", len(prep_df.columns))

                    with st.expander("📋 Журнал изменений", expanded=True):
                        if log:
                            for entry in log:
                                st.markdown(f"- {entry}")
                        else:
                            st.info("Изменений не было (все методы — «Без изменений»).")

                    st.dataframe(prep_df.head(10), width='stretch')
                except Exception as exc:
                    import traceback
                    st.error(f"**{type(exc).__name__}:** {exc}")
                    with st.expander("📋 Подробности ошибки"):
                        st.code(traceback.format_exc(), language="python")

        elif st.session_state.get("prep_df") is not None:
            prep_df = st.session_state["prep_df"]
            st.success(
                f"Последний результат: {len(prep_df):,} строк × {len(prep_df.columns)} колонок."
            )
            st.dataframe(prep_df.head(10), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TFT + Export
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">🏷️ Распределение ролей TFT + Экспорт</p>', unsafe_allow_html=True)

    # ── Status bar ────────────────────────────────────────────────────────────
    _t5_merged = st.session_state.get("merged_df")
    _t5_prep   = st.session_state.get("prep_df")
    _t5_disp   = _t5_prep if _t5_prep is not None else _t5_merged
    if _t5_disp is not None:
        _t5_src  = "из сохранённых файлов" if st.session_state.get("load_mode") == "saved_files" else "через вкладки 1–2"
        st.success(
            f"Загружено ({_t5_src}): {len(_t5_disp):,} строк × {len(_t5_disp.columns)} колонок"
            + (" · препроцессинг применён" if _t5_prep is not None else "")
        )

    # ── Load from saved session folder ────────────────────────────────────────
    # After saving a new session the save handler stores the folder name in
    # _t5_pending_session_select.  We apply it here — before the selectbox
    # widget is rendered — to avoid the "cannot modify after instantiation" error.
    _pending_sel = st.session_state.pop("_t5_pending_session_select", None)
    if _pending_sel is not None:
        st.session_state["load_session_select"] = _pending_sel

    # Session-state toggle — replaces st.expander whose expanded= param forces state
    # on every rerender, causing it to close when the folder selectbox is changed.
    if st.session_state.get("_t5_load_exp_open") is None:
        st.session_state["_t5_load_exp_open"] = (_t5_disp is None)
    _lexp_open = bool(st.session_state["_t5_load_exp_open"])

    _lexp_arrow = "▲" if _lexp_open else "▼"
    if st.button(
        f"{_lexp_arrow} 📂 Загрузить сохранённую сессию",
        key="t5_load_toggle",
        use_container_width=True,
    ):
        st.session_state["_t5_load_exp_open"] = not _lexp_open
        st.rerun()

    if _lexp_open:
        with st.container(border=True):
            _project_root_load = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            _exports_dir = os.path.join(_project_root_load, "exports")

            if not os.path.exists(_exports_dir):
                st.info(
                    "Папка exports/ не найдена. "
                    "Сначала сохраните сессию кнопкой «💾 Сохранить всё в папку проекта»."
                )
            else:
                _available = sorted(
                    [d for d in os.listdir(_exports_dir)
                     if os.path.isdir(os.path.join(_exports_dir, d))],
                    reverse=True,
                )
                if not _available:
                    st.info(
                        "В папке exports/ нет сохранённых сессий. "
                        "Нажмите «💾 Сохранить всё в папку проекта» чтобы создать первую."
                    )
                else:
                    st.caption(
                        "Выберите папку из exports/ — дашборд автоматически восстановит "
                        "все вкладки так, как будто работа велась с нуля."
                    )
                    def _keep_load_panel_open():
                        st.session_state["_t5_load_exp_open"] = True

                    _sel_session = st.selectbox(
                        "Сохранённая сессия",
                        options=_available,
                        key="load_session_select",
                        on_change=_keep_load_panel_open,
                    )
                    _session_path = os.path.join(_exports_dir, _sel_session)

                    # File inventory — mandatory (❌ if missing) vs optional (— if missing)
                    _mandatory_files = (
                        "processed_data.csv",
                        "tft_config.json",
                        "prep_config.json",
                        "session_config.json",
                    )
                    _optional_files = (
                        "inverse_transforms.pkl",
                        "merged_data.csv",
                        "data_prep_report.md",
                        "split_config.json",
                    )
                    # status: "ok" | "err" (mandatory missing) | "opt" (optional missing)
                    _inv_rows = []
                    for _fn in _mandatory_files:
                        _exists = os.path.exists(os.path.join(_session_path, _fn))
                        _inv_rows.append(("ok" if _exists else "err", _fn, "обязательный"))
                    for _fn in _optional_files:
                        _exists = os.path.exists(os.path.join(_session_path, _fn))
                        _inv_rows.append(("ok" if _exists else "opt", _fn, "опциональный"))
                    _src_dir_inv = os.path.join(_session_path, "source_files")
                    _n_src_inv = len(os.listdir(_src_dir_inv)) if os.path.exists(_src_dir_inv) else 0
                    _inv_rows.append((
                        "ok" if _n_src_inv > 0 else "opt",
                        f"source_files/ ({_n_src_inv} файлов)",
                        "опциональный",
                    ))

                    _n_missing_mandatory = sum(1 for s, _, _ in _inv_rows if s == "err")
                    with st.expander(
                        f"📋 Содержимое папки"
                        + (f" — ⚠️ {_n_missing_mandatory} обязательных файла отсутствуют"
                           if _n_missing_mandatory else ""),
                        expanded=False,
                    ):
                        for _inv_st, _inv_name, _inv_role in _inv_rows:
                            if _inv_st == "ok":
                                st.markdown(f"✅ `{_inv_name}`")
                            elif _inv_st == "err":
                                st.markdown(
                                    f'❌ <span style="color:{RED}; font-family:monospace; font-size:0.9em">'
                                    f'{_inv_name}</span>'
                                    f' <span style="color:{RED}; font-size:0.8em">({_inv_role}, отсутствует)</span>',
                                    unsafe_allow_html=True,
                                )
                            else:  # opt
                                st.markdown(
                                    f'<span style="color:{GRAY}">—</span> '
                                    f'<span style="color:{GRAY}; font-family:monospace; font-size:0.9em">'
                                    f'{_inv_name}</span>'
                                    f' <span style="color:{GRAY}; font-size:0.8em">({_inv_role}, не сохранён)</span>',
                                    unsafe_allow_html=True,
                                )

                    if st.button("✅ Загрузить сессию", type="primary", key="apply_sf", width="stretch"):
                        # ── Mandatory-file check (all four must exist) ─────────
                        _mandatory = {
                            "processed_data.csv":  os.path.join(_session_path, "processed_data.csv"),
                            "tft_config.json":     os.path.join(_session_path, "tft_config.json"),
                            "prep_config.json":    os.path.join(_session_path, "prep_config.json"),
                            "session_config.json": os.path.join(_session_path, "session_config.json"),
                        }
                        _missing = [name for name, path in _mandatory.items() if not os.path.exists(path)]
                        if _missing:
                            st.error(
                                "Папка неполная — отсутствуют обязательные файлы: "
                                + ", ".join(f"`{f}`" for f in _missing)
                                + ". Загрузите полную папку из exports/."
                            )
                        else:
                            _load_errors: list = []

                            # 1 ── session_config.json
                            _sess_cfg: dict = {}
                            try:
                                with open(_mandatory["session_config.json"], "r", encoding="utf-8") as _f:
                                    _sess_cfg = json.load(_f)
                            except Exception as _e:
                                _load_errors.append(f"session_config.json: {_e}")

                            # 2 ── processed_data.csv
                            _proc_df_r = None
                            try:
                                _proc_df_r = pd.read_csv(_mandatory["processed_data.csv"], encoding="utf-8")
                                if len(_proc_df_r) == 0 or len(_proc_df_r.columns) == 0:
                                    _load_errors.append("processed_data.csv пустой.")
                                    _proc_df_r = None
                            except Exception as _e:
                                _load_errors.append(f"processed_data.csv: {_e}")

                            # 3 ── tft_config.json
                            _tft_data_r: dict = {}
                            try:
                                with open(_mandatory["tft_config.json"], "r", encoding="utf-8") as _f:
                                    _tft_data_r = json.load(_f)
                            except Exception as _e:
                                _load_errors.append(f"tft_config.json: {_e}")

                            # 4 ── prep_config.json
                            _prep_data_r: dict = {}
                            try:
                                with open(_mandatory["prep_config.json"], "r", encoding="utf-8") as _f:
                                    _prep_data_r = json.load(_f)
                            except Exception as _e:
                                _load_errors.append(f"prep_config.json: {_e}")

                            # 5 ── inverse_transforms.pkl (optional)
                            _inv_r: dict = {}
                            _inv_pkl_path = os.path.join(_session_path, "inverse_transforms.pkl")
                            if os.path.exists(_inv_pkl_path):
                                try:
                                    import pickle as _pkl_r
                                    with open(_inv_pkl_path, "rb") as _f:
                                        _inv_r = _pkl_r.load(_f)
                                except Exception as _e:
                                    _load_errors.append(f"inverse_transforms.pkl: {_e}")

                            # 6 ── split_config.json (optional)
                            _split_cfg_r: dict = {}
                            _split_cfg_path = os.path.join(_session_path, "split_config.json")
                            if os.path.exists(_split_cfg_path):
                                try:
                                    with open(_split_cfg_path, "r", encoding="utf-8") as _f:
                                        _split_cfg_r = json.load(_f)
                                except Exception as _e:
                                    _load_errors.append(f"split_config.json: {_e}")

                            # 7 ── merged_data.csv (optional)
                            _merged_df_r = None
                            _merged_path = os.path.join(_session_path, "merged_data.csv")
                            if os.path.exists(_merged_path):
                                try:
                                    _merged_df_r = pd.read_csv(_merged_path, encoding="utf-8")
                                    _merged_df_r = filter_merged_duplicates(_merged_df_r, _sess_cfg)
                                except Exception as _e:
                                    _load_errors.append(f"merged_data.csv: {_e}")

                            # 8 ── source_files/ (optional)
                            _src_loaded: dict = {}
                            _src_dir_path = os.path.join(_session_path, "source_files")
                            if os.path.exists(_src_dir_path):
                                _src_order = _sess_cfg.get("source_files_order") or sorted(os.listdir(_src_dir_path))
                                for _sfn in _src_order:
                                    _sfp = os.path.join(_src_dir_path, _sfn)
                                    if os.path.exists(_sfp):
                                        try:
                                            _src_loaded[_sfn] = pd.read_csv(_sfp, encoding="utf-8")
                                        except Exception as _e:
                                            _load_errors.append(f"source_files/{_sfn}: {_e}")

                            if _load_errors:
                                for _lerr in _load_errors:
                                    st.error(_lerr)
                            else:
                                # ── Restore session state ──────────────────────
                                st.session_state["uploaded_files"]          = _src_loaded
                                st.session_state["_uploader_was_populated"] = False

                                _prep_applied = _sess_cfg.get("preprocessing_applied", True)
                                if _prep_applied:
                                    st.session_state["merged_df"] = _merged_df_r
                                    st.session_state["prep_df"]   = _proc_df_r
                                else:
                                    st.session_state["merged_df"] = (
                                        _merged_df_r if _merged_df_r is not None else _proc_df_r
                                    )
                                    st.session_state["prep_df"] = None

                                st.session_state["prep_config"]         = _prep_data_r
                                st.session_state["applied_prep_config"] = _prep_data_r
                                st.session_state["inverse_params"]      = _inv_r

                                # Restore temporal split (recompute masks from merged_df)
                                if _split_cfg_r:
                                    st.session_state["split_config"]         = _split_cfg_r
                                    st.session_state["applied_split_config"] = _split_cfg_r
                                    if _split_cfg_r.get("mode") == "date":
                                        st.session_state["split_mode_radio"] = "По датам"
                                    _ref_df = _merged_df_r if _merged_df_r is not None else _proc_df_r
                                    _recomputed = compute_split_masks(_ref_df, _split_cfg_r) if _ref_df is not None else None
                                    if _recomputed:
                                        st.session_state["split_masks"] = _recomputed
                                    else:
                                        st.session_state["split_masks"] = None
                                        st.warning(
                                            "⚠️ Параметры сплита загружены, но маски не удалось пересчитать "
                                            f"(колонка `{_split_cfg_r.get('date_col')}` не найдена или данные изменились). "
                                            "Перейдите на вкладку 4 и нажмите «✅ Применить сплит» повторно."
                                        )
                                else:
                                    st.session_state["split_config"]         = {}
                                    st.session_state["split_masks"]          = None
                                    st.session_state["applied_split_config"] = None

                                st.session_state["load_mode"]        = "saved_files"
                                st.session_state["load_choice_used"] = "full"

                                _saved_ctypes = _sess_cfg.get("col_types", {})
                                if _saved_ctypes:
                                    st.session_state["col_types"] = _saved_ctypes
                                elif _merged_df_r is not None:
                                    st.session_state["col_types"] = detect_col_types(_merged_df_r)
                                else:
                                    st.session_state["col_types"] = {}

                                _afc = _sess_cfg.get("added_col_formulas", {})
                                st.session_state["added_col_formulas"] = _afc
                                st.session_state["added_cols"]         = list(_afc.keys())
                                _tidx_r = _sess_cfg.get("tidx_config", {})
                                st.session_state["tidx_config"]        = _tidx_r
                                _dtx_r = _sess_cfg.get("dtx_configs", [])
                                st.session_state["dtx_configs"] = _dtx_r
                                if _dtx_r:
                                    st.session_state["_pending_dtx_src_col"] = _dtx_r[-1]["src_col"]

                                _merged_cols_r = _merged_df_r.columns.tolist() if _merged_df_r is not None else []
                                if _tidx_r.get("src_col") and _tidx_r["src_col"] in _merged_cols_r:
                                    st.session_state["_pending_tidx_src_col"] = _tidx_r["src_col"]
                                if _tidx_r.get("gran"):
                                    st.session_state["_pending_tidx_gran"] = _tidx_r["gran"]
                                if _tidx_r.get("name"):
                                    st.session_state["_pending_tidx_name"] = _tidx_r["name"]

                                _fs_r = _sess_cfg.get("file_settings", {})
                                if _fs_r.get("sep"):
                                    st.session_state["_pending_file_sep"] = _fs_r["sep"]
                                if _fs_r.get("enc"):
                                    st.session_state["_pending_file_enc"] = _fs_r["enc"]
                                if _fs_r.get("dec"):
                                    st.session_state["_pending_file_dec"] = _fs_r["dec"]
                                if _fs_r.get("detected_encs"):
                                    st.session_state["_detected_encs"] = _fs_r["detected_encs"]

                                for _mi, _mc in enumerate(_sess_cfg.get("merge_configs", [])):
                                    if _mc.get("on"):
                                        st.session_state[f"_pending_join_col_{_mi}"] = _mc["on"]
                                    if _mc.get("how"):
                                        st.session_state[f"_pending_join_how_{_mi}"] = _mc["how"]

                                _ref_df = _proc_df_r if _proc_df_r is not None else _merged_df_r
                                if _tft_data_r and _ref_df is not None:
                                    _cset_r = set(_ref_df.columns)
                                    def _filt_r(v):
                                        if v is None: return None
                                        if isinstance(v, list): return [c for c in v if c in _cset_r]
                                        return v if v in _cset_r else None
                                    st.session_state["tft_roles"] = {
                                        "time_col":    _filt_r(_tft_data_r.get("time_col")),
                                        "group_col":   _filt_r(_tft_data_r.get("group_col")),
                                        "target":      _filt_r(_tft_data_r.get("target", [])),
                                        "static_cat":  _filt_r(_tft_data_r.get("static_cats", [])),
                                        "static_real": _filt_r(_tft_data_r.get("static_reals", [])),
                                        "known_cat":   _filt_r(_tft_data_r.get("time_varying_known_categoricals", [])),
                                        "known_real":  _filt_r(_tft_data_r.get("time_varying_known_reals", [])),
                                        "unknown_real":_filt_r(_tft_data_r.get("time_varying_unknown_reals", [])),
                                        "dropped":     _filt_r(_tft_data_r.get("dropped", [])),
                                    }

                                st.session_state["tft_reset_v"] = st.session_state.get("tft_reset_v", 0) + 1
                                st.session_state["_t5_load_exp_open"] = False  # Close panel

                                _parts = []
                                if _src_loaded:
                                    _parts.append(f"{len(_src_loaded)} исходных файлов")
                                if _merged_df_r is not None:
                                    _parts.append(f"merged_data ({len(_merged_df_r):,}×{len(_merged_df_r.columns)})")
                                if _proc_df_r is not None:
                                    _parts.append(f"processed_data ({len(_proc_df_r):,}×{len(_proc_df_r.columns)})")
                                if _tft_data_r:
                                    _parts.append("роли TFT")
                                if _prep_data_r:
                                    _parts.append("конфиг препроцессинга")
                                st.success("Загружено: " + " · ".join(_parts))
                                st.rerun()

    st.divider()

    # ── Clear All with confirmation ────────────────────────────────────────────
    if st.session_state.get("t5_confirm_clear"):
        st.warning("Все данные, препроцессинг и роли TFT будут сброшены. Продолжить?")
        _cc1, _cc2, _cc3 = st.columns([2, 2, 6])
        with _cc1:
            if st.button("Да, сбросить", type="primary", key="t5_clear_confirm"):
                st.session_state["upload_key"] += 1
                st.session_state["uploaded_files"] = {}
                st.session_state["saved_files_upload_key"] = (
                    st.session_state.get("saved_files_upload_key", 0) + 1
                )
                st.session_state.pop("t5_confirm_clear", None)
                st.session_state["_t5_load_exp_open"] = True  # Re-open load panel
                reset_downstream()
                st.rerun()
        with _cc2:
            if st.button("Отмена", key="t5_clear_cancel"):
                st.session_state.pop("t5_confirm_clear", None)
                st.rerun()
    else:
        _, _t5_clear_col = st.columns([5, 1])
        with _t5_clear_col:
            if st.button("🗑️ Очистить всё", key="t5_clear_init", width="stretch"):
                st.session_state["t5_confirm_clear"] = True
                st.rerun()

    _prep = st.session_state.get("prep_df")
    work_df = _prep if _prep is not None else st.session_state.get("merged_df")

    if work_df is None:
        st.info("Загрузите данные через панель выше или через вкладки «📂 Файлы» и «🔗 Объединение».")
    else:
        # Exclude original columns that were replaced by derived versions after preprocessing
        _pcfg = st.session_state.get("prep_config", {})
        _superseded: set = set()
        for _col, _cfg in _pcfg.items():
            if not isinstance(_cfg, dict):
                continue
            _m = _cfg.get("method", "none")
            if _m == "label_enc" and f"{_col}_enc" in work_df.columns:
                _superseded.add(_col)
            # cyclical and onehot: original is already absent from work_df

        all_cols = [c for c in work_df.columns if c not in _superseded]
        roles = st.session_state["tft_roles"]
        col_types = st.session_state.get("col_types", {})

        # Version suffix — changed by reset_tft_roles() to force widget recreation
        trv = st.session_state.get("tft_reset_v", 0)

        # ── Service columns ───────────────────────────────────────────────────
        tc_options = [None] + all_cols
        gc_options = [None] + all_cols

        sc1, sc2 = st.columns(2)
        with sc1:
            tc_idx = tc_options.index(roles.get("time_col")) if roles.get("time_col") in tc_options else 0
            time_col = st.selectbox(
                "⏱️ Временная колонка (time_idx / дата)",
                options=tc_options,
                index=tc_idx,
                key=f"tft_time_col_{trv}",
            )
            st.session_state["tft_roles"]["time_col"] = time_col

        with sc2:
            gc_idx = gc_options.index(roles.get("group_col")) if roles.get("group_col") in gc_options else 0
            group_col = st.selectbox(
                "🏢 Группирующая колонка (group_id / station_id)",
                options=gc_options,
                index=gc_idx,
                key=f"tft_group_col_{trv}",
            )
            st.session_state["tft_roles"]["group_col"] = group_col

        avail_cols = [c for c in all_cols if c not in (time_col, group_col)]

        st.divider()

        # ── TFT roles ─────────────────────────────────────────────────────────
        ROLE_DEFS = [
            ("target",       "🎯 Целевые переменные",                             "Что предсказывает модель"),
            ("static_cat",   "🏷️ Статические категориальные (STATIC_CATS)",       "Неизменные категории объекта"),
            ("static_real",  "📌 Статические вещественные (STATIC_REALS)",         "Неизменные числа объекта"),
            ("known_cat",    "📅 Известные будущие категориальные (KNOWN_CATS)",    "Категории, известные на горизонте"),
            ("known_real",   "📈 Известные будущие вещественные (KNOWN_REALS)",     "Числа, известные на горизонте"),
            ("unknown_real", "🔮 Наблюдаемые прошлые (UNKNOWN_REALS)",             "Числа только в прошлом (энкодер)"),
            ("dropped",      "🗑️ Исключить из модели",                             "Служебные, дублирующие колонки"),
        ]

        type_suggestions: dict = {
            "target":       [c for c in avail_cols if col_types.get(c) == "numerical"],
            "static_cat":   [c for c in avail_cols if col_types.get(c) in ("categorical", "binary")],
            "static_real":  [c for c in avail_cols if col_types.get(c) == "numerical"],
            "known_cat":    [c for c in avail_cols if col_types.get(c) in ("categorical", "binary")],
            "known_real":   [c for c in avail_cols if col_types.get(c) == "numerical"],
            "unknown_real": [c for c in avail_cols if col_types.get(c) == "numerical"],
            "dropped":      [c for c in avail_cols if col_types.get(c) == "text"],
        }

        # Pre-collect ALL current role values from widget state BEFORE rendering any
        # multiselect — so upper roles already see what lower roles currently hold.
        _role_snapshot: dict = {}
        for rk, _, _ in ROLE_DEFS:
            _wk = f"tft_role_{rk}_{trv}"
            _role_snapshot[rk] = list(
                st.session_state[_wk]
                if _wk in st.session_state
                else (st.session_state["tft_roles"].get(rk) or [])
            )

        for role_key, role_title, role_desc in ROLE_DEFS:
            with st.expander(role_title, expanded=False):
                st.markdown(f'<span style="color:{GRAY}">{role_desc}</span>', unsafe_allow_html=True)
                # Exclude cols already assigned to other roles (mutual exclusion)
                _other_assigned: set = set()
                for _other_key, _, _ in ROLE_DEFS:
                    if _other_key != role_key:
                        _other_assigned.update(_role_snapshot[_other_key])
                _base_opts = [c for c in avail_cols if c not in _other_assigned]
                # known_real is the only role that may include time_col
                if role_key == "known_real" and time_col and time_col not in _base_opts:
                    role_options = [time_col] + _base_opts
                else:
                    role_options = _base_opts
                current_vals = [v for v in _role_snapshot[role_key] if v in role_options]
                hint = ", ".join(type_suggestions.get(role_key, [])) or "—"
                sel = st.multiselect(
                    "Выберите колонки",
                    options=role_options,
                    default=current_vals,
                    help=f"Рекомендованы по типу: {hint}",
                    key=f"tft_role_{role_key}_{trv}",
                )
                st.session_state["tft_roles"][role_key] = sel

        # ── Coverage metrics ──────────────────────────────────────────────────
        st.divider()
        assigned: set = set()
        for rk, _, _ in ROLE_DEFS:
            assigned.update(st.session_state["tft_roles"].get(rk, []))
        not_assigned = [c for c in avail_cols if c not in assigned]

        cov1, cov2, cov3 = st.columns(3)
        _total_cols = len(avail_cols) + (1 if time_col and time_col in assigned else 0)
        cov1.metric("Всего колонок", _total_cols)
        cov2.metric("Распределено", len(assigned))
        cov3.metric("Не распределено", len(not_assigned))

        if not_assigned:
            st.warning("Нераспределённые колонки: " + ", ".join(not_assigned))

        # ── Readiness check ───────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-title">Готовность к загрузке в модель</p>', unsafe_allow_html=True)

        _roles_now = st.session_state["tft_roles"]
        _prep_done = st.session_state.get("prep_df") is not None
        _tc = _roles_now.get("time_col")
        # time_col must also appear in known_real for pytorch-forecasting TFT
        _time_col_in_known_real = bool(_tc) and _tc in _roles_now.get("known_real", [])

        _checks = [
            # (ok, required, label)
            (bool(_tc),                                       True,  "Задана временная колонка (time_col)"),
            (bool(_roles_now.get("group_col")),               True,  "Задана группирующая колонка (group_col)"),
            (len(_roles_now.get("target", [])) > 0,           True,  "Задана хотя бы одна целевая переменная (target)"),
            (_time_col_in_known_real,                         False, "Временная колонка добавлена в «known_real» — требуется pytorch-forecasting"),
            (len(_roles_now.get("unknown_real", [])) > 0
             or len(_roles_now.get("known_real", [])) > 1
             or len(_roles_now.get("known_cat", [])) > 0,    False, "Есть ковариаты помимо time_col (known_real / unknown_real / known_cat)"),
            (_prep_done,                                      False, "Препроцессинг применён (вкладка «⚙️ Препроцессинг»)"),
            (len(not_assigned) == 0,                         False, "Все колонки распределены по ролям"),
        ]

        for _ok, _req, _label in _checks:
            if _ok:
                _icon, _color = "✅", GREEN
            elif _req:
                _icon, _color = "❌", RED
            else:
                _icon, _color = "⚠️", GOLD
            st.markdown(
                f'<span style="color:{_color}">{_icon}&nbsp; {_label}</span>',
                unsafe_allow_html=True,
            )

        _errors   = [lbl for ok, req, lbl in _checks if not ok and req]
        _warnings = [lbl for ok, req, lbl in _checks if not ok and not req]

        st.markdown("")
        if _errors:
            st.error(
                f"**{len(_errors)} обязательных пункта не выполнены.** "
                "Исправьте перед использованием в модели."
            )
        elif _warnings:
            st.warning(
                "Обязательные поля заполнены — можно экспортировать. "
                "Выполните рекомендации для повышения качества модели."
            )
        else:
            st.success(
                "Все проверки пройдены — конфигурация готова к экспорту. "
                "Дашборд проверяет структуру ролей, но не гарантирует корректность данных для конкретной задачи."
            )

        # ── TFT config preview ────────────────────────────────────────────────
        tft_config = build_tft_config(st.session_state["tft_roles"])

        with st.expander("👁️ Просмотр TFT-конфига", expanded=False):
            st.json(tft_config)

        # ── Export ────────────────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-title">Экспорт</p>', unsafe_allow_html=True)

        def _report_md() -> str:
            return build_report(
                uploaded=st.session_state.get("uploaded_files", {}),
                merged=st.session_state.get("merged_df"),
                prep_df=st.session_state.get("prep_df"),
                col_types=st.session_state.get("col_types", {}),
                prep_config=st.session_state.get("applied_prep_config") or st.session_state.get("prep_config", {}),
                tft_roles=st.session_state.get("tft_roles", {}),
                method_labels=METHOD_LABELS,
                split_config=st.session_state.get("applied_split_config") or None,
                inverse_params=st.session_state.get("inverse_params") or None,
            )

        # ── Row 1: data files ─────────────────────────────────────────────────
        _ex_merged = st.session_state.get("merged_df")
        ex1, ex2, ex3 = st.columns(3)

        with ex1:
            _buf_proc = io.StringIO()
            work_df.to_csv(_buf_proc, index=False, encoding="utf-8")
            st.download_button(
                "⬇️ processed_data.csv",
                data=_buf_proc.getvalue().encode("utf-8"),
                file_name="processed_data.csv",
                mime="text/csv",
                width='stretch',
            )
            st.caption(
                f"Датасет после препроцессинга "
                f"({len(work_df):,} стр × {len(work_df.columns)} кол). "
                "Подаётся напрямую в модель."
            )

        with ex2:
            if _ex_merged is not None:
                _buf_merged = io.StringIO()
                _ex_merged.to_csv(_buf_merged, index=False, encoding="utf-8")
                st.download_button(
                    "⬇️ merged_data.csv",
                    data=_buf_merged.getvalue().encode("utf-8"),
                    file_name="merged_data.csv",
                    mime="text/csv",
                    width='stretch',
                )
                st.caption(
                    f"Объединённый датасет до препроцессинга "
                    f"({len(_ex_merged):,} стр × {len(_ex_merged.columns)} кол). "
                    "Нужен для полного восстановления сессии."
                )
            else:
                st.button("⬇️ merged_data.csv", disabled=True, width='stretch')
                st.caption("Недоступно: объединённый датафрейм ещё не создан.")

        with ex3:
            _src_names_ex = list(st.session_state.get("uploaded_files", {}).keys())
            _n_pairs_ex = max(0, len(_src_names_ex) - 1)
            _merge_cfgs_ex = [
                {
                    "right_name": _src_names_ex[_si + 1],
                    "on": st.session_state.get(f"join_col_{_si}"),
                    "how": st.session_state.get(f"join_how_{_si}", "left"),
                }
                for _si in range(_n_pairs_ex)
            ]
            _sess_cfg_ex = build_session_config(
                source_file_names=_src_names_ex,
                merge_configs=_merge_cfgs_ex,
                added_col_formulas=st.session_state.get("added_col_formulas", {}),
                tidx_config=st.session_state.get("tidx_config", {}),
                dtx_configs=st.session_state.get("dtx_configs", []),
                col_types=st.session_state.get("col_types", {}),
                preprocessing_applied=st.session_state.get("prep_df") is not None,
                file_settings={
                    "sep": st.session_state.get("file_sep", ","),
                    "enc": st.session_state.get("file_enc", "auto"),
                    "dec": st.session_state.get("file_dec", "."),
                    "detected_encs": st.session_state.get("_detected_encs", {}),
                },
            )
            st.download_button(
                "⬇️ session_config.json",
                data=json.dumps(_sess_cfg_ex, ensure_ascii=False, indent=2, cls=_NpEncoder).encode("utf-8"),
                file_name="session_config.json",
                mime="application/json",
                width='stretch',
            )
            st.caption(
                "Конфиг сессии: порядок файлов, конфиг объединения, "
                "формулы вычисляемых колонок, конфиг time_idx, типы колонок. "
                "Нужен для полного восстановления сессии."
            )

        # ── Row 2: config files ───────────────────────────────────────────────
        ex4, ex5, ex6 = st.columns(3)

        with ex4:
            st.download_button(
                "⬇️ tft_config.json",
                data=json.dumps(tft_config, ensure_ascii=False, indent=2, cls=_NpEncoder).encode("utf-8"),
                file_name="tft_config.json",
                mime="application/json",
                width='stretch',
            )
            st.caption(
                "Роли TFT: time_col, group_col, target, static_cats, known_reals и т.д. "
                "+ методы препроцессинга."
            )

        with ex5:
            _prep_export = st.session_state.get("applied_prep_config") or st.session_state.get("prep_config", {})
            st.download_button(
                "⬇️ prep_config.json",
                data=json.dumps(_prep_export, ensure_ascii=False, indent=2, cls=_NpEncoder).encode("utf-8"),
                file_name="prep_config.json",
                mime="application/json",
                width='stretch',
            )
            st.caption(
                "Метод и параметры препроцессинга для каждой колонки, "
                "настройки заполнения пропусков."
            )

        with ex6:
            st.download_button(
                "⬇️ data_prep_report.md",
                data=_report_md().encode("utf-8"),
                file_name="data_prep_report.md",
                mime="text/markdown",
                width='stretch',
            )
            st.caption(
                "Markdown-отчёт: исходные файлы, типы колонок, "
                "методы препроцессинга, роли TFT, итоговая статистика."
            )

        # ── Row 3: inverse transforms + split config ──────────────────────────
        _inv_params_dl  = st.session_state.get("inverse_params", {})
        _split_cfg_dl   = st.session_state.get("applied_split_config") or {}
        _prep_active    = st.session_state.get("prep_df") is not None
        _show_inv       = bool(_inv_params_dl) and _prep_active
        _show_split     = bool(_split_cfg_dl) and _prep_active

        if _show_inv or _show_split:
            ex7, ex8, _ = st.columns(3)
            if _show_inv:
                with ex7:
                    import pickle as _pickle
                    try:
                        _inv_bytes = _pickle.dumps(_inv_params_dl)
                        st.download_button(
                            "⬇️ inverse_transforms.pkl",
                            data=_inv_bytes,
                            file_name="inverse_transforms.pkl",
                            mime="application/octet-stream",
                            width='stretch',
                        )
                        st.caption(
                            "Параметры для обратного преобразования: метод, "
                            "подобранные значения (mean/std, min/max, классы и т.д.), "
                            "источник подбора (train/full)."
                        )
                    except Exception as _pkl_e:
                        st.error(f"Ошибка сериализации inverse_transforms: {_pkl_e}")
            if _show_split:
                with ex8:
                    st.download_button(
                        "⬇️ split_config.json",
                        data=json.dumps(_split_cfg_dl, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="split_config.json",
                        mime="application/json",
                        width='stretch',
                    )
                    _sc = _split_cfg_dl
                    st.caption(
                        f"Параметры временного сплита: колонка `{_sc.get('date_col', '—')}`, "
                        f"train до `{_sc.get('train_end', '—')}`, val до `{_sc.get('val_end', '—')}`."
                    )

        # ══════════════════════════════════════════════════════════════════════
        # Column Groups — конфиг аналитического дашборда
        # Сохраняется в корень проекта (column_groups.json), не в exports/.
        # value_labels не включаются — берутся из inverse_transforms.pkl.
        # ══════════════════════════════════════════════════════════════════════
        st.divider()
        _GROUPS_PATH = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "column_groups.json")
        )

        def _auto_detect_groups(columns: list) -> dict:
            cols = columns
            g: dict = {
                "display_names": {}, "fuel_types": {}, "traffic_lanes": {},
                "vehicle_types": {}, "shop_categories": {},
                "competitor_prices": {}, "special_cols": {},
            }
            for sfx, lbl in {
                "AI92": "АИ-92", "AI92_bio": "АИ-92 bio",
                "AI95": "АИ-95", "AI95_bio": "АИ-95 bio", "AI100_bio": "АИ-100 bio",
                "DT": "ДТ", "DT_bio": "ДТ bio", "SUG": "СУГ", "KPG": "КПГ", "SPG": "СПГ",
            }.items():
                entry = {}
                if f"sales_{sfx}" in cols: entry["sales"] = f"sales_{sfx}"
                if f"price_{sfx}" in cols: entry["price"] = f"price_{sfx}"
                if entry: g["fuel_types"][lbl] = entry
            for col in cols:
                if col.startswith("traffic_intensiv_fiz_"):
                    parts = col[len("traffic_intensiv_fiz_"):].split("_")
                    if len(parts) >= 2:
                        d = "попутная" if parts[1] == "poputn" else "встречная"
                        g["traffic_lanes"][f"Полоса {parts[0]} {d}"] = col
            for vtype, lbl in {
                "Passengers_cars": "Легковые", "Truck_short": "Малые грузовые",
                "Truck_long": "Большегрузы", "Truck": "Грузовые",
                "Transporter": "Транспортёры", "Undefined": "Неопределённые",
            }.items():
                vc = [c for c in cols if f"traffic_{vtype}_" in c]
                if vc: g["vehicle_types"][lbl] = vc
            for col in cols:
                if col.startswith("shop_"):
                    g["shop_categories"][col[5:].replace("_", " ")] = col
            for col in cols:
                if col.startswith("competitor_price_") and not col.lower().endswith("brend"):
                    fuel = col[len("competitor_price_"):]
                    brend = f"competitor_price_{fuel}_brend"
                    if brend not in cols: brend = f"competitor_price_{fuel}_Brend"
                    entry = {"standard": col}
                    if brend in cols: entry["brend"] = brend
                    g["competitor_prices"][fuel] = entry
            for key, col in {
                "road_col":               "road_type",
                "direction_col":          "direction",
                "distance_col":           "distance_to_city_km",
                "settlement_col":         "settlement_size",
                "weekend_col":            "is_weekend",
                "season_col":             "season",
                "holiday_col":            "is_holiday",
                "holiday_name_col":       "holiday_name",
                "temperature_col":        "temperature",
                "precipitation_col":      "precipitation_mm",
                "weather_condition_col":  "weather_condition",
                "corporate_ratio_col":    "corporate_customer_ratio",
                "loyalty_score_col":      "customer_loyalty_score",
                "promotion_fuel_col":     "promotion_fuel_active",
                "promotion_shop_col":     "promotion_shop_active",
                "promotion_cafe_col":     "promotion_cafe_active",
            }.items():
                if col in cols: g["special_cols"][key] = col
            g["display_names"] = {}
            return g

        with st.expander("🗂️ Группировки для аналитического дашборда", expanded=False):
            st.caption(
                "Создаётся **один раз** → сохраняется в `column_groups.json` в корне проекта. "
                "Расшифровки значений не указываются — они берутся из `inverse_transforms.pkl`."
            )
            st.info(
                "💡 **Файл опциональный.** Аналитический дашборд работает и без него — "
                "показывает базовые графики из `tft_config.json` (динамика таргета, корреляции, сравнение станций). "
                "С файлом дополнительно доступны: фильтр по видам топлива, полосам трафика, "
                "категориям магазина и специфичные графики из EDA-плана.",
                icon=None,
            )
            if _t5_merged is None:
                st.info("Загрузите и объедините файлы чтобы настроить группировки.")
            else:
                _cg_cols = _t5_merged.columns.tolist()
                _col_opts = ["—"] + _cg_cols
                _force_auto = st.session_state.pop("cg_force_auto", False)
                if os.path.exists(_GROUPS_PATH) and not _force_auto:
                    with open(_GROUPS_PATH, "r", encoding="utf-8") as _gf:
                        _cg = json.load(_gf)
                    st.caption("✅ Загружен существующий column_groups.json")
                else:
                    _cg = _auto_detect_groups(_cg_cols)
                    st.caption("🔍 Авто-определено по именам колонок — проверьте и сохраните.")

                _ft = _cg.get("fuel_types", {})
                _tl = _cg.get("traffic_lanes", {})
                _vt = _cg.get("vehicle_types", {})
                _sc = _cg.get("shop_categories", {})
                _cp = _cg.get("competitor_prices", {})
                _sp = _cg.get("special_cols", {})
                _dn = _cg.get("display_names", {})

                st.markdown("**🛢️ Виды топлива**")
                _ft_ed = st.data_editor(
                    pd.DataFrame([{"Название": k, "Продажи": v.get("sales","—"), "Цена": v.get("price","—")} for k,v in _ft.items()]) if _ft else pd.DataFrame(columns=["Название","Продажи","Цена"]),
                    column_config={
                        "Название": st.column_config.TextColumn(required=True),
                        "Продажи": st.column_config.SelectboxColumn("Колонка продаж", options=_col_opts),
                        "Цена": st.column_config.SelectboxColumn("Колонка цены", options=_col_opts),
                    }, num_rows="dynamic", hide_index=True, key="cg_fuel", width="stretch",
                )
                st.markdown("**🛣️ Полосы трафика**")
                _tl_ed = st.data_editor(
                    pd.DataFrame([{"Название": k, "Колонка": v} for k,v in _tl.items()]) if _tl else pd.DataFrame(columns=["Название","Колонка"]),
                    column_config={
                        "Название": st.column_config.TextColumn(required=True),
                        "Колонка": st.column_config.SelectboxColumn(options=_col_opts),
                    }, num_rows="dynamic", hide_index=True, key="cg_lanes", width="stretch",
                )
                st.markdown("**🚛 Типы транспортных средств**")
                st.caption("Каждый тип объединяет колонки по всем полосам.")
                _vt_def = _vt if _vt else {"Легковые":[],"Малые грузовые":[],"Грузовые":[],"Большегрузы":[],"Транспортёры":[],"Неопределённые":[]}
                _vt_res: dict = {}
                _vtl, _vtr = st.columns(2)
                for _i, (_vtype, _vcols) in enumerate(_vt_def.items()):
                    with (_vtl if _i % 2 == 0 else _vtr):
                        _vt_res[_vtype] = st.multiselect(_vtype, options=_cg_cols, default=[c for c in (_vcols if isinstance(_vcols, list) else []) if c in _cg_cols], key=f"cg_vt_{_vtype}")
                st.markdown("**🏪 Категории магазина**")
                _sc_ed = st.data_editor(
                    pd.DataFrame([{"Название": k, "Колонка": v} for k,v in _sc.items()]) if _sc else pd.DataFrame(columns=["Название","Колонка"]),
                    column_config={
                        "Название": st.column_config.TextColumn(required=True),
                        "Колонка": st.column_config.SelectboxColumn(options=_col_opts),
                    }, num_rows="dynamic", hide_index=True, key="cg_shop", width="stretch",
                )
                st.markdown("**💰 Цены конкурентов**")
                _cp_ed = st.data_editor(
                    pd.DataFrame([{"Топливо": k, "Стандарт": v.get("standard","—"), "Бренд": v.get("brend","—")} for k,v in _cp.items()]) if _cp else pd.DataFrame(columns=["Топливо","Стандарт","Бренд"]),
                    column_config={
                        "Топливо": st.column_config.TextColumn(required=True),
                        "Стандарт": st.column_config.SelectboxColumn(options=_col_opts),
                        "Бренд": st.column_config.SelectboxColumn(options=_col_opts),
                    }, num_rows="dynamic", hide_index=True, key="cg_comp", width="stretch",
                )
                st.markdown("**🌤️ Специальные колонки**")
                _sp_labels = {
                    "total_sales_col":       "Суммарные продажи топлива",
                    "road_col":              "Тип дороги",
                    "direction_col":         "Направление АЗС",
                    "distance_col":          "Удалённость от города",
                    "settlement_col":        "Размер населённого пункта",
                    "weekend_col":           "Флаг выходного дня (0/1)",
                    "season_col":            "Сезон",
                    "holiday_col":           "Флаг праздника (0/1)",
                    "holiday_name_col":      "Название праздника",
                    "temperature_col":       "Температура",
                    "precipitation_col":     "Осадки (мм)",
                    "weather_condition_col": "Тип погоды",
                    "corporate_ratio_col":   "Доля корп. клиентов",
                    "loyalty_score_col":     "Лояльность клиентов",
                    "promotion_fuel_col":    "Акция на топливо",
                    "promotion_shop_col":    "Акция в магазине",
                    "promotion_cafe_col":    "Акция в кафе",
                }
                _sp_res: dict = {}
                _spl, _spr = st.columns(2)
                for _i, (_key, _lbl) in enumerate(_sp_labels.items()):
                    _cur = _sp.get(_key, "—")
                    with (_spl if _i % 2 == 0 else _spr):
                        _sp_res[_key] = st.selectbox(_lbl, options=_col_opts, index=_col_opts.index(_cur) if _cur in _col_opts else 0, key=f"cg_sp_{_key}")
                st.markdown("**🏷️ Отображаемые названия колонок**")
                st.caption("Подписи осей/заголовков. Расшифровки значений — в `inverse_transforms.pkl`.")
                _dn_ed = st.data_editor(
                    pd.DataFrame([{"Колонка": c, "Название": _dn.get(c,"")} for c in _cg_cols]),
                    column_config={
                        "Колонка": st.column_config.TextColumn(disabled=True),
                        "Название": st.column_config.TextColumn("Отображаемое название"),
                    }, hide_index=True, key="cg_dn", width="stretch",
                )
                st.markdown("**🔤 Расшифровки строковых значений**")
                st.caption(
                    "Для колонок, где значения уже строки (не числа): задайте читаемые названия. "
                    "Сохраняются в `value_remaps` и используются в фильтрах аналитического дашборда."
                )
                _vr = _cg.get("value_remaps", {})
                _vr_editors: dict = {}
                _str_cols = [
                    c for c in _cg_cols
                    if pd.api.types.is_string_dtype(_t5_merged[c])
                    and 1 < _t5_merged[c].nunique() <= 30
                ]
                if _str_cols:
                    for _vrc in _str_cols:
                        _vr_vals = sorted(_t5_merged[_vrc].dropna().unique().tolist())
                        _vr_cur  = _vr.get(_vrc, {})
                        _vr_df   = pd.DataFrame([
                            {"Значение": str(v), "Отображение": _vr_cur.get(str(v), str(v))}
                            for v in _vr_vals
                        ])
                        with st.expander(f"`{_vrc}`", expanded=bool(_vr_cur)):
                            _vr_editors[_vrc] = st.data_editor(
                                _vr_df,
                                column_config={
                                    "Значение":    st.column_config.TextColumn(disabled=True),
                                    "Отображение": st.column_config.TextColumn("Отображаемое название"),
                                },
                                hide_index=True,
                                key=f"cg_vr_{_vrc}",
                                width="stretch",
                            )
                else:
                    st.caption("Строковых категориальных колонок не обнаружено.")
                st.divider()
                _cg_bl, _cg_br = st.columns(2)
                with _cg_bl:
                    if st.button("💾 Сохранить column_groups.json", width="stretch", key="cg_save_btn"):
                        def _nv(val):
                            return None if (val == "—" or (not isinstance(val, str) and pd.isna(val))) else str(val)
                        _vr_result = {}
                        for _vrc, _vr_df_ed in _vr_editors.items():
                            _mapping = {
                                str(r["Значение"]): str(r["Отображение"])
                                for _, r in _vr_df_ed.iterrows()
                                if str(r.get("Отображение", "")).strip()
                            }
                            if _mapping:
                                _vr_result[_vrc] = _mapping
                        _result = {
                            "display_names": {str(r["Колонка"]): str(r["Название"]) for _,r in _dn_ed.iterrows() if str(r.get("Название","")).strip()},
                            "fuel_types": {str(r["Название"]): {k:w for k,w in {"sales":_nv(r["Продажи"]),"price":_nv(r["Цена"])}.items() if w} for _,r in _ft_ed.iterrows() if str(r.get("Название","")).strip()},
                            "traffic_lanes": {str(r["Название"]): str(r["Колонка"]) for _,r in _tl_ed.iterrows() if str(r.get("Название","")).strip() and _nv(r["Колонка"])},
                            "vehicle_types": {k:v for k,v in _vt_res.items() if v},
                            "shop_categories": {str(r["Название"]): str(r["Колонка"]) for _,r in _sc_ed.iterrows() if str(r.get("Название","")).strip() and _nv(r["Колонка"])},
                            "competitor_prices": {str(r["Топливо"]): {k:w for k,w in {"standard":_nv(r["Стандарт"]),"brend":_nv(r["Бренд"])}.items() if w} for _,r in _cp_ed.iterrows() if str(r.get("Топливо","")).strip()},
                            "special_cols": {k:v for k,v in _sp_res.items() if v and v != "—"},
                            "value_remaps": _vr_result,
                        }
                        with open(_GROUPS_PATH, "w", encoding="utf-8") as _gf:
                            json.dump(_result, _gf, ensure_ascii=False, indent=2)
                        st.success(f"Сохранено → {_GROUPS_PATH}")
                with _cg_br:
                    if st.button("🔄 Сбросить к авто-определению", width="stretch", key="cg_reset_btn"):
                        for _k in ["cg_fuel","cg_lanes","cg_shop","cg_comp","cg_dn"]:
                            st.session_state.pop(_k, None)
                        for _vtype in _vt_def:
                            st.session_state.pop(f"cg_vt_{_vtype}", None)
                        for _key in _sp_labels:
                            st.session_state.pop(f"cg_sp_{_key}", None)
                        for _vrc in _str_cols:
                            st.session_state.pop(f"cg_vr_{_vrc}", None)
                        st.session_state["cg_force_auto"] = True
                        st.rerun()

        st.divider()
        _sess_name_input = st.text_input(
            "Название сессии",
            key="save_session_name",
            placeholder="Моя сессия",
        )
        _sess_name_ok, _sess_name_err = validate_report_name(_sess_name_input)
        if _sess_name_input and not _sess_name_ok:
            st.error(_sess_name_err)
        if st.button("💾 Сохранить всё в папку проекта", type="primary", width="stretch",
                     disabled=bool(_sess_name_input) and not _sess_name_ok):
            from datetime import datetime as _dt
            _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            _ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
            _folder_name = f"{_sess_name_input.strip()}_{_ts}" if _sess_name_input.strip() else _ts
            _out_dir = os.path.join(_project_root, "exports", _folder_name)
            try:
                os.makedirs(_out_dir, exist_ok=True)

                work_df.to_csv(
                    os.path.join(_out_dir, "processed_data.csv"),
                    index=False, encoding="utf-8",
                )
                with open(os.path.join(_out_dir, "tft_config.json"), "w", encoding="utf-8") as _f:
                    json.dump(tft_config, _f, ensure_ascii=False, indent=2, cls=_NpEncoder)
                with open(os.path.join(_out_dir, "prep_config.json"), "w", encoding="utf-8") as _f:
                    json.dump(
                        st.session_state.get("applied_prep_config") or st.session_state.get("prep_config", {}),
                        _f, ensure_ascii=False, indent=2, cls=_NpEncoder,
                    )
                with open(os.path.join(_out_dir, "data_prep_report.md"), "w", encoding="utf-8") as _f:
                    _f.write(_report_md())

                # Save merged_data.csv (before preprocessing — needed for full restore)
                _merged_to_save = st.session_state.get("merged_df")
                if _merged_to_save is not None:
                    _merged_to_save.to_csv(
                        os.path.join(_out_dir, "merged_data.csv"),
                        index=False, encoding="utf-8",
                    )

                # Save original source CSV files if uploaded via Tab 1
                _src_files = st.session_state.get("uploaded_files", {})
                _n_src = 0
                if _src_files:
                    _src_dir = os.path.join(_out_dir, "source_files")
                    os.makedirs(_src_dir, exist_ok=True)
                    for _sf_name, _sf_df in _src_files.items():
                        _sf_df.to_csv(
                            os.path.join(_src_dir, _sf_name),
                            index=False, encoding="utf-8",
                        )
                        _n_src += 1

                # Save session_config.json (merge config, computed col formulas, col types, tidx)
                _s_names = list(_src_files.keys())
                _n_pairs = max(0, len(_s_names) - 1)
                _merge_cfgs_save = [
                    {
                        "right_name": _s_names[_si + 1],
                        "on": st.session_state.get(f"join_col_{_si}"),
                        "how": st.session_state.get(f"join_how_{_si}", "left"),
                    }
                    for _si in range(_n_pairs)
                ]
                _session_cfg = build_session_config(
                    source_file_names=_s_names,
                    merge_configs=_merge_cfgs_save,
                    added_col_formulas=st.session_state.get("added_col_formulas", {}),
                    tidx_config=st.session_state.get("tidx_config", {}),
                    dtx_configs=st.session_state.get("dtx_configs", []),
                    col_types=st.session_state.get("col_types", {}),
                    preprocessing_applied=st.session_state.get("prep_df") is not None,
                    file_settings={
                        "sep": st.session_state.get("file_sep", ","),
                        "enc": st.session_state.get("file_enc", "auto"),
                        "dec": st.session_state.get("file_dec", "."),
                        "detected_encs": st.session_state.get("_detected_encs", {}),
                    },
                )
                with open(os.path.join(_out_dir, "session_config.json"), "w", encoding="utf-8") as _f:
                    json.dump(_session_cfg, _f, ensure_ascii=False, indent=2, cls=_NpEncoder)

                # Save inverse_transforms.pkl if preprocessing was applied
                _inv_to_save = st.session_state.get("inverse_params", {})
                _inv_note = ""
                if _inv_to_save:
                    import pickle as _pkl
                    with open(os.path.join(_out_dir, "inverse_transforms.pkl"), "wb") as _f:
                        _pkl.dump(_inv_to_save, _f)
                    _inv_note = " · inverse_transforms.pkl"

                # Save split_config.json — use the snapshot frozen at preprocessing time
                _split_to_save = st.session_state.get("applied_split_config") or {}
                _split_note = ""
                if _split_to_save:
                    with open(os.path.join(_out_dir, "split_config.json"), "w", encoding="utf-8") as _f:
                        json.dump(_split_to_save, _f, ensure_ascii=False, indent=2)
                    _split_note = " · split_config.json"

                _src_note = (
                    f" + {_n_src} исходных файлов в source_files/"
                    if _n_src else ""
                )
                st.session_state["_save_flash"] = (
                    f"Сохранено в папку:  \n`{_out_dir}`  \n"
                    f"processed_data.csv · merged_data.csv · tft_config.json · "
                    f"prep_config.json · session_config.json · data_prep_report.md"
                    f"{_inv_note}{_split_note}{_src_note}"
                )
                st.session_state["_t5_pending_session_select"] = _folder_name
                st.rerun()
            except Exception as _exc:
                st.error(f"Ошибка сохранения: {_exc}")
        st.caption(
            "Название необязательно — если оставить пустым, папка называется по дате и времени. "
            "Создаёт подпапку exports/[название_]ГГГГ-ММ-ДД_ЧЧ-ММ-СС/ в корне проекта. "
            "Эта папка видна в селекторе «📂 Загрузить сохранённую сессию» для полного восстановления."
        )
        if "_save_flash" in st.session_state:
            st.success(st.session_state.pop("_save_flash"))
