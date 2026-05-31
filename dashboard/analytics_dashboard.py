"""
EDA Analytics Dashboard — АЗС Татнефть.
Run: streamlit run dashboard/analytics_dashboard.py  (from project root)
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*save_hyperparameters.*")
warnings.filterwarnings("ignore", message=".*nn.Module.*checkpointing.*")
warnings.filterwarnings("ignore", message=".*NumPy array is not writable.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*deprecated.*")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.analytics_utils import (
    available_fuel_types,
    available_shop_categories,
    decode_col_series,
    denormalize,
    filter_merged_duplicates,
    label,
    list_sessions,
    load_session,
)

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Аналитика Татнефть АЗС",
    page_icon="⛽",
    layout="wide",
)

EXPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exports"))

# ══════════════════════════════════════════════════════════════════════════════
# Session loading (sidebar — выбор сессии)
# ══════════════════════════════════════════════════════════════════════════════
sessions = list_sessions(EXPORTS_DIR)
if not sessions:
    st.error("Папка exports/ не найдена или пуста. Сохраните сессию в дашборде предобработки.")
    st.stop()

_title_col, _sess_col = st.columns([6, 2])
_title_col.markdown("# ⛽ Аналитика Татнефть АЗС")
selected_session = _sess_col.selectbox(
    "Набор данных", sessions, index=0
)

@st.cache_data(show_spinner="Загрузка сессии...")
def _load(session_name: str) -> dict:
    return load_session(os.path.join(EXPORTS_DIR, session_name))

S = _load(selected_session)
tft      = S["tft"]
inv      = S["inv"]
session  = S["session"]
prep_cfg = S.get("prep_cfg") or {}
merged   = S["merged"]
proc     = S.get("proc")

_grp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "column_groups.json"))
if os.path.exists(_grp_path):
    with open(_grp_path, "r", encoding="utf-8") as _f:
        groups = json.load(_f)
else:
    groups = S["groups"]

if tft is None:
    st.error("tft_config.json не найден в выбранной сессии.")
    st.stop()
if merged is None:
    st.error("merged_data.csv не найден. Пересохраните сессию.")
    st.stop()

merged = filter_merged_duplicates(merged, session)

# ══════════════════════════════════════════════════════════════════════════════
# Base helpers
# ══════════════════════════════════════════════════════════════════════════════
lbl       = lambda col: label(col, groups)
group_col = tft.get("group_col", "station_id")
targets   = tft.get("target", [])

date_col = None
if session:
    _tc = session.get("tidx_config", {})
    if _tc.get("src_col"):
        date_col = _tc["src_col"]
if date_col is None:
    date_col = tft.get("time_col")
if date_col and date_col in merged.columns:
    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")

_sp = groups.get("special_cols", {})
_cols = merged.columns.tolist()

def _resolve(key: str, fallback: str | None = None) -> str | None:
    """Берём имя колонки из special_cols конфига, проверяем наличие в данных."""
    col = _sp.get(key, fallback)
    return col if col and col in _cols else None

fuel_types = available_fuel_types(groups, _cols)
shop_cats  = available_shop_categories(groups, _cols)

_fuel_sales_cols = [v.get("sales") for v in fuel_types.values()
                    if v.get("sales") and v.get("sales") in merged.columns]
if "total_fuel_sales" in merged.columns:
    total_sales_col = "total_fuel_sales"
elif _fuel_sales_cols:
    merged["total_fuel_sales"] = merged[_fuel_sales_cols].sum(axis=1)
    total_sales_col = "total_fuel_sales"
else:
    total_sales_col = targets[0] if targets else None

road_col = _resolve("road_col",      "road_type")
dir_col  = _resolve("direction_col", "direction")
dist_col = _resolve("distance_col",  "distance_to_city_km")
sett_col = _resolve("settlement_col","settlement_size")

def _decoded_opts(col: str) -> tuple[list, dict]:
    """Return (decoded_labels, {decoded_label: raw_value}) for a column."""
    if col not in merged.columns:
        return [], {}
    raw = sorted(merged[col].dropna().unique().tolist())
    _remaps = groups.get("value_remaps", {}).get(col, {})
    dec = [
        _remaps.get(str(v), str(decode_col_series(pd.Series([v]), col, inv).iloc[0]))
        for v in raw
    ]
    lmap: dict = {}
    for d, r in zip(dec, raw):
        lmap.setdefault(d, r)
    return dec, lmap

_ck = [0]
def _pchart(fig): st.plotly_chart(fig, width="stretch", key=f"pc{_ck[0]}"); _ck[0] += 1

tab_eda, tab_stat, tab_forecast, tab_reco = st.tabs([
    "📊 EDA-анализ",
    "📈 Статистика",
    "🔮 Прогнозы",
    "💡 Рекомендации",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB верхнего уровня: EDA-анализ
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:

    # ── Глобальные фильтры (горизонтально) ───────────────────────────────────
    with st.expander("🔍 Глобальные фильтры", expanded=True):
        # Строка 1
        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 1])

        all_stations = (
            sorted(merged[group_col].unique().tolist())
            if group_col in merged.columns else []
        )
        # Суффикс сессии в ключах — при смене датасета фильтры сбрасываются к дефолту
        _sk = selected_session

        sel_stations = c1.multiselect(
            lbl(group_col), all_stations, default=all_stations, key=f"g_st_{_sk}"
        )

        date_from = date_to = None
        if date_col and date_col in merged.columns:
            _min_d = merged[date_col].min().date()
            _max_d = merged[date_col].max().date()
            _rng = c2.date_input(
                "Период", (_min_d, _max_d),
                min_value=_min_d, max_value=_max_d, key=f"g_dt_{_sk}",
            )
            if isinstance(_rng, (list, tuple)) and len(_rng) == 2:
                date_from, date_to = pd.Timestamp(_rng[0]), pd.Timestamp(_rng[1])
            else:
                date_from, date_to = pd.Timestamp(_min_d), pd.Timestamp(_max_d)

        _season_col  = _resolve("season_col",  "season")
        _weekend_col = _resolve("weekend_col", "is_weekend")
        _holiday_col = _resolve("holiday_col", "is_holiday")

        sel_season = None
        if _season_col:
            _opts_s, _lmap_s = _decoded_opts(_season_col)
            _sel_s = c3.multiselect(lbl(_season_col), _opts_s, default=_opts_s, key=f"g_seas_{_sk}")
            sel_season = [_lmap_s[s] for s in _sel_s]

        day_type = "Все"
        if _weekend_col:
            day_type = c4.selectbox("Тип дня", ["Все", "Будни", "Выходные"], key=f"g_day_{_sk}")

        holiday_f = "Все"
        if _holiday_col:
            holiday_f = c5.selectbox("Праздники", ["Все", "Да", "Нет"], key=f"g_hol_{_sk}")

        # Строка 2
        c6, c7, c8, c9 = st.columns(4)

        sel_road = None
        if road_col:
            _opts_r = sorted(merged[road_col].dropna().unique().tolist())
            sel_road = c6.multiselect(lbl(road_col), _opts_r, default=_opts_r, key=f"g_rd_{_sk}")

        sel_dir = None
        if dir_col:
            _opts_d = sorted(merged[dir_col].dropna().unique().tolist())
            sel_dir = c7.multiselect(lbl(dir_col), _opts_d, default=_opts_d, key=f"g_dir_{_sk}")

        sel_dist = None
        if dist_col:
            _opts_di, _lmap_di = _decoded_opts(dist_col)
            _sel_di = c8.multiselect(lbl(dist_col), _opts_di, default=_opts_di, key=f"g_dist_{_sk}")
            sel_dist = [_lmap_di[s] for s in _sel_di]

        sel_sett = None
        if sett_col:
            _opts_se, _lmap_se = _decoded_opts(sett_col)
            _sel_se = c9.multiselect(lbl(sett_col), _opts_se, default=_opts_se, key=f"g_sett_{_sk}")
            sel_sett = [_lmap_se[s] for s in _sel_se]

    # ── Apply global filters ──────────────────────────────────────────────────
    def apply_filters(src: pd.DataFrame) -> pd.DataFrame:
        out = src.copy()
        if sel_stations and group_col in out.columns:
            out = out[out[group_col].isin(sel_stations)]
        if date_from and date_to and date_col and date_col in out.columns:
            out = out[(out[date_col] >= date_from) & (out[date_col] <= date_to)]
        if sel_season is not None and _season_col and _season_col in out.columns:
            out = out[out[_season_col].isin(sel_season)]
        if day_type == "Будни" and _weekend_col and _weekend_col in out.columns:
            out = out[out[_weekend_col] == 0]
        elif day_type == "Выходные" and _weekend_col and _weekend_col in out.columns:
            out = out[out[_weekend_col] == 1]
        if holiday_f == "Да" and _holiday_col and _holiday_col in out.columns:
            out = out[out[_holiday_col] == 1]
        elif holiday_f == "Нет" and _holiday_col and _holiday_col in out.columns:
            out = out[out[_holiday_col] == 0]
        if sel_road is not None and road_col and road_col in out.columns:
            out = out[out[road_col].isin(sel_road)]
        if sel_dir is not None and dir_col and dir_col in out.columns:
            out = out[out[dir_col].isin(sel_dir)]
        if sel_dist is not None and dist_col and dist_col in out.columns:
            out = out[out[dist_col].isin(sel_dist)]
        if sel_sett is not None and sett_col and sett_col in out.columns:
            out = out[out[sett_col].isin(sel_sett)]
        return out

    df = apply_filters(merged)

    if df.empty:
        st.warning("Нет данных для выбранных фильтров.")

    _n_st   = df[group_col].nunique() if group_col in df.columns else "?"
    _n_days = df[date_col].nunique()  if date_col and date_col in df.columns else "?"
    st.caption(f"{len(df):,} строк · {_n_st} станций · {_n_days} дней")

    # ── Подвкладки EDA ────────────────────────────────────────────────────────
    sub_ov, sub_sales, sub_prices, sub_weather, sub_shop, sub_promo, sub_stations = st.tabs([
        "📊 Сводка",
        "⛽ Продажи топлива",
        "💰 Цены и конкуренты",
        "🌤️ Погода и трафик",
        "🏪 Магазин и клиенты",
        "🎯 Акции и праздники",
        "📍 Станции",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # Подвкладка: Обзор
    # ════════════════════════════════════════════════════════════════════════════
    with sub_ov:

        def _card(title: str, value: str, sub: str = "", color: str = "#2196F3") -> str:
            sub_html = f'<div style="color:#9ca3af;font-size:12px;margin-top:4px">{sub}</div>' if sub else ""
            return f"""
            <div style="
                background:linear-gradient(135deg,{color}18,{color}08);
                border:1px solid {color}33;
                border-left:4px solid {color};
                border-radius:10px;
                padding:16px 18px;
                margin-bottom:8px;
                height:100%;
            ">
                <div style="color:#9ca3af;font-size:11px;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.8px">{title}</div>
                <div style="color:#f1f5f9;font-size:22px;font-weight:700;
                            margin-top:6px;line-height:1.2">{value}</div>
                {sub_html}
            </div>"""

        kc1, kc2, kc3, kc4, kc5 = st.columns([2.2, 2.2, 1.5, 2, 1.8])

        # ── Колонка 1: суммарные + средние продажи (стопкой) ─────────────────
        with kc1:
            if total_sales_col and total_sales_col in df.columns:
                _total = df[total_sales_col].sum()
                _daily = (
                    df.groupby(date_col)[total_sales_col].sum().mean()
                    if date_col and date_col in df.columns
                    else df[total_sales_col].mean()
                )
                st.markdown(_card(
                    "⛽ Суммарные продажи топлива",
                    f"{_total / 1_000_000:.2f} млн л",
                    color="#2196F3",
                ), unsafe_allow_html=True)
                st.markdown(_card(
                    "📅 Среднесуточные продажи",
                    f"{_daily:,.0f} л/сут",
                    color="#2196F3",
                ), unsafe_allow_html=True)

        # ── Колонка 2: общая + средняя выручка магазина (стопкой) ────────────
        with kc2:
            if shop_cats:
                _sc = [c for c in shop_cats.values() if c in df.columns]
                if _sc:
                    _shop_series = df[_sc].sum(axis=1)
                    _shop_total  = _shop_series.sum()
                    _shop_daily  = _shop_series.mean()
                    st.markdown(_card(
                        "🏪 Общая выручка магазина",
                        f"{_shop_total / 1_000_000:.2f} млн руб",
                        color="#4CAF50",
                    ), unsafe_allow_html=True)
                    st.markdown(_card(
                        "🏪 Средняя выручка магазина",
                        f"{_shop_daily:,.0f} руб/сут",
                        color="#4CAF50",
                    ), unsafe_allow_html=True)

        # ── Колонка 3: количество станций ────────────────────────────────────
        with kc3:
            _n_stations = df[group_col].nunique() if group_col in df.columns else "—"
            st.markdown(_card(
                "📍 Кол-во станций",
                str(_n_stations),
                color="#9C27B0",
            ), unsafe_allow_html=True)

        # ── Колонка 4: лучшая станция ─────────────────────────────────────────
        with kc4:
            if total_sales_col and total_sales_col in df.columns and group_col in df.columns:
                _by_st = df.groupby(group_col)[total_sales_col].mean()
                if not _by_st.empty:
                    st.markdown(_card(
                        "🏆 Лучшая станция",
                        str(_by_st.idxmax()),
                        f"{_by_st.max():,.0f} л/сут (ср.)",
                        color="#FFC107",
                    ), unsafe_allow_html=True)

        # ── Колонка 5: средний трафик ─────────────────────────────────────────
        with kc5:
            _fiz = [c for c in df.columns if "intensiv_fiz" in c]
            if _fiz:
                st.markdown(_card(
                    "🚗 Средний трафик",
                    f"{df[_fiz].sum(axis=1).mean():,.0f}",
                    "авт/сут",
                    color="#00BCD4",
                ), unsafe_allow_html=True)

        st.divider()

        _has_date_ov = bool(date_col and date_col in df.columns)
        _has_ts_ov   = _has_date_ov and bool(total_sales_col and total_sales_col in df.columns)
        _sc_ov       = [v for v in shop_cats.values() if v in df.columns]

        # ── Динамика: monthly stacked bar ─────────────────────────────────────
        if _has_date_ov and fuel_types:
            st.markdown("### 📅 Динамика продаж топлива")
            _rows_m_ov: list[pd.DataFrame] = []
            for _ft, _finfo in fuel_types.items():
                _sc = _finfo.get("sales")
                if _sc and _sc in df.columns:
                    _m = df.copy()
                    _m["_month"] = _m[date_col].dt.to_period("M").astype(str)
                    _a = _m.groupby("_month")[_sc].sum().reset_index()
                    _a["fuel"] = _ft
                    _a.rename(columns={_sc: "sales"}, inplace=True)
                    _rows_m_ov.append(_a)
            if _rows_m_ov:
                fig = px.bar(
                    pd.concat(_rows_m_ov), x="_month", y="sales", color="fuel", barmode="stack",
                    labels={"_month": "Месяц", "sales": "Продажи (л)", "fuel": "Вид топлива"},
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(legend_title="Вид топлива")
                _pchart(fig)

        st.divider()

        # ── Структура: два pie ────────────────────────────────────────────────
        st.markdown("### 🧩 Структура")
        _ov_s1, _ov_s2 = st.columns(2)

        with _ov_s1:
            _ov_struct = [
                {"fuel": _ft, "total": df[_finfo["sales"]].sum()}
                for _ft, _finfo in fuel_types.items()
                if _finfo.get("sales") and _finfo["sales"] in df.columns
            ]
            if _ov_struct:
                st.markdown("#### Доля видов топлива")
                fig = px.pie(
                    pd.DataFrame(_ov_struct), names="fuel", values="total", hole=0.45,
                    labels={"fuel": "Вид топлива", "total": "Продажи (л)"},
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                _pchart(fig)

        with _ov_s2:
            if _sc_ov:
                st.markdown("#### Структура выручки магазина")
                _ov_struct_sh = [
                    {"cat": k, "total": df[v].sum()}
                    for k, v in shop_cats.items() if v in df.columns
                ]
                fig = px.pie(
                    pd.DataFrame(_ov_struct_sh), names="cat", values="total", hole=0.45,
                    labels={"cat": "Категория", "total": "Выручка (руб.)"},
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                _pchart(fig)

        st.divider()

        # ── По станциям: bar + grouped bar топливо vs магазин ─────────────────
        st.markdown("### 📍 По станциям")
        _ov_st1, _ov_st2 = st.columns(2)

        with _ov_st1:
            if _has_ts_ov:
                st.markdown("#### Средние продажи топлива")
                _ov_sales_st = (
                    df.groupby(group_col)[total_sales_col].mean()
                    .reset_index()
                    .sort_values(total_sales_col, ascending=False)
                )
                fig = px.bar(
                    _ov_sales_st, x=group_col, y=total_sales_col,
                    color=group_col, text_auto=",.0f",
                    labels={group_col: lbl(group_col), total_sales_col: lbl(total_sales_col) + " (ср.)"},
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                _pchart(fig)

        with _ov_st2:
            if _has_ts_ov and _sc_ov:
                st.markdown("#### Топливо vs Магазин")
                _ov_cmp = df.copy()
                _ov_cmp["_shop"] = _ov_cmp[_sc_ov].sum(axis=1)
                _ov_cmp_agg = _ov_cmp.groupby(group_col).agg(
                    shop_total=("_shop", "mean"),
                    fuel_sales=(total_sales_col, "mean"),
                ).reset_index()
                _max_f = _ov_cmp_agg["fuel_sales"].max() or 1
                _max_s = _ov_cmp_agg["shop_total"].max() or 1
                _ov_cmp_agg["fuel_pct"] = _ov_cmp_agg["fuel_sales"] / _max_f * 100
                _ov_cmp_agg["shop_pct"] = _ov_cmp_agg["shop_total"] / _max_s * 100
                _ov_st_lbl = _ov_cmp_agg[group_col].astype(str)
                fig = go.Figure([
                    go.Bar(
                        name="Продажи топлива", x=_ov_st_lbl, y=_ov_cmp_agg["fuel_pct"],
                        customdata=_ov_cmp_agg["fuel_sales"],
                        hovertemplate="%{x}<br>Топливо: %{customdata:,.0f} л/сут<extra></extra>",
                    ),
                    go.Bar(
                        name="Выручка магазина", x=_ov_st_lbl, y=_ov_cmp_agg["shop_pct"],
                        customdata=_ov_cmp_agg["shop_total"],
                        hovertemplate="%{x}<br>Магазин: %{customdata:,.0f} руб/сут<extra></extra>",
                    ),
                ])
                fig.update_layout(
                    barmode="group",
                    yaxis_title="% от максимума",
                    xaxis_title=lbl(group_col),
                    legend=dict(orientation="h", y=-0.25, x=0),
                    margin=dict(b=80),
                )
                _pchart(fig)

        st.divider()

        # ── Паттерны: heatmap + avg по дням недели ────────────────────────────
        st.markdown("### 📊 Паттерны")
        _ov_p1, _ov_p2 = st.columns(2)

        with _ov_p1:
            if _has_ts_ov:
                st.markdown("#### Тепловая карта: неделя × день недели")
                _hm = df.copy()
                _hm["_dow"]  = _hm[date_col].dt.dayofweek
                _hm["_week"] = _hm[date_col].dt.isocalendar().week.astype(int)
                _dow_names = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
                _pivot = (
                    _hm.groupby(["_week", "_dow"])[total_sales_col].mean()
                    .reset_index()
                    .pivot(index="_week", columns="_dow", values=total_sales_col)
                    .reindex(columns=[d for d in range(7) if d in _hm["_dow"].unique()])
                )
                _pivot.columns = [_dow_names[d] for d in _pivot.columns]
                fig = px.imshow(
                    _pivot,
                    color_continuous_scale="YlOrRd",
                    labels={"color": lbl(total_sales_col), "x": "День недели", "y": "Неделя года"},
                    aspect="auto",
                )
                _pchart(fig)

        with _ov_p2:
            if _has_ts_ov:
                st.markdown("#### Средние продажи по дням недели")
                _dow_ov = df.copy()
                _dow_map = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
                _dow_ov["День недели"] = _dow_ov[date_col].dt.dayofweek.map(_dow_map)
                _present_dow = [_dow_map[d] for d in range(7)
                                if _dow_map[d] in _dow_ov["День недели"].values]
                _dow_agg = (
                    _dow_ov.groupby("День недели")[total_sales_col].mean()
                    .reindex(_present_dow)
                    .reset_index()
                )
                fig = px.bar(
                    _dow_agg, x="День недели", y=total_sales_col,
                    text_auto=",.0f",
                    category_orders={"День недели": _present_dow},
                    labels={"День недели": "", total_sales_col: lbl(total_sales_col) + " (ср.)"},
                )
                fig.update_traces(textposition="outside")
                _pchart(fig)

    # ════════════════════════════════════════════════════════════════════════════
    # Подвкладка: Продажи топлива
    # ════════════════════════════════════════════════════════════════════════════
    with sub_sales:

        sel_fuel: list[str] = []
        if fuel_types:
            sel_fuel = st.multiselect(
                "Вид топлива",
                options=list(fuel_types.keys()),
                default=list(fuel_types.keys()),
                key="t_fuel",
            )
        if not sel_fuel:
            st.info("Выберите хотя бы один вид топлива.")
        else:
            _has_date = bool(date_col and date_col in df.columns)
            _has_ts   = _has_date and bool(total_sales_col and total_sales_col in df.columns)

            # ── Блок 1: Динамика ─────────────────────────────────────────────────
            if _has_ts:
                st.markdown("### 📅 Динамика")
                _wk = df.copy()
                _wk["_week"] = _wk[date_col].dt.to_period("W").astype(str)
                _wk_agg = _wk.groupby(["_week", group_col])[total_sales_col].sum().reset_index()
                st.markdown("#### Еженедельная динамика")
                fig = px.line(
                    _wk_agg, x="_week", y=total_sales_col, color=group_col,
                    labels={"_week": "Неделя", total_sales_col: lbl(total_sales_col), group_col: lbl(group_col)},
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(legend_title=lbl(group_col))
                _pchart(fig)

            if _has_ts and df[date_col].dt.year.nunique() > 1:
                st.markdown("#### Сравнение год к году")
                _yoy = df.copy()
                _yoy["_year"] = _yoy[date_col].dt.year.astype(str)
                _yoy["_doy"]  = _yoy[date_col].dt.dayofyear
                _yoy_agg = _yoy.groupby(["_doy", "_year"])[total_sales_col].mean().reset_index()
                fig = px.line(
                    _yoy_agg, x="_doy", y=total_sales_col, color="_year",
                    labels={"_doy": "День года", total_sales_col: lbl(total_sales_col), "_year": "Год"},
                )
                _pchart(fig)

            # ── Блок 2: Профиль ───────────────────────────────────────────────────
            if _has_ts:
                st.divider()
                st.markdown("### 🕐 Профиль")
                _p1, _p2 = st.columns(2)

                with _p1:
                    st.markdown("#### По дням недели")
                    _dow = df.copy()
                    _dow["День недели"] = _dow[date_col].dt.dayofweek.map(
                        {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
                    )
                    fig = px.box(
                        _dow, x="День недели", y=total_sales_col, color=group_col,
                        category_orders={"День недели": ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]},
                        labels={total_sales_col: lbl(total_sales_col), group_col: lbl(group_col)},
                    )
                    fig.update_layout(legend_title=lbl(group_col), showlegend=False)
                    _pchart(fig)

                with _p2:
                    st.markdown("#### По кварталам")
                    _qt = df.copy()
                    _qt["_q"] = _qt[date_col].dt.to_period("Q").astype(str)
                    _qt_agg = _qt.groupby(["_q", group_col])[total_sales_col].sum().reset_index()
                    fig = px.bar(
                        _qt_agg, x="_q", y=total_sales_col, color=group_col, barmode="group",
                        labels={"_q": "Квартал", total_sales_col: lbl(total_sales_col), group_col: lbl(group_col)},
                    )
                    fig.update_layout(legend_title=lbl(group_col))
                    _pchart(fig)

            # ── Блок 3: Структура ─────────────────────────────────────────────────
            st.divider()
            st.markdown("### 🧩 Структура")

            _rows_az: list[pd.DataFrame] = []
            for _ft in sel_fuel:
                _sc = fuel_types[_ft].get("sales")
                if _sc and _sc in df.columns:
                    _a = df.groupby(group_col)[_sc].sum().reset_index()
                    _a["fuel"] = _ft
                    _a.rename(columns={_sc: "sales"}, inplace=True)
                    _rows_az.append(_a)
            if _rows_az:
                st.markdown("#### Структура по станциям")
                fig = px.bar(
                    pd.concat(_rows_az), x=group_col, y="sales", color="fuel", barmode="stack",
                    labels={group_col: lbl(group_col), "sales": "Продажи (л)", "fuel": "Вид топлива"},
                )
                _pchart(fig)

            _rows: list[pd.DataFrame] = []
            for _ft in sel_fuel:
                _sc = fuel_types[_ft].get("sales")
                if _sc and _sc in df.columns:
                    _a = df.groupby(group_col)[_sc].mean().reset_index()
                    _a["fuel"] = _ft
                    _a.rename(columns={_sc: "sales"}, inplace=True)
                    _rows.append(_a)
            if _rows:
                st.markdown("#### Средние продажи по станциям")
                fig = px.bar(
                    pd.concat(_rows), x=group_col, y="sales", color="fuel", barmode="group",
                    labels={group_col: lbl(group_col), "sales": "Средние продажи (л/сут)", "fuel": "Вид топлива"},
                )
                _pchart(fig)

            # ── Блок 4: Распределение ─────────────────────────────────────────────
            if total_sales_col and total_sales_col in df.columns and group_col in df.columns:
                st.divider()
                st.markdown("### 📐 Распределение")
                st.markdown("#### Violin по станциям")
                fig = px.violin(
                    df, x=group_col, y=total_sales_col, color=group_col,
                    box=True, points="outliers",
                    labels={total_sales_col: lbl(total_sales_col), group_col: lbl(group_col)},
                )
                fig.update_layout(showlegend=False)
                _pchart(fig)

            _d1, _d2 = st.columns(2)
            with _d1:
                if road_col and road_col in df.columns and total_sales_col and total_sales_col in df.columns:
                    st.markdown("#### По типу дороги")
                    fig = px.box(
                        df, x=road_col, y=total_sales_col, color=road_col,
                        labels={road_col: lbl(road_col), total_sales_col: lbl(total_sales_col)},
                    )
                    fig.update_layout(showlegend=False)
                    _pchart(fig)

            with _d2:
                if sett_col and sett_col in df.columns and total_sales_col and total_sales_col in df.columns:
                    st.markdown("#### По размеру населённого пункта")
                    _sdf = df.copy()
                    _sdf[sett_col] = decode_col_series(_sdf[sett_col], sett_col, inv)
                    fig = px.box(
                        _sdf, x=sett_col, y=total_sales_col, color=sett_col,
                        labels={sett_col: lbl(sett_col), total_sales_col: lbl(total_sales_col)},
                    )
                    fig.update_layout(showlegend=False)
                    _pchart(fig)

    # ════════════════════════════════════════════════════════════════════════════
    # Подвкладка: Цены и конкуренты
    # ════════════════════════════════════════════════════════════════════════════
    with sub_prices:

        comp_prices = groups.get("competitor_prices", {})

        # Per-tab фильтр по виду топлива
        sel_fuel_p: list[str] = []
        if fuel_types:
            sel_fuel_p = st.multiselect(
                "Вид топлива",
                options=list(fuel_types.keys()),
                default=list(fuel_types.keys()),
                key="t_price_fuel",
            )
        _has_date_p = bool(date_col and date_col in df.columns)
        if not sel_fuel_p:
            st.info("Выберите хотя бы один вид топлива.")

        # ── Блок 1: Собственные цены ─────────────────────────────────────────
        st.markdown("### 🏷️ Собственные цены")

        if _has_date_p:
            _price_rows: list[pd.DataFrame] = []
            for _ft in sel_fuel_p:
                _pc = fuel_types[_ft].get("price")
                if _pc and _pc in df.columns:
                    _p = df.groupby(date_col)[_pc].mean().reset_index()
                    _p["fuel"] = _ft
                    _p.rename(columns={_pc: "price"}, inplace=True)
                    _price_rows.append(_p)
            if _price_rows:
                st.markdown("#### Динамика цен")
                fig = px.line(
                    pd.concat(_price_rows), x=date_col, y="price", color="fuel",
                    labels={date_col: lbl(date_col), "price": "Цена (руб./л)", "fuel": "Вид топлива"},
                )
                fig.update_layout(legend_title="Вид топлива")
                _pchart(fig)

        _cp1, _cp2 = st.columns(2)
        with _cp1:
            _st_price_rows: list[pd.DataFrame] = []
            for _ft in sel_fuel_p:
                _pc = fuel_types[_ft].get("price")
                if _pc and _pc in df.columns:
                    _a = df.groupby(group_col)[_pc].mean().reset_index()
                    _a["fuel"] = _ft
                    _a.rename(columns={_pc: "price"}, inplace=True)
                    _st_price_rows.append(_a)
            if _st_price_rows:
                st.markdown("#### Средние цены по станциям")
                fig = px.bar(
                    pd.concat(_st_price_rows), x=group_col, y="price",
                    color="fuel", barmode="group",
                    labels={group_col: lbl(group_col), "price": "Цена (руб./л)", "fuel": "Вид топлива"},
                )
                _pchart(fig)

        with _cp2:
            _violin_rows: list[pd.DataFrame] = []
            for _ft in sel_fuel_p:
                _pc = fuel_types[_ft].get("price")
                if _pc and _pc in df.columns:
                    _v = df[[_pc]].copy()
                    _v["fuel"] = _ft
                    _v.rename(columns={_pc: "price"}, inplace=True)
                    _violin_rows.append(_v)
            if _violin_rows:
                st.markdown("#### Распределение цен")
                fig = px.violin(
                    pd.concat(_violin_rows), x="fuel", y="price", color="fuel",
                    box=True, points="outliers",
                    labels={"fuel": "Вид топлива", "price": "Цена (руб./л)"},
                )
                fig.update_layout(showlegend=False)
                _pchart(fig)

        # ── Блок 2: Конкуренты ────────────────────────────────────────────────
        if comp_prices:
            st.divider()
            st.markdown("### 🏁 Конкуренты")

            # Строим список топлива у которого есть и своя цена и цена конкурента
            _matched: list[dict] = []
            for _ft in sel_fuel_p:
                _own_col = fuel_types[_ft].get("price")
                if not _own_col or _own_col not in df.columns:
                    continue
                # Ищем ключ конкурента по суффиксу колонки (price_AI92 → AI92)
                _suffix = _own_col.replace("price_", "")
                _comp_entry = comp_prices.get(_suffix, {})
                _std_col   = _comp_entry.get("standard")
                _brend_col = _comp_entry.get("brend")
                if _std_col and _std_col in df.columns:
                    _matched.append({
                        "fuel": _ft,
                        "own": _own_col,
                        "std": _std_col,
                        "brend": _brend_col if _brend_col and _brend_col in df.columns else None,
                    })

            if _matched and _has_date_p:
                # Ценовой спред по времени (на всю ширину)
                st.markdown("#### Ценовой спред (свои − конкурент) по времени")
                _spread_rows: list[pd.DataFrame] = []
                for _m in _matched:
                    _sp = df.groupby(date_col).apply(
                        lambda x: (x[_m["own"]] - x[_m["std"]]).mean()
                    ).reset_index(name="spread")
                    _sp["fuel"] = _m["fuel"]
                    _spread_rows.append(_sp)
                if _spread_rows:
                    _sp_df = pd.concat(_spread_rows)
                    _sp_sel_fuel = st.selectbox(
                        "Вид топлива", [_m["fuel"] for _m in _matched], key="t_spread_fuel"
                    )
                    _sp_one = _sp_df[_sp_df["fuel"] == _sp_sel_fuel].copy()
                    _sp_one["_month"] = pd.to_datetime(_sp_one[date_col]).dt.to_period("M").astype(str)
                    _sp_mon = _sp_one.groupby("_month")["spread"].mean().reset_index()

                    fig = go.Figure()
                    # Заливка выше нуля — красная (дороже)
                    fig.add_trace(go.Scatter(
                        x=_sp_mon["_month"], y=_sp_mon["spread"].clip(lower=0),
                        fill="tozeroy", mode="none",
                        fillcolor="rgba(239,83,80,0.35)", name="Дороже конкурента",
                    ))
                    # Заливка ниже нуля — зелёная (дешевле)
                    fig.add_trace(go.Scatter(
                        x=_sp_mon["_month"], y=_sp_mon["spread"].clip(upper=0),
                        fill="tozeroy", mode="none",
                        fillcolor="rgba(102,187,106,0.35)", name="Дешевле конкурента",
                    ))
                    # Линия спреда
                    fig.add_trace(go.Scatter(
                        x=_sp_mon["_month"], y=_sp_mon["spread"],
                        mode="lines", line=dict(color="#546e7a", width=2),
                        name="Спред",
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
                    fig.update_layout(
                        yaxis_title="Спред (руб./л)",
                        xaxis_title="Месяц",
                        legend=dict(orientation="h", y=-0.25, x=0),
                        margin=dict(b=80),
                    )
                    fig.update_xaxes(tickangle=45)
                    _pchart(fig)

            if _matched:
                _cmp1, _cmp2 = st.columns(2)
                with _cmp1:
                    # Средний спред по видам топлива (bar)
                    st.markdown("#### Средний спред по видам топлива")
                    _bar_sp = []
                    for _m in _matched:
                        _diff = (df[_m["own"]] - df[_m["std"]]).mean()
                        _bar_sp.append({"fuel": _m["fuel"], "spread": round(_diff, 3)})
                    if _bar_sp:
                        _bsp_df = pd.DataFrame(_bar_sp)
                        fig = px.bar(
                            _bsp_df, x="fuel", y="spread", color="spread",
                            color_continuous_scale=["red", "white", "green"],
                            color_continuous_midpoint=0,
                            labels={"fuel": "Вид топлива", "spread": "Спред (руб./л)"},
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig.update_layout(coloraxis_showscale=False)
                        _pchart(fig)

                with _cmp2:
                    # Scatter: цена → продажи (ценовая эластичность)
                    st.markdown("#### Цена vs Продажи")
                    _el_sel = st.selectbox(
                        "Вид топлива", [_m["fuel"] for _m in _matched], key="t_elast"
                    )
                    _el_m = next((_m for _m in _matched if _m["fuel"] == _el_sel), None)
                    if _el_m:
                        _sc_col = fuel_types[_el_m["fuel"]].get("sales")
                        if _sc_col and _sc_col in df.columns:
                            fig = px.scatter(
                                df, x=_el_m["own"], y=_sc_col,
                                color=group_col, trendline="ols",
                                labels={
                                    _el_m["own"]: lbl(_el_m["own"]),
                                    _sc_col: lbl(_sc_col),
                                    group_col: lbl(group_col),
                                },
                            )
                            fig.update_layout(legend_title=lbl(group_col))
                            _pchart(fig)

            if not _matched:
                st.info("Нет топлива с совпадающими колонками цен конкурента.")

    # ════════════════════════════════════════════════════════════════════════════
    # Подвкладка: Погода и трафик
    # ════════════════════════════════════════════════════════════════════════════
    with sub_weather:
        sp_w      = groups.get("special_cols", {})
        temp_col  = sp_w.get("temperature_col")
        prec_col  = sp_w.get("precipitation_col")
        wcond_col = sp_w.get("weather_condition_col")
        _t_lanes  = groups.get("traffic_lanes", {})
        _v_types  = groups.get("vehicle_types", {})

        _has_date_w = bool(date_col and date_col in df.columns)
        _fiz_cols   = [c for c in df.columns if "intensiv_fiz" in c]

        # Per-tab фильтр по типу погоды
        _wcond_filter = "Все"
        if wcond_col and wcond_col in df.columns:
            _wc_opts, _wc_lmap = _decoded_opts(wcond_col)
            _wcond_filter = st.selectbox(
                "Тип погоды", ["Все"] + _wc_opts, key="t_wcond"
            )
        dfw = df.copy()
        if _wcond_filter != "Все" and wcond_col and wcond_col in dfw.columns:
            _raw_wc = _wc_lmap.get(_wcond_filter)
            if _raw_wc is not None:
                dfw = dfw[dfw[wcond_col] == _raw_wc]

        # ── Блок 1: Погода ───────────────────────────────────────────────────
        if temp_col and temp_col in dfw.columns:
            st.markdown("### 🌡️ Погода")

            # Двойная ось: температура + продажи по времени
            if _has_date_w and total_sales_col and total_sales_col in dfw.columns:
                st.markdown("#### Температура и продажи по времени")
                _tmp_d = dfw.groupby(date_col).agg(
                    {total_sales_col: "sum", temp_col: "mean"}
                ).reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=_tmp_d[date_col], y=_tmp_d[total_sales_col],
                    name=lbl(total_sales_col), yaxis="y1",
                ))
                fig.add_trace(go.Scatter(
                    x=_tmp_d[date_col], y=_tmp_d[temp_col],
                    name=lbl(temp_col), yaxis="y2",
                    line=dict(dash="dot", color="tomato"),
                ))
                fig.update_layout(
                    yaxis=dict(title=lbl(total_sales_col)),
                    yaxis2=dict(title=lbl(temp_col), overlaying="y", side="right"),
                    legend=dict(orientation="h"),
                )
                _pchart(fig)

            _w1, _w2 = st.columns(2)
            with _w1:
                if total_sales_col and total_sales_col in dfw.columns:
                    st.markdown("#### Температура → Продажи")
                    try:
                        fig = px.scatter(
                            dfw, x=temp_col, y=total_sales_col,
                            color=group_col, trendline="lowess",
                            labels={
                                temp_col: lbl(temp_col),
                                total_sales_col: lbl(total_sales_col),
                                group_col: lbl(group_col),
                            },
                        )
                    except Exception:
                        fig = px.scatter(
                            dfw, x=temp_col, y=total_sales_col, color=group_col,
                            labels={temp_col: lbl(temp_col), total_sales_col: lbl(total_sales_col)},
                        )
                    fig.update_layout(showlegend=False)
                    _pchart(fig)

            with _w2:
                if prec_col and prec_col in dfw.columns and total_sales_col and total_sales_col in dfw.columns:
                    st.markdown("#### Осадки → Продажи")
                    _rain = dfw.copy()
                    _rain["Осадки"] = pd.cut(
                        _rain[prec_col], bins=[-1, 0, 10, 9999],
                        labels=["0 мм", "0–10 мм", ">10 мм"]
                    )
                    fig = px.box(
                        _rain, x="Осадки", y=total_sales_col, color="Осадки",
                        labels={"Осадки": lbl(prec_col), total_sales_col: lbl(total_sales_col)},
                        category_orders={"Осадки": ["0 мм", "0–10 мм", ">10 мм"]},
                    )
                    fig.update_layout(showlegend=False)
                    _pchart(fig)

            if wcond_col and wcond_col in dfw.columns and total_sales_col and total_sales_col in dfw.columns:
                _w3, _w4 = st.columns(2)
                with _w3:
                    st.markdown("#### Тип погоды → Продажи")
                    _wc_df = dfw.copy()
                    _wc_df[wcond_col] = decode_col_series(_wc_df[wcond_col], wcond_col, inv)
                    fig = px.box(
                        _wc_df, x=wcond_col, y=total_sales_col, color=wcond_col,
                        labels={wcond_col: lbl(wcond_col), total_sales_col: lbl(total_sales_col)},
                    )
                    fig.update_layout(showlegend=False)
                    _pchart(fig)

                with _w4:
                    if _has_date_w:
                        st.markdown("#### Температура по сезонам")
                        _tmp_s = dfw.copy()
                        if "season" in _tmp_s.columns:
                            _tmp_s["season_d"] = _tmp_s["season"].map(
                                groups.get("value_remaps", {}).get("season", {})
                            ).fillna(_tmp_s["season"])
                            fig = px.box(
                                _tmp_s, x="season_d", y=temp_col, color="season_d",
                                labels={"season_d": lbl("season"), temp_col: lbl(temp_col)},
                            )
                            fig.update_layout(showlegend=False)
                            _pchart(fig)

        # ── Блок 2: Трафик ───────────────────────────────────────────────────
        if _fiz_cols or _t_lanes or _v_types:
            st.divider()
            st.markdown("### 🚗 Трафик")

            # Интенсивность по полосам (full width)
            _lane_cols = {n: c for n, c in _t_lanes.items() if c in dfw.columns}
            if _lane_cols and _has_date_w:
                st.markdown("#### Интенсивность трафика по полосам")
                _lane_rows: list[pd.DataFrame] = []
                for _ln, _lc in _lane_cols.items():
                    _la = dfw.groupby(date_col)[_lc].mean().reset_index()
                    _la["lane"] = _ln
                    _la.rename(columns={_lc: "intensity"}, inplace=True)
                    _lane_rows.append(_la)
                if _lane_rows:
                    fig = px.line(
                        pd.concat(_lane_rows), x=date_col, y="intensity", color="lane",
                        labels={date_col: lbl(date_col), "intensity": "Авт/ч", "lane": "Полоса"},
                    )
                    _pchart(fig)

            _tr1, _tr2 = st.columns(2)
            with _tr1:
                # Трафик по типам ТС (stacked bar)
                _vt_avail = {
                    n: [c for c in cols if c in dfw.columns]
                    for n, cols in _v_types.items()
                    if isinstance(cols, list)
                }
                _vt_avail = {n: cols for n, cols in _vt_avail.items() if cols}
                if _vt_avail:
                    st.markdown("#### Структура трафика по типам ТС")
                    _vt_data = [
                        {"Тип ТС": n, "Авт/сут": dfw[cols].sum(axis=1).mean()}
                        for n, cols in _vt_avail.items()
                    ]
                    fig = px.bar(
                        pd.DataFrame(_vt_data), x="Тип ТС", y="Авт/сут", color="Тип ТС",
                        labels={"Тип ТС": "Тип ТС", "Авт/сут": "Среднее авт/сут"},
                    )
                    fig.update_layout(showlegend=False)
                    _pchart(fig)

            with _tr2:
                # Трафик → Продажи (scatter)
                if _fiz_cols and total_sales_col and total_sales_col in dfw.columns:
                    st.markdown("#### Трафик → Продажи")
                    _trs = dfw.copy()
                    _trs["_traffic"] = _trs[_fiz_cols].sum(axis=1)
                    fig = px.scatter(
                        _trs, x="_traffic", y=total_sales_col,
                        color=group_col, trendline="ols",
                        labels={
                            "_traffic": "Суммарный трафик (авт/ч)",
                            total_sales_col: lbl(total_sales_col),
                            group_col: lbl(group_col),
                        },
                    )
                    fig.update_layout(showlegend=False)
                    _pchart(fig)

        if not (temp_col and temp_col in df.columns) and not _fiz_cols:
            st.info("Нет данных о погоде и трафике в текущей сессии.")

    # ════════════════════════════════════════════════════════════════════════════
    # Подвкладка: Магазин и клиенты
    # ════════════════════════════════════════════════════════════════════════════
    with sub_shop:
        sp_sh     = groups.get("special_cols", {})
        corp_col  = sp_sh.get("corporate_ratio_col")
        loyal_col = sp_sh.get("loyalty_score_col")
        _has_date_s = bool(date_col and date_col in df.columns)

        # Per-tab фильтр по категориям магазина
        sel_shop_cats = list(shop_cats.keys())
        if shop_cats:
            sel_shop_cats = st.multiselect(
                "Категории магазина",
                options=list(shop_cats.keys()),
                default=list(shop_cats.keys()),
                key="t_shop_cats",
            )
        _active_cats = {k: v for k, v in shop_cats.items() if k in sel_shop_cats and v in df.columns}

        if not _active_cats and not (corp_col and corp_col in df.columns):
            st.info("Нет данных о магазине и клиентах в текущей сессии.")
        else:
            # ── Блок 1: Выручка магазина ─────────────────────────────────────
            if _active_cats:
                st.markdown("### 🛒 Выручка магазина")

                _sh1, _sh2 = st.columns(2)
                with _sh1:
                    st.markdown("#### Структура выручки по категориям")
                    _struct_sh = [
                        {"cat": k, "total": df[v].sum()}
                        for k, v in _active_cats.items()
                    ]
                    fig = px.pie(
                        pd.DataFrame(_struct_sh), names="cat", values="total", hole=0.45,
                        labels={"cat": "Категория", "total": "Выручка (руб.)"},
                    )
                    fig.update_traces(textposition="inside", textinfo="percent+label")
                    _pchart(fig)

                with _sh2:
                    st.markdown("#### Средняя выручка по категориям")
                    _avg_sh = [
                        {"cat": k, "avg": df[v].mean()}
                        for k, v in _active_cats.items()
                    ]
                    _avg_sh_df = pd.DataFrame(_avg_sh).sort_values("avg", ascending=True)
                    fig = px.bar(
                        _avg_sh_df, x="avg", y="cat", orientation="h",
                        labels={"avg": "Средняя выручка (руб./сут)", "cat": "Категория"},
                    )
                    _pchart(fig)

                # Выручка по станциям (full width)
                st.markdown("#### Выручка магазина по станциям")
                _st_sh_rows: list[pd.DataFrame] = []
                for k, v in _active_cats.items():
                    _a = df.groupby(group_col)[v].sum().reset_index()
                    _a["cat"] = k
                    _a.rename(columns={v: "revenue"}, inplace=True)
                    _st_sh_rows.append(_a)
                if _st_sh_rows:
                    fig = px.bar(
                        pd.concat(_st_sh_rows), x=group_col, y="revenue",
                        color="cat", barmode="stack",
                        labels={group_col: lbl(group_col), "revenue": "Выручка (руб.)", "cat": "Категория"},
                    )
                    _pchart(fig)

                _sh3, _sh4 = st.columns(2)
                with _sh3:
                    if _has_date_s:
                        st.markdown("#### Динамика выручки по месяцам")
                        _mon_sh_rows: list[pd.DataFrame] = []
                        for k, v in _active_cats.items():
                            _m = df.copy()
                            _m["_month"] = _m[date_col].dt.to_period("M").astype(str)
                            _a = _m.groupby("_month")[v].sum().reset_index()
                            _a["cat"] = k
                            _a.rename(columns={v: "revenue"}, inplace=True)
                            _mon_sh_rows.append(_a)
                        if _mon_sh_rows:
                            fig = px.line(
                                pd.concat(_mon_sh_rows), x="_month", y="revenue", color="cat",
                                labels={"_month": "Месяц", "revenue": "Выручка (руб.)", "cat": "Категория"},
                            )
                            fig.update_xaxes(tickangle=45)
                            _pchart(fig)

                with _sh4:
                    if total_sales_col and total_sales_col in df.columns:
                        st.markdown("#### Топливо vs Магазин по станциям")
                        _cmp_sh = df.copy()
                        _cmp_sh["shop_total"] = _cmp_sh[list(_active_cats.values())].sum(axis=1)
                        _cmp_agg = _cmp_sh.groupby(group_col).agg(
                            shop_total=("shop_total", "mean"),
                            fuel_sales=(total_sales_col, "mean"),
                        ).reset_index()
                        # Нормализация к % от максимума, чтобы разные единицы сравнимы
                        _max_fuel = _cmp_agg["fuel_sales"].max() or 1
                        _max_shop = _cmp_agg["shop_total"].max() or 1
                        _cmp_agg["fuel_pct"] = _cmp_agg["fuel_sales"] / _max_fuel * 100
                        _cmp_agg["shop_pct"] = _cmp_agg["shop_total"] / _max_shop * 100
                        _stations = _cmp_agg[group_col].astype(str)
                        fig = go.Figure([
                            go.Bar(
                                name=lbl(total_sales_col),
                                x=_stations,
                                y=_cmp_agg["fuel_pct"],
                                customdata=_cmp_agg["fuel_sales"],
                                hovertemplate="%{x}<br>Топливо: %{customdata:,.0f} л/сут<extra></extra>",
                            ),
                            go.Bar(
                                name="Выручка магазина",
                                x=_stations,
                                y=_cmp_agg["shop_pct"],
                                customdata=_cmp_agg["shop_total"],
                                hovertemplate="%{x}<br>Магазин: %{customdata:,.0f} руб/сут<extra></extra>",
                            ),
                        ])
                        fig.update_layout(
                            barmode="group",
                            yaxis_title="% от максимума по станциям",
                            xaxis_title=lbl(group_col),
                            legend=dict(orientation="h", y=-0.25, x=0),
                            margin=dict(b=80),
                        )
                        _pchart(fig)

            # ── Блок 2: Клиенты ───────────────────────────────────────────────
            if (corp_col and corp_col in df.columns) or (loyal_col and loyal_col in df.columns):
                st.divider()
                st.markdown("### 👥 Клиенты")

                _cl1, _cl2 = st.columns(2)
                with _cl1:
                    if corp_col and corp_col in df.columns and _has_date_s:
                        st.markdown("#### Доля корпоративных клиентов по месяцам")
                        _corp_m = df.copy()
                        _corp_m["_month"] = _corp_m[date_col].dt.to_period("M").astype(str)
                        _corp_m = _corp_m.groupby(["_month", group_col])[corp_col].mean().reset_index()
                        fig = px.line(
                            _corp_m, x="_month", y=corp_col, color=group_col,
                            markers=True,
                            labels={
                                "_month": "Месяц",
                                corp_col: lbl(corp_col),
                                group_col: lbl(group_col),
                            },
                        )
                        fig.update_xaxes(tickangle=45)
                        fig.update_layout(legend_title=lbl(group_col))
                        _pchart(fig)

                with _cl2:
                    if loyal_col and loyal_col in df.columns and _has_date_s:
                        st.markdown("#### Лояльность клиентов по месяцам")
                        _loyal_m = df.copy()
                        _loyal_m["_month"] = _loyal_m[date_col].dt.to_period("M").astype(str)
                        _loyal_m = _loyal_m.groupby(["_month", group_col])[loyal_col].mean().reset_index()
                        fig = px.line(
                            _loyal_m, x="_month", y=loyal_col, color=group_col,
                            markers=True,
                            labels={
                                "_month": "Месяц",
                                loyal_col: lbl(loyal_col),
                                group_col: lbl(group_col),
                            },
                        )
                        fig.update_xaxes(tickangle=45)
                        fig.update_layout(legend_title=lbl(group_col))
                        _pchart(fig)

                _cl3, _cl4 = st.columns(2)
                with _cl3:
                    if corp_col and corp_col in df.columns:
                        st.markdown("#### Корп. клиенты по станциям")
                        _corp_st = (
                            df.groupby(group_col)[corp_col].mean()
                            .reset_index()
                            .sort_values(corp_col, ascending=False)
                        )
                        fig = px.bar(
                            _corp_st, x=group_col, y=corp_col, color=group_col,
                            text_auto=".3g",
                            labels={group_col: lbl(group_col), corp_col: lbl(corp_col)},
                        )
                        fig.update_traces(textposition="outside")
                        fig.update_layout(showlegend=False)
                        _pchart(fig)

                with _cl4:
                    if loyal_col and loyal_col in df.columns and total_sales_col and total_sales_col in df.columns:
                        st.markdown("#### Лояльность vs Продажи топлива по станциям")
                        _loyal_agg = df.groupby(group_col).agg(
                            loyalty=(loyal_col, "mean"),
                            fuel=(total_sales_col, "mean"),
                        ).reset_index()
                        _max_loy  = _loyal_agg["loyalty"].max() or 1
                        _max_fuel = _loyal_agg["fuel"].max() or 1
                        _loyal_agg["loy_pct"]  = _loyal_agg["loyalty"] / _max_loy  * 100
                        _loyal_agg["fuel_pct"] = _loyal_agg["fuel"]    / _max_fuel * 100
                        _st_labels = _loyal_agg[group_col].astype(str)
                        fig = go.Figure([
                            go.Bar(
                                name=lbl(loyal_col),
                                x=_st_labels,
                                y=_loyal_agg["loy_pct"],
                                customdata=_loyal_agg["loyalty"],
                                hovertemplate="%{x}<br>Лояльность: %{customdata:.3g}<extra></extra>",
                            ),
                            go.Bar(
                                name=lbl(total_sales_col),
                                x=_st_labels,
                                y=_loyal_agg["fuel_pct"],
                                customdata=_loyal_agg["fuel"],
                                hovertemplate="%{x}<br>Топливо: %{customdata:,.0f} л/сут<extra></extra>",
                            ),
                        ])
                        fig.update_layout(
                            barmode="group",
                            yaxis_title="% от максимума по станциям",
                            xaxis_title=lbl(group_col),
                            legend=dict(orientation="h", y=-0.25, x=0),
                            margin=dict(b=80),
                        )
                        _pchart(fig)

    # ════════════════════════════════════════════════════════════════════════════
    # Подвкладка: Акции и праздники
    # ════════════════════════════════════════════════════════════════════════════
    with sub_promo:
        _sp = groups.get("special_cols", {})
        promo_fuel_col = _sp.get("promotion_fuel_col")
        promo_shop_col = _sp.get("promotion_shop_col")
        promo_cafe_col = _sp.get("promotion_cafe_col")
        hol_col        = _sp.get("holiday_col", "is_holiday")
        hname_col      = _sp.get("holiday_name_col", "holiday_name")

        _promo_cols  = [c for c in [promo_fuel_col, promo_shop_col, promo_cafe_col]
                        if c and c in df.columns]
        _has_promo   = bool(_promo_cols)
        _has_hol     = bool(hol_col and hol_col in df.columns)
        _has_hname   = bool(hname_col and hname_col in df.columns)
        _has_date_pr = bool(date_col and date_col in df.columns)

        if not _has_promo and not _has_hol:
            st.info("Нет данных об акциях и праздниках в текущей сессии.")
        else:
            def _promo_map(col: str, src: pd.DataFrame) -> pd.Series:
                return src[col].fillna(0).astype(float).astype(int).map(
                    {0: "Без акции", 1: "С акцией"}
                )

            # ── Блок: Акции ───────────────────────────────────────────────────
            if _has_promo:
                st.markdown("### 🎯 Акции")

                _pr1, _pr2 = st.columns(2)
                with _pr1:
                    if promo_fuel_col and promo_fuel_col in df.columns and total_sales_col and total_sales_col in df.columns:
                        st.markdown("#### Эффект акции на продажи топлива")
                        _pf = df.copy()
                        _pf["_promo"] = _promo_map(promo_fuel_col, _pf)
                        fig = px.box(
                            _pf, x="_promo", y=total_sales_col, color="_promo",
                            category_orders={"_promo": ["Без акции", "С акцией"]},
                            labels={"_promo": "Акция на топливо", total_sales_col: lbl(total_sales_col)},
                            points="outliers",
                        )
                        fig.update_layout(showlegend=False)
                        _pchart(fig)

                with _pr2:
                    _shop_cols = [v for v in shop_cats.values() if v in df.columns]
                    if promo_shop_col and promo_shop_col in df.columns and _shop_cols:
                        st.markdown("#### Эффект акции в магазине")
                        _ps = df.copy()
                        _ps["_shop_total"] = _ps[_shop_cols].sum(axis=1)
                        _ps["_promo"] = _promo_map(promo_shop_col, _ps)
                        fig = px.box(
                            _ps, x="_promo", y="_shop_total", color="_promo",
                            category_orders={"_promo": ["Без акции", "С акцией"]},
                            labels={"_promo": "Акция в магазине", "_shop_total": "Выручка магазина (руб.)"},
                            points="outliers",
                        )
                        fig.update_layout(showlegend=False)
                        _pchart(fig)

                _pr3, _pr4 = st.columns(2)
                with _pr3:
                    _cafe_col = shop_cats.get("Кафе")
                    if promo_cafe_col and promo_cafe_col in df.columns and _cafe_col and _cafe_col in df.columns:
                        st.markdown("#### Эффект акции в кафе")
                        _pc = df.copy()
                        _pc["_promo"] = _promo_map(promo_cafe_col, _pc)
                        fig = px.box(
                            _pc, x="_promo", y=_cafe_col, color="_promo",
                            category_orders={"_promo": ["Без акции", "С акцией"]},
                            labels={"_promo": "Акция в кафе", _cafe_col: lbl(_cafe_col)},
                            points="outliers",
                        )
                        fig.update_layout(showlegend=False)
                        _pchart(fig)

                with _pr4:
                    if _has_date_pr:
                        st.markdown("#### Активность акций по месяцам")
                        _act = df.copy()
                        _act["_month"] = _act[date_col].dt.to_period("M").astype(str)
                        _promo_labels = {
                            promo_fuel_col: "Топливо",
                            promo_shop_col: "Магазин",
                            promo_cafe_col: "Кафе",
                        }
                        _act_rows: list[pd.DataFrame] = []
                        for _pcol in _promo_cols:
                            _a = (
                                _act.groupby([date_col, "_month"])[_pcol].max()
                                .reset_index()
                                .groupby("_month")[_pcol].sum()
                                .reset_index(name="Дней")
                            )
                            _a["Акция"] = _promo_labels.get(_pcol, _pcol)
                            _act_rows.append(_a)
                        if _act_rows:
                            fig = px.bar(
                                pd.concat(_act_rows), x="_month", y="Дней", color="Акция",
                                barmode="group",
                                labels={"_month": "Месяц", "Дней": "Дней с акцией"},
                            )
                            fig.update_xaxes(tickangle=45)
                            fig.update_layout(
                                legend=dict(orientation="h", y=-0.25, x=0),
                                margin=dict(b=80),
                            )
                            _pchart(fig)

                # Одновременные акции (full width)
                if len(_promo_cols) > 1 and total_sales_col and total_sales_col in df.columns:
                    st.markdown("#### Эффект одновременных акций на продажи топлива")
                    _multi = df.copy()
                    _multi["_n"] = _multi[_promo_cols].fillna(0).astype(float).astype(int).sum(axis=1)
                    _lmap_n = {0: "0 акций", 1: "1 акция", 2: "2 акции", 3: "3 акции"}
                    _multi["_n_label"] = _multi["_n"].map(_lmap_n).fillna("3+ акции")
                    _order = [_lmap_n[k] for k in range(len(_promo_cols) + 1) if k in _lmap_n]
                    fig = px.box(
                        _multi, x="_n_label", y=total_sales_col, color="_n_label",
                        category_orders={"_n_label": _order},
                        labels={"_n_label": "Кол-во активных акций", total_sales_col: lbl(total_sales_col)},
                        points="outliers",
                    )
                    fig.update_layout(showlegend=False)
                    _pchart(fig)

            # ── Блок: Праздники ───────────────────────────────────────────────
            if _has_hol:
                if _has_promo:
                    st.divider()
                st.markdown("### 🎉 Праздники")

                _hol_rows = df[df[hol_col].fillna(0).astype(int) == 1]

                _hl1, _hl2 = st.columns(2)
                with _hl1:
                    if total_sales_col and total_sales_col in df.columns:
                        _hdf = df.copy()
                        _hdf["_dtype"] = _hdf[hol_col].fillna(0).astype(int).map(
                            {0: "Обычный день", 1: "Праздник"}
                        )
                        _present_types = sorted(_hdf["_dtype"].dropna().unique(),
                                                key=lambda x: x == "Праздник")
                        if len(_present_types) > 1:
                            st.markdown("#### Продажи топлива: праздники vs обычные дни")
                            fig = px.box(
                                _hdf, x="_dtype", y=total_sales_col, color="_dtype",
                                category_orders={"_dtype": ["Обычный день", "Праздник"]},
                                labels={"_dtype": "", total_sales_col: lbl(total_sales_col)},
                                points="outliers",
                            )
                            fig.update_layout(showlegend=False)
                            _pchart(fig)

                with _hl2:
                    if (_has_hname and total_sales_col and total_sales_col in df.columns
                            and not _hol_rows.empty and hname_col in _hol_rows.columns):
                        _hn_agg = (
                            _hol_rows.groupby(hname_col)[total_sales_col].mean()
                            .reset_index()
                            .sort_values(total_sales_col, ascending=True)
                        )
                        if not _hn_agg.empty:
                            st.markdown("#### Средние продажи по праздникам")
                            fig = px.bar(
                                _hn_agg, x=total_sales_col, y=hname_col,
                                orientation="h", text_auto=",.0f",
                                labels={
                                    total_sales_col: lbl(total_sales_col) + " (ср.)",
                                    hname_col: "Праздник",
                                },
                            )
                            fig.update_traces(textposition="outside")
                            _pchart(fig)

                if _hol_rows.empty:
                    st.info("В выбранном периоде праздничных дней нет.")
                elif _has_date_pr:
                    _month_names = {
                        1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр",
                        5: "Май", 6: "Июн", 7: "Июл", 8: "Авг",
                        9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
                    }
                    _hl3, _hl4 = st.columns(2)
                    with _hl3:
                        _hm = _hol_rows.copy()
                        _hm["_mnum"] = _hm[date_col].dt.month
                        _hm["_mname"] = _hm["_mnum"].map(_month_names)
                        _hm_freq = (
                            _hm.groupby(["_mnum", "_mname"])[date_col].nunique()
                            .reset_index(name="Праздничных дней")
                            .sort_values("_mnum")
                        )
                        if not _hm_freq.empty:
                            st.markdown("#### Частота праздников по месяцам")
                            fig = px.bar(
                                _hm_freq, x="_mname", y="Праздничных дней",
                                text_auto=True,
                                labels={"_mname": "Месяц"},
                            )
                            fig.update_traces(textposition="outside")
                            _pchart(fig)

                    with _hl4:
                        if total_sales_col and total_sales_col in df.columns:
                            _hm2 = _hol_rows.copy()
                            if not _hm2.empty:
                                _hm2["_mnum"] = _hm2[date_col].dt.month
                                _hm2["_mname"] = _hm2["_mnum"].map(_month_names)
                                _hm2_agg = (
                                    _hm2.groupby(["_mnum", "_mname"])[total_sales_col]
                                    .mean()
                                    .reset_index()
                                    .sort_values("_mnum")
                                )
                                if not _hm2_agg.empty:
                                    st.markdown("#### Средние продажи в праздники по месяцам")
                                    fig = px.bar(
                                        _hm2_agg, x="_mname", y=total_sales_col,
                                        text_auto=",.0f",
                                        labels={
                                            "_mname": "Месяц",
                                            total_sales_col: lbl(total_sales_col) + " (ср.)",
                                        },
                                    )
                                    fig.update_traces(textposition="outside")
                                    _pchart(fig)

    # ════════════════════════════════════════════════════════════════════════════
    # Подвкладка: Станции
    # ════════════════════════════════════════════════════════════════════════════
    with sub_stations:
        _has_sales_st = bool(total_sales_col and total_sales_col in df.columns)
        _fiz_cols_st  = [c for c in df.columns if "intensiv_fiz" in c]
        _shop_cols_st = [v for v in shop_cats.values() if v in df.columns]
        _short = lambda c: (lbl(c)[:18] + "…") if len(lbl(c)) > 20 else lbl(c)

        # ── Блок 1: Продажи ───────────────────────────────────────────────────
        st.markdown("### 📊 Продажи по станциям")
        _sst1, _sst2 = st.columns(2)

        with _sst1:
            if _has_sales_st:
                st.markdown("#### Средние продажи по станциям")
                _sales_st = (
                    df.groupby(group_col)[total_sales_col].mean()
                    .reset_index()
                    .sort_values(total_sales_col, ascending=False)
                )
                fig = px.bar(
                    _sales_st, x=group_col, y=total_sales_col,
                    color=group_col, text_auto=",.0f",
                    labels={group_col: lbl(group_col), total_sales_col: lbl(total_sales_col) + " (ср.)"},
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                _pchart(fig)

        with _sst2:
            if road_col and road_col in df.columns and _has_sales_st:
                _sst2_title = "#### Продажи по типу дороги"
                _df_rd = df.copy()
                _df_rd[road_col] = decode_col_series(_df_rd[road_col], road_col, inv)
                if dir_col and dir_col in df.columns:
                    _sst2_title += " и направлению"
                    _df_rd[dir_col] = decode_col_series(_df_rd[dir_col], dir_col, inv)
                    st.markdown(_sst2_title)
                    fig = px.box(
                        _df_rd, x=road_col, y=total_sales_col, color=dir_col,
                        labels={
                            road_col: lbl(road_col),
                            total_sales_col: lbl(total_sales_col),
                            dir_col: lbl(dir_col),
                        },
                        points="outliers",
                    )
                    fig.update_layout(legend_title=lbl(dir_col))
                else:
                    st.markdown(_sst2_title)
                    fig = px.box(
                        _df_rd, x=road_col, y=total_sales_col, color=road_col,
                        labels={road_col: lbl(road_col), total_sales_col: lbl(total_sales_col)},
                        points="outliers",
                    )
                    fig.update_layout(showlegend=False)
                _pchart(fig)

        if dist_col and dist_col in df.columns and _has_sales_st:
            st.markdown("#### Продажи по удалённости от города")
            _df_dist = df.copy()
            if _df_dist[dist_col].nunique() > 8:
                _df_dist["_dist_bin"] = pd.cut(
                    _df_dist[dist_col], bins=5, precision=0
                ).astype(str)
                _x_dist, _x_lbl_dist = "_dist_bin", lbl(dist_col) + " (км, диапазон)"
            else:
                _df_dist[dist_col] = decode_col_series(_df_dist[dist_col], dist_col, inv)
                _x_dist, _x_lbl_dist = dist_col, lbl(dist_col)
            fig = px.box(
                _df_dist, x=_x_dist, y=total_sales_col, color=_x_dist,
                labels={_x_dist: _x_lbl_dist, total_sales_col: lbl(total_sales_col)},
                points="outliers",
            )
            fig.update_layout(showlegend=False)
            _pchart(fig)

        # ── Блок 2: Корреляции ────────────────────────────────────────────────
        _num_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in ([date_col] if date_col else [])
        ]

        if _has_sales_st and total_sales_col in _num_cols:
            st.divider()
            st.markdown("### 🔗 Корреляции")
            _corr_all = df[_num_cols].corr()[total_sales_col].drop(total_sales_col, errors="ignore").dropna()
            _corr_top = _corr_all.abs().sort_values(ascending=False).head(20)
            _corr_show = _corr_all[_corr_top.index].reset_index()
            _corr_show.columns = ["col", "Корреляция"]
            _corr_show["Признак"] = _corr_show["col"].map(lbl)
            _corr_show = _corr_show.sort_values("Корреляция")

            _scr1, _scr2 = st.columns([2, 1])
            with _scr1:
                st.markdown("#### Значимые факторы (топ-20 по |r| с целевой)")
                fig = px.bar(
                    _corr_show, x="Корреляция", y="Признак", orientation="h",
                    color="Корреляция",
                    color_continuous_scale=["#ef5350", "#ffffff", "#66bb6a"],
                    color_continuous_midpoint=0,
                    labels={"Корреляция": "Корреляция Пирсона"},
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.update_layout(coloraxis_showscale=False, yaxis_title="")
                _pchart(fig)

            with _scr2:
                st.markdown("#### Матрица корреляций (топ-10)")
                _heat_cols = _corr_top.head(10).index.tolist()
                if total_sales_col not in _heat_cols:
                    _heat_cols = [total_sales_col] + _heat_cols[:9]
                _cm = df[_heat_cols].corr()
                _cm.index   = [_short(c) for c in _cm.index]
                _cm.columns = [_short(c) for c in _cm.columns]
                fig = px.imshow(
                    _cm,
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    zmin=-1, zmax=1,
                    text_auto=".2f",
                    aspect="auto",
                    labels={"color": "r"},
                )
                fig.update_layout(
                    coloraxis_colorbar=dict(title="r"),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                _pchart(fig)

        # ── Блок 3: Radar ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🕸️ Профиль станций")

        _radar_metrics: list[str] = []
        if _has_sales_st:
            _radar_metrics.append("Продажи топлива")
        if _fiz_cols_st:
            _radar_metrics.append("Трафик")
        if _shop_cols_st:
            _radar_metrics.append("Выручка магазина")

        if len(_radar_metrics) >= 3:
            _radar_data: dict[str, list[float]] = {}
            for _st in sorted(df[group_col].unique()):
                _sdf = df[df[group_col] == _st]
                _vals: list[float] = []
                if "Продажи топлива" in _radar_metrics:
                    _vals.append(float(_sdf[total_sales_col].mean()))
                if "Трафик" in _radar_metrics:
                    _vals.append(float(_sdf[_fiz_cols_st].sum(axis=1).mean()))
                if "Выручка магазина" in _radar_metrics:
                    _vals.append(float(_sdf[_shop_cols_st].sum(axis=1).mean()))
                _radar_data[str(_st)] = _vals

            _max_vals = [
                max(v[i] for v in _radar_data.values()) or 1
                for i in range(len(_radar_metrics))
            ]
            _theta = _radar_metrics + [_radar_metrics[0]]

            fig = go.Figure()
            for _st_name, _vals in _radar_data.items():
                _r = [v / mx for v, mx in zip(_vals, _max_vals)]
                fig.add_trace(go.Scatterpolar(
                    r=_r + [_r[0]],
                    theta=_theta,
                    fill="toself",
                    name=_st_name,
                    opacity=0.65,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
                legend=dict(orientation="h", y=-0.15, x=0),
                margin=dict(b=80),
            )
            _pchart(fig)
        else:
            st.info("Недостаточно метрик для radar chart (нужны продажи, трафик и выручка магазина).")

        # ── Блок 4: Таблица характеристик ─────────────────────────────────────
        st.divider()
        st.markdown("### 📋 Характеристики станций")

        _static_cols = [
            c for c in df.columns
            if c != group_col and c != date_col
            and df[group_col].notna().any()
            and df.groupby(group_col)[c].nunique().max() == 1
        ]
        if _static_cols:
            _tbl = df.groupby(group_col)[_static_cols].first().reset_index()
            _tbl.columns = [lbl(c) for c in _tbl.columns]
            st.dataframe(_tbl, width="stretch", hide_index=True)
        else:
            st.info("Статические характеристики станций не обнаружены.")

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка: Статистика
# ══════════════════════════════════════════════════════════════════════════════
with tab_stat:
    _exc = {date_col} if date_col else set()
    _num_m = [c for c in merged.select_dtypes(include="number").columns if c not in _exc]
    _num_p = (
        [c for c in proc.select_dtypes(include="number").columns if c not in _exc]
        if proc is not None else []
    )

    # ── Описательная статистика ────────────────────────────────────────────────
    st.markdown("### 📋 Описательная статистика")

    def _desc_table(src: pd.DataFrame, cols: list) -> pd.DataFrame:
        d = src[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].copy()
        d["missing"]   = src[cols].isnull().sum()
        d["missing_%"] = (src[cols].isnull().mean() * 100).round(1)
        d.index = [lbl(c) for c in d.index]
        d.columns = ["N", "Среднее", "Ст. откл.", "Мин", "Медиана", "Макс", "Пропуски", "Пропуски (%)"]
        return d

    if proc is not None:
        _st_tab_before, _st_tab_after = st.tabs(["До нормализации", "После нормализации"])
        with _st_tab_before:
            st.dataframe(_desc_table(merged, _num_m).style.format(precision=3),
                         width="stretch")
        with _st_tab_after:
            st.dataframe(_desc_table(proc, _num_p).style.format(precision=3),
                         width="stretch")
    else:
        st.dataframe(_desc_table(merged, _num_m).style.format(precision=3),
                     width="stretch")

    st.divider()

    # ── Распределение до / после нормализации ─────────────────────────────────
    if proc is not None and inv:
        st.markdown("### 📊 Распределение до / после нормализации")
        _common = [c for c in inv if c in merged.columns and c in proc.columns]
        if _common:
            _sel_hist = st.selectbox(
                "Колонка", _common, format_func=lbl, key="stat_hist_col"
            )
            _hh1, _hh2 = st.columns(2)
            with _hh1:
                st.markdown(f"**До** — {lbl(_sel_hist)}")
                fig = px.histogram(
                    merged, x=_sel_hist, nbins=40,
                    labels={_sel_hist: lbl(_sel_hist)},
                )
                _pchart(fig)
            with _hh2:
                st.markdown("**После** (нормализовано)")
                fig = px.histogram(
                    proc, x=_sel_hist, nbins=40,
                    labels={_sel_hist: lbl(_sel_hist)},
                    color_discrete_sequence=["#ff7043"],
                )
                _pchart(fig)
        else:
            st.info("Нет общих колонок между merged и proc.")
        st.divider()

    # ── Пропущенные значения ───────────────────────────────────────────────────
    st.markdown("### ❓ Пропущенные значения")
    _miss = merged.isnull().sum()
    _miss = _miss[_miss > 0].sort_values(ascending=True)
    if _miss.empty:
        st.success("Пропущенных значений нет.")
    else:
        _miss_df = _miss.reset_index()
        _miss_df.columns = ["col", "missing"]
        _miss_df["Колонка"]  = _miss_df["col"].map(lbl)
        _miss_df["missing_%"] = (_miss_df["missing"] / len(merged) * 100).round(1)
        fig = px.bar(
            _miss_df, x="missing", y="Колонка", orientation="h",
            labels={"missing": "Пропущено (строк)", "Колонка": ""},
            color="missing_%",
            color_continuous_scale="Reds",
        )
        fig.update_traces(
            text=_miss_df["missing_%"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
        )
        fig.update_layout(coloraxis_showscale=False)
        _pchart(fig)

    st.divider()

    # ── Список преобразований ──────────────────────────────────────────────────
    st.markdown("### 🔄 Применённые преобразования")
    if inv:
        _trans = []
        for _col, _meta in inv.items():
            _method = _meta.get("method", "none")
            if _method in ("none", None):
                continue
            _params = _meta.get("params", {})
            _pstr = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in _params.items()
            )
            _trans.append({"Колонка": lbl(_col), "Метод": _method, "Параметры": _pstr})
        if _trans:
            st.dataframe(pd.DataFrame(_trans), width="stretch", hide_index=True)
        else:
            st.info("Преобразования не зафиксированы в inverse_transforms.")
    else:
        st.info("Информация о преобразованиях не найдена (inv=None).")

    st.divider()

    # ── Выбросы по колонкам ────────────────────────────────────────────────────
    st.markdown("### 📦 Выбросы по колонкам")
    _default_box = (
        [total_sales_col] + [c for c in _num_m if c != total_sales_col]
    )[:5] if total_sales_col and total_sales_col in _num_m else _num_m[:5]

    _sel_box = st.multiselect(
        "Колонки для анализа выбросов",
        options=_num_m,
        default=_default_box,
        format_func=lbl,
        key="stat_box_sel",
    )
    if _sel_box:
        _box_rows: list[pd.DataFrame] = []
        for _bc in _sel_box:
            _tmp = merged[[_bc]].copy()
            _tmp["Колонка"] = lbl(_bc)
            _tmp.rename(columns={_bc: "_val"}, inplace=True)
            _box_rows.append(_tmp)
        fig = px.box(
            pd.concat(_box_rows), x="Колонка", y="_val", color="Колонка",
            labels={"Колонка": "", "_val": "Значение"},
            points="outliers",
        )
        fig.update_layout(showlegend=False)
        _pchart(fig)
    else:
        st.info("Выберите хотя бы одну колонку.")

# ══════════════════════════════════════════════════════════════════════════════
# Прогнозы / Рекомендации — заглушки
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    import glob as _glob

    _TRAINING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training"))
    _pred_dir      = os.path.join(_TRAINING_ROOT, selected_session, "predictions")
    _pred_files    = sorted(
        _glob.glob(os.path.join(_pred_dir, "*.csv")),
        key=os.path.getmtime, reverse=True,
    ) if os.path.isdir(_pred_dir) else []

    if not _pred_files:
        st.info(
            "Нет файлов прогнозов для этой сессии. "
            "Запустите прогнозирование в дашборде обучения (`train_dashboard.py`)."
        )
    else:
        # ── Выбор файла прогноза ──────────────────────────────────────────────
        _fc_col1, _fc_col2 = st.columns([3, 1])
        with _fc_col1:
            _pred_labels   = [os.path.basename(f) for f in _pred_files]
            _sel_pred_lbl  = st.selectbox("📂 Файл прогноза", _pred_labels, key="fc_file")
            _sel_pred_path = _pred_files[_pred_labels.index(_sel_pred_lbl)]

        @st.cache_data(show_spinner="Загрузка прогноза...")
        def _load_pred(path: str) -> pd.DataFrame:
            return pd.read_csv(path, parse_dates=["date"], encoding="utf-8")

        pred_df = _load_pred(_sel_pred_path)

        # Обнаружение таргетов по суффиксу _q10
        _fc_targets = sorted({c[:-4] for c in pred_df.columns if c.endswith("_q10")})

        # Дополняем _pred как среднее q10+q90 если колонки нет — все разом через concat
        _new_pred_cols = {
            f"{_ft}_pred": (pred_df[f"{_ft}_q10"] + pred_df[f"{_ft}_q90"]) / 2
            for _ft in _fc_targets
            if f"{_ft}_pred" not in pred_df.columns
            and f"{_ft}_q10" in pred_df.columns
            and f"{_ft}_q90" in pred_df.columns
        }
        if _new_pred_cols:
            pred_df = pd.concat([pred_df, pd.DataFrame(_new_pred_cols, index=pred_df.index)], axis=1)

        with _fc_col2:
            _fc_d0 = pred_df["date"].min().date() if "date" in pred_df.columns else "?"
            _fc_d1 = pred_df["date"].max().date() if "date" in pred_df.columns else "?"
            _fc_ns = pred_df[group_col].nunique() if group_col in pred_df.columns else "?"
            st.markdown(
                f"<div style='background:#1e2235;border-radius:8px;padding:10px 14px;"
                f"font-size:12px;line-height:1.8;'>"
                f"<span style='color:#9ca3af;'>📅 Период</span><br>"
                f"<b style='color:#f1f5f9;'>{_fc_d0} → {_fc_d1}</b><br>"
                f"<span style='color:#9ca3af;'>📍 Станций: </span><b style='color:#f1f5f9;'>{_fc_ns}</b>&nbsp;&nbsp;"
                f"<span style='color:#9ca3af;'>🎯 Таргетов: </span><b style='color:#f1f5f9;'>{len(_fc_targets)}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if not _fc_targets:
            st.warning("Колонки прогнозов (_q10/_q90) не найдены в файле.")
            st.stop()

        sub_eval, sub_whatif, sub_vsn = st.tabs([
            "📊 Оценка модели",
            "🔀 What-IF анализ",
            "🧠 Интерпретация VSN",
        ])

        with sub_eval:
            st.divider()

            # ── Метрики качества ──────────────────────────────────────────────────
            st.markdown("### 📊 Метрики качества")

            _met_rows: list[dict] = []
            _mape_by_st: dict[str, dict] = {}

            for _ft in _fc_targets:
                _ac, _pr = _ft, f"{_ft}_pred"
                if _ac not in pred_df.columns or _pr not in pred_df.columns:
                    continue
                _cdf = pred_df[[group_col, _ac, _pr]].dropna()
                if _cdf.empty:
                    continue
                _a, _p = _cdf[_ac], _cdf[_pr]
                _mae  = float(np.abs(_a - _p).mean())
                _rmse = float(np.sqrt(((_a - _p) ** 2).mean()))

                # MAPE только на строках где факт значимо > 0 (избегаем деление на ~0)
                _thresh = max(_a.quantile(0.05), 1e-2)
                _mask   = _a > _thresh
                if _mask.sum() >= 5:
                    _mape = float((np.abs(_a[_mask] - _p[_mask]) / _a[_mask]).mean() * 100)
                else:
                    # Слишком мало ненулевых — используем sMAPE (симметричный, не взрывается)
                    _denom = (np.abs(_a) + np.abs(_p)).replace(0, np.nan)
                    _mape  = float((2 * np.abs(_a - _p) / _denom).mean() * 100)

                _ss_res = float(((_a - _p) ** 2).sum())
                _ss_tot = float(((_a - _a.mean()) ** 2).sum())
                _r2 = (1 - _ss_res / _ss_tot) if _ss_tot > 0 else float("nan")
                _met_rows.append({
                    "Таргет":     lbl(_ft),
                    "_mape_raw":  _mape,
                    "_r2_raw":    _r2,
                    "MAPE (%)":   f"{_mape:.1f}%",
                    "R²":         f"{_r2:.3f}" if not np.isnan(_r2) else "—",
                    "MAE":        f"{_mae:,.1f}",
                    "RMSE":       f"{_rmse:,.1f}",
                })

                def _st_mape(grp: pd.DataFrame) -> float:
                    _ga, _gp = grp[_ac], grp[_pr]
                    _th = max(_ga.quantile(0.05), 1e-2)
                    _m  = _ga > _th
                    if _m.sum() >= 3:
                        return float((np.abs(_ga[_m] - _gp[_m]) / _ga[_m]).mean() * 100)
                    _dn = (np.abs(_ga) + np.abs(_gp)).replace(0, np.nan)
                    return float((2 * np.abs(_ga - _gp) / _dn).mean() * 100)

                _mape_by_st[_ft] = {
                    str(_st): _st_mape(_g) for _st, _g in _cdf.groupby(group_col)
                }

            if _met_rows:
                # ── KPI карточки ──────────────────────────────────────────────────
                _all_mapes = [r["_mape_raw"] for r in _met_rows]
                _all_r2s   = [r["_r2_raw"] for r in _met_rows if not np.isnan(r["_r2_raw"])]
                _med_mape  = float(np.median(_all_mapes))
                _med_r2    = float(np.median(_all_r2s)) if _all_r2s else float("nan")
                _pct_good  = sum(m <= 15 for m in _all_mapes) / len(_all_mapes) * 100

                def _fc_kpi(title: str, value: str, sub: str = "", color: str = "#2196F3") -> str:
                    _sub = (f'<div style="color:#9ca3af;font-size:11px;margin-top:3px;">{sub}</div>'
                            if sub else "")
                    return (
                        f'<div style="background:linear-gradient(135deg,{color}18,{color}08);'
                        f'border:1px solid {color}33;border-left:4px solid {color};'
                        f'border-radius:10px;padding:14px 16px;">'
                        f'<div style="color:#9ca3af;font-size:10px;font-weight:600;'
                        f'text-transform:uppercase;letter-spacing:0.8px;">{title}</div>'
                        f'<div style="color:#f1f5f9;font-size:20px;font-weight:700;'
                        f'margin-top:4px;">{value}</div>{_sub}</div>'
                    )

                _mape_clr = "#4CAF50" if _med_mape <= 15 else "#FF9800" if _med_mape <= 30 else "#f44336"
                _r2_clr   = "#4CAF50" if _med_r2 >= 0.8 else "#FF9800" if _med_r2 >= 0.5 else "#f44336"
                _gd_clr   = "#4CAF50" if _pct_good >= 70 else "#FF9800" if _pct_good >= 40 else "#f44336"

                _kc1, _kc2, _kc3, _kc4 = st.columns(4)
                with _kc1:
                    st.markdown(_fc_kpi("🎯 Таргетов", str(len(_met_rows)),
                                        "в прогнозе", "#9C27B0"), unsafe_allow_html=True)
                with _kc2:
                    st.markdown(_fc_kpi("📊 Медиана MAPE", f"{_med_mape:.1f}%",
                                        "по всем таргетам", _mape_clr), unsafe_allow_html=True)
                with _kc3:
                    st.markdown(_fc_kpi("📈 Медиана R²",
                                        f"{_med_r2:.3f}" if not np.isnan(_med_r2) else "—",
                                        "по всем таргетам", _r2_clr), unsafe_allow_html=True)
                with _kc4:
                    st.markdown(_fc_kpi("✅ MAPE ≤ 15%", f"{_pct_good:.0f}%",
                                        "доля «хороших» таргетов", _gd_clr), unsafe_allow_html=True)

                # ── Интерпретация ─────────────────────────────────────────────────
                with st.expander("ℹ️ Как интерпретировать метрики?"):
                    st.markdown("""
    **MAPE** *(Mean Absolute Percentage Error)* — средняя абсолютная ошибка в % от факта.
    Показывает насколько прогноз отклоняется от реального значения в относительных единицах.

    | MAPE | Интерпретация |
    |---|---|
    | < 10% | 🟢 Отличная точность |
    | 10–15% | 🟡 Хорошая точность |
    | 15–30% | 🟠 Приемлемая точность |
    | > 30% | 🔴 Низкая точность — модель требует доработки |

    **R²** — коэффициент детерминации. Доля дисперсии факта, объяснённая моделью.
    R²=1 идеальный прогноз · R²=0 модель не лучше среднего · R²<0 хуже среднего.

    **MAE / RMSE** — ошибка в оригинальных единицах (л/сут, руб/сут). RMSE сильнее штрафует крупные выбросы.
                    """)

                st.divider()

                # ── Bar MAPE по таргетам + таблица ───────────────────────────────
                _mb1, _mb2 = st.columns([1, 1])
                with _mb1:
                    st.markdown("#### MAPE по таргетам")
                    _mape_bar = (
                        pd.DataFrame([{"Таргет": r["Таргет"], "MAPE": r["_mape_raw"]}
                                      for r in _met_rows])
                        .sort_values("MAPE", ascending=True)
                    )
                    fig = px.bar(
                        _mape_bar, x="MAPE", y="Таргет", orientation="h",
                        color="MAPE",
                        color_continuous_scale=["#4CAF50", "#FF9800", "#f44336"],
                        range_color=[0, max(40, _mape_bar["MAPE"].max())],
                        labels={"MAPE": "MAPE, %"},
                    )
                    fig.add_vline(x=15, line_dash="dash", line_color="gray", opacity=0.7,
                                  annotation_text="15%", annotation_position="top right")
                    fig.update_traces(
                        texttemplate="%{x:.1f}%", textposition="outside",
                        text=_mape_bar["MAPE"],
                    )
                    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
                    _pchart(fig)

                with _mb2:
                    st.markdown("#### Таблица метрик")
                    _disp_df = pd.DataFrame([{
                        "Таргет":   r["Таргет"],
                        "MAPE (%)": r["MAPE (%)"],
                        "R²":       r["R²"],
                        "MAE":      r["MAE"],
                        "RMSE":     r["RMSE"],
                    } for r in _met_rows])
                    st.dataframe(_disp_df, hide_index=True, width="stretch")

                # ── Heatmap MAPE станция × таргет ────────────────────────────────
                if _mape_by_st:
                    st.markdown("#### MAPE: станция × таргет")
                    _hm_df = pd.DataFrame(
                        {lbl(_ft): _mape_by_st[_ft] for _ft in _fc_targets if _ft in _mape_by_st}
                    ).T
                    fig = px.imshow(
                        _hm_df,
                        color_continuous_scale=["#4CAF50", "#FF9800", "#f44336"],
                        zmin=0, zmax=40,
                        text_auto=".1f",
                        aspect="auto",
                        labels={"color": "MAPE (%)", "x": lbl(group_col), "y": "Таргет"},
                    )
                    _pchart(fig)

            st.divider()

            # ── Факт vs Прогноз ───────────────────────────────────────────────────
            st.markdown("### 📈 Факт vs Прогноз")
            _fv1, _fv2, _fv3 = st.columns([2, 2, 1])
            with _fv1:
                _sel_fc_tgt = st.selectbox(
                    "Таргет", _fc_targets, format_func=lbl, key="fc_target"
                )
            with _fv2:
                _fc_st_list = sorted(pred_df[group_col].unique()) if group_col in pred_df.columns else []
                _sel_fc_st  = st.selectbox(lbl(group_col), _fc_st_list, key="fc_station")
            with _fv3:
                _hist_days = st.select_slider(
                    "История (дней)", options=[30, 60, 90, 180], value=60, key="fc_hist_days"
                )

            _fdf = pred_df[pred_df[group_col] == _sel_fc_st].sort_values("date").copy()
            _ac  = _sel_fc_tgt
            _pr  = f"{_sel_fc_tgt}_pred"
            _q10 = f"{_sel_fc_tgt}_q10"
            _q90 = f"{_sel_fc_tgt}_q90"

            if "date" in _fdf.columns and _ac in _fdf.columns:
                _pred_start = _fdf["date"].min()

                # История из merged до начала прогноза
                _hist_df = None
                if date_col and date_col in merged.columns and _ac in merged.columns:
                    _hist_raw = merged[
                        (merged[group_col] == _sel_fc_st) &
                        (merged[date_col] < _pred_start) &
                        (merged[date_col] >= _pred_start - pd.Timedelta(days=_hist_days))
                    ][[date_col, _ac]].dropna().sort_values(date_col)
                    if not _hist_raw.empty:
                        _hist_df = _hist_raw

                fig = go.Figure()

                # История (серая)
                if _hist_df is not None:
                    fig.add_trace(go.Scatter(
                        x=_hist_df[date_col], y=_hist_df[_ac],
                        mode="lines", name=f"История ({_hist_days} дн.)",
                        line=dict(color="#6b7280", width=1.5),
                        opacity=0.7,
                    ))

                # Вертикальная линия: начало прогноза
                fig.add_shape(
                    type="line",
                    x0=str(_pred_start), x1=str(_pred_start),
                    y0=0, y1=1, yref="paper",
                    line=dict(dash="dot", color="#9ca3af", width=1.5),
                    opacity=0.8,
                )
                fig.add_annotation(
                    x=str(_pred_start), y=1, yref="paper",
                    text="Начало прогноза",
                    showarrow=False,
                    font=dict(color="#9ca3af", size=11),
                    xanchor="left", yanchor="bottom",
                )

                # Доверительный интервал
                if _q10 in _fdf.columns and _q90 in _fdf.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([_fdf["date"], _fdf["date"][::-1]]),
                        y=pd.concat([_fdf[_q90], _fdf[_q10][::-1]]),
                        fill="toself", fillcolor="rgba(33,150,243,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Интервал q10–q90", hoverinfo="skip",
                    ))

                # Факт в периоде прогноза
                fig.add_trace(go.Scatter(
                    x=_fdf["date"], y=_fdf[_ac],
                    mode="lines", name="Факт",
                    line=dict(color="#4CAF50", width=2),
                ))

                # Прогноз (медиана)
                if _pr in _fdf.columns:
                    fig.add_trace(go.Scatter(
                        x=_fdf["date"], y=_fdf[_pr],
                        mode="lines", name="Прогноз (медиана)",
                        line=dict(color="#2196F3", width=2, dash="dash"),
                    ))

                fig.update_layout(
                    xaxis_title="Дата",
                    yaxis_title=lbl(_sel_fc_tgt),
                    legend=dict(orientation="h", y=-0.22, x=0),
                    margin=dict(b=80),
                )
                _pchart(fig)

            st.divider()

            # ── Анализ ошибок ─────────────────────────────────────────────────────
            st.markdown("### 🔍 Анализ ошибок")
            _ac = _sel_fc_tgt
            _pr = f"{_sel_fc_tgt}_pred"

            if _ac in pred_df.columns and _pr in pred_df.columns:
                _err = pred_df[[group_col, "date", _ac, _pr]].dropna().copy()
                _err["_err"] = _err[_pr] - _err[_ac]

                _ea1, _ea2 = st.columns(2)
                with _ea1:
                    st.markdown("#### Scatter: факт vs прогноз")
                    fig = px.scatter(
                        _err, x=_ac, y=_pr, color=group_col,
                        trendline="ols",
                        labels={
                            _ac: lbl(_ac) + " (факт)",
                            _pr: lbl(_ac) + " (прогноз)",
                            group_col: lbl(group_col),
                        },
                    )
                    _xy_min = min(_err[_ac].min(), _err[_pr].min())
                    _xy_max = max(_err[_ac].max(), _err[_pr].max())
                    fig.add_trace(go.Scatter(
                        x=[_xy_min, _xy_max], y=[_xy_min, _xy_max],
                        mode="lines", name="Идеал",
                        line=dict(color="gray", dash="dash", width=1),
                    ))
                    fig.update_layout(showlegend=False)
                    _pchart(fig)

                with _ea2:
                    st.markdown("#### Распределение ошибок")
                    fig = px.histogram(
                        _err, x="_err", nbins=30,
                        labels={"_err": "Ошибка (прогноз − факт)"},
                        color_discrete_sequence=["#2196F3"],
                    )
                    fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    _pchart(fig)

                if "date" in _err.columns:
                    st.markdown("#### Средняя ошибка: станция × месяц")
                    _hme = _err.copy()
                    _hme["_month"] = _hme["date"].dt.to_period("M").astype(str)
                    _hme_piv = (
                        _hme.groupby([group_col, "_month"])["_err"].mean()
                        .reset_index()
                        .pivot(index=group_col, columns="_month", values="_err")
                    )
                    fig = px.imshow(
                        _hme_piv,
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        text_auto=".0f",
                        aspect="auto",
                        labels={"color": "Ошибка", "x": "Месяц", "y": lbl(group_col)},
                    )
                    _pchart(fig)

        with sub_whatif:
            st.divider()

            # ── What-IF анализ ────────────────────────────────────────────────────
            st.markdown("### 🔀 What-IF анализ")
            st.caption("Измените факторы ниже и запустите сценарий — модель пересчитает прогноз.")

            try:
                from pytorch_forecasting import TemporalFusionTransformer as _TFT
                from pytorch_forecasting import TimeSeriesDataSet as _TSD
                from pytorch_forecasting.data.encoders import (
                    EncoderNormalizer as _EncNorm, MultiNormalizer as _MNorm,
                )
                _pf_ok = True
            except ImportError:
                _pf_ok = False

            if not _pf_ok:
                st.warning("⚠️ pytorch-forecasting не установлен. Запустите: `.venv\\Scripts\\pip install pytorch-forecasting`")
            else:
                _wi_model_p  = os.path.join(_TRAINING_ROOT, selected_session, "model.ckpt")
                _wi_ckpt_dir = os.path.join(_TRAINING_ROOT, selected_session, "checkpoints")
                # Авто-выбор: берём model.ckpt, иначе последний по имени из checkpoints/
                if os.path.exists(_wi_model_p):
                    _wi_ckpt_path = _wi_model_p
                else:
                    _ckpt_files = sorted(
                        [f for f in os.listdir(_wi_ckpt_dir) if f.endswith(".ckpt")],
                        reverse=True,
                    ) if os.path.isdir(_wi_ckpt_dir) else []
                    _wi_ckpt_path = os.path.join(_wi_ckpt_dir, _ckpt_files[0]) if _ckpt_files else None

                if not _wi_ckpt_path:
                    st.info("Нет чекпоинтов для сессии. Запустите обучение в train_dashboard.py.")
                else:
                    _wi_station = st.selectbox(
                        lbl(group_col),
                        sorted(pred_df[group_col].unique()) if group_col in pred_df.columns else [],
                        key="wi_station",
                    )

                    _wi_horizon: int = st.radio(
                        "🕐 Горизонт прогноза",
                        options=[1, 7, 30],
                        format_func=lambda x: {1: "📅 День (1 д.)", 7: "📆 Неделя (7 д.)", 30: "🗓️ Месяц (30 д.)"}[x],
                        horizontal=True, key="wi_horizon",
                    )

                    # Первый день прогноза
                    _wi_spl_ui   = S.get("split") or {}
                    _wi_def_date = (
                        (pd.Timestamp(_wi_spl_ui["val_end"]) + pd.Timedelta(days=1)).date()
                        if _wi_spl_ui.get("val_end") else None
                    )
                    _wi_min_date = (
                        (pd.Timestamp(_wi_spl_ui["train_end"]) + pd.Timedelta(days=1)).date()
                        if _wi_spl_ui.get("train_end") else None
                    )
                    _wi_start_date = st.date_input(
                        "📅 Первый день прогноза",
                        value=_wi_def_date,
                        min_value=_wi_min_date,
                        key="wi_start_date",
                        help=(
                            f"Модель читает последние {S.get('tft', {}).get('encoder_length', 'N')} "
                            "дней реальных данных до этой даты как контекст, "
                            "затем предсказывает с неё вперёд на выбранный горизонт."
                        ),
                    )

                    # Предупреждение о разрыве между данными и датой прогноза
                    _wi_data_end = (
                        merged["date"].max().date()
                        if date_col and date_col in merged.columns else None
                    )
                    if _wi_data_end and _wi_start_date > _wi_data_end:
                        _gap_days = (_wi_start_date - _wi_data_end).days
                        st.warning(
                            f"⚠️ Прогноз начинается на **{_gap_days} дн.** позже конца данных "
                            f"({_wi_data_end}). Модель использует конец {_wi_data_end.year} "
                            f"как контекст — промежуток ей не виден."
                        )

                    st.markdown("#### Корректировки факторов")
                    _wif1, _wif2, _wif3 = st.columns(3)

                    _wi_price_vals: dict[str, float] = {}
                    with _wif1:
                        st.markdown("**Цены на топливо (руб/л)**")
                        _wi_price_cols = [
                            c for c in (tft.get("time_varying_known_reals") or [])
                            if c.startswith("price_") and not c.endswith("_enc")
                        ]
                        for _wpc in _wi_price_cols:
                            if _wpc in pred_df.columns and group_col in pred_df.columns:
                                _st_mask = pred_df[group_col] == _wi_station
                                if _st_mask.any() and "date" in pred_df.columns:
                                    _last_d = pred_df.loc[_st_mask, "date"].max()
                                    _last_row = pred_df.loc[_st_mask & (pred_df["date"] == _last_d), _wpc]
                                    _cur_p = float(_last_row.iloc[0]) if not _last_row.empty else float(pred_df.loc[_st_mask, _wpc].mean())
                                elif _st_mask.any():
                                    _cur_p = float(pred_df.loc[_st_mask, _wpc].mean())
                                else:
                                    _cur_p = float(pred_df[_wpc].mean())
                            else:
                                _cur_p = 60.0
                            _new_p = st.number_input(
                                lbl(_wpc),
                                min_value=0.0,
                                value=round(_cur_p, 2),
                                step=0.5,
                                format="%.2f",
                                key=f"wi_p_{_wpc}",
                            )
                            if abs(_new_p - _cur_p) > 0.01:
                                _wi_price_vals[_wpc] = _new_p

                    _sp_wi       = groups.get("special_cols", {})
                    _temp_col_wi = _sp_wi.get("temperature_col")
                    _prec_col_wi = _sp_wi.get("precipitation_col")
                    _wi_weather_schedule: dict[str, list[float]] = {}

                    with _wif2:
                        st.markdown("**Погода**")
                        _has_weather = bool(_temp_col_wi or _prec_col_wi)
                        if _has_weather and st.checkbox("Задать погоду", key="wi_use_weather"):
                            if _wi_horizon == 1:
                                if _temp_col_wi:
                                    _wi_weather_schedule[_temp_col_wi] = [
                                        float(st.number_input("Температура (°C)", -30, 40, 15, key="wi_temp_0"))
                                    ]
                                if _prec_col_wi:
                                    _wi_weather_schedule[_prec_col_wi] = [
                                        float(st.number_input("Осадки (мм)", 0, 200, 0, key="wi_prec_0"))
                                    ]
                            else:
                                _days_ru = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
                                _w_lbl = "паттерн по дням недели" if _wi_horizon == 30 else "по дням"
                                st.caption(f"Погода {_w_lbl}:")
                                _wcols: dict[str, tuple[str, int, int, int]] = {}
                                if _temp_col_wi:
                                    _wcols[_temp_col_wi] = ("Темп.(°C)", -30, 40, 10)
                                if _prec_col_wi:
                                    _wcols[_prec_col_wi] = ("Осадки(мм)", 0, 200, 0)
                                _wdf_init = pd.DataFrame(
                                    {"День": _days_ru,
                                     **{_nm: [_def] * 7 for _, (_nm, _mn, _mx, _def) in _wcols.items()}}
                                )
                                _wdf_cfg = {
                                    "День": st.column_config.TextColumn(disabled=True),
                                    **{
                                        _nm: st.column_config.NumberColumn(_nm, min_value=_mn, max_value=_mx, step=1)
                                        for _, (_nm, _mn, _mx, _def) in _wcols.items()
                                    }
                                }
                                _edited_weather = st.data_editor(
                                    _wdf_init, hide_index=True,
                                    key="wi_weather_table",
                                    column_config=_wdf_cfg,
                                )
                                for _wcol, (_nm, *_) in _wcols.items():
                                    _wi_weather_schedule[_wcol] = [float(v) for v in _edited_weather[_nm].tolist()]
                        elif not _has_weather:
                            st.caption("Нет погодных колонок в сессии")

                    # Сбор колонок акций + value_labels из prep_config
                    # {enc_col: (ui_label, {int_val: str_label}, raw_col)}
                    _pm_cols_wi: dict[str, tuple[str, dict[int, str], str]] = {}
                    for _pmk, _pml in [
                        ("promotion_fuel_col", "Топливо"),
                        ("promotion_shop_col", "Магазин"),
                        ("promotion_cafe_col", "Кафе"),
                    ]:
                        _pm_src = _sp_wi.get(_pmk)
                        _pm_enc = f"{_pm_src}_enc" if _pm_src else None
                        if _pm_enc:
                            _raw_labels = prep_cfg.get(_pm_src, {}).get("value_labels", {})
                            _vmap = {int(k): v for k, v in _raw_labels.items()} if _raw_labels else {0: "нет", 1: "да"}
                            _pm_cols_wi[_pm_enc] = (_pml, _vmap, _pm_src)

                    # Расписание акций: {enc_col: [int_значение_на_каждый_день]}
                    _wi_promo_schedule: dict[str, list[int]] = {}
                    with _wif3:
                        st.markdown("**Акции**")
                        if _pm_cols_wi:
                            if _wi_horizon == 1:
                                for _pme, (_pml, _vmap, _) in _pm_cols_wi.items():
                                    _opts = list(_vmap.values())
                                    _sel  = st.selectbox(f"Акция: {_pml}", _opts, key=f"wi_pm_{_pme}")
                                    _wi_promo_schedule[_pme] = [
                                        next(k for k, v in _vmap.items() if v == _sel)
                                    ]
                            else:
                                _days_ru = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
                                _lbl = "паттерн по дням недели" if _wi_horizon == 30 else "по дням"
                                st.caption(f"Акции {_lbl}:")
                                _promo_init  = pd.DataFrame({
                                    "День": _days_ru,
                                    **{_pml: [_vmap[0]] * 7 for _, (_pml, _vmap, _) in _pm_cols_wi.items()},
                                })
                                _promo_col_cfg = {"День": st.column_config.TextColumn(disabled=True)}
                                for _, (_pml, _vmap, _) in _pm_cols_wi.items():
                                    _promo_col_cfg[_pml] = st.column_config.SelectboxColumn(
                                        _pml, options=list(_vmap.values()), required=True,
                                    )
                                _edited_promo = st.data_editor(
                                    _promo_init, hide_index=True,
                                    key="wi_promo_table",
                                    column_config=_promo_col_cfg,
                                )
                                for _pme, (_pml, _vmap, _) in _pm_cols_wi.items():
                                    _lbl_to_int = {v: k for k, v in _vmap.items()}
                                    _wi_promo_schedule[_pme] = [
                                        _lbl_to_int.get(v, 0) for v in _edited_promo[_pml].tolist()
                                    ]

                    # Авто-сброс: если что-то изменилось — очищаем старый результат
                    import hashlib as _hl, json as _json
                    _wi_cur_key = _hl.md5(_json.dumps({
                        "st": str(_wi_station),
                        "dt": str(_wi_start_date),
                        "hz": _wi_horizon,
                        "pr": sorted(_wi_price_vals.items()),
                        "wt": {k: v for k, v in sorted(_wi_weather_schedule.items())},
                        "pm": {k: v for k, v in sorted(_wi_promo_schedule.items())},
                    }, sort_keys=True, default=str).encode()).hexdigest()
                    if st.session_state.get("wi_input_key") != _wi_cur_key:
                        st.session_state.pop("wi_result", None)
                        st.session_state.pop("wi_result_station", None)
                        st.session_state["wi_input_key"] = _wi_cur_key

                    if st.button("🔀 Запустить сценарий", type="primary", key="wi_run"):

                        def _shift_norm(col: str, series: pd.Series, factor: float) -> pd.Series:
                            if not inv or col not in inv:
                                return series * factor
                            _p = inv[col].get("params", {})
                            if inv[col].get("method") == "zscore":
                                _s = _p.get("std", 1.0) or 1.0
                                return series * factor + _p.get("mean", 0.0) * (factor - 1) / _s
                            return series * factor

                        def _set_norm(col: str, val: float) -> float:
                            if not inv or col not in inv:
                                return val
                            _p = inv[col].get("params", {})
                            if inv[col].get("method") == "zscore":
                                _s = _p.get("std", 1.0) or 1.0
                                return (val - _p.get("mean", 0.0)) / _s
                            if inv[col].get("method") == "minmax":
                                _mn, _mx = _p.get("min", 0.0), _p.get("max", 1.0)
                                return (val - _mn) / (_mx - _mn) if (_mx - _mn) else 0.0
                            return val

                        @st.cache_resource
                        def _load_wi_model(ckpt: str):
                            import torch as _t
                            _orig = _t.load
                            def _patch(f, *a, **kw):
                                kw.setdefault("weights_only", False)
                                return _orig(f, *a, **kw)
                            _t.load = _patch
                            _m = _TFT.load_from_checkpoint(ckpt)
                            _m.eval()
                            return _m

                        with st.status("🔀 Выполняется сценарий...", expanded=True) as _wi_status:
                            _wi_status.write("⏳ Шаг 1/5 — Загрузка модели")
                            _wi_model = _load_wi_model(_wi_ckpt_path)

                            _wi_status.write("⏳ Шаг 2/5 — Загрузка данных")
                            _wi_exp_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exports"))
                            _wi_proc_path = os.path.join(_wi_exp_root, selected_session, "processed_data.csv")
                            _wi_mrg_path  = os.path.join(_wi_exp_root, selected_session, "merged_data.csv")
                            _wi_spl_path  = os.path.join(_wi_exp_root, selected_session, "split_config.json")
                            _wi_tcfg_path = os.path.join(_TRAINING_ROOT, selected_session, "train_config.json")

                            _wi_proc = pd.read_csv(_wi_proc_path, encoding="utf-8")
                            with open(_wi_spl_path) as _f: _wi_spl  = json.load(_f)
                            with open(_wi_tcfg_path) as _f: _wi_tcfg = json.load(_f)

                            TC   = tft["time_col"]
                            TGTS = tft["target"]
                            ENC  = int(_wi_tcfg.get("encoder_length", 30))
                            PRD  = int(_wi_tcfg.get("prediction_length", 7))
                            SC   = tft.get("static_cats", [])
                            SR   = tft.get("static_reals", [])
                            KC   = tft.get("time_varying_known_categoricals", [])
                            KR   = [c for c in tft.get("time_varying_known_reals", []) if c != TC]
                            UR   = [c for c in tft.get("time_varying_unknown_reals", []) if c not in TGTS]

                            _wi_proc[TC]        = _wi_proc[TC].astype(int)
                            _wi_proc[group_col] = _wi_proc[group_col].astype(str)

                            _dmap = pd.read_csv(_wi_mrg_path, usecols=["date", TC],
                                                parse_dates=["date"], encoding="utf-8")
                            _dmap = _dmap.drop_duplicates(subset=[TC])
                            _td   = _dmap.set_index(TC)["date"]

                            _tr_max = int(_td[_td <= pd.Timestamp(_wi_spl["train_end"])].index.max())
                            _vl_max = int(_td[_td <= pd.Timestamp(_wi_spl["val_end"])].index.max())
                            _al_max = int(_wi_proc[TC].max())

                            _sel_ts      = pd.Timestamp(_wi_start_date)
                            _td_last_ti  = int(_td.index.max())
                            _td_last_dt  = pd.Timestamp(_td.iloc[-1])
                            _ps_match    = _td[_td >= _sel_ts]
                            if _ps_match.empty:
                                # Выбрана дата за концом данных — считаем time_idx как смещение
                                _ps = _td_last_ti + (_sel_ts - _td_last_dt).days
                            else:
                                _ps = int(_ps_match.index[0])
                            _pe = _ps + _wi_horizon - 1  # горизонт не ограничиваем _al_max

                            _wi_status.write("⏳ Шаг 3/5 — Подготовка TSD")

                            for _c in SC + KC:
                                if _c in _wi_proc.columns:
                                    _wi_proc[_c] = _wi_proc[_c].astype(str)

                            _tr_df = _wi_proc[_wi_proc[TC] <= _tr_max].copy()
                            _wi_tsd = _TSD(
                                _tr_df, time_idx=TC, target=TGTS, group_ids=[group_col],
                                min_encoder_length=ENC // 2, max_encoder_length=ENC,
                                min_prediction_length=1, max_prediction_length=PRD,
                                static_categoricals=SC, static_reals=SR,
                                time_varying_known_categoricals=KC, time_varying_known_reals=KR,
                                time_varying_unknown_reals=UR,
                                target_normalizer=_MNorm([_EncNorm() for _ in TGTS]),
                                add_relative_time_idx=True, add_target_scales=True,
                                add_encoder_length=True, allow_missing_timesteps=True,
                            )

                            import warnings as _warnings

                            # ── Всегда rolling ───────────────────────────────────
                            _pe = _ps + _wi_horizon - 1

                            _wi_status.write("⏳ Шаг 3/5 — Подготовка энкодера")
                            _st_hist  = _wi_proc[_wi_proc[group_col] == str(_wi_station)].copy()
                            _syn_hist = _st_hist[_st_hist[TC] < _ps].tail(ENC).reset_index(drop=True)
                            if _syn_hist.empty:
                                raise ValueError("Нет исторических данных для выбранной станции.")

                            _td_idx = _td.reset_index(); _td_idx.columns = [TC, "_date"]
                            _sh_dated = _syn_hist.merge(_td_idx, on=TC, how="left")
                            _sh_dated["_date"] = pd.to_datetime(_sh_dated["_date"])

                            _season_map: dict = {}
                            if "season_enc" in _sh_dated.columns:
                                _season_map = (
                                    _sh_dated.groupby(_sh_dated["_date"].dt.month)["season_enc"]
                                    .first().to_dict()
                                )
                            _hol_map: dict = {}
                            _no_hol_enc = "0"
                            if "holiday_name_enc" in _sh_dated.columns:
                                for _, _hr in _sh_dated.dropna(subset=["_date"]).iterrows():
                                    _hol_map[(int(_hr["_date"].month), int(_hr["_date"].day))] = {
                                        "is_holiday":       int(_hr.get("is_holiday", 0)),
                                        "holiday_name_enc": str(_hr["holiday_name_enc"]),
                                    }
                                _no_hol_rows = (
                                    _sh_dated[_sh_dated["is_holiday"] == 0]
                                    if "is_holiday" in _sh_dated.columns else pd.DataFrame()
                                )
                                if len(_no_hol_rows) > 0:
                                    _no_hol_enc = str(_no_hol_rows["holiday_name_enc"].iloc[0])

                            _tmpl = _syn_hist.iloc[-1].copy()
                            _known_reals_set = set(tft.get("time_varying_known_reals") or [])

                            # Маппинг base_col → pandas-аксессор (неизбежный фиксированный словарь)
                            _date_acc_map = {
                                "day_of_week":    lambda d: d.dayofweek,
                                "week_of_year":   lambda d: int(d.isocalendar()[1]),
                                "month":          lambda d: d.month,
                                "date_dayofyear": lambda d: d.timetuple().tm_yday,
                                "hour":           lambda d: d.hour,
                                "quarter":        lambda d: d.quarter,
                            }

                            # Циклические признаки из inverse_transforms (единый источник правды):
                            # [(sin_col, cos_col, accessor_fn, period)]
                            _cyc_feats = []
                            for _base, _pcfg in (inv or {}).items():
                                if _pcfg.get("method") == "cyclical" and _base in _date_acc_map:
                                    _p = _pcfg.get("params", {})
                                    _period = float(_p.get("period", 1.0))
                                    _sc = _p.get("sin_col", f"{_base}_sin")
                                    _cc = _p.get("cos_col", f"{_base}_cos")
                                    if _sc in _known_reals_set or _cc in _known_reals_set:
                                        _cyc_feats.append((_sc, _cc, _date_acc_map[_base], _period))

                            # Компонентные колонки из session dtx_configs:
                            # [(col_name, accessor_fn)]
                            _dtx_acc_map = {
                                "dayofyear": lambda d: d.timetuple().tm_yday,
                                "day":       lambda d: d.day,
                                "month":     lambda d: d.month,
                                "year":      lambda d: d.year,
                                "hour":      lambda d: d.hour,
                                "week":      lambda d: int(d.isocalendar()[1]),
                                "weekday":   lambda d: d.weekday(),
                            }
                            _dtx_feats = []
                            for _dcfg in (session or {}).get("dtx_configs", []):
                                _src = _dcfg.get("src_col", "date")
                                for _comp in _dcfg.get("components", []):
                                    if _comp in _dtx_acc_map:
                                        _col = f"{_src}_{_comp}"
                                        if _col in _known_reals_set:
                                            _dtx_feats.append((_col, _dtx_acc_map[_comp]))

                            # Бинарные флаги, если присутствуют в known_reals
                            _bin_feats = [
                                (c, f) for c, f in [
                                    ("is_weekend", lambda d: int(d.dayofweek >= 5)),
                                    ("is_holiday", lambda d: 0),  # будущее неизвестно
                                ] if c in _known_reals_set
                            ]

                            def _make_future_row(day_date: pd.Timestamp, day_tidx: int) -> pd.Series:
                                row = _tmpl.copy()
                                row[TC] = day_tidx
                                # Циклические признаки из конфига препроцессинга
                                for _sc, _cc, _acc, _period in _cyc_feats:
                                    _val = _acc(day_date)
                                    if _sc in row.index: row[_sc] = np.sin(2*np.pi*_val/_period)
                                    if _cc in row.index: row[_cc] = np.cos(2*np.pi*_val/_period)
                                # Компонентные колонки из dtx_configs
                                for _col, _acc in _dtx_feats:
                                    if _col in row.index: row[_col] = _acc(day_date)
                                # Бинарные флаги
                                for _col, _func in _bin_feats:
                                    if _col in row.index: row[_col] = _func(day_date)
                                # Сезон и праздник из исторических карт
                                if "season_enc" in row.index:
                                    row["season_enc"] = str(_season_map.get(day_date.month, row["season_enc"]))
                                _hk = (day_date.month, day_date.day)
                                if _hk in _hol_map:
                                    if "is_holiday" in row.index:
                                        row["is_holiday"] = _hol_map[_hk]["is_holiday"]
                                    if "holiday_name_enc" in row.index:
                                        row["holiday_name_enc"] = _hol_map[_hk]["holiday_name_enc"]
                                else:
                                    if "holiday_name_enc" in row.index:
                                        row["holiday_name_enc"] = _no_hol_enc
                                for _tg in TGTS:
                                    if _tg in row.index: row[_tg] = 0.0
                                return row

                            _wi_status.write("⏳ Шаг 4/5 — Скользящий инференс")

                            # Хелпер: получить известные значения категориального энкодера
                            def _enc_classes(enc):
                                if enc is None or not hasattr(enc, "classes_"):
                                    return None, None
                                cls = enc.classes_
                                lst = list(cls.keys()) if isinstance(cls, dict) else (
                                    cls.tolist() if hasattr(cls, "tolist") else list(cls)
                                )
                                return (
                                    {str(c) for c in lst},
                                    str(lst[0]) if lst else None,
                                )

                            # Множество всех категориальных колонок по TSD
                            _all_cat_cols = set(
                                list(getattr(_wi_tsd, "static_categoricals",     None) or [])
                                + list(getattr(_wi_tsd, "time_varying_known_categoricals",   None) or [])
                                + list(getattr(_wi_tsd, "time_varying_unknown_categoricals", None) or [])
                            )
                            _cat_encs = _wi_tsd.categorical_encoders or {}

                            # {(tidx, tgt): (date, q50, q10, q90)}
                            _all_preds: dict[tuple, tuple] = {}

                            for _batch_s in range(0, _wi_horizon, PRD):
                                _batch_sz = min(PRD, _wi_horizon - _batch_s)
                                _dec_rows: list[pd.Series] = []
                                for _di in range(PRD):
                                    _abs_di   = _batch_s + _di
                                    _dtidx    = _ps + _abs_di
                                    _raw_date = _td.get(_dtidx) or (
                                        _td_last_dt + pd.Timedelta(days=(_dtidx - _td_last_ti))
                                    )
                                    _dec_rows.append(_make_future_row(pd.Timestamp(_raw_date), _dtidx))
                                for _di, _row in enumerate(_dec_rows):
                                    _abs_di = _batch_s + _di
                                    for _wpc, _new_p in _wi_price_vals.items():
                                        if _wpc in _row.index: _row[_wpc] = _set_norm(_wpc, _new_p)
                                    for _wc, _wvals in _wi_weather_schedule.items():
                                        if _wc in _row.index:
                                            _row[_wc] = _set_norm(_wc, _wvals[_abs_di % len(_wvals)])
                                    for _pme, _pmv_list in _wi_promo_schedule.items():
                                        if _pme in _row.index:
                                            _row[_pme] = str(int(_pmv_list[_abs_di % len(_pmv_list)]))

                                _ctx = pd.concat(
                                    [_syn_hist.tail(ENC).reset_index(drop=True),
                                     pd.DataFrame(_dec_rows)], ignore_index=True,
                                )

                                # Добавляем отсутствующие cat-колонки, заполняем первым известным
                                for _cc in _all_cat_cols:
                                    if _cc not in _ctx.columns:
                                        _, _cf = _enc_classes(_cat_encs.get(_cc))
                                        _ctx[_cc] = _cf if _cf is not None else "0"

                                # Приводим к строкам и заменяем неизвестные значения на первое допустимое
                                for _cc in _all_cat_cols:
                                    if _cc not in _ctx.columns:
                                        continue
                                    try:
                                        _ctx[_cc] = (_ctx[_cc].ffill().bfill()
                                                     .astype(float).astype(int).astype(str))
                                    except Exception:
                                        try: _ctx[_cc] = _ctx[_cc].ffill().bfill().astype(str)
                                        except Exception: pass
                                    _ck, _cf = _enc_classes(_cat_encs.get(_cc))
                                    if _ck and _cf is not None:
                                        _ctx[_cc] = _ctx[_cc].map(
                                            lambda v, k=_ck, f=_cf: v if v in k else f
                                        )

                                with _warnings.catch_warnings():
                                    _warnings.filterwarnings("ignore", category=UserWarning)
                                    _pd_ds  = _TSD.from_dataset(_wi_tsd, _ctx, stop_randomization=True)
                                    _loader = _pd_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
                                    _res    = _wi_model.predict(
                                        _loader, mode="quantiles",
                                        trainer_kwargs={"logger": False, "enable_progress_bar": False},
                                    )
                                _wpreds = _res.output if hasattr(_res, "output") else _res
                                if isinstance(_wpreds, (list, tuple)):
                                    _wpnp = [p.detach().cpu().numpy() for p in _wpreds]
                                else:
                                    _wpnp = [_wpreds.detach().cpu().numpy()]
                                _nq  = _wpnp[0].shape[-1]
                                _QM  = _nq // 2
                                _QLO = 1
                                _QHI = _nq - 2
                                for _di in range(_batch_sz):
                                    _abs_di = _batch_s + _di
                                    _dtidx  = _ps + _abs_di
                                    _ddate  = _td.get(_dtidx) or (
                                        _td_last_dt + pd.Timedelta(days=(_dtidx - _td_last_ti))
                                    )
                                    for _ti2, _tc2 in enumerate(TGTS):
                                        _arr = _wpnp[_ti2]
                                        def _dn(_v, _c=_tc2):
                                            return float(denormalize(pd.Series([_v]), _c, inv).iloc[0])
                                        _all_preds[(_dtidx, _tc2)] = (
                                            _ddate,
                                            _dn(float(_arr[0, _di, _QM])),
                                            _dn(float(_arr[0, _di, _QLO])),
                                            _dn(float(_arr[0, _di, _QHI])),
                                        )
                                        # Feedback: P50 в log-пространстве обратно в энкодер
                                        if _tc2 in _dec_rows[_di].index:
                                            _dec_rows[_di][_tc2] = float(_arr[0, _di, _QM])
                                _syn_hist = pd.concat(
                                    [_syn_hist, pd.DataFrame(_dec_rows[:_batch_sz])],
                                    ignore_index=True,
                                )

                            _wi_status.write("⏳ Шаг 5/5 — Обработка результатов")
                            _wi_rows = []
                            for _dtidx in range(_ps, _pe + 1):
                                _ddate = _td.get(_dtidx) or (
                                    _td_last_dt + pd.Timedelta(days=(_dtidx - _td_last_ti))
                                )
                                _row = {group_col: str(_wi_station), "date": _ddate}
                                for _tc2 in TGTS:
                                    _k = (_dtidx, _tc2)
                                    if _k in _all_preds:
                                        _, _v50, _v10, _v90 = _all_preds[_k]
                                        _row[f"{_tc2}_wi"]     = _v50
                                        _row[f"{_tc2}_wi_q10"] = _v10
                                        _row[f"{_tc2}_wi_q90"] = _v90
                                _wi_rows.append(_row)
                            _wi_out = pd.DataFrame(_wi_rows)

                            st.session_state["wi_result"]         = _wi_out
                            st.session_state["wi_result_station"] = _wi_station
                            _wi_status.update(label="✅ Сценарий выполнен!", state="complete")
                            st.rerun()

                    if "wi_result" in st.session_state:
                        _wi_out = st.session_state["wi_result"]
                        _wi_st  = st.session_state.get("wi_result_station", "")

                        st.divider()
                        st.markdown("#### Результаты сценария")

                        # Таргеты из самого wi_out (все, что модель предсказала)
                        _wi_all_tgts = [c[:-3] for c in _wi_out.columns if c.endswith("_wi")]
                        _wit_sel  = st.selectbox("Таргет", _wi_all_tgts, format_func=lbl, key="wi_tgt")
                        _wis_col  = f"{_wit_sel}_wi"

                        _wi_q10_col = f"{_wit_sel}_wi_q10"
                        _wi_q90_col = f"{_wit_sel}_wi_q90"

                        # Фильтр по станции — нормализуем тип для надёжного сравнения
                        _wi_s = pd.DataFrame()
                        if _wis_col in _wi_out.columns:
                            _mask_st = _wi_out[group_col].astype(str) == str(_wi_st)
                            _cols    = ["date", _wis_col] + [
                                c for c in [_wi_q10_col, _wi_q90_col] if c in _wi_out.columns
                            ]
                            _wi_s = _wi_out[_mask_st][_cols].dropna(subset=["date", _wis_col])

                        if not _wi_s.empty:
                            _wim = _wi_s.copy()
                            _wim["date"] = pd.to_datetime(_wim["date"])

                            _n_pts = len(_wim)
                            _mode  = "lines+markers" if _n_pts <= 3 else "lines"
                            _msize = 8 if _n_pts <= 3 else 5

                            fig = go.Figure()

                            # Зелёная линия факта: 30 дней до прогноза + реальные данные
                            # внутри прогнозного периода (если есть) — всё одним цветом
                            if date_col and date_col in merged.columns and _wit_sel in merged.columns:
                                _mst = merged[
                                    merged[group_col].astype(str) == str(_wi_st)
                                ].copy() if group_col in merged.columns else merged.copy()
                                _fc_start = pd.Timestamp(_wim["date"].min())
                                _fact_beg = _fc_start - pd.Timedelta(days=30)
                                # конец = последняя дата прогноза или последняя в данных
                                _fact_end = pd.Timestamp(_wim["date"].max())
                                _mfact_all = _mst[
                                    (pd.to_datetime(_mst[date_col]) >= _fact_beg) &
                                    (pd.to_datetime(_mst[date_col]) <= _fact_end)
                                ][[date_col, _wit_sel]].dropna().rename(columns={date_col: "date"})
                                _mfact_all["date"] = pd.to_datetime(_mfact_all["date"])
                                if not _mfact_all.empty:
                                    # Соединяем последнюю точку истории с первой точкой прогноза
                                    # добавив граничную точку если её нет
                                    _last_hist = _mfact_all[_mfact_all["date"] < _fc_start]
                                    _in_fc     = _mfact_all[_mfact_all["date"] >= _fc_start]
                                    if not _last_hist.empty and not _in_fc.empty:
                                        # граничная точка уже есть — рисуем единой линией
                                        fig.add_trace(go.Scatter(
                                            x=_mfact_all["date"], y=_mfact_all[_wit_sel],
                                            mode="lines", name="Факт",
                                            line=dict(color="#4CAF50", width=2),
                                        ))
                                    elif not _last_hist.empty:
                                        fig.add_trace(go.Scatter(
                                            x=_last_hist["date"], y=_last_hist[_wit_sel],
                                            mode="lines", name="Факт (история)",
                                            line=dict(color="#4CAF50", width=2),
                                        ))

                            # Квантильная полоса q10–q90 (оранжевая, полупрозрачная)
                            if _wi_q10_col in _wim.columns and _wi_q90_col in _wim.columns:
                                fig.add_trace(go.Scatter(
                                    x=_wim["date"], y=_wim[_wi_q90_col],
                                    mode="lines", line=dict(width=0),
                                    showlegend=False, hoverinfo="skip",
                                ))
                                fig.add_trace(go.Scatter(
                                    x=_wim["date"], y=_wim[_wi_q10_col],
                                    mode="lines", line=dict(width=0),
                                    fill="tonexty", fillcolor="rgba(255,152,0,0.18)",
                                    name="Оптимист/Пессимист (q10–q90)",
                                    hoverinfo="skip",
                                ))

                            # Прогноз — медиана, соединяется с историей через первую точку
                            # Добавляем последнюю точку факта-истории как начало линии прогноза
                            _fc_dates = _wim["date"].tolist()
                            _fc_vals  = _wim[_wis_col].tolist()
                            if date_col and date_col in merged.columns and _wit_sel in merged.columns:
                                _pre = _mst[
                                    pd.to_datetime(_mst[date_col]) == pd.Timestamp(_fc_dates[0]) - pd.Timedelta(days=1)
                                ][[date_col, _wit_sel]].rename(columns={date_col: "date"})
                                if not _pre.empty:
                                    _fc_dates = [pd.Timestamp(_pre["date"].iloc[0])] + _fc_dates
                                    _fc_vals  = [float(_pre[_wit_sel].iloc[0])] + _fc_vals

                            fig.add_trace(go.Scatter(
                                x=_fc_dates, y=_fc_vals,
                                mode=_mode, name="Прогноз (медиана)",
                                line=dict(color="#FF9800", width=2, dash="dash"),
                                marker=dict(size=_msize),
                            ))

                            fig.update_layout(
                                xaxis_title="Дата", yaxis_title=lbl(_wit_sel),
                                legend=dict(orientation="h", y=-0.22),
                                margin=dict(b=80),
                            )
                            _pchart(fig)

                            _ss = _wim[_wis_col]
                            st.dataframe(pd.DataFrame({
                                "":        ["Прогноз (медиана)", "Оптимист (q90)", "Пессимист (q10)"],
                                "Старт":   [f"{_ss.iloc[0]:,.0f}",
                                            f"{_wim[_wi_q90_col].iloc[0]:,.0f}" if _wi_q90_col in _wim.columns else "—",
                                            f"{_wim[_wi_q10_col].iloc[0]:,.0f}" if _wi_q10_col in _wim.columns else "—"],
                                "Среднее": [f"{_ss.mean():,.0f}",
                                            f"{_wim[_wi_q90_col].mean():,.0f}" if _wi_q90_col in _wim.columns else "—",
                                            f"{_wim[_wi_q10_col].mean():,.0f}" if _wi_q10_col in _wim.columns else "—"],
                                "Пик":     [f"{_ss.max():,.0f}",
                                            f"{_wim[_wi_q90_col].max():,.0f}" if _wi_q90_col in _wim.columns else "—",
                                            f"{_wim[_wi_q10_col].max():,.0f}" if _wi_q10_col in _wim.columns else "—"],
                            }), hide_index=True, width="stretch")
                        else:
                            st.info(f"Нет данных сценария для станции «{_wi_st}».")

        with sub_vsn:
            st.markdown("### 🧠 Интерпретация Variable Selection Network")
            st.caption(
                "VSN показывает, какие признаки модель считает наиболее важными. "
                "Веса нормированы: сумма по каждой группе ≈ 1."
            )

            if not _pf_ok:
                st.warning("pytorch-forecasting не установлен.")
            else:
                _vsn_station = st.selectbox(
                    lbl(group_col),
                    sorted(pred_df[group_col].unique()) if group_col in pred_df.columns else [],
                    key="vsn_station",
                )
                _vsn_n_windows = st.select_slider(
                    "Окон для усреднения",
                    options=[1, 5, 10, 20, 50],
                    value=10,
                    key="vsn_n_windows",
                    help="Больше окон → стабильнее веса, дольше вычисление. "
                         "1 = только последнее окно (старое поведение).",
                )

                if st.button("🔍 Запустить интерпретацию", key="vsn_run"):
                    with st.status("🧠 Вычисление весов VSN...", expanded=True) as _vsn_st:
                        try:
                            _vsn_st.write("⏳ Загрузка модели")
                            _vsn_ckpt_p = os.path.join(_TRAINING_ROOT, selected_session, "model.ckpt")
                            _vsn_ckpt_d = os.path.join(_TRAINING_ROOT, selected_session, "checkpoints")
                            if not os.path.exists(_vsn_ckpt_p):
                                _vsn_files = sorted(
                                    [f for f in os.listdir(_vsn_ckpt_d) if f.endswith(".ckpt")],
                                    reverse=True,
                                ) if os.path.isdir(_vsn_ckpt_d) else []
                                _vsn_ckpt_p = os.path.join(_vsn_ckpt_d, _vsn_files[0]) if _vsn_files else None
                            if not _vsn_ckpt_p:
                                st.error("Нет чекпоинтов."); st.stop()

                            @st.cache_resource
                            def _load_vsn_model(ckpt: str):
                                import torch as _t
                                _orig = _t.load
                                def _patch(f, *a, **kw):
                                    kw.setdefault("weights_only", False); return _orig(f, *a, **kw)
                                _t.load = _patch
                                _m = _TFT.load_from_checkpoint(ckpt); _m.eval(); return _m

                            _vsn_model = _load_vsn_model(_vsn_ckpt_p)

                            _vsn_st.write("⏳ Подготовка данных")
                            _vsn_exp   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exports"))
                            _vsn_proc  = pd.read_csv(
                                os.path.join(_vsn_exp, selected_session, "processed_data.csv"),
                                encoding="utf-8",
                            )
                            _vsn_spl   = json.load(open(
                                os.path.join(_vsn_exp, selected_session, "split_config.json")
                            ))
                            _vsn_tcfg  = json.load(open(
                                os.path.join(_TRAINING_ROOT, selected_session, "train_config.json")
                            ))

                            _vTC  = tft["time_col"];  _vTG = tft["target"]
                            _vENC = int(_vsn_tcfg.get("encoder_length", 30))
                            _vPRD = int(_vsn_tcfg.get("prediction_length", 7))
                            _vSC  = tft.get("static_cats", [])
                            _vSR  = tft.get("static_reals", [])
                            _vKC  = tft.get("time_varying_known_categoricals", [])
                            _vKR  = [c for c in tft.get("time_varying_known_reals", []) if c != _vTC]
                            _vUR  = [c for c in tft.get("time_varying_unknown_reals", []) if c not in _vTG]

                            _vsn_proc[_vTC]        = _vsn_proc[_vTC].astype(int)
                            _vsn_proc[group_col]   = _vsn_proc[group_col].astype(str)
                            for _c in _vSC + _vKC:
                                if _c in _vsn_proc.columns:
                                    _vsn_proc[_c] = _vsn_proc[_c].astype(str)

                            _vsn_dmap = pd.read_csv(
                                os.path.join(_vsn_exp, selected_session, "merged_data.csv"),
                                usecols=["date", _vTC], parse_dates=["date"], encoding="utf-8",
                            ).drop_duplicates(subset=[_vTC])
                            _vsn_td = _vsn_dmap.set_index(_vTC)["date"]
                            _vsn_tr_max = int(
                                _vsn_td[_vsn_td <= pd.Timestamp(_vsn_spl["train_end"])].index.max()
                            )

                            _vsn_tr_df = _vsn_proc[_vsn_proc[_vTC] <= _vsn_tr_max].copy()
                            _vsn_tsd   = _TSD(
                                _vsn_tr_df, time_idx=_vTC, target=_vTG, group_ids=[group_col],
                                min_encoder_length=_vENC, max_encoder_length=_vENC,
                                min_prediction_length=_vPRD, max_prediction_length=_vPRD,
                                static_categoricals=_vSC, static_reals=_vSR,
                                time_varying_known_categoricals=_vKC, time_varying_known_reals=_vKR,
                                time_varying_unknown_reals=_vUR,
                                target_normalizer=_MNorm([_EncNorm() for _ in _vTG]),
                                add_relative_time_idx=True, add_target_scales=True,
                                add_encoder_length=True, allow_missing_timesteps=True,
                            )

                            _vsn_st.write(f"⏳ Инференс для интерпретации ({_vsn_n_windows} окон)")
                            _vsn_st_df = _vsn_proc[_vsn_proc[group_col] == str(_vsn_station)]
                            _vsn_n_rows = (_vsn_n_windows - 1) + _vENC + _vPRD
                            _vsn_pd_df = _vsn_st_df.tail(_vsn_n_rows).copy()

                            import warnings as _w
                            with _w.catch_warnings():
                                _w.filterwarnings("ignore", category=UserWarning)
                                _vsn_ds  = _TSD.from_dataset(_vsn_tsd, _vsn_pd_df, stop_randomization=True)
                                _vsn_actual_windows = len(_vsn_ds)
                                _vsn_ldr = _vsn_ds.to_dataloader(
                                    train=False, batch_size=max(1, _vsn_actual_windows), num_workers=0
                                )
                                import torch as _torch
                                _vsn_raw = _vsn_model.predict(
                                    _vsn_ldr, mode="raw",
                                    trainer_kwargs={"logger": False, "enable_progress_bar": False},
                                )

                            _vsn_st.write("⏳ Извлечение весов VSN")
                            _raw_out = _vsn_raw.output if hasattr(_vsn_raw, "output") else _vsn_raw
                            # reduction="none" → shape [batch, n_vars]; усредняем по батчу вручную
                            _interp  = _vsn_model.interpret_output(_raw_out, reduction="none")

                            def _to_np(t):
                                return t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.array(t)

                            def _avg_norm(t):
                                """Среднее по батч-размерности, нормировка к сумме = 1."""
                                arr = _to_np(t)
                                if arr.ndim > 1:
                                    arr = arr.mean(axis=0)
                                arr = arr.flatten()
                                s = arr.sum()
                                return arr / s if s > 0 else arr

                            # Внимание: shape [batch, enc+dec_length] → усредняем по батчу
                            _att_raw = _to_np(_interp["attention"])
                            _att_avg = _att_raw.mean(axis=0) if _att_raw.ndim > 1 else _att_raw
                            # Берём только часть энкодера (первые max_encoder_length позиций)
                            _enc_len = _vsn_model.hparams.max_encoder_length
                            _att_enc = _att_avg[:_enc_len] if len(_att_avg) >= _enc_len else _att_avg

                            st.session_state["vsn_result"] = {
                                "encoder":  (_vsn_model.encoder_variables,
                                             _avg_norm(_interp["encoder_variables"])),
                                "decoder":  (_vsn_model.decoder_variables,
                                             _avg_norm(_interp["decoder_variables"])),
                                "static":   (_vsn_model.static_variables,
                                             _avg_norm(_interp["static_variables"])),
                                "attention": _att_enc,
                                "enc_len":   _enc_len,
                                "station":   _vsn_station,
                                "n_windows": _vsn_actual_windows,
                            }
                            _vsn_st.update(label="✅ Готово!", state="complete")

                        except Exception as _ex:
                            import traceback as _tb
                            _vsn_st.update(label="❌ Ошибка", state="error")
                            st.error(f"{_ex}\n\n{_tb.format_exc()}")

                if "vsn_result" in st.session_state:
                    _vr = st.session_state["vsn_result"]
                    st.markdown(
                        f"**Станция:** {_vr['station']} &nbsp;·&nbsp; "
                        f"**Усреднено по:** {_vr.get('n_windows', 1)} окн.",
                        unsafe_allow_html=True,
                    )

                    def _vsn_chart(names: list, weights, title: str):
                        _w = np.array(weights).flatten()
                        if len(_w) != len(names):
                            st.caption(f"⚠️ Размер весов ({len(_w)}) ≠ количеству переменных ({len(names)})")
                            return
                        # Нормализуем к сумме = 1 (на случай числовых погрешностей)
                        _s = _w.sum()
                        if _s > 0: _w = _w / _s
                        _df = pd.DataFrame({"Признак": [lbl(n) for n in names], "Вес": _w})
                        _df = _df.sort_values("Вес", ascending=True).tail(30)
                        _fig = go.Figure(go.Bar(
                            x=_df["Вес"], y=_df["Признак"],
                            orientation="h",
                            marker_color="#2196F3",
                            text=_df["Вес"].apply(lambda v: f"{v*100:.1f}%"),
                            textposition="outside",
                        ))
                        _fig.update_layout(
                            title=title,
                            xaxis_title="Доля важности (сумма = 100%)",
                            xaxis=dict(tickformat=".0%"),
                            margin=dict(l=200, r=80, t=40, b=30),
                            height=max(300, len(_df) * 22 + 80),
                        )
                        _pchart(_fig)

                    _tab_enc, _tab_dec, _tab_sta, _tab_att = st.tabs([
                        "📊 Энкодер", "📊 Декодер", "📊 Статика", "🔥 Внимание"
                    ])

                    with _tab_enc:
                        st.caption(
                            "Насколько каждый признак влияет на анализ **исторических** данных. "
                            "Высокий вес = модель активно использует этот признак в окне истории."
                        )
                        _vsn_chart(*_vr["encoder"], "Важность признаков энкодера")

                    with _tab_dec:
                        st.caption(
                            "Насколько каждый **известный будущий** признак (цены, календарь, акции) "
                            "влияет на прогноз. Эти признаки задаются в сценарии What-IF."
                        )
                        _vsn_chart(*_vr["decoder"], "Важность признаков декодера")

                    with _tab_sta:
                        st.caption(
                            "Какие **постоянные характеристики станции** (тип дороги, число колонок и т.д.) "
                            "модель считает важными. Не меняются со временем."
                        )
                        _vsn_chart(*_vr["static"], "Важность статических признаков")

                    with _tab_att:
                        _enc_len_att = _vr.get("enc_len", len(_vr["attention"]))
                        st.caption(
                            f"На какие **дни из истории** модель обращает внимание при формировании прогноза. "
                            f"Ось X: дни назад от даты прогноза (−{_enc_len_att} = самый ранний, −1 = вчера). "
                            f"Высокий столбец = модель больше опирается на этот день."
                        )
                        _att = _vr["attention"].flatten()
                        if len(_att) > 0:
                            _x_labels = list(range(-len(_att), 0))
                            _att_fig = go.Figure(go.Bar(
                                x=_x_labels, y=_att,
                                marker_color=[
                                    "#FF9800" if v >= np.percentile(_att, 75) else "#90CAF9"
                                    for v in _att
                                ],
                                hovertemplate="День %{x}: вес %{y:.4f}<extra></extra>",
                            ))
                            _att_fig.update_layout(
                                xaxis_title="Дней назад от точки прогноза",
                                yaxis_title="Вес внимания",
                                margin=dict(t=20, b=50),
                                height=300,
                            )
                            _pchart(_att_fig)

with tab_reco:
    import glob as _glob_r

    _reco_tr_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training"))
    _reco_pd_dir  = os.path.join(_reco_tr_root, selected_session, "predictions")
    _reco_pfiles  = sorted(
        _glob_r.glob(os.path.join(_reco_pd_dir, "*.csv")),
        key=os.path.getmtime, reverse=True,
    ) if os.path.isdir(_reco_pd_dir) else []

    if not _reco_pfiles:
        st.info(
            "Нет файлов прогнозов для этой сессии. "
            "Запустите прогнозирование в дашборде обучения (`train_dashboard.py`)."
        )
    else:
        _reco_lbls    = [os.path.basename(f) for f in _reco_pfiles]
        _reco_sel_lbl = st.selectbox("📂 Файл прогноза", _reco_lbls, key="reco_pred_file")
        _reco_path    = _reco_pfiles[_reco_lbls.index(_reco_sel_lbl)]

        @st.cache_data(show_spinner="Загрузка прогноза...")
        def _reco_load(p: str) -> pd.DataFrame:
            return pd.read_csv(p, parse_dates=["date"], encoding="utf-8")

        rdf = _reco_load(_reco_path)

        _rt = sorted({c[:-4] for c in rdf.columns if c.endswith("_q10")})
        _new_rp = {
            f"{t}_pred": (rdf[f"{t}_q10"] + rdf[f"{t}_q90"]) / 2
            for t in _rt
            if f"{t}_pred" not in rdf.columns
            and f"{t}_q10" in rdf.columns and f"{t}_q90" in rdf.columns
        }
        if _new_rp:
            rdf = pd.concat([rdf, pd.DataFrame(_new_rp, index=rdf.index)], axis=1)

        # Порог > 1 исключает таргеты с почти нулевыми продажами (КПГ/СПГ/СУГ у станций без оборудования)
        _fuel_tgts = [t for t in _rt if t.startswith("sales_") and rdf[t].mean() > 1]
        _shop_tgts = [t for t in _rt if t.startswith("shop_")  and rdf[t].mean() > 1]
        _reco_sts  = sorted(rdf[group_col].unique()) if group_col in rdf.columns else []

        # Объединяем прогноз с реальными значениями из merged для корректного MAPE
        _reco_actual: pd.DataFrame | None = None
        if "date" in rdf.columns and date_col and date_col in merged.columns and group_col in merged.columns:
            _reco_dates = rdf["date"].dt.date if hasattr(rdf["date"], "dt") else pd.to_datetime(rdf["date"]).dt.date
            _merged_sub = merged[pd.to_datetime(merged[date_col]).dt.date.isin(set(_reco_dates))]
            if not _merged_sub.empty:
                _reco_actual = _merged_sub[[group_col, date_col] + [t for t in _rt if t in merged.columns]].copy()
                _reco_actual = _reco_actual.rename(columns={date_col: "date"})
                _reco_actual["date"] = pd.to_datetime(_reco_actual["date"])

        def _r_mape(sdf, tgts):
            """MAPE по реальным значениям из merged; fallback — сравнение медианы с (q10+q90)/2."""
            vals = []
            _st_id = sdf[group_col].iloc[0] if group_col in sdf.columns and not sdf.empty else None
            for t in tgts:
                pc = f"{t}_pred"
                if pc not in sdf.columns:
                    continue
                p = sdf[pc]
                # Берём реальные значения из merged если доступны
                if (_reco_actual is not None and t in _reco_actual.columns
                        and _st_id is not None and "date" in sdf.columns):
                    _act_st = _reco_actual[_reco_actual[group_col] == _st_id]
                    _joined = sdf[["date", pc]].merge(_act_st[["date", t]], on="date", how="inner")
                    a, p = _joined[t], _joined[pc]
                elif t in sdf.columns:
                    a = sdf[t]
                else:
                    continue
                m = a > 1  # исключаем почти нулевые значения
                if m.sum() > 0:
                    vals.append(((a[m] - p[m]).abs() / a[m]).mean() * 100)
            return float(np.mean(vals)) if vals else None

        def _r_unc(sdf, tgts):
            vals = []
            for t in tgts:
                q10c, q90c = f"{t}_q10", f"{t}_q90"
                if q10c in sdf.columns and q90c in sdf.columns:
                    mid = (sdf[q10c] + sdf[q90c]) / 2
                    m = mid > 0
                    if m.sum() > 0:
                        vals.append(((sdf.loc[m, q90c] - sdf.loc[m, q10c]) / mid[m]).mean())
            return float(np.mean(vals)) if vals else None

        # ── БЛОК 1: Профиль станций ───────────────────────────────────────────
        st.markdown("### 📋 Профиль станций")
        st.caption(
            "**Тренд** — сравнение средних продаж первой и второй половины горизонта прогноза: "
            "📈 Растёт (>+5%), 📉 Падает (>−5%), ➡️ Стабильно (±5%). "
            "**MAPE** — средняя абсолютная процентная ошибка модели на тестовом периоде: "
            "✅ <10%, ⚠️ 10–20%, ❌ >20%. "
            "**Неопределённость** — относительная ширина интервала q10–q90: "
            "🟢 <30%, 🟡 30–60%, 🔴 >60%. "
            "**Риск** — итоговый балл 0–100 (сумма штрафов за MAPE и неопределённость)."
        )

        _profile = []
        for _st in _reco_sts:
            _sdf = rdf[rdf[group_col] == _st].sort_values("date")

            _trend = "—"
            _pri = _fuel_tgts[0] if _fuel_tgts else None
            if _pri and f"{_pri}_pred" in _sdf.columns:
                _v = _sdf[f"{_pri}_pred"].dropna().values
                if len(_v) >= 4:
                    _mid = len(_v) // 2
                    _m1, _m2 = _v[:_mid].mean(), _v[_mid:].mean()
                    _trend = ("📈 Растёт" if _m2 > _m1 * 1.05
                              else ("📉 Падает" if _m2 < _m1 * 0.95 else "➡️ Стабильно"))

            _mv    = _r_mape(_sdf, _fuel_tgts)
            _acc   = ("—" if _mv is None else
                      ("✅ Отлично" if _mv < 10 else ("⚠️ Норма" if _mv < 20 else "❌ Плохо")))

            _mv_sh = _r_mape(_sdf, _shop_tgts)
            _acc_sh = ("—" if _mv_sh is None else
                       ("✅ Отлично" if _mv_sh < 10 else ("⚠️ Норма" if _mv_sh < 20 else "❌ Плохо")))

            _uv    = _r_unc(_sdf, _fuel_tgts)
            _unc   = ("—" if _uv is None else
                      ("🟢 Низкая" if _uv < 0.3 else ("🟡 Средняя" if _uv < 0.6 else "🔴 Высокая")))

            _uv_sh = _r_unc(_sdf, _shop_tgts)
            _unc_sh = ("—" if _uv_sh is None else
                       ("🟢 Низкая" if _uv_sh < 0.3 else ("🟡 Средняя" if _uv_sh < 0.6 else "🔴 Высокая")))

            _risk = 0.0
            _all_mv = [v for v in [_mv, _mv_sh] if v is not None]
            _all_uv = [v for v in [_uv, _uv_sh] if v is not None]
            if _all_mv: _risk += min(50.0, float(np.mean(_all_mv)) * 2)
            if _all_uv: _risk += min(50.0, float(np.mean(_all_uv)) * 50)
            _rlbl = "🟢 Низкий" if _risk < 25 else ("🟡 Средний" if _risk < 50 else "🔴 Высокий")

            _profile.append({
                "Станция":            str(_st),
                "Тренд":              _trend,
                "MAPE топливо":       f"{_mv:.1f}%"    if _mv    is not None else "—",
                "MAPE магазин":       f"{_mv_sh:.1f}%" if _mv_sh is not None else "—",
                "Неопред. топливо":   _unc,
                "Неопред. магазин":   _unc_sh,
                "Риск":               _rlbl,
                "_mape":              _mv    if _mv    is not None else 999.0,
                "_mape_sh":           _mv_sh if _mv_sh is not None else 999.0,
                "_unc":               _uv    if _uv    is not None else 0.0,
                "_risk":              _risk,
            })

        st.dataframe(
            pd.DataFrame(_profile).drop(columns=["_mape", "_mape_sh", "_unc", "_risk"]),
            hide_index=True, width="stretch",
        )

        st.divider()

        # ── БЛОК 2: Топливные рекомендации ────────────────────────────────────
        st.markdown("### ⛽ Топливные рекомендации")
        st.caption(
            "Все расчёты основаны на прогнозах модели. "
            "**Дефицит** определяется по реальным данным: факт продаж был ниже прогноза более чем в 50% дней. "
            "**Рекомендуемый запас** = сумма верхнего квантиля прогноза (q90) по всем дням горизонта. "
            "q90 — это уровень при котором с вероятностью 90% реальный спрос не превысит запас. "
            "Единицы измерения: топливо — литры, магазин — штуки/порции."
        )

        if _fuel_tgts:
            _deficit = []
            for _st in _reco_sts:
                _sdf = rdf[rdf[group_col] == _st]
                for _ft in _fuel_tgts:
                    _pc = f"{_ft}_pred"
                    if _pc not in _sdf.columns:
                        continue
                    # Факт из merged; если нет — пропускаем (нельзя делать вывод без реальных данных)
                    if (_reco_actual is not None and _ft in _reco_actual.columns
                            and "date" in _sdf.columns):
                        _act_st = _reco_actual[_reco_actual[group_col] == _st]
                        _joined = _sdf[["date", _pc]].merge(_act_st[["date", _ft]], on="date", how="inner")
                        _a, _p = _joined[_ft], _joined[_pc]
                    elif _ft in _sdf.columns:
                        _a, _p = _sdf[_ft], _sdf[_pc]
                    else:
                        continue
                    _m = _a > 1
                    if _m.sum() >= 3:
                        _ds = (_a[_m] < _p[_m] * 0.9).mean()
                        if _ds > 0.5:
                            _deficit.append({
                                "Станция":    str(_st),
                                "Топливо":    lbl(_ft),
                                "Дней с дефицитом": f"{_ds*100:.0f}%",
                                "Рекомендация": "⚠️ Увеличить завоз",
                            })

            if _deficit:
                st.markdown("**Вероятный дефицит запасов** (факт < прогноза в >50% дней):")
                st.dataframe(pd.DataFrame(_deficit), hide_index=True, width="stretch")
            else:
                st.success("Систематического дефицита не обнаружено.")

            st.markdown("**Рекомендуемый запас** (сумма q90 на горизонте прогноза):")
            st.caption("Сколько литров каждого вида топлива нужно иметь в наличии на весь период прогноза, чтобы покрыть спрос с вероятностью 90%.")
            _stock = []
            for _st in _reco_sts:
                _sdf = rdf[rdf[group_col] == _st]
                _row = {"Станция": str(_st)}
                for _ft in _fuel_tgts:
                    _q90c = f"{_ft}_q90"
                    if _q90c in _sdf.columns:
                        _row[lbl(_ft)] = round(_sdf[_q90c].sum(), 0)
                _stock.append(_row)

            _stock_df = pd.DataFrame(_stock)
            st.dataframe(_stock_df, hide_index=True, width="stretch")

            _bar_data = []
            for _, _r in _stock_df.iterrows():
                for _fl in [lbl(f) for f in _fuel_tgts if f"{f}_q90" in rdf.columns]:
                    if _fl in _r and _r[_fl] > 0:
                        _bar_data.append({"Станция": _r["Станция"], "Топливо": _fl, "Запас (q90)": _r[_fl]})
            if _bar_data:
                _fig_s = px.bar(
                    pd.DataFrame(_bar_data), x="Топливо", y="Запас (q90)", color="Станция",
                    barmode="group",
                    labels={"Запас (q90)": "Рекомендуемый запас (ед.)"},
                    title="Рекомендуемый запас по топливу × станция",
                )
                _pchart(_fig_s)

        if _shop_tgts:
            st.markdown("#### 🛒 Магазин и кафе")

            _shop_deficit = []
            for _st in _reco_sts:
                _sdf = rdf[rdf[group_col] == _st]
                for _ft in _shop_tgts:
                    _pc = f"{_ft}_pred"
                    if _pc not in _sdf.columns:
                        continue
                    if (_reco_actual is not None and _ft in _reco_actual.columns
                            and "date" in _sdf.columns):
                        _act_st = _reco_actual[_reco_actual[group_col] == _st]
                        _joined = _sdf[["date", _pc]].merge(_act_st[["date", _ft]], on="date", how="inner")
                        _a, _p = _joined[_ft], _joined[_pc]
                    elif _ft in _sdf.columns:
                        _a, _p = _sdf[_ft], _sdf[_pc]
                    else:
                        continue
                    _m = _a > 1
                    if _m.sum() >= 3:
                        _ds = (_a[_m] < _p[_m] * 0.9).mean()
                        if _ds > 0.5:
                            _shop_deficit.append({
                                "Станция":          str(_st),
                                "Категория":        lbl(_ft),
                                "Дней с дефицитом": f"{_ds*100:.0f}%",
                                "Рекомендация":     "⚠️ Увеличить закупку",
                            })

            if _shop_deficit:
                st.markdown("**Вероятный дефицит товаров** (факт < прогноза в >50% дней):")
                st.dataframe(pd.DataFrame(_shop_deficit), hide_index=True, width="stretch")
            else:
                st.success("Систематического дефицита товаров не обнаружено.")

            st.markdown("**Рекомендуемый запас товаров** (сумма q90 на горизонте прогноза):")
            st.caption("Сколько единиц товара каждой категории нужно закупить на весь период прогноза, чтобы покрыть спрос с вероятностью 90%.")
            _shop_stock = []
            for _st in _reco_sts:
                _sdf = rdf[rdf[group_col] == _st]
                _row = {"Станция": str(_st)}
                for _ft in _shop_tgts:
                    _q90c = f"{_ft}_q90"
                    if _q90c in _sdf.columns:
                        _row[lbl(_ft)] = round(_sdf[_q90c].sum(), 0)
                _shop_stock.append(_row)

            _shop_stock_df = pd.DataFrame(_shop_stock)
            st.dataframe(_shop_stock_df, hide_index=True, width="stretch")

            _shop_bar = []
            for _, _r in _shop_stock_df.iterrows():
                for _fl in [lbl(f) for f in _shop_tgts if f"{f}_q90" in rdf.columns]:
                    if _fl in _r and _r[_fl] > 0:
                        _shop_bar.append({"Станция": _r["Станция"], "Категория": _fl, "Запас (q90)": _r[_fl]})
            if _shop_bar:
                _fig_sh = px.bar(
                    pd.DataFrame(_shop_bar), x="Категория", y="Запас (q90)", color="Станция",
                    barmode="group",
                    labels={"Запас (q90)": "Рекомендуемый запас (ед.)"},
                    title="Рекомендуемый запас товаров × станция",
                )
                _pchart(_fig_sh)

        st.divider()

        # ── БЛОК 3: Операционные рекомендации ─────────────────────────────────
        st.markdown("### 📅 Операционные рекомендации")
        st.caption(
            "Пиковые дни определяются по суммарному прогнозному спросу по всем станциям. "
            "Эффект акций и разница выходные/будни — из исторических данных за весь период обучения."
        )

        _pri = (
            max(_fuel_tgts, key=lambda t: rdf[t].mean() if t in rdf.columns else 0)
            if _fuel_tgts else None
        )
        if _pri and f"{_pri}_pred" in rdf.columns and "date" in rdf.columns:
            _pc_lbl = lbl(_pri)
            _daily = (
                rdf.groupby("date")[f"{_pri}_pred"].sum()
                .reset_index()
                .rename(columns={"date": "Дата", f"{_pri}_pred": f"Суммарный спрос ({_pc_lbl})"})
                .sort_values(f"Суммарный спрос ({_pc_lbl})", ascending=False)
                .head(5)
            )
            _daily["Дата"] = _daily["Дата"].dt.date
            st.markdown("**Топ-5 дней пикового спроса** (усилить завоз):")
            st.dataframe(_daily, hide_index=True, width="stretch")

        if _shop_tgts:
            _pri_sh = _shop_tgts[0]
            _pc_sh  = f"{_pri_sh}_pred"
            if _pc_sh in rdf.columns and "date" in rdf.columns:
                _sh_lbl = lbl(_pri_sh)
                _daily_sh = (
                    rdf.groupby("date")[_pc_sh].sum()
                    .reset_index()
                    .rename(columns={"date": "Дата", _pc_sh: f"Суммарный спрос ({_sh_lbl})"})
                    .sort_values(f"Суммарный спрос ({_sh_lbl})", ascending=False)
                    .head(5)
                )
                _daily_sh["Дата"] = _daily_sh["Дата"].dt.date
                st.markdown(f"**Топ-5 дней пикового спроса на товары магазина** ({_sh_lbl}):")
                st.dataframe(_daily_sh, hide_index=True, width="stretch")

        if "promotion_shop_active" in merged.columns:
            _shop_total_cols = [c for c in _shop_tgts if c in merged.columns]
            if _shop_total_cols:
                _sh_sum = merged[_shop_total_cols].sum(axis=1)
                _promo_sh_avg = merged.groupby("promotion_shop_active").apply(
                    lambda g: _sh_sum.loc[g.index].mean()
                ).sort_index()
                if len(_promo_sh_avg) > 1:
                    _sh_min, _sh_max = _promo_sh_avg.iloc[0], _promo_sh_avg.iloc[-1]
                    if _sh_min > 0:
                        _sh_lift = (_sh_max - _sh_min) / _sh_min * 100
                        st.markdown(
                            f"🛒 Акции в магазине дают прирост продаж **+{_sh_lift:.1f}%** "
                            "относительно дней без акции."
                        )

        if "promotion_fuel_active" in merged.columns and total_sales_col and total_sales_col in merged.columns:
            _promo_avg = merged.groupby("promotion_fuel_active")[total_sales_col].mean()
            if len(_promo_avg) > 1:
                _pv = _promo_avg.sort_index()
                _p_min, _p_max = _pv.iloc[0], _pv.iloc[-1]
                if _p_min > 0:
                    _p_lift = (_p_max - _p_min) / _p_min * 100
                    st.markdown(
                        f"🎯 Акции на топливо дают прирост продаж **+{_p_lift:.1f}%** "
                        "относительно дней без акции."
                    )

        if date_col and date_col in merged.columns and total_sales_col and total_sales_col in merged.columns:
            _wd_df = merged[[date_col, total_sales_col]].copy()
            _wd_df["_we"] = pd.to_datetime(_wd_df[date_col]).dt.dayofweek >= 5
            _wd_avg = _wd_df.groupby("_we")[total_sales_col].mean()
            if True in _wd_avg.index and False in _wd_avg.index:
                _diff = (_wd_avg[True] - _wd_avg[False]) / _wd_avg[False] * 100
                st.markdown(
                    (f"📦 **Дифференцированная логистика рекомендуется**: "
                     f"продажи в выходные {'выше' if _diff > 0 else 'ниже'} будних на **{abs(_diff):.1f}%**.")
                    if abs(_diff) > 10 else
                    "✅ Существенной разницы выходные/будни нет — единый график завоза достаточен."
                )

        st.divider()

        # ── БЛОК 4: Зоны риска ────────────────────────────────────────────────
        st.markdown("### 🚨 Зоны риска")
        st.caption(
            "Станции с высокой неопределённостью требуют буферного запаса сверх q90. "
            "Станции с MAPE >20% — прогноз ненадёжен, используйте его с осторожностью и проверяйте вручную."
        )

        _risk_found = False

        for _row in _profile:
            _high_unc_fuel = "Высокая" in _row.get("Неопред. топливо", "")
            _high_unc_shop = "Высокая" in _row.get("Неопред. магазин", "")
            if _high_unc_fuel or _high_unc_shop:
                _unc_parts = []
                if _high_unc_fuel: _unc_parts.append("топливо")
                if _high_unc_shop: _unc_parts.append("магазин")
                st.warning(
                    f"**{_row['Станция']}** — высокая неопределённость прогноза "
                    f"({', '.join(_unc_parts)}): рекомендуется буферный запас +20–30%."
                )
                _risk_found = True

        for _row in _profile:
            if _row["_mape"] < 900 and _row["_mape"] > 20:
                st.error(
                    f"**{_row['Станция']}** — MAPE топливо = {_row['_mape']:.1f}%: "
                    "прогноз топлива ненадёжен, проверяйте вручную."
                )
                _risk_found = True
            if _row["_mape_sh"] < 900 and _row["_mape_sh"] > 20:
                st.error(
                    f"**{_row['Станция']}** — MAPE магазин = {_row['_mape_sh']:.1f}%: "
                    "прогноз товаров магазина ненадёжен, проверяйте вручную."
                )
                _risk_found = True

        if "date" in rdf.columns and "is_holiday" in rdf.columns:
            _hol_dates = (
                rdf[rdf["is_holiday"] == 1]["date"].dt.date.unique()
                if hasattr(rdf["date"], "dt") else []
            )
            if len(_hol_dates) > 0:
                st.info(
                    f"📅 Праздничные дни в горизонте прогноза: "
                    f"{', '.join(str(d) for d in sorted(_hol_dates)[:10])}. "
                    "Исторически возможны аномальные продажи."
                )
                _risk_found = True

        if not _risk_found:
            st.success("🟢 Серьёзных зон риска не обнаружено.")
