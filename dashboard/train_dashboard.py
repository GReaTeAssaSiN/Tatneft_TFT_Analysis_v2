"""
TFT Training Dashboard.
Run from project root:
    streamlit run dashboard/train_dashboard.py
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import plotly.graph_objects as go
import streamlit as st

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.train_utils import (
    CPU_DEFAULTS,
    GPU_DEFAULTS,
    PREDICT_PY,
    TRAIN_PY,
    check_session_files,
    delete_train_config,
    fmt_duration,
    get_ckpt_dir,
    get_export_path,
    get_group_count,
    get_model_path,
    get_predictions_dir,
    get_train_config_path,
    get_training_dir,
    list_ckpt_files,
    list_export_sessions,
    list_tb_versions,
    load_session_config,
    load_tft_config,
    load_split_config,
    load_train_config,
    get_data_info,
    read_tb_losses,
    save_split_config_to_training,
    save_train_config,
)

# ─── Colors ────────────────────────────────────────────────────────────────────
GOLD      = "#c8a84b"
GREEN     = "#4ECB71"
RED       = "#E24B4A"
BLUE      = "#2E75B6"
TEAL      = "#1ABC9C"
GRAY      = "#8B949E"
CARD_BG   = "#13161f"
GRID      = "#1e2235"

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TFT Обучение",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{ background-color: #0d1117; color: #c9d1d9; }}
[data-testid="stHeader"] {{ background-color: #0d1117; }}
.stTabs [data-baseweb="tab-list"] {{
    background-color: {CARD_BG}; border-radius: 8px; padding: 2px 4px; gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {GRAY}; background-color: transparent; border-radius: 6px; padding: 6px 16px;
}}
.stTabs [aria-selected="true"] {{
    color: {GOLD} !important; background-color: {GRID} !important;
}}
div[data-testid="metric-container"] {{
    background-color: {CARD_BG}; border: 1px solid {GRID};
    border-radius: 8px; padding: 12px 16px;
}}

/* ── Управление обучением: кнопка «Остановить» ── */
.stop-btn [data-testid="stBaseButton-secondary"] {{
    background: linear-gradient(135deg, {RED} 0%, #c93b3a 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 2px 8px rgba(226,75,74,0.25) !important;
    transition: all 0.18s ease !important;
}}
.stop-btn [data-testid="stBaseButton-secondary"]:hover {{
    background: linear-gradient(135deg, #c93b3a 0%, {RED} 100%) !important;
    box-shadow: 0 4px 16px rgba(226,75,74,0.42) !important;
    transform: translateY(-1px) !important;
}}
.stop-btn [data-testid="stBaseButton-secondary"]:active {{
    transform: translateY(0) !important;
    box-shadow: 0 1px 4px rgba(226,75,74,0.25) !important;
}}

</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "selected_session":   None,
        "train_proc":         None,
        "train_output":       [],
        "train_start_time":   None,
        "train_end_time":     None,
        "output_queue":       queue.Queue(),
        "reader_thread":      None,
        "_preset_values":     None,
        # session-lock state
        "_params_locked":     None,   # None=auto, True=locked, False=unlocked
        "_locked_for":        None,   # which export session the lock applies to
        "_files_scanned":     False,
        "_split_train_end":   "",
        "_split_val_end":     "",
        "_force_session":     None,   # revert selectbox to this value on next run
        "_session_loaded":    False,  # True once user clicks "Приступить"
        "_change_confirm":    False,  # True = showing "unsaved changes" warning
        "_training_done":     False,  # True when last run completed with exit_code 0
        # prediction subprocess state
        "pred_proc":          None,
        "pred_output":        [],
        "pred_start_time":    None,
        "pred_end_time":      None,
        "pred_output_queue":  queue.Queue(),
        "_last_pred_file":    None,   # path of last successfully completed prediction
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─── Training process helpers ──────────────────────────────────────────────────
def is_training() -> bool:
    proc = st.session_state.train_proc
    return proc is not None and proc.poll() is None


def start_training(session_name: str):
    if is_training():
        return
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["TFT_SESSION"] = session_name

    proc = subprocess.Popen(
        [sys.executable, TRAIN_PY],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        env=env,
    )
    st.session_state.train_proc       = proc
    st.session_state.train_output     = []
    st.session_state.train_start_time = time.time()
    st.session_state.train_end_time   = None
    st.session_state.output_queue     = queue.Queue()
    st.session_state._training_done   = False
    _marker = os.path.join(get_training_dir(session_name), "training_complete.flag")
    try:
        os.remove(_marker)
    except FileNotFoundError:
        pass

    def _reader(p, q):
        for line in p.stdout:
            q.put(line.rstrip("\r\n"))
        q.put(None)

    t = threading.Thread(
        target=_reader, args=(proc, st.session_state.output_queue), daemon=True
    )
    t.start()
    st.session_state.reader_thread = t


def stop_training():
    proc = st.session_state.train_proc
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    if st.session_state.train_end_time is None:
        st.session_state.train_end_time = time.time()


def drain_queue() -> bool:
    q = st.session_state.output_queue
    new = False
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            if st.session_state.train_end_time is None:
                st.session_state.train_end_time = time.time()
            proc = st.session_state.train_proc
            if proc and proc.returncode == 0:
                st.session_state._training_done = True
                _sess = st.session_state.selected_session
                if _sess:
                    _marker = os.path.join(get_training_dir(_sess), "training_complete.flag")
                    try:
                        open(_marker, "w").close()
                    except Exception:
                        pass
            break
        st.session_state.train_output.append(line)
        new = True
    return new


# ─── Prediction process helpers ────────────────────────────────────────────────
def is_predicting() -> bool:
    proc = st.session_state.pred_proc
    return proc is not None and proc.poll() is None


def start_prediction(session_name: str, ckpt_path: str, output_file: str, split: str):
    if is_predicting():
        return
    st.session_state._pending_pred_file = output_file  # saved on success
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["TFT_SESSION"]       = session_name
    env["TFT_CHECKPOINT"]    = ckpt_path
    env["TFT_OUTPUT_FILE"]   = output_file
    env["TFT_PREDICT_SPLIT"] = split

    proc = subprocess.Popen(
        [sys.executable, PREDICT_PY],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        env=env,
    )
    st.session_state.pred_proc        = proc
    st.session_state.pred_output      = []
    st.session_state.pred_start_time  = time.time()
    st.session_state.pred_end_time    = None
    st.session_state.pred_output_queue = queue.Queue()
    st.session_state._last_pred_file  = None  # reset until confirmed complete

    q = st.session_state.pred_output_queue

    def _reader(p, q):
        for line in p.stdout:
            q.put(line.rstrip("\r\n"))
        q.put(None)

    threading.Thread(target=_reader, args=(proc, q), daemon=True).start()


def stop_prediction():
    proc = st.session_state.pred_proc
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    if st.session_state.pred_end_time is None:
        st.session_state.pred_end_time = time.time()


def drain_pred_queue() -> bool:
    q = st.session_state.pred_output_queue
    new = False
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            if st.session_state.pred_end_time is None:
                st.session_state.pred_end_time = time.time()
            proc = st.session_state.pred_proc
            if proc:
                proc.poll()  # ensure returncode is populated
                if proc.returncode == 0:
                    st.session_state._last_pred_file = st.session_state.get("_pending_pred_file")
            break
        st.session_state.pred_output.append(line)
        new = True
    return new


def elapsed_str(start, end=None) -> str:
    if start is None:
        return "—"
    secs = int((end or time.time()) - start)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ─── UI helpers ────────────────────────────────────────────────────────────────
def section_header(icon: str, title: str, subtitle: str = "", color: str = GOLD):
    sub = (f'<span style="color:#4b5563;font-size:11px;white-space:nowrap;">'
           f'{subtitle}</span>') if subtitle else ""
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:16px 0 12px 0;">
  <span style="color:{color};font-size:11px;font-weight:700;letter-spacing:.1em;
               text-transform:uppercase;white-space:nowrap;">{icon} {title}</span>
  <span style="flex:1;height:1px;background:#2a2f45;"></span>
  {sub}
</div>""", unsafe_allow_html=True)


def render_output_html(lines: list) -> tuple[str, int]:
    import html as _html
    import re as _re

    _ansi_re   = _re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b[^[]?')
    _pb_re     = _re.compile(r'^(.+?):\s+(?:\d+%\||\|)')
    _pct_re    = _re.compile(r'(\d+)%\|')
    _val_outer = _re.compile(r'^Validation:\s+(?:\d+%\||\|)')
    _val_dl    = _re.compile(r'^Validation\s+DataLoader\s+\d+:\s+(\d+)%\|')
    _val_info  = _re.compile(r'(\d+)/(\d+)\s+\[([^\]]+)\]')

    deduped: list = []
    key_pos: dict = {}  # pb key → fixed index in deduped
    _val_started = False  # True while validation is in progress

    for raw in lines:
        line = _ansi_re.sub('', raw.replace('\r', ''))

        # Validation outer bar — drop completely
        if _val_outer.match(line):
            continue

        # Validation DataLoader bar → two lines max: start + completion
        vdl_m = _val_dl.match(line)
        if vdl_m:
            pct = int(vdl_m.group(1))
            if pct == 100:
                info_m = _val_info.search(line)
                if info_m:
                    cur_n, tot_n, timing = info_m.group(1), info_m.group(2), info_m.group(3)
                    deduped.append(f"✓ Валидация завершена  {cur_n}/{tot_n} батч.  [{timing}]")
                else:
                    deduped.append("✓ Валидация завершена")
                _val_started = False
            elif not _val_started:
                deduped.append("⏳ Идёт валидация, подождите...")
                _val_started = True
            # else: intermediate update — skip
            continue

        m = _pb_re.match(line)
        if m:
            key = m.group(1).strip()
            if key in key_pos:
                idx = key_pos[key]
                cur_m = _pct_re.search(deduped[idx])
                new_m = _pct_re.search(line)
                cur_pct = int(cur_m.group(1)) if cur_m else 0
                new_pct = int(new_m.group(1)) if new_m else 0
                if new_pct >= cur_pct:
                    deduped[idx] = line
                # else skip: tqdm сбросил бар на 0% перед следующей эпохой
            else:
                key_pos[key] = len(deduped)
                deduped.append(line)
        else:
            deduped.append(line)

    # схлопываем подряд идущие пустые строки
    _clean: list = []
    _prev_blank = False
    for _ln in deduped:
        _blank = not _ln.strip()
        if _blank and _prev_blank:
            continue
        _clean.append(_ln)
        _prev_blank = _blank
    deduped = _clean

    _tbl_hdr   = _re.compile(r'\|\s*Name\s*\|')
    _tbl_row   = _re.compile(r'^\s*\d+\s+\|\s+\w')
    _param_sum = _re.compile(r'(Trainable|Non-trainable|Total)\s+params')
    _separator = _re.compile(r'^[=\-]{8,}\s*$')
    _epoch_pb  = _re.compile(r'^Epoch\s+\d+:')
    _val_loss  = _re.compile(r'val_loss', _re.IGNORECASE)
    _kv_line   = _re.compile(r'^\s{2,}\S[^:]{2,}:\s+\S')

    in_table = False
    parts = []

    def _span(text, color, bold=False):
        w = "font-weight:700;" if bold else ""
        return f'<span style="color:{color};{w}">{text}</span>'

    for line in deduped:
        stripped = line.strip()
        esc = _html.escape(line)
        if not stripped:
            parts.append('')
            continue
        if stripped.startswith('💡') or ('Tip:' in stripped):
            continue
        if _re.match(r'^(GPU|TPU|HPU)\s+available:', stripped):
            parts.append(_span(esc, '#5a6270'))
            continue
        if _tbl_hdr.search(stripped):
            in_table = True
            parts.append(_span('  ── Архитектура модели ─────────────────', GRAY))
            continue
        if in_table and (_tbl_row.match(stripped) or stripped.startswith('|')):
            parts.append(_span(esc, '#4a5260'))
            continue
        if _param_sum.search(stripped):
            in_table = False
            parts.append(_span(esc, BLUE, bold=True))
            continue
        if in_table and not stripped.startswith('|'):
            in_table = False
        if _separator.match(stripped):
            parts.append(_span(esc, '#2a2f45'))
            continue
        if _re.search(r'(Error|Traceback|Exception)', stripped):
            parts.append(_span(esc, RED, bold=True))
            continue
        if _val_loss.search(stripped):
            parts.append(_span(esc, GREEN))
            continue
        if _re.match(r'^✅ BEST\b', stripped):
            parts.append(
                f'<div style="margin:3px 0;padding:4px 12px;'
                f'background:#0a2218;border-left:3px solid {GREEN};'
                f'border-radius:0 6px 6px 0;">'
                f'<span style="color:{GREEN};font-weight:700;">{esc}</span></div>'
            )
            continue
        if _re.search(r'(model\.ckpt|Лучший|best)', stripped, _re.IGNORECASE):
            parts.append(_span(esc, GREEN, bold=True))
            continue
        if _epoch_pb.match(stripped):
            parts.append(_span(esc, GOLD))
            continue
        if stripped.startswith(('⏳', '✓ Валидация')):
            parts.append(_span(esc, TEAL))
            continue
        if _re.search(r'\d+%\|', stripped) or 'it/s' in stripped:
            parts.append(_span(esc, TEAL))
            continue
        if _re.match(r'Rest(oring|ored)', stripped):
            parts.append(_span(esc, GRAY))
            continue
        if _kv_line.match(line):
            ci = esc.find(':')
            if ci > 0:
                parts.append(
                    f'<span style="color:{GRAY};">{esc[:ci+1]}</span>'
                    f'<span style="color:#c9d1d9;">{esc[ci+1:]}</span>'
                )
                continue
        parts.append(f'<span style="color:#c9d1d9;">{esc}</span>')

    return '\n'.join(parts), len(deduped)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h1 style='color:{GOLD};margin:0 0 2px 0;font-size:26px;'>🧠 TFT — Управление обучением</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:{GRAY};margin:0 0 14px 0;font-size:13px;'>"
    "Выбор сессии · Гиперпараметры · Запуск обучения · Мониторинг результатов</p>",
    unsafe_allow_html=True,
)

# ─── Session selector ──────────────────────────────────────────────────────────
sessions = list_export_sessions()

if not sessions:
    st.error(
        "❌ Не найдено ни одной сессии в папке `exports/` с обязательными файлами "
        "(`processed_data.csv` + `tft_config.json`). "
        "Сначала выполните подготовку данных на дашборде данных."
    )
    st.stop()

# If training was active and user tried to switch session last run, revert now
# (must happen BEFORE the selectbox widget renders — Streamlit disallows setting
#  a widget's key after it has been instantiated in the same run)
if st.session_state._force_session is not None and is_training():
    st.session_state.selected_session = st.session_state._force_session
    st.session_state._force_session = None

# Ensure session state has a valid selection before the widget renders.
# When key="selected_session" is used, Streamlit uses session_state value
# instead of index= parameter — so None would show "Choose an option".
if st.session_state.selected_session not in sessions:
    st.session_state.selected_session = sessions[0]

_prev_session = st.session_state.selected_session
sel_session = st.selectbox(
    "📂 Сессия экспорта",
    options=sessions,
    index=sessions.index(_prev_session) if _prev_session in sessions else 0,
    key="selected_session",
    disabled=st.session_state._session_loaded,
    help="Сессия зафиксирована. Для смены используйте кнопку «← Сменить сессию»."
         if st.session_state._session_loaded else
         "Выберите папку из exports/ с обязательными файлами.",
)

if sel_session is None:
    sel_session = sessions[0]

if sel_session != _prev_session and is_training():
    st.warning("⚠️ Идёт обучение по предыдущей сессии. Остановите его перед сменой.")
    st.session_state._force_session = _prev_session
    sel_session = _prev_session
    st.rerun()

# Reset all per-session state when session changes
if sel_session != st.session_state._locked_for:
    st.session_state._params_locked    = None
    st.session_state._files_scanned    = False
    st.session_state._session_loaded   = False
    st.session_state._change_confirm   = False
    st.session_state._locked_for       = sel_session
    st.session_state["_split_train_end"] = ""
    st.session_state["_split_val_end"]   = ""

training_dir  = get_training_dir(sel_session)
config_exists = os.path.exists(get_train_config_path(sel_session))
params_locked = (
    config_exists if st.session_state._params_locked is None
    else bool(st.session_state._params_locked)
)

# ══════════════════════════════════════════════════════════════════════════════
# SELECTION SCREEN  (shown until user clicks "Приступить")
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state._session_loaded:
    _finfo = check_session_files(sel_session)

    st.markdown(
        f"<p style='color:{GRAY};font-size:13px;margin:0 0 12px 0;'>"
        "Выберите папку экспорта — ниже отображается её содержимое. "
        "После проверки нажмите «Приступить к обучению».</p>",
        unsafe_allow_html=True,
    )

    with st.expander("📁 Содержимое папки сессии", expanded=True):
        def _badge_html(text, bg, fg):
            return (f"<span style='background:{bg};color:{fg};font-size:10px;"
                    f"padding:1px 7px;border-radius:3px;margin-left:6px;'>{text}</span>")

        _sections = [
            (_finfo["train"],   "обучение",  GREEN,  "#1e3a1e", "✅", "❌"),
            (_finfo["predict"], "прогноз",   TEAL,   "#1a2e2e", "🔮", "⚠️"),
            (_finfo["optional"],"опционально",GOLD,  "#2e2a10", "📅", "📅"),
            (_finfo["unused"],  "не используется", GRAY, CARD_BG, "📄", "📄"),
            (_finfo["other"],   "",          GRAY,   CARD_BG,   "📄", "📄"),
        ]
        for _group, _label, _fg, _bg, _icon_ok, _icon_miss in _sections:
            for _fi in _group:
                _exists = _fi["exists"]
                _icon   = _icon_ok if _exists else _icon_miss
                _color  = _fg if _exists else (RED if _label in ("обучение","прогноз") else GRAY)
                _badge  = _badge_html(_label, _bg, _fg) if _label else ""
                if not _exists and _label in ("обучение", "прогноз"):
                    _badge = _badge_html("отсутствует!", "#3a1e1e", RED)
                _sz = (f"<span style='color:{GRAY};font-size:11px;margin-left:8px;'>"
                       f"{_fi['size']/1024:.1f} КБ</span>") if _fi.get("size") else ""
                st.markdown(
                    f"<div style='padding:4px 0;border-bottom:1px solid {GRID};'>"
                    f"{_icon} <span style='color:{_color};font-size:13px;'>{_fi['name']}</span>"
                    f"{_badge}{_sz}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Content validation results ────────────────────────────────────────────
    _c_issues   = _finfo.get("content_issues", [])
    _c_warnings = _finfo.get("content_warnings", [])
    if _c_issues or _c_warnings:
        with st.expander("🔍 Проверка содержимого файлов", expanded=bool(_c_issues)):
            for _msg in _c_issues:
                st.error(f"❌ {_msg}")
            for _msg in _c_warnings:
                st.warning(f"⚠️ {_msg}")

    _can_start = _finfo["all_required_present"] and not _c_issues
    if _can_start:
        if st.button(
            "▶  Приступить к обучению TFT-модели",
            type="primary",
            use_container_width=True,
        ):
            st.session_state._session_loaded = True
            st.rerun()
    elif not _finfo["all_required_present"]:
        st.error(
            "❌ Обязательные файлы отсутствуют. "
            "Сначала выполните подготовку данных на дашборде данных."
        )
    else:
        st.error(
            "❌ Сессия содержит ошибки содержимого (см. выше). "
            "Исправьте их в дашборде предобработки и повторно экспортируйте."
        )

    st.stop()

# ── "Change session" button (shown when tabs are loaded) ──────────────────────
_btn_col, _info_col = st.columns([1, 5])
with _btn_col:
    if st.button("← Сменить сессию", use_container_width=True):
        # Safe to switch if: no config saved yet, OR config saved and locked
        _safe = not config_exists or params_locked
        if _safe:
            st.session_state._session_loaded  = False
            st.session_state._change_confirm  = False
            st.rerun()
        else:
            st.session_state._change_confirm = not st.session_state._change_confirm
            st.rerun()

if st.session_state._change_confirm:
    st.warning(
        "⚠️ Гиперпараметры изменены, но сессия не сохранена. "
        "При смене несохранённые изменения будут потеряны."
    )
    _cc1, _cc2 = st.columns(2)
    with _cc1:
        if st.button("✅ Всё равно сменить", type="primary", use_container_width=True):
            st.session_state._session_loaded = False
            st.session_state._change_confirm = False
            st.rerun()
    with _cc2:
        if st.button("↩ Отмена", use_container_width=True):
            st.session_state._change_confirm = False
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["⚙️  Конфигурация", "▶  Обучение", "📊  Результаты", "🔮  Прогнозы"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Конфигурация
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    cfg = load_train_config(sel_session)
    if st.session_state._preset_values is not None:
        cfg.update(st.session_state._preset_values)
        st.session_state._preset_values = None

    data_info    = get_data_info(sel_session)
    tft_cfg      = load_tft_config(sel_session)
    split_cfg    = load_split_config(sel_session)
    session_cfg  = load_session_config(sel_session)

    # ── Информация о сессии ───────────────────────────────────────────────────
    section_header("📋", "СЕССИЯ ЭКСПОРТА")

    if tft_cfg:
        _targets    = tft_cfg.get("target", [])
        _sc_cats    = tft_cfg.get("static_cats", [])
        _sc_reals   = tft_cfg.get("static_reals", [])
        _kn_cats    = tft_cfg.get("time_varying_known_categoricals", [])
        _kn_reals   = tft_cfg.get("time_varying_known_reals", [])
        _un_reals   = tft_cfg.get("time_varying_unknown_reals", [])
        _group_col  = tft_cfg.get("group_col", "—")
        _time_col   = tft_cfg.get("time_col", "—")

        # ── Строка данных ─────────────────────────────────────────────────
        _n_rows   = data_info["n_rows"] if data_info else None
        _n_groups = get_group_count(sel_session)
        _gran     = (session_cfg.get("tidx_config") or {}).get("gran", "—")
        _steps    = (_n_rows // _n_groups) if (_n_rows and _n_groups) else None

        _stats_items = []
        if _n_rows:
            _stats_items.append(
                f"<span style='color:#c9d1d9;'><b style='color:{GOLD};font-size:15px;'>"
                f"{_n_rows:,}</b> <span style='color:{GRAY};font-size:12px;'>строк</span></span>"
            )
        if _n_groups:
            _stats_items.append(
                f"<span style='color:#c9d1d9;'><b style='color:{GOLD};font-size:15px;'>"
                f"{_n_groups}</b> <span style='color:{GRAY};font-size:12px;'>групп ({_group_col})</span></span>"
            )
        if _steps:
            _stats_items.append(
                f"<span style='color:#c9d1d9;'><b style='color:{GOLD};font-size:15px;'>"
                f"~{_steps:,}</b> <span style='color:{GRAY};font-size:12px;'>шагов/группу</span></span>"
            )
        if _gran != "—":
            _stats_items.append(
                f"<span style='color:#c9d1d9;'><b style='color:{GOLD};font-size:15px;'>"
                f"{_gran}</b> <span style='color:{GRAY};font-size:12px;'>гранулярность</span></span>"
            )

        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {GRID};"
            f"border-radius:6px;padding:12px 20px;margin-bottom:14px;"
            f"display:flex;gap:36px;align-items:center;flex-wrap:wrap;'>"
            + "  <span style='color:#2a2f45;'>|</span>  ".join(_stats_items)
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Цели и признаки ───────────────────────────────────────────────
        _ci1, _ci2 = st.columns(2)

        with _ci1:
            _fuel   = [t for t in _targets if t.startswith("sales_")]
            _shop   = [t for t in _targets if t.startswith("shop_")]
            _other  = [t for t in _targets if not t.startswith(("sales_", "shop_"))]

            def _tag(name: str) -> str:
                short = name.replace("sales_", "").replace("shop_", "")
                return f"<span style='background:{GRID};color:#c9d1d9;font-size:11px;" \
                       f"padding:2px 7px;border-radius:4px;margin:2px 2px;'>{short}</span>"

            _target_html = (
                f"<div style='background:{CARD_BG};border:1px solid {GRID};"
                f"border-radius:6px;padding:12px 14px;height:100%;'>"
                f"<div style='color:{GOLD};font-size:11px;font-weight:700;"
                f"letter-spacing:.08em;margin-bottom:10px;'>🎯 ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ ({len(_targets)})</div>"
            )
            if _fuel:
                _target_html += (
                    f"<div style='color:{GRAY};font-size:11px;margin-bottom:4px;'>"
                    f"Топливо ({len(_fuel)})</div>"
                    f"<div style='margin-bottom:10px;'>{''.join(_tag(t) for t in _fuel)}</div>"
                )
            if _shop:
                _target_html += (
                    f"<div style='color:{GRAY};font-size:11px;margin-bottom:4px;'>"
                    f"НСК / Магазин ({len(_shop)})</div>"
                    f"<div style='margin-bottom:10px;'>{''.join(_tag(t) for t in _shop)}</div>"
                )
            if _other:
                _target_html += (
                    f"<div style='color:{GRAY};font-size:11px;margin-bottom:4px;'>"
                    f"Прочее ({len(_other)})</div>"
                    f"<div>{''.join(_tag(t) for t in _other)}</div>"
                )
            _target_html += "</div>"
            st.markdown(_target_html, unsafe_allow_html=True)

        with _ci2:
            def _feat_row(label: str, n_cat: int, n_real: int) -> str:
                parts = []
                if n_cat:
                    parts.append(
                        f"<span style='color:{TEAL};font-weight:600;'>{n_cat}</span>"
                        f"<span style='color:{GRAY};'> кат</span>"
                    )
                if n_real:
                    parts.append(
                        f"<span style='color:{BLUE};font-weight:600;'>{n_real}</span>"
                        f"<span style='color:{GRAY};'> вещ</span>"
                    )
                val = " + ".join(parts) if parts else f"<span style='color:{GRAY};'>—</span>"
                return (
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:5px 0;border-bottom:1px solid {GRID};font-size:12px;'>"
                    f"<span style='color:#c9d1d9;'>{label}</span><span>{val}</span></div>"
                )

            _feat_html = (
                f"<div style='background:{CARD_BG};border:1px solid {GRID};"
                f"border-radius:6px;padding:12px 14px;'>"
                f"<div style='color:{GOLD};font-size:11px;font-weight:700;"
                f"letter-spacing:.08em;margin-bottom:10px;'>🔧 ПРИЗНАКИ</div>"
                + _feat_row("Статические", len(_sc_cats), len(_sc_reals))
                + _feat_row("Известные в будущем", len(_kn_cats), len(_kn_reals))
                + _feat_row("Только из прошлого", 0, len(_un_reals))
                + f"<div style='margin-top:10px;padding-top:8px;border-top:1px solid {GRID};'>"
                f"<span style='color:{GRAY};font-size:11px;'>Группировка:&nbsp;</span>"
                f"<code style='color:{TEAL};font-size:11px;'>{_group_col}</code>"
                f"&nbsp;&nbsp;"
                f"<span style='color:{GRAY};font-size:11px;'>Индекс:&nbsp;</span>"
                f"<code style='color:{TEAL};font-size:11px;'>{_time_col}</code>"
                f"</div></div>"
            )
            st.markdown(_feat_html, unsafe_allow_html=True)

    with st.expander("🔍 Полный TFT конфиг", expanded=False):
        if tft_cfg:
            st.json(tft_cfg)
        else:
            st.caption("tft_config.json не найден.")

    # ── Темпоральный сплит ────────────────────────────────────────────────────
    section_header("📅", "ТЕМПОРАЛЬНЫЙ СПЛИТ")
    if split_cfg:
        _te = str(split_cfg.get("train_end", "—"))
        _ve = str(split_cfg.get("val_end",   "—"))
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {GRID};"
            f"border-radius:6px;padding:11px 18px;display:flex;gap:0;"
            f"align-items:stretch;font-size:12px;'>"
            # Train
            f"<div style='flex:1;padding-right:16px;border-right:1px solid {GRID};'>"
            f"<div style='color:{GRAY};font-size:10px;letter-spacing:.08em;margin-bottom:4px;'>TRAIN</div>"
            f"<div style='color:#c9d1d9;'>начало&nbsp;→&nbsp;"
            f"<b style='color:{GREEN};'>{_te}</b></div></div>"
            # Val
            f"<div style='flex:1;padding:0 16px;border-right:1px solid {GRID};'>"
            f"<div style='color:{GRAY};font-size:10px;letter-spacing:.08em;margin-bottom:4px;'>VALIDATION</div>"
            f"<div style='color:#c9d1d9;'>"
            f"<b style='color:{GOLD};'>{_te}</b>&nbsp;→&nbsp;<b style='color:{GOLD};'>{_ve}</b>"
            f"</div></div>"
            # Test
            f"<div style='flex:1;padding-left:16px;'>"
            f"<div style='color:{GRAY};font-size:10px;letter-spacing:.08em;margin-bottom:4px;'>TEST</div>"
            f"<div style='color:#c9d1d9;'><b style='color:{TEAL};'>{_ve}</b>&nbsp;→&nbsp;конец</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"🔒 Сплит из `exports/{sel_session}/split_config.json` — изменение недоступно.")
    else:
        # Check if split was already saved to training dir
        _tr_split_path = os.path.join(get_training_dir(sel_session), "split_config.json")
        if os.path.exists(_tr_split_path):
            import json as _json
            with open(_tr_split_path, encoding="utf-8") as _f:
                _saved_split = _json.load(_f)
            _te2 = str(_saved_split.get("train_end", "—"))
            _ve2 = str(_saved_split.get("val_end",   "—"))
            st.markdown(
                f"<div style='background:{CARD_BG};border:1px solid {GRID};"
                f"border-radius:6px;padding:11px 18px;display:flex;gap:0;"
                f"align-items:stretch;font-size:12px;'>"
                f"<div style='flex:1;padding-right:16px;border-right:1px solid {GRID};'>"
                f"<div style='color:{GRAY};font-size:10px;letter-spacing:.08em;margin-bottom:4px;'>TRAIN</div>"
                f"<div style='color:#c9d1d9;'>начало → <b style='color:{GREEN};'>{_te2}</b></div></div>"
                f"<div style='flex:1;padding:0 16px;border-right:1px solid {GRID};'>"
                f"<div style='color:{GRAY};font-size:10px;letter-spacing:.08em;margin-bottom:4px;'>VALIDATION</div>"
                f"<div style='color:#c9d1d9;'><b style='color:{GOLD};'>{_te2}</b> → <b style='color:{GOLD};'>{_ve2}</b></div></div>"
                f"<div style='flex:1;padding-left:16px;'>"
                f"<div style='color:{GRAY};font-size:10px;letter-spacing:.08em;margin-bottom:4px;'>TEST</div>"
                f"<div style='color:#c9d1d9;'><b style='color:{TEAL};'>{_ve2}</b> → конец</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            st.caption("🔒 Сплит сохранён в папке обучения.")
        else:
            st.markdown(
                f"<p style='color:{GOLD};font-size:12px;margin:4px 0 10px 0;'>"
                "⚠️ split_config.json не найден в папке экспорта. "
                "Настройте разбивку — она сохранится в папку обучения.</p>",
                unsafe_allow_html=True,
            )

            # Load date range from merged_data.csv
            import pandas as _pd2
            _merged_path = os.path.join(get_export_path(sel_session), "merged_data.csv")
            _dt_s = None
            _dt_col = None
            if os.path.exists(_merged_path):
                try:
                    _mdf_head = _pd2.read_csv(_merged_path, nrows=0, encoding="utf-8")
                    for _c in _mdf_head.columns:
                        if "date" in _c.lower():
                            _dt_col = _c
                            break
                    if _dt_col:
                        _dt_s = _pd2.to_datetime(
                            _pd2.read_csv(_merged_path, usecols=[_dt_col],
                                         encoding="utf-8")[_dt_col],
                            errors="coerce",
                        ).dropna()
                except Exception:
                    pass

            if _dt_s is None or len(_dt_s) == 0:
                st.error(
                    "❌ Не удалось определить диапазон дат — файл `merged_data.csv` "
                    "не найден в папке экспорта или не содержит колонку с датами. "
                    "Убедитесь, что сессия была экспортирована с сохранением "
                    "`merged_data.csv` (опция «Сохранить объединённые данные» в дашборде "
                    "предобработки)."
                )
            else:
                _dt_min = _dt_s.min()
                _dt_max = _dt_s.max()
                st.caption(f"Диапазон дат в данных: `{_dt_min.date()}` → `{_dt_max.date()}`")

                _split_mode = st.radio(
                    "Режим разбивки",
                    ["По процентам", "По датам"],
                    horizontal=True,
                    key=f"_split_mode_{sel_session}",
                )

                _prev_te = st.session_state.get("_split_train_end", "")
                _prev_ve = st.session_state.get("_split_val_end", "")

                if _split_mode == "По процентам":
                    _spc1, _spc2, _spc3 = st.columns(3)
                    with _spc1:
                        _tr_pct = st.slider("Train %", 1, 97, 70,
                                            key=f"_spl_tr_pct_{sel_session}")
                    with _spc2:
                        _vl_pct = st.slider("Val %", 1, 99 - _tr_pct,
                                            min(15, 99 - _tr_pct),
                                            key=f"_spl_vl_pct_{sel_session}")
                    with _spc3:
                        st.metric("Test %", 100 - _tr_pct - _vl_pct)

                    _sorted_u = sorted(_dt_s.unique())
                    _nd = len(_sorted_u)
                    _tr_cut = _sorted_u[max(0, int(_nd * _tr_pct / 100) - 1)]
                    _vl_cut = _sorted_u[max(0, int(_nd * (_tr_pct + _vl_pct) / 100) - 1)]
                    _train_m = _dt_s <= _tr_cut
                    _val_m   = (_dt_s > _tr_cut) & (_dt_s <= _vl_cut)
                    _test_m  = _dt_s > _vl_cut
                    _new_split = {"train_end": str(_pd2.Timestamp(_tr_cut).date()),
                                  "val_end":   str(_pd2.Timestamp(_vl_cut).date())}
                    st.caption(f"Граница train: **{_pd2.Timestamp(_tr_cut).date()}** · "
                               f"граница val: **{_pd2.Timestamp(_vl_cut).date()}**")
                else:
                    _spc1, _spc2 = st.columns(2)
                    with _spc1:
                        _tr_end = st.date_input(
                            "Train до (включительно)",
                            value=(_pd2.to_datetime(_prev_te).date()
                                   if _prev_te else _dt_min.date()),
                            min_value=_dt_min.date(), max_value=_dt_max.date(),
                            key=f"_spl_tr_end_di_{sel_session}",
                        )
                    with _spc2:
                        _vl_end = st.date_input(
                            "Val до (включительно)",
                            value=(_pd2.to_datetime(_prev_ve).date()
                                   if _prev_ve else _dt_min.date()),
                            min_value=_dt_min.date(), max_value=_dt_max.date(),
                            key=f"_spl_vl_end_di_{sel_session}",
                        )
                    _tr_cut = _pd2.Timestamp(_tr_end)
                    _vl_cut = _pd2.Timestamp(_vl_end)
                    _train_m = _dt_s <= _tr_cut
                    _val_m   = (_dt_s > _tr_cut) & (_dt_s <= _vl_cut)
                    _test_m  = _dt_s > _vl_cut
                    _new_split = {"train_end": str(_tr_end), "val_end": str(_vl_end)}
                    _n_valid = len(_dt_s)
                    if _n_valid > 0:
                        _tr_p = int(round(_train_m.sum() / _n_valid * 100))
                        _vl_p = int(round(_val_m.sum()   / _n_valid * 100))
                        _ts_p = int(round(_test_m.sum()  / _n_valid * 100))
                        st.caption(f"Примерное соотношение: "
                                   f"Train **{_tr_p}%** · Val **{_vl_p}%** · Test **{_ts_p}%**")

                _sm1, _sm2, _sm3 = st.columns(3)
                _sm1.metric("Train", f"{int(_train_m.sum()):,} строк")
                _sm2.metric("Val",   f"{int(_val_m.sum()):,} строк")
                _sm3.metric("Test",  f"{int(_test_m.sum()):,} строк")

                _sp_errs = []
                if _train_m.sum() == 0: _sp_errs.append("Train пустой")
                if _val_m.sum()   == 0: _sp_errs.append("Val пустой")
                if _test_m.sum()  == 0: _sp_errs.append("Test пустой")
                if _split_mode == "По датам" and _tr_cut >= _vl_cut:
                    _sp_errs.append("Дата Train должна быть раньше даты Val")

                if _sp_errs:
                    st.error("❌ " + "; ".join(_sp_errs) + ". Скорректируйте параметры.")
                else:
                    if st.button("✅ Применить сплит", type="primary",
                                 key=f"_apply_split_{sel_session}"):
                        save_split_config_to_training(
                            sel_session,
                            _new_split["train_end"],
                            _new_split["val_end"],
                        )
                        st.session_state["_split_train_end"] = _new_split["train_end"]
                        st.session_state["_split_val_end"]   = _new_split["val_end"]
                        st.toast("Сплит сохранён!", icon="✅")
                        st.rerun()

    # ── Гиперпараметры ────────────────────────────────────────────────────────
    section_header("⚙️", "ГИПЕРПАРАМЕТРЫ")

    if params_locked:
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {GRID};"
            f"border-radius:6px;padding:8px 14px;margin-bottom:10px;"
            f"display:flex;align-items:center;gap:10px;'>"
            f"<span style='font-size:16px;'>🔒</span>"
            f"<span style='color:#c9d1d9;font-size:12px;'>"
            f"Загружена сохранённая сессия обучения — параметры заблокированы.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("✏️ Изменить параметры",
                     help="Разблокировать редактирование гиперпараметров"):
            st.session_state._params_locked = False
            st.rerun()

    arch1, arch2, arch3 = st.columns(3)

    with arch1:
        st.markdown(
            f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 8px 0;'>"
            "Архитектура</p>", unsafe_allow_html=True)
        hidden_size = st.select_slider(
            "Hidden size", options=[32, 64, 128, 256],
            value=cfg["hidden_size"],
            help="Размер скрытого состояния LSTM и VSN.",
            disabled=params_locked,
        )
        attn_heads = st.select_slider(
            "Attention heads", options=[1, 2, 4, 8],
            value=cfg["attention_head_size"],
            help="hidden_size должен делиться на heads.",
            disabled=params_locked,
        )
        hidden_cont = st.number_input(
            "Hidden continuous size",
            min_value=8, max_value=256, step=8,
            value=cfg["hidden_continuous_size"],
            help="Рекомендуется ≤ hidden_size / 2.",
            disabled=params_locked,
        )

    with arch2:
        st.markdown(
            f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 8px 0;'>"
            "Регуляризация и оптимизация</p>", unsafe_allow_html=True)
        lr = st.number_input(
            "Learning rate",
            min_value=1e-5, max_value=1e-2, step=1e-5,
            value=float(cfg["learning_rate"]), format="%.5f",
            disabled=params_locked,
        )
        dropout = st.slider(
            "Dropout", min_value=0.0, max_value=0.5, step=0.05,
            value=float(cfg["dropout"]),
            disabled=params_locked,
        )
        grad_clip = st.slider(
            "Gradient clip", min_value=0.1, max_value=5.0, step=0.1,
            value=float(cfg["gradient_clip"]),
            disabled=params_locked,
        )

    with arch3:
        st.markdown(
            f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 8px 0;'>"
            "Расписание обучения</p>", unsafe_allow_html=True)
        batch_size = st.select_slider(
            "Batch size", options=[16, 32, 64, 128],
            value=cfg["batch_size"],
            help="CPU: 32, GPU: 64.",
            disabled=params_locked,
        )
        max_epochs = st.number_input(
            "Max epochs",
            min_value=1, max_value=300, step=5,
            value=int(cfg["max_epochs"]),
            disabled=params_locked,
        )
        patience = st.number_input(
            "Early stopping patience",
            min_value=1, max_value=50, step=1,
            value=int(cfg["patience"]),
            disabled=params_locked,
        )
        encoder_length = st.number_input(
            "Encoder length (шагов истории)",
            min_value=7, max_value=365, step=7,
            value=int(cfg.get("encoder_length", 30)),
            help="Сколько прошлых шагов видит модель.",
            disabled=params_locked,
        )
        prediction_length = st.number_input(
            "Prediction length (горизонт прогноза)",
            min_value=1, max_value=90, step=1,
            value=int(cfg.get("prediction_length", 7)),
            help="Сколько шагов вперёд предсказывает модель.",
            disabled=params_locked,
        )

    if not params_locked:
        if hidden_cont > hidden_size // 2:
            st.warning(
                f"⚠️ Hidden continuous ({hidden_cont}) > hidden_size/2 ({hidden_size // 2}). "
                "Рекомендация TFT: hidden_continuous ≤ hidden_size / 2."
            )
        if hidden_size % attn_heads != 0:
            st.error(
                f"❌ hidden_size ({hidden_size}) не делится на attention_heads ({attn_heads})."
            )

    new_cfg = {
        "batch_size":             batch_size,
        "max_epochs":             max_epochs,
        "hidden_size":            hidden_size,
        "attention_head_size":    attn_heads,
        "hidden_continuous_size": hidden_cont,
        "learning_rate":          lr,
        "dropout":                dropout,
        "gradient_clip":          grad_clip,
        "patience":               patience,
        "encoder_length":         encoder_length,
        "prediction_length":      prediction_length,
    }

    # ── Кнопки действий ──────────────────────────────────────────────────────
    section_header("💾", "СЕССИЯ ОБУЧЕНИЯ", color=GRAY)
    _b1, _b2, _b3, _b4, _b5 = st.columns(5)

    with _b1:
        if st.button("💾 Сохранить сессию", type="primary", width="stretch",
                     disabled=params_locked):
            save_train_config(sel_session, new_cfg)
            if not split_cfg:
                _te = st.session_state.get("_split_train_end", "").strip()
                _ve = st.session_state.get("_split_val_end", "").strip()
                if _te and _ve:
                    save_split_config_to_training(sel_session, _te, _ve)
            st.session_state._params_locked = True
            st.toast("Сессия обучения сохранена!", icon="✅")
            st.rerun()

    with _b2:
        if config_exists:
            if st.button("📂 Загрузить сессию", width="stretch"):
                saved = load_train_config(sel_session)
                st.session_state._preset_values = saved
                st.session_state._params_locked = True
                st.toast("Сессия обучения загружена!", icon="📂")
                st.rerun()
        else:
            st.button("📂 Загрузить сессию", disabled=True, width="stretch",
                      help="Нет сохранённой сессии")

    with _b3:
        if st.button("🖥️ CPU", width="stretch",
                     disabled=params_locked, help="Пресет по умолчанию (CPU)"):
            st.session_state._preset_values = dict(CPU_DEFAULTS)
            st.rerun()

    with _b4:
        if st.button("🎮 GPU", width="stretch",
                     disabled=params_locked, help="Пресет GPU"):
            st.session_state._preset_values = dict(GPU_DEFAULTS)
            st.rerun()

    with _b5:
        if config_exists:
            if st.button("🗑️ Удалить", width="stretch"):
                delete_train_config(sel_session)
                st.session_state._params_locked = None
                st.rerun()
        else:
            st.markdown(
                f"<div style='color:{GOLD};font-size:12px;font-weight:600;"
                f"padding:8px 0;'>⚠️ Не сохранено</div>"
                f"<div style='color:{GRAY};font-size:11px;margin-top:-4px;'>"
                "используются авто-дефолты</div>",
                unsafe_allow_html=True,
            )

    with st.expander("👁 Превью конфига (JSON)", expanded=False):
        st.json(new_cfg)

    _cfg_path = get_train_config_path(sel_session)
    if config_exists:
        _status = "🔒 Загружена (заблокировано)" if params_locked else "✏️ Редактируется"
        st.caption(f"📁 Активная сессия: `{_cfg_path}` · {_status}")
    else:
        st.caption("⚠️ Сессия не сохранена — нажмите «Сохранить сессию» перед запуском.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Обучение
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    drain_queue()

    training_active = is_training()
    has_output      = bool(st.session_state.train_output)
    proc            = st.session_state.train_proc
    exit_code       = proc.returncode if (proc and not training_active) else None
    start_t         = st.session_state.train_start_time
    end_t           = st.session_state.train_end_time
    elapsed         = elapsed_str(start_t, end_t if not training_active else None)

    # ── Проверка готовности ───────────────────────────────────────────────────
    section_header("🔍", "ГОТОВНОСТЬ К ЗАПУСКУ")
    _train_dir_exists = os.path.exists(get_training_dir(sel_session))
    _script_ok        = os.path.exists(TRAIN_PY)

    # Script check — hard error if missing
    _script_icon = "✅" if _script_ok else "❌"
    _script_col  = GREEN if _script_ok else RED
    st.markdown(
        f'<span style="color:{_script_col};">{_script_icon}</span> '
        f'<span style="color:#c9d1d9;font-size:13px;">'
        f'Скрипт обучения найден (<code>tft/train.py</code>)</span>',
        unsafe_allow_html=True,
    )

    # Config check — ℹ️ if missing (auto-saved on launch, not a blocker)
    if config_exists:
        st.markdown(
            f'<span style="color:{GREEN};">✅</span> '
            f'<span style="color:#c9d1d9;font-size:13px;">Конфиг обучения сохранён '
            f'<span style="color:{GRAY};">(training/{sel_session}/train_config.json)</span></span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:{GOLD};">ℹ️</span> '
            f'<span style="color:#c9d1d9;font-size:13px;">Конфиг не сохранён — '
            f'<span style="color:{GRAY};">автоматически сохранится при запуске</span></span>',
            unsafe_allow_html=True,
        )

    # Folder check — ℹ️ if missing (auto-created on launch)
    if _train_dir_exists:
        st.markdown(
            f'<span style="color:{GREEN};">✅</span> '
            f'<span style="color:#c9d1d9;font-size:13px;">Папка обучения для сессии существует '
            f'<span style="color:{GRAY};">(training/{sel_session}/)</span></span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:{GOLD};">ℹ️</span> '
            f'<span style="color:#c9d1d9;font-size:13px;">Папка обучения не создана — '
            f'<span style="color:{GRAY};">автоматически создастся при запуске</span></span>',
            unsafe_allow_html=True,
        )

    # Split config check — required for training
    _split_in_export   = os.path.exists(
        os.path.join(get_export_path(sel_session), "split_config.json"))
    _split_in_training = os.path.exists(
        os.path.join(get_training_dir(sel_session), "split_config.json"))
    _split_ok = _split_in_export or _split_in_training
    if _split_ok:
        _split_src = (f"exports/{sel_session}" if _split_in_export
                      else f"training/{sel_session}")
        st.markdown(
            f'<span style="color:{GREEN};">✅</span> '
            f'<span style="color:#c9d1d9;font-size:13px;">'
            f'Конфиг сплита найден '
            f'<span style="color:{GRAY};">({_split_src}/split_config.json)</span></span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:{RED};">❌</span> '
            f'<span style="color:#c9d1d9;font-size:13px;">'
            f'split_config.json не найден — '
            f'<span style="color:{GOLD};">настройте сплит на вкладке «Конфигурация»</span></span>',
            unsafe_allow_html=True,
        )

    if not _script_ok:
        st.warning("⚠️ Файл `tft/train.py` не найден — запуск обучения невозможен.")
    if not _split_ok:
        st.warning("⚠️ Сплит не настроен — перейдите на вкладку «Конфигурация» и задайте границы train/val/test.")

    st.divider()

    # ── Управление ───────────────────────────────────────────────────────────
    section_header("▶", "УПРАВЛЕНИЕ ОБУЧЕНИЕМ")
    _done_marker    = os.path.join(get_training_dir(sel_session), "training_complete.flag")
    _session_done   = st.session_state._training_done or os.path.exists(_done_marker)

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 6])

    with ctrl1:
        can_start = _script_ok

        _ckpt_dir_t2 = get_ckpt_dir(sel_session)
        _ckpts_t2    = sorted(
            __import__("glob").glob(os.path.join(_ckpt_dir_t2, "*.ckpt"))
        )
        _resume_ckpt = _ckpts_t2[-1] if _ckpts_t2 else None
        _has_ckpt    = bool(_resume_ckpt)

        # Единый контейнер — при каждом рендере содержимое полностью заменяется
        _ctrl1 = st.empty()
        with _ctrl1.container():
            if training_active:
                st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
                if st.button("⏹ Остановить", type="secondary",
                             use_container_width=True, key="btn_main_train_action"):
                    stop_training()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            elif _session_done:
                st.markdown(
                    f"<div style='padding:8px 12px;background:#0a1a0f;"
                    f"border:1px solid {GREEN}40;border-radius:8px;"
                    f"font-size:12px;color:{GREEN};font-weight:600;'>"
                    "✅ Обучение завершено</div>"
                    f"<div style='margin-top:6px;font-size:11px;color:{GRAY};'>"
                    "Удалите чекпоинты в разделе «Результаты», чтобы запустить заново.</div>",
                    unsafe_allow_html=True,
                )
            elif has_output:
                # Прерванный / ошибка — можно продолжить
                _lbl2 = ("▶️ Продолжить обучение" if _has_ckpt
                         else "▶️ Запустить обучение заново")
                if st.button(_lbl2, type="primary", use_container_width=True,
                             disabled=not can_start, key="btn_main_train_action"):
                    save_train_config(sel_session, new_cfg)
                    start_training(sel_session)
                    st.rerun()
            else:
                # Начальное состояние — только одна кнопка
                _lbl1 = ("▶️ Продолжить обучение" if _has_ckpt
                         else "▶️ Запустить обучение")
                if st.button(_lbl1, type="primary", use_container_width=True,
                             disabled=not can_start, key="btn_main_train_action"):
                    save_train_config(sel_session, new_cfg)
                    start_training(sel_session)
                    st.rerun()

    with ctrl2:
        if training_active:
            st.metric("Статус", "⚡ Обучение", f"прошло {elapsed}")
        elif _session_done:
            _done_elapsed = elapsed if has_output else "—"
            st.metric("Статус", "✅ Завершено", f"за {_done_elapsed}")
        elif has_output:
            code_str = f"код {exit_code}" if exit_code is not None else "остановлено"
            st.metric("Статус", "⛔ Прервано", code_str)
        else:
            st.metric("Статус", "💤 Ожидание", "")

    # ── Вывод процесса ────────────────────────────────────────────────────────
    section_header("📟", "ВЫВОД ПРОЦЕССА",
                   subtitle=f"{len(st.session_state.train_output)} строк" if has_output else "")

    if has_output:
        _raw_total  = len(st.session_state.train_output)
        _full_log   = st.toggle("Полный лог", value=False, key="train_full_log")

        if _full_log:
            _show_lines = st.session_state.train_output[-2000:]
            import re as _re2
            _ansi_strip = _re2.compile(r'\x1b\[[0-9;]*m')
            _raw_html = "<br>".join(
                _re2.sub(r'<', '&lt;', _re2.sub(r'&', '&amp;',
                    _ansi_strip.sub('', l).replace('\r', '')))
                for l in _show_lines
            )
            _caption = f"Показаны последние {len(_show_lines)} из {_raw_total} сырых строк."
        else:
            display_lines = st.session_state.train_output[-300:]
            _raw_html, _deduped_count = render_output_html(display_lines)
            _caption = (
                f"Показаны последние {_deduped_count} строк"
                f" (из {_raw_total} сырых, после дедупликации)."
            ) if _raw_total > 300 else None

        st.markdown(f"""
<div style="background:#0a0e17;border:1px solid {GRID};border-radius:6px;
            padding:12px 16px;height:440px;overflow-y:auto;
            font-family:'Consolas','Courier New',monospace;
            font-size:11px;line-height:1.6;white-space:pre-wrap;"
     id="train-out">
{_raw_html}
</div>
<script>
const el = document.getElementById('train-out');
if(el) el.scrollTop = el.scrollHeight;
</script>""", unsafe_allow_html=True)
        if _caption:
            st.caption(_caption)
    else:
        st.markdown(
            f"<div style='color:{GRAY};font-size:12px;padding:40px 0;text-align:center;'>"
            "Нажмите «Запустить обучение» — вывод появится здесь.</div>",
            unsafe_allow_html=True,
        )

    if training_active:
        time.sleep(2)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Результаты
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    section_header("📈", "СОСТОЯНИЕ МОДЕЛИ")

    import datetime as _dt

    model_path  = get_model_path(sel_session)
    ckpt_dir    = get_ckpt_dir(sel_session)
    ckpt_files  = sorted(
        __import__("glob").glob(os.path.join(ckpt_dir, "*.ckpt"))
    ) if os.path.exists(ckpt_dir) else []
    tb_versions = list_tb_versions(sel_session)

    def _kpi(label, value, sub="", sub_color=GRAY):
        # always render subtitle line so all cards have identical height
        sub_html = (f'<div style="font-size:11px;color:{sub_color};margin-top:3px;">'
                    f'{sub if sub else "&nbsp;"}</div>')
        return (
            f'<div style="background:{CARD_BG};border:1px solid {GRID};'
            f'border-radius:8px;padding:12px 16px;height:100%;">'
            f'<div style="font-size:11px;color:{GRAY};margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:15px;font-weight:700;color:#c9d1d9;'
            f'word-break:break-word;line-height:1.35;">{value}</div>'
            f'{sub_html}</div>'
        )

    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)

    with kpi_c1:
        # model.ckpt preferred; fall back to latest .ckpt if not synced yet
        _best_src = None
        if os.path.exists(model_path):
            _best_src = model_path
        elif ckpt_files:
            _best_src = ckpt_files[-1]

        if _best_src:
            _ts      = _dt.datetime.fromtimestamp(
                os.path.getmtime(_best_src)).strftime("%d.%m.%Y %H:%M")
            _size_mb = os.path.getsize(_best_src) / 1e6
            _label   = "model.ckpt" if _best_src == model_path else "последний .ckpt"
            st.markdown(
                _kpi("Лучшая модель", f"{_size_mb:.1f} МБ",
                     f"↑ {_label} · {_ts}", GREEN),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _kpi("Лучшая модель", "нет", "запустите обучение"),
                unsafe_allow_html=True,
            )

    with kpi_c2:
        if ckpt_files:
            _ckpt_name = os.path.basename(ckpt_files[-1])
            st.markdown(
                _kpi("Последний чекпоинт", _ckpt_name,
                     f"↑ {len(ckpt_files)} файл(ов) в папке", GRAY),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _kpi("Последний чекпоинт", "нет"),
                unsafe_allow_html=True,
            )

    with kpi_c3:
        _tb_logdir = os.path.relpath(
            get_training_dir(sel_session) + "/logs", ROOT)
        st.markdown(
            _kpi("TensorBoard версии", str(len(tb_versions)),
                 f"tensorboard --logdir {_tb_logdir}", GRAY),
            unsafe_allow_html=True,
        )

    # ── Кривые потерь ─────────────────────────────────────────────────────────
    section_header("📉", "КРИВЫЕ ПОТЕРЬ", subtitle="из TensorBoard логов")

    try:
        all_versions = read_tb_losses(sel_session)
    except Exception as _e:
        all_versions = []
        st.caption(f"Ошибка чтения логов: {_e}")

    if all_versions:
        import pandas as _pd

        fig = go.Figure()

        def _epochs(df):
            """Replace step numbers with epoch indices 1, 2, 3..."""
            if df is None or len(df) == 0:
                return df
            import pandas as _pd2
            return _pd2.DataFrame({
                "step":      list(range(1, len(df) + 1)),
                "value":     df["value"].tolist(),
                "wall_time": df["wall_time"].tolist() if "wall_time" in df.columns else [None]*len(df),
            })

        # Склеиваем все версии в единый ряд по эпохам
        import pandas as _pd
        _all_tr_vals, _all_vl_vals = [], []
        for _, td, vd in all_versions:
            if td is not None and len(td) > 0:
                _all_tr_vals.extend(td["value"].tolist())
            if vd is not None and len(vd) > 0:
                _all_vl_vals.extend(vd["value"].tolist())

        if _all_tr_vals:
            _tr_x = list(range(1, len(_all_tr_vals) + 1))
            _single_tr = len(_all_tr_vals) == 1
            fig.add_trace(go.Scatter(
                x=_tr_x, y=_all_tr_vals,
                name="Обучающая выборка",
                mode="markers" if _single_tr else "lines+markers",
                line=dict(color=GOLD, width=2.5),
                marker=dict(size=8 if _single_tr else 5, color=GOLD),
            ))

        if _all_vl_vals:
            _vl_x = list(range(1, len(_all_vl_vals) + 1))
            _single_vl = len(_all_vl_vals) == 1
            fig.add_trace(go.Scatter(
                x=_vl_x, y=_all_vl_vals,
                name="Валидационная выборка",
                mode="markers" if _single_vl else "lines+markers",
                line=dict(color=GREEN, width=2),
                marker=dict(size=10 if _single_vl else 6, color=GREEN),
            ))
            if not _single_vl:
                _best_idx = int(_pd.Series(_all_vl_vals).idxmin())
                fig.add_trace(go.Scatter(
                    x=[_vl_x[_best_idx]], y=[_all_vl_vals[_best_idx]],
                    mode="markers+text",
                    name=f"Лучший результат: {_all_vl_vals[_best_idx]:.4f}",
                    marker=dict(symbol="star", size=16, color=GREEN),
                    text=[f"  {_all_vl_vals[_best_idx]:.4f}"],
                    textfont=dict(color=GREEN, size=11),
                    textposition="middle right",
                ))

        fig.update_layout(
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color="#c9d1d9", size=11),
            xaxis=dict(title="Эпоха", gridcolor=GRID,
                       zerolinecolor=GRID, tickformat=",d", dtick=1),
            yaxis=dict(title="Потери (QuantileLoss)", gridcolor=GRID, zerolinecolor=GRID),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=55, r=20, t=55, b=50),
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        # ── Таблица запусков ──────────────────────────────────────────────────
        section_header("📋", "МЕТРИКИ ЗАПУСКОВ", color=GRAY)
        _max_epochs = load_train_config(sel_session).get("max_epochs", None)
        _done_marker_t3 = os.path.join(
            get_training_dir(sel_session), "training_complete.flag")
        rows = []
        for i, (label, train_df, val_df) in enumerate(all_versions):
            n_tr = len(train_df) if train_df is not None else 0
            n_vl = len(val_df)   if val_df   is not None else 0
            dur  = None
            if train_df is not None and n_tr >= 2 and "wall_time" in train_df.columns:
                dur = train_df["wall_time"].iloc[-1] - train_df["wall_time"].iloc[0]
            elif val_df is not None and n_vl >= 2 and "wall_time" in val_df.columns:
                dur = val_df["wall_time"].iloc[-1] - val_df["wall_time"].iloc[0]

            is_latest = (i == len(all_versions) - 1)
            if is_latest:
                if os.path.exists(_done_marker_t3):
                    status = "✅ Завершено"
                elif n_vl == 0:
                    status = "⚠️ Прервано"
                else:
                    status = "⏹ Остановлено"
            else:
                if _max_epochs and n_vl >= _max_epochs:
                    status = "✅ Завершено"
                elif n_vl == 0:
                    status = "⚠️ Прервано"
                else:
                    status = "⏹ Остановлено"

            rows.append({
                "Запуск":          label,
                "Статус":          status,
                "Время":           fmt_duration(dur),
                "Val эпох":        str(n_vl) if n_vl else "—",
                "Last train loss": f"{train_df['value'].iloc[-1]:.4f}" if n_tr else "—",
                "Best val loss":   f"{val_df['value'].min():.4f}" if n_vl else "—",
            })
        stats_df = _pd.DataFrame(rows)

        def _style(row):
            if "(текущий)" in str(row["Запуск"]):
                return [f"color:{GOLD};font-weight:700"] * len(row)
            return ["color:#c9d1d9"] * len(row)

        st.dataframe(
            stats_df.style.apply(_style, axis=1),
            hide_index=True, width="stretch",
        )
    else:
        st.info(
            "📭 Данные обучения не найдены. Запустите обучение — "
            "кривые потерь появятся здесь автоматически."
        )

    # ── Удаление версий логов ─────────────────────────────────────────────────
    if tb_versions:
        section_header("🗑️", "УПРАВЛЕНИЕ ЛОГАМИ", color=GRAY)
        with st.expander(f"Удалить версии логов TensorBoard ({len(tb_versions)} доступно)"):
            st.markdown(
                f"<p style='color:{GRAY};font-size:12px;margin:0 0 10px 0;'>"
                f"Операция <b style='color:{RED};'>необратима</b>.</p>",
                unsafe_allow_html=True,
            )
            _to_del = []
            _cols = 3
            _rows = [tb_versions[i:i+_cols] for i in range(0, len(tb_versions), _cols)]
            for _row in _rows:
                _vcols = st.columns(_cols)
                for _ci, _v in enumerate(_row):
                    with _vcols[_ci]:
                        _vn = os.path.basename(_v)
                        if st.checkbox(_vn, key=f"del_{sel_session}_{_vn}"):
                            _to_del.append(_v)
            if _to_del:
                st.warning(f"⚠️ Будет удалено: {', '.join(os.path.basename(v) for v in _to_del)}")
                if st.checkbox("✔ Подтверждаю удаление", key=f"confirm_del_{sel_session}"):
                    if st.button("🗑️ Удалить выбранные", type="primary",
                                 key=f"btn_del_{sel_session}"):
                        for _v in _to_del:
                            shutil.rmtree(_v, ignore_errors=True)
                        st.toast(f"Удалено {len(_to_del)} версий", icon="🗑️")
                        st.rerun()

    # ── Активный конфиг ───────────────────────────────────────────────────────
    section_header("⚙️", "АКТИВНЫЙ КОНФИГ ОБУЧЕНИЯ", color=GRAY)
    cur_cfg = load_train_config(sel_session)
    _cc1, _cc2 = st.columns(2)
    with _cc1:
        st.markdown(
            f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 6px 0;'>"
            "Архитектура</p>", unsafe_allow_html=True)
        for _lbl, _val in [
            ("Hidden size",        cur_cfg["hidden_size"]),
            ("Attention heads",    cur_cfg["attention_head_size"]),
            ("Hidden continuous",  cur_cfg["hidden_continuous_size"]),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:3px 0;"
                f"border-bottom:1px solid {GRID};font-size:12px;'>"
                f"<span style='color:{GRAY};'>{_lbl}</span>"
                f"<span style='color:#c9d1d9;font-weight:600;'>{_val}</span></div>",
                unsafe_allow_html=True,
            )
    with _cc2:
        st.markdown(
            f"<p style='color:{TEAL};font-size:12px;font-weight:600;margin:0 0 6px 0;'>"
            "Обучение</p>", unsafe_allow_html=True)
        for _lbl, _val in [
            ("Learning rate",  f"{cur_cfg['learning_rate']:.5f}"),
            ("Dropout",        f"{cur_cfg['dropout']:.2f}"),
            ("Gradient clip",  f"{cur_cfg['gradient_clip']:.1f}"),
            ("Batch size",     cur_cfg["batch_size"]),
            ("Max epochs",     cur_cfg["max_epochs"]),
            ("Patience",       cur_cfg["patience"]),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:3px 0;"
                f"border-bottom:1px solid {GRID};font-size:12px;'>"
                f"<span style='color:{GRAY};'>{_lbl}</span>"
                f"<span style='color:#c9d1d9;font-weight:600;'>{_val}</span></div>",
                unsafe_allow_html=True,
            )

    if is_training():
        st.markdown(
            f"<p style='color:{TEAL};font-size:11px;margin-top:8px;'>"
            "⚡ Обучение идёт — страница обновляется каждые 10 сек</p>",
            unsafe_allow_html=True,
        )
        time.sleep(10)
        st.rerun()
    else:
        if st.button("🔄 Обновить результаты", width="content"):
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Прогнозы
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    drain_pred_queue()

    pred_active  = is_predicting()
    pred_has_out = bool(st.session_state.pred_output)
    pred_proc    = st.session_state.pred_proc
    pred_exit    = pred_proc.returncode if (pred_proc and not pred_active) else None
    pred_start_t = st.session_state.pred_start_time
    pred_end_t   = st.session_state.pred_end_time
    pred_elapsed = elapsed_str(pred_start_t, pred_end_t if not pred_active else None)

    # ── Чекпоинт ─────────────────────────────────────────────────────────────
    section_header("🔖", "ВЫБОР ЧЕКПОИНТА")
    _ckpts = list_ckpt_files(sel_session)

    if not _ckpts:
        st.warning(
            "⚠️ Нет доступных чекпоинтов. "
            "Запустите обучение — чекпоинты появятся здесь."
        )
        _ckpt_disabled = True
        _sel_ckpt_path = None
    else:
        _ckpt_disabled = False
        _ckpt_labels   = [c["label"] for c in _ckpts]
        _sel_ckpt_label = st.selectbox(
            "Чекпоинт",
            options=_ckpt_labels,
            index=0,
            key=f"sel_ckpt_{sel_session}",
            help="Выберите чекпоинт для генерации предсказаний",
            disabled=pred_active,
        )
        _sel_ckpt_path = next(
            (c["path"] for c in _ckpts if c["label"] == _sel_ckpt_label), None
        )
        if _sel_ckpt_path:
            _ckpt_size = os.path.getsize(_sel_ckpt_path) / 1e6 if os.path.exists(_sel_ckpt_path) else 0
            st.caption(f"📄 `{_sel_ckpt_path}` · {_ckpt_size:.1f} МБ")

    # ── Параметры вывода ─────────────────────────────────────────────────────
    section_header("📁", "ПАРАМЕТРЫ ВЫВОДА")
    _pred_c1, _pred_c2 = st.columns([1, 2])

    with _pred_c1:
        _pred_split = st.radio(
            "Период прогноза",
            options=["test", "val"],
            format_func=lambda x: "🧪 Тест (после val_end)" if x == "test" else "📊 Валидация",
            horizontal=False,
            disabled=pred_active,
            key=f"pred_split_{sel_session}",
        )

    with _pred_c2:
        _pred_dir = get_predictions_dir(sel_session)
        import datetime as _dt
        if _sel_ckpt_path and os.path.exists(_sel_ckpt_path):
            _ts = _dt.datetime.fromtimestamp(
                os.path.getmtime(_sel_ckpt_path)
            ).strftime("%Y%m%d_%H%M%S")
        else:
            _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        _ckpt_tag = (
            os.path.splitext(os.path.basename(_sel_ckpt_path or "best"))[0]
            .replace("model", "best")
            .replace(" ", "_")
            [:20]
        )
        _default_fname = f"{_ts}_{_pred_split}_{_ckpt_tag}.csv"
        _pred_fname = st.text_input(
            "Имя файла предсказаний",
            value=_default_fname,
            disabled=pred_active,
            key=f"pred_fname_{sel_session}_{_pred_split}",
            help=f"Сохранится в {_pred_dir}\\",
        )
        _pred_out_file = os.path.join(_pred_dir, _pred_fname)
        st.caption(f"💾 Полный путь: `{_pred_out_file}`")

    # ── Управление ───────────────────────────────────────────────────────────
    section_header("🚀", "ЗАПУСК ПРОГНОЗИРОВАНИЯ")
    _pctrl1, _pctrl2, _pctrl3 = st.columns([2, 2, 6])

    with _pctrl1:
        if pred_active:
            if st.button("⏹ Остановить", type="secondary", width="stretch"):
                stop_prediction()
                st.rerun()
        else:
            _can_predict = bool(_sel_ckpt_path) and os.path.exists(PREDICT_PY)
            if st.button(
                "🔮 Сформировать прогноз",
                type="primary",
                width="stretch",
                disabled=not _can_predict or _ckpt_disabled,
            ):
                os.makedirs(_pred_dir, exist_ok=True)
                start_prediction(sel_session, _sel_ckpt_path, _pred_out_file, _pred_split)
                st.rerun()

    with _pctrl2:
        if pred_active:
            st.metric("Статус", "⚡ Выполняется", f"прошло {pred_elapsed}")
        elif pred_has_out:
            if pred_exit == 0:
                st.metric("Статус", "✅ Готово", f"за {pred_elapsed}")
            else:
                _code_str = f"код {pred_exit}" if pred_exit is not None else "остановлено"
                st.metric("Статус", "⛔ Ошибка", _code_str)
        else:
            st.metric("Статус", "💤 Ожидание", "")

        if not os.path.exists(PREDICT_PY):
            st.warning(
                f"⚠️ `tft/predict.py` не реализован. "
                "Откройте файл и заполните раздел '# TODO'."
            )

    # ── Вывод процесса ────────────────────────────────────────────────────────
    section_header("📟", "ВЫВОД",
                   subtitle=f"{len(st.session_state.pred_output)} строк" if pred_has_out else "")

    if pred_has_out:
        _pred_lines = st.session_state.pred_output[-200:]
        _pred_html, _ = render_output_html(_pred_lines)
        st.markdown(f"""
<div style="background:#0a0e17;border:1px solid {GRID};border-radius:6px;
            padding:12px 16px;height:320px;overflow-y:auto;
            font-family:'Consolas','Courier New',monospace;
            font-size:11px;line-height:1.6;white-space:pre-wrap;"
     id="pred-out">
{_pred_html}
</div>
<script>
const el = document.getElementById('pred-out');
if(el) el.scrollTop = el.scrollHeight;
</script>""", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='color:{GRAY};font-size:12px;padding:30px 0;text-align:center;'>"
            "Выберите чекпоинт и нажмите «Сформировать прогноз».</div>",
            unsafe_allow_html=True,
        )

    # ── Сохранённые файлы прогнозов ───────────────────────────────────────────
    section_header("📂", "ФАЙЛЫ ПРОГНОЗОВ", color=GRAY)
    _pred_files = sorted(
        __import__("glob").glob(os.path.join(_pred_dir, "*.csv")),
        key=os.path.getmtime,
        reverse=True,
    ) if os.path.isdir(_pred_dir) else []

    if _pred_files:
        import datetime as _dt
        _last_file = st.session_state.get("_last_pred_file")
        for _pf in _pred_files[:10]:
            _pf_size = os.path.getsize(_pf) / 1024
            _pf_time = _dt.datetime.fromtimestamp(os.path.getmtime(_pf)).strftime("%d.%m.%Y %H:%M")
            _is_last = _last_file and os.path.normpath(_pf) == os.path.normpath(_last_file)
            _icon       = "✅" if _is_last else "📄"
            _name_color = GREEN if _is_last else "#c9d1d9"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid {GRID};font-size:12px;'>"
                f"<span style='color:{_name_color};'>"
                f"{_icon} {os.path.basename(_pf)}</span>"
                f"<span style='color:{GRAY};'>{_pf_size:.1f} КБ · {_pf_time}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if len(_pred_files) > 10:
            st.caption(f"...и ещё {len(_pred_files) - 10} файл(ов) в `{_pred_dir}`")
    else:
        st.markdown(
            f"<div style='color:{GRAY};font-size:12px;'>"
            f"Нет сохранённых предсказаний. Папка: `{_pred_dir}`</div>",
            unsafe_allow_html=True,
        )

    if pred_active:
        time.sleep(2)
        st.rerun()
