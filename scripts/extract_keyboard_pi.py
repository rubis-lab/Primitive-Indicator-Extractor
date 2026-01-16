"""
Keyboard Primitive Indicator Extractor

Input directory layout example:
  /data_248/pdss/hospital_data/YYYYMMDD/<client_id>/<task>/<sensor>/<uuid>/keyboard.csv

Output directory:
  <output_base_path>/<client_id>/<task>/keyboard_pi.json

Notes:
- If multiple keyboard.csv exist for the same (client_id, task), this path will be overwritten.
  If you need to avoid overwrites, include date/uuid in the filename or create deeper dirs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# Parameters
# =========================

# Screen size (default)
SCREEN_WIDTH_PX = 1440
SCREEN_HEIGHT_PX = 900

# Windowing parameters
WINDOW_SIZE_SEC = 60.0  # default: 60s
STRIDE_SEC = 30.0       # default: 30s

# Keyboard PI-specific
PAUSE_THRESHOLD_MS = 2000.0  # θ_pause
NGRAM_N = 2                  # n for n-gram features

# Output filename
OUTPUT_FILENAME = "keyboard_pi.json"

# =========================
# Indicator list
# =========================

BASE_INDICATOR_NAMES = [
    "Total Keystroke Count",
    "Character Keystroke Count",
    "Non-character Keystroke Count",
    "First Keystroke Latency",
    "Typing Duration",
    "Typing Velocity",
    "Down-Down Interval",
    "Up-Down Interval",
    "Up-Up Interval",
    "Key Overlap Duration",
    "Key Overlap Ratio",
    "Ngram Typing Duration",
    "Inter-Word Interval",
    "Pause Duration",
    "Pause Ratio",
    "Burst Length",
    "Burst Duration",
    "Burst Velocity",
    "Key Hold Duration",
    "Alphabetic Key Hold Duration",
    "Alphabetic Key Ratio",
    "Spacebar Key Hold Duration",
    "Spacebar Key Ratio",
    "Backspace Key Hold Duration",
    "Backspace Key Ratio",
    "Delete Key Hold Duration",
    "Delete Key Ratio",
    "Enter Key Hold Duration",
    "Enter Key Ratio",
    "CapsLock Key Hold Duration",
    "CapsLock Key Ratio",
    "Shift Key Hold Duration",
    "Shift Key Ratio",
    "Control Key Hold Duration",
    "Control Key Ratio",
    "Alt Key Hold Duration",
    "Alt Key Ratio",
    "Tab Key Hold Duration",
    "Tab Key Ratio",
    "Escape Key Hold Duration",
    "Escape Key Ratio",
    "Insert Key Hold Duration",
    "Insert Key Ratio",
    "Punctuation Key Hold Duration",
    "Punctuation Key Ratio",
    "Digit Key Hold Duration",
    "Digit Key Ratio",
    "Cursor Move Key Duration",
    "Cursor Move Key Ratio",
    "Undo Duration",
    "Undo Ratio",
    "Paste Duration",
    "Paste Ratio",
    "Cut Duration",
    "Cut Ratio",
    "Copy Duration",
    "Copy Ratio",
    "Key Category Entropy",
    "Ngram Distribution Entropy",
    "Key Sequence Levenshtein Distance",
    "Ngram Distribution Similarity",
    "ROUGE-L Similarity",
]
STATS = ("mean", "std", "min", "max")


def to_snake(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("rouge-l", "rouge_l")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


SNAKE_BASES = [to_snake(n) for n in BASE_INDICATOR_NAMES]

# =========================
# Key normalization / categories
# =========================

CTRL_SET = {"CTRL", "CONTROL", "CONTROLLEFT", "CONTROLRIGHT", "CTRLL", "CTRLR"}
SHIFT_SET = {"SHIFT", "SHIFTLEFT", "SHIFTRIGHT"}
ALT_SET = {"ALT", "ALTLEFT", "ALTRIGHT", "OPTION"}
CAPS_SET = {"CAPSLOCK", "CAPS_LOCK"}
ENTER_SET = {"ENTER", "RETURN"}
BACKSPACE_SET = {"BACKSPACE"}
DELETE_SET = {"DELETE", "DEL"}
TAB_SET = {"TAB"}
ESC_SET = {"ESC", "ESCAPE"}
INSERT_SET = {"INSERT", "INS"}

CURSOR_SET = {
    "ARROWLEFT", "ARROWRIGHT", "ARROWUP", "ARROWDOWN",
    "LEFT", "RIGHT", "UP", "DOWN",
    "HOME", "END",
    "PAGEUP", "PAGEDOWN",
}

SPACE_TOKENS = {"SPACE", " "}


def normalize_token(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    if len(s) == 1:
        if s.isalpha():
            return s.upper()
        return s
    return s.upper()


def is_alpha_token(tok: str) -> bool:
    return len(tok) == 1 and tok.isalpha()


def is_digit_token(tok: str) -> bool:
    return len(tok) == 1 and tok.isdigit()


def is_punctuation_token(tok: str) -> bool:
    return len(tok) == 1 and (not tok.isalnum()) and (tok not in SPACE_TOKENS)


def key_category(tok: str) -> str:
    if tok in SPACE_TOKENS:
        return "space"
    if tok in BACKSPACE_SET:
        return "backspace"
    if tok in DELETE_SET:
        return "delete"
    if tok in ENTER_SET:
        return "enter"
    if tok in CAPS_SET:
        return "capslock"
    if tok in SHIFT_SET:
        return "shift"
    if tok in CTRL_SET:
        return "control"
    if tok in ALT_SET:
        return "alt"
    if tok in TAB_SET:
        return "tab"
    if tok in ESC_SET:
        return "escape"
    if tok in INSERT_SET:
        return "insert"
    if tok in CURSOR_SET:
        return "cursor"
    if is_alpha_token(tok):
        return "alphabetic"
    if is_digit_token(tok):
        return "digit"
    if is_punctuation_token(tok):
        return "punctuation"
    return "other"


# =========================
# Utilities
# =========================

def ms_from_timedelta(td: pd.Timedelta) -> float:
    return float(td.total_seconds() * 1000.0)


def safe_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if values is None or len(values) == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def empty_pi_dict() -> Dict[str, Optional[float]]:
    pi = {}
    for base in SNAKE_BASES:
        for st in STATS:
            pi[f"{base}_{st}"] = None
    return pi


def fill_pi_stats(pi: Dict[str, Optional[float]], base_snake: str, values: List[float]) -> None:
    st = safe_stats(values)
    for k in STATS:
        pi[f"{base_snake}_{k}"] = st[k]


@dataclass
class WindowParams:
    window_size_sec: float = WINDOW_SIZE_SEC
    stride_sec: float = STRIDE_SEC
    pause_threshold_ms: float = PAUSE_THRESHOLD_MS
    ngram_n: int = NGRAM_N


# =========================
# Hold-time pairing (split KEY_DOWN / KEY_UP)
# =========================

def _parse_event_timestamp(df: pd.DataFrame) -> pd.Series:
    for col in ["timeStamp", "downTime", "upTime"]:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series([pd.NaT] * len(df))


def pair_key_events_stack(
    df_raw: pd.DataFrame,
    down_label: str = "KEY_DOWN",
    up_label: str = "KEY_UP",
    action_col: str = "key_action",
    key_col: str = "letter",
    ts_col: str = "event_dt",
) -> pd.DataFrame:
    out_rows = []
    pending: Dict[str, List[pd.Timestamp]] = defaultdict(list)

    df = df_raw.sort_values(ts_col).reset_index(drop=True)
    for _, r in df.iterrows():
        act = str(r.get(action_col, "")).upper()
        key = normalize_token(r.get(key_col, ""))
        ts = r.get(ts_col)
        if pd.isna(ts) or key == "":
            continue

        if act == down_label:
            pending[key].append(ts)
        elif act == up_label:
            if pending[key]:
                down_ts = pending[key].pop()
                up_ts = ts
                hold_ms = ms_from_timedelta(up_ts - down_ts)
                if hold_ms >= 0:
                    out_rows.append((key, down_ts, up_ts, hold_ms))

    out = pd.DataFrame(out_rows, columns=["token", "down_dt", "up_dt", "hold_ms"])
    return out


# =========================
# Feature computation per window
# =========================

def compute_keyboard_pi_for_window(
    dfw: pd.DataFrame,
    w_start: pd.Timestamp,
    w_end: pd.Timestamp,
    params: WindowParams,
) -> Dict[str, Optional[float]]:
    pi = empty_pi_dict()
    if dfw is None or dfw.empty:
        return pi

    dfw = dfw.sort_values("down_dt").reset_index(drop=True)

    down = dfw["down_dt"]
    up = dfw["up_dt"]
    hold_ms = dfw["hold_ms"].astype(float).tolist()
    tokens = dfw["token"].tolist()
    cats = dfw["category"].tolist()

    N = int(len(dfw))
    N_pairs = max(0, N - 1)

    total_count = float(N)
    char_mask = [(c in {"alphabetic", "digit", "punctuation"}) for c in cats]
    char_count = float(sum(char_mask))
    nonchar_count = float(N) - char_count

    fill_pi_stats(pi, to_snake("Total Keystroke Count"), [total_count])
    fill_pi_stats(pi, to_snake("Character Keystroke Count"), [char_count])
    fill_pi_stats(pi, to_snake("Non-character Keystroke Count"), [nonchar_count])

    first_latency_ms = ms_from_timedelta(down.iloc[0] - w_start)
    fill_pi_stats(pi, to_snake("First Keystroke Latency"), [first_latency_ms])

    typing_duration_ms = ms_from_timedelta(up.iloc[-1] - down.iloc[0])
    fill_pi_stats(pi, to_snake("Typing Duration"), [typing_duration_ms])

    typing_span_s = max(typing_duration_ms / 1000.0, 1e-9)
    typing_velocity = float(N) / typing_span_s
    fill_pi_stats(pi, to_snake("Typing Velocity"), [typing_velocity])

    if N_pairs > 0:
        dd_ms = ((down.shift(-1) - down).iloc[:-1].dt.total_seconds() * 1000.0).tolist()
        ud_ms = ((down.shift(-1) - up).iloc[:-1].dt.total_seconds() * 1000.0).tolist()
        uu_ms = ((up.shift(-1) - up).iloc[:-1].dt.total_seconds() * 1000.0).tolist()

        overlap_ms = (np.maximum(
            0.0,
            (up.iloc[:-1].to_numpy() - down.iloc[1:].to_numpy())
            .astype("timedelta64[ns]").astype(np.int64) / 1e6
        )).tolist()

        fill_pi_stats(pi, to_snake("Down-Down Interval"), dd_ms)
        fill_pi_stats(pi, to_snake("Up-Down Interval"), ud_ms)
        fill_pi_stats(pi, to_snake("Up-Up Interval"), uu_ms)
        fill_pi_stats(pi, to_snake("Key Overlap Duration"), overlap_ms)

        overlap_ratio = float(sum(1 for i in range(N_pairs) if down.iloc[i + 1] < up.iloc[i])) / float(N_pairs)
        fill_pi_stats(pi, to_snake("Key Overlap Ratio"), [overlap_ratio])

        p_ms = np.array(ud_ms, dtype=float)
        pause_mask = p_ms >= float(params.pause_threshold_ms)

        pause_durations = p_ms[pause_mask].tolist()
        pause_sum_ms = float(p_ms[pause_mask].sum())

        fill_pi_stats(pi, to_snake("Pause Duration"), pause_durations)

        typing_span_ms = max(typing_duration_ms, 1e-9)
        pause_ratio = pause_sum_ms / typing_span_ms
        fill_pi_stats(pi, to_snake("Pause Ratio"), [pause_ratio])

        break_idxs = np.where(pause_mask)[0].tolist()
        bursts: List[Tuple[int, int]] = []
        start_idx = 0
        for i in break_idxs:
            end_idx = i
            if end_idx >= start_idx:
                bursts.append((start_idx, end_idx))
            start_idx = i + 1
        if start_idx <= N - 1:
            bursts.append((start_idx, N - 1))

        burst_lengths = [float(e - s + 1) for s, e in bursts]
        burst_durations_ms = [ms_from_timedelta(down.iloc[e] - down.iloc[s]) for s, e in bursts]
        burst_velocities = []
        for bl, bd in zip(burst_lengths, burst_durations_ms):
            if bd <= 0.0:
                burst_velocities.append(float("nan"))
            else:
                burst_velocities.append(float(bl) / (bd / 1000.0))

        fill_pi_stats(pi, to_snake("Burst Length"), burst_lengths)
        fill_pi_stats(pi, to_snake("Burst Duration"), burst_durations_ms)
        fill_pi_stats(pi, to_snake("Burst Velocity"), burst_velocities)
    else:
        fill_pi_stats(pi, to_snake("Key Overlap Ratio"), [])
        fill_pi_stats(pi, to_snake("Pause Duration"), [])
        fill_pi_stats(pi, to_snake("Pause Ratio"), [])
        fill_pi_stats(pi, to_snake("Burst Length"), [])
        fill_pi_stats(pi, to_snake("Burst Duration"), [])
        fill_pi_stats(pi, to_snake("Burst Velocity"), [])

    n = int(params.ngram_n)
    ngram_durs_ms: List[float] = []
    if N >= n and n >= 2:
        for i in range(0, N - n + 1):
            ngram_durs_ms.append(ms_from_timedelta(down.iloc[i + n - 1] - down.iloc[i]))
    fill_pi_stats(pi, to_snake("Ngram Typing Duration"), ngram_durs_ms)

    word_starts: List[int] = []
    for i in range(N):
        if tokens[i] in SPACE_TOKENS:
            continue
        if i == 0 or tokens[i - 1] in SPACE_TOKENS:
            word_starts.append(i)
    inter_word_ms: List[float] = []
    for j in range(1, len(word_starts)):
        inter_word_ms.append(ms_from_timedelta(down.iloc[word_starts[j]] - down.iloc[word_starts[j - 1]]))
    fill_pi_stats(pi, to_snake("Inter-Word Interval"), inter_word_ms)

    fill_pi_stats(pi, to_snake("Key Hold Duration"), hold_ms)

    def holds_where(pred) -> List[float]:
        return [float(h) for h, t, c in zip(hold_ms, tokens, cats) if pred(t, c)]

    fill_pi_stats(pi, to_snake("Alphabetic Key Hold Duration"), holds_where(lambda t, c: c == "alphabetic"))
    fill_pi_stats(pi, to_snake("Spacebar Key Hold Duration"), holds_where(lambda t, c: t in SPACE_TOKENS))
    fill_pi_stats(pi, to_snake("Backspace Key Hold Duration"), holds_where(lambda t, c: t in BACKSPACE_SET))
    fill_pi_stats(pi, to_snake("Delete Key Hold Duration"), holds_where(lambda t, c: t in DELETE_SET))
    fill_pi_stats(pi, to_snake("Enter Key Hold Duration"), holds_where(lambda t, c: t in ENTER_SET))
    fill_pi_stats(pi, to_snake("CapsLock Key Hold Duration"), holds_where(lambda t, c: t in CAPS_SET))
    fill_pi_stats(pi, to_snake("Shift Key Hold Duration"), holds_where(lambda t, c: t in SHIFT_SET))
    fill_pi_stats(pi, to_snake("Control Key Hold Duration"), holds_where(lambda t, c: t in CTRL_SET))
    fill_pi_stats(pi, to_snake("Alt Key Hold Duration"), holds_where(lambda t, c: t in ALT_SET))
    fill_pi_stats(pi, to_snake("Tab Key Hold Duration"), holds_where(lambda t, c: t in TAB_SET))
    fill_pi_stats(pi, to_snake("Escape Key Hold Duration"), holds_where(lambda t, c: t in ESC_SET))
    fill_pi_stats(pi, to_snake("Insert Key Hold Duration"), holds_where(lambda t, c: t in INSERT_SET))
    fill_pi_stats(pi, to_snake("Punctuation Key Hold Duration"), holds_where(lambda t, c: c == "punctuation"))
    fill_pi_stats(pi, to_snake("Digit Key Hold Duration"), holds_where(lambda t, c: c == "digit"))
    fill_pi_stats(pi, to_snake("Cursor Move Key Duration"), holds_where(lambda t, c: t in CURSOR_SET))

    def ratio_where(pred) -> Optional[float]:
        if N <= 0:
            return None
        cnt = sum(1 for t, c in zip(tokens, cats) if pred(t, c))
        return float(cnt) / float(N)

    def set_ratio(name: str, val: Optional[float]) -> None:
        fill_pi_stats(pi, to_snake(name), [val] if val is not None else [])

    set_ratio("Alphabetic Key Ratio", ratio_where(lambda t, c: c == "alphabetic"))
    set_ratio("Spacebar Key Ratio", ratio_where(lambda t, c: t in SPACE_TOKENS))
    set_ratio("Backspace Key Ratio", ratio_where(lambda t, c: t in BACKSPACE_SET))
    set_ratio("Delete Key Ratio", ratio_where(lambda t, c: t in DELETE_SET))
    set_ratio("Enter Key Ratio", ratio_where(lambda t, c: t in ENTER_SET))
    set_ratio("CapsLock Key Ratio", ratio_where(lambda t, c: t in CAPS_SET))
    set_ratio("Shift Key Ratio", ratio_where(lambda t, c: t in SHIFT_SET))
    set_ratio("Control Key Ratio", ratio_where(lambda t, c: t in CTRL_SET))
    set_ratio("Alt Key Ratio", ratio_where(lambda t, c: t in ALT_SET))
    set_ratio("Tab Key Ratio", ratio_where(lambda t, c: t in TAB_SET))
    set_ratio("Escape Key Ratio", ratio_where(lambda t, c: t in ESC_SET))
    set_ratio("Insert Key Ratio", ratio_where(lambda t, c: t in INSERT_SET))
    set_ratio("Punctuation Key Ratio", ratio_where(lambda t, c: c == "punctuation"))
    set_ratio("Digit Key Ratio", ratio_where(lambda t, c: c == "digit"))
    set_ratio("Cursor Move Key Ratio", ratio_where(lambda t, c: t in CURSOR_SET))

    def chord_durations_and_ratio(target_key: str) -> Tuple[List[float], Optional[float]]:
        if N <= 0:
            return [], None
        durs: List[float] = []
        cnt = 0
        for i in range(N - 1):
            if tokens[i] in CTRL_SET and tokens[i + 1] == target_key:
                dur = ms_from_timedelta(max(up.iloc[i], up.iloc[i + 1]) - down.iloc[i])
                durs.append(dur)
                cnt += 1
        ratio = float(cnt) / float(N)
        return durs, ratio

    undo_durs, undo_ratio = chord_durations_and_ratio("Z")
    paste_durs, paste_ratio = chord_durations_and_ratio("V")
    cut_durs, cut_ratio = chord_durations_and_ratio("X")
    copy_durs, copy_ratio = chord_durations_and_ratio("C")

    fill_pi_stats(pi, to_snake("Undo Duration"), undo_durs)
    set_ratio("Undo Ratio", undo_ratio)
    fill_pi_stats(pi, to_snake("Paste Duration"), paste_durs)
    set_ratio("Paste Ratio", paste_ratio)
    fill_pi_stats(pi, to_snake("Cut Duration"), cut_durs)
    set_ratio("Cut Ratio", cut_ratio)
    fill_pi_stats(pi, to_snake("Copy Duration"), copy_durs)
    set_ratio("Copy Ratio", copy_ratio)

    if N > 0:
        cat_counts = Counter(cats)
        ent = 0.0
        for _, cnt in cat_counts.items():
            p = float(cnt) / float(N)
            if p > 0.0:
                ent -= p * math.log2(p)
        fill_pi_stats(pi, to_snake("Key Category Entropy"), [ent])
    else:
        fill_pi_stats(pi, to_snake("Key Category Entropy"), [])

    if N >= n and n >= 2:
        ngrams = [tuple(tokens[i:i + n]) for i in range(0, N - n + 1)]
        ng_counts = Counter(ngrams)
        total_ng = sum(ng_counts.values())
        ent_ng = 0.0
        for _, cnt in ng_counts.items():
            p = float(cnt) / float(total_ng)
            if p > 0.0:
                ent_ng -= p * math.log2(p)
        fill_pi_stats(pi, to_snake("Ngram Distribution Entropy"), [ent_ng])
    else:
        fill_pi_stats(pi, to_snake("Ngram Distribution Entropy"), [])

    return pi


# =========================
# Windowing + file IO
# =========================

def format_ts_like_input(ts: pd.Timestamp) -> str:
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    base = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")
    off = ts.strftime("%z")  # +0900
    if len(off) == 5 and off.endswith("00"):
        off = off[:-2]  # +09
    elif len(off) == 5:
        off = off[:3] + ":" + off[3:]
    return base + off


def iter_keyboard_csv_paths(root_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower() == "keyboard.csv":
                out.append(os.path.join(dirpath, fn))
    return out


def parse_path_components(root_dir: str, csv_path: str) -> Tuple[str, str, str, str]:
    """
    Return (date, client_id, task, uuid) inferred from:
      YYYYMMDD/<client_id>/<task>/keyboard/<uuid>/keyboard.csv
    """
    rel = os.path.relpath(csv_path, root_dir)
    parts = rel.split(os.sep)
    lower = [p.lower() for p in parts]

    if "keyboard" not in lower:
        raise ValueError(f"Path does not contain 'keyboard': {csv_path}")
    kidx = lower.index("keyboard")

    date = parts[kidx - 3] if kidx - 3 >= 0 else "unknown_date"
    client_id = parts[kidx - 2] if kidx - 2 >= 0 else "unknown_client"
    task = parts[kidx - 1] if kidx - 1 >= 0 else "unknown_task"
    uuid = parts[kidx + 1] if kidx + 1 < len(parts) else "unknown_uuid"
    return date, client_id, task, uuid


def read_keyboard_csv(
    csv_path: str,
    holdtime_mode: str,
) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    if "letter" not in df_raw.columns:
        raise ValueError(f"Missing 'letter' column: {csv_path}")
    df_raw["token"] = df_raw["letter"].apply(normalize_token)
    df_raw["category"] = df_raw["token"].apply(key_category)

    has_down_up_cols = ("downTime" in df_raw.columns) and ("upTime" in df_raw.columns)
    has_key_action = ("key_action" in df_raw.columns)

    mode = holdtime_mode.lower()
    if mode not in {"paired_rows", "event_pairing", "auto"}:
        raise ValueError(f"Invalid holdtime_mode={holdtime_mode}")

    def build_from_paired_rows(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["down_dt"] = pd.to_datetime(d["downTime"], errors="coerce")
        d["up_dt"] = pd.to_datetime(d["upTime"], errors="coerce")
        d = d.dropna(subset=["down_dt", "up_dt"]).copy()

        if "holdTime" in d.columns:
            d["hold_ms"] = pd.to_numeric(d["holdTime"], errors="coerce")
        else:
            d["hold_ms"] = np.nan

        miss = d["hold_ms"].isna()
        if miss.any():
            d.loc[miss, "hold_ms"] = (
                (d.loc[miss, "up_dt"] - d.loc[miss, "down_dt"]).dt.total_seconds() * 1000.0
            )

        d = d.sort_values("down_dt").reset_index(drop=True)
        return d[["token", "down_dt", "up_dt", "hold_ms", "category"]]

    def build_from_event_pairing(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["event_dt"] = _parse_event_timestamp(d)
        d = d.dropna(subset=["event_dt"]).copy()

        if not has_key_action:
            raise ValueError("Split-event pairing requires 'key_action' column")
        paired = pair_key_events_stack(d, ts_col="event_dt")

        if paired.empty:
            return paired

        paired["category"] = paired["token"].apply(key_category)
        paired = paired.sort_values("down_dt").reset_index(drop=True)
        return paired[["token", "down_dt", "up_dt", "hold_ms", "category"]]

    if mode == "paired_rows":
        if has_down_up_cols:
            out = build_from_paired_rows(df_raw)
            if not out.empty:
                return out
        if has_key_action:
            return build_from_event_pairing(df_raw)
        raise ValueError("paired_rows requested but downTime/upTime unavailable and no key_action to pair")

    if mode == "event_pairing":
        return build_from_event_pairing(df_raw)

    # auto
    if has_down_up_cols:
        out = build_from_paired_rows(df_raw)
        if not out.empty:
            return out
    if has_key_action:
        return build_from_event_pairing(df_raw)
    raise ValueError("auto mode failed: no usable paired rows and no key_action for pairing")


def build_windows(df: pd.DataFrame, params: WindowParams) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if df.empty:
        return []
    global_start = df["down_dt"].min()
    global_end = df["up_dt"].max()

    wsize = pd.Timedelta(seconds=float(params.window_size_sec))
    stride = pd.Timedelta(seconds=float(params.stride_sec))

    windows = []
    w_start = global_start
    while w_start < global_end:
        w_end = w_start + wsize
        windows.append((w_start, w_end))
        w_start = w_start + stride
        if stride <= pd.Timedelta(0):
            break
        if w_start > global_end and len(windows) > 0:
            break
    return windows


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def default_output_root_from_hospital(base_path: str) -> str:
    parent = os.path.dirname(os.path.abspath(base_path))
    return os.path.join(parent, "primitive_indicator")


def make_output_path(
    output_root: str,
    client_id: str,
    task: str,
) -> str:
    """
    Output:
      <output_root>/<client_id>/<task>/keyboard_pi.json
    """
    out_dir = os.path.join(output_root, client_id, task)
    ensure_dir(out_dir)
    return os.path.join(out_dir, OUTPUT_FILENAME)


def process_one_keyboard_csv(
    csv_path: str,
    base_path: str,
    output_root: str,
    template_obj: dict,
    params: WindowParams,
    holdtime_mode: str,
    screen_width_px: int,
    screen_height_px: int,
) -> Optional[str]:
    try:
        _, client_id, task, _ = parse_path_components(base_path, csv_path)
        df = read_keyboard_csv(csv_path, holdtime_mode=holdtime_mode)
    except Exception:
        return None

    windows = build_windows(df, params)
    if len(windows) == 0:
        return None

    out_obj = copy.deepcopy(template_obj)
    out_obj["client_id"] = client_id
    out_obj["context"] = task
    out_obj["sensor_modality"] = "keyboard"
    out_obj["screen"] = {
        "width": float(screen_width_px),
        "height": float(screen_height_px),
    }
    out_obj["windows"] = []

    for idx, (w_start, w_end) in enumerate(windows, start=1):
        dfw = df[(df["down_dt"] >= w_start) & (df["down_dt"] < w_end)].copy()
        pi = compute_keyboard_pi_for_window(dfw, w_start, w_end, params)
        out_obj["windows"].append({
            "window_number": idx,
            "start_timestamp": format_ts_like_input(w_start),
            "end_timestamp": format_ts_like_input(w_end),
            "pi": pi,
        })

    out_path = make_output_path(output_root, client_id, task)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    return out_path


def main():
    ap = argparse.ArgumentParser()

    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--base_path",
        help="hospital_data root, e.g. /data_248/pdss/hospital_data",
    )
    src_group.add_argument(
        "--root_dir",
        help="(deprecated) alias of --base_path",
    )

    ap.add_argument(
        "--output_base_path",
        default=None,
        help="Output root directory. Default: <parent_of_hospital_data>/primitive_indicator",
    )
    ap.add_argument(
        "--output_root",
        dest="output_base_path",
        default=None,
        help=argparse.SUPPRESS,
    )

    ap.add_argument("--window_size_sec", type=float, default=WINDOW_SIZE_SEC)
    ap.add_argument("--stride_sec", type=float, default=STRIDE_SEC)
    ap.add_argument("--pause_threshold_ms", type=float, default=PAUSE_THRESHOLD_MS)
    ap.add_argument("--ngram_n", type=int, default=NGRAM_N)

    ap.add_argument("--screen_width_px", type=int, default=SCREEN_WIDTH_PX)
    ap.add_argument("--screen_height_px", type=int, default=SCREEN_HEIGHT_PX)

    ap.add_argument(
        "--holdtime_mode",
        choices=["paired_rows", "event_pairing", "auto"],
        default="paired_rows",
        help="paired_rows: current KEY_DOWN_UP rows. event_pairing: pair KEY_DOWN/KEY_UP. auto: try paired_rows then pairing.",
    )

    ap.add_argument("--only_client_id", default=None, help="If set, process only this client_id")

    args = ap.parse_args()

    base_path = args.base_path or args.root_dir
    output_root = args.output_base_path or default_output_root_from_hospital(base_path)

    params = WindowParams(
        window_size_sec=args.window_size_sec,
        stride_sec=args.stride_sec,
        pause_threshold_ms=args.pause_threshold_ms,
        ngram_n=args.ngram_n,
    )

    def make_template_obj() -> dict:
        return {
            "client_id": "",
            "context": "",
            "sensor_modality": "keyboard",
            "windows": [],
        }

    template_obj = make_template_obj()

    csv_paths_all = iter_keyboard_csv_paths(base_path)
    if not csv_paths_all:
        print("[DONE] No keyboard.csv found.")
        return

    csv_paths = []
    client_ids = set()
    for p in csv_paths_all:
        try:
            _, client_id, _, _ = parse_path_components(base_path, p)
        except Exception:
            continue
        if args.only_client_id is not None and client_id != args.only_client_id:
            continue
        csv_paths.append(p)
        client_ids.add(client_id)

    print(f"[INFO] clients={len(client_ids)} csv_files={len(csv_paths)}")
    print(f"[INFO] SCREEN: {args.screen_width_px}x{args.screen_height_px} (px)")
    print(f"[INFO] window_size={params.window_size_sec}s stride={params.stride_sec}s output_root={output_root}")

    ok = 0
    for p in tqdm(csv_paths, desc="keyboard.csv -> keyboard_pi.json", unit="file"):
        out_path = process_one_keyboard_csv(
            csv_path=p,
            base_path=base_path,
            output_root=output_root,
            template_obj=template_obj,
            params=params,
            holdtime_mode=args.holdtime_mode,
            screen_width_px=args.screen_width_px,
            screen_height_px=args.screen_height_px,
        )
        if out_path:
            ok += 1

    print(f"[DONE] processed={ok} / total={len(csv_paths)}")


if __name__ == "__main__":
    main()
