"""
Touch Primitive Indicator Extractor

Input directory layout example:
  /data_248/pdss/hospital_data/YYYYMMDD/<client_id>/<task>/stylus/<archive_id>/stylus.csv

Output directory:
  /data_248/pdss/primitive_indicator/<client_id>/<task>/touch_pi.json

각 stylus.csv 를 읽어서 윈도우 단위로 Primitive Indicator 계산하고,
84개(FEATURE_NAMES) × 통계량 4개(STATS)을 JSON 형태로 저장

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Parameters (edit here)
# =========================

# 화면 크기 기본값 (유저가 CLI에서 override 가능)
SCREEN_WIDTH_PX = 1440.0
SCREEN_HEIGHT_PX = 900.0

# 윈도우 기본값 (초 단위, CLI에서 override 가능)
WINDOW_SIZE_SEC = 60.0
STRIDE_SEC = 30.0

# 제스처 분류용 heuristic
MIN_SWIPE_DISPLACEMENT_PX = 50.0
MAX_TAP_DISPLACEMENT_PX = 20.0
LONG_PRESS_DURATION_S = 0.5

HIGH_PRESSURE_THRESHOLD = 0.8
HIGH_VELOCITY_THRESHOLD = 1000.0  # px/s

DIRECTION_CHANGE_THRESHOLD_RAD = np.deg2rad(45.0)
TOUCH_ENTROPY_GRID_SIZE = 10  # 10x10 grid

# JSON 파일명
OUTPUT_FILENAME = "touch_pi.json"


# =========================
# Indicator list (84 × 4 stats)
# =========================

FEATURE_NAMES = [
    "Touch Position X",
    "Touch Position Y",
    "Touch Pressure",
    "Total Touch Stroke Count",
    "Touch Stroke Rate",
    "Total Touch Count",
    "Touch Event Rate",
    "Touch Event-Touch Stroke Ratio",
    "Touch Pressure Change Rate",
    "High Pressure Touch Count",
    "Touch Area",
    "Touch Major Axis",
    "Touch Minor Axis",
    "Touch Area Change Rate",
    "Touch Pressure-Area Correlation",
    "First Touch Latency",
    "First Touch Duration",
    "Tap Ratio",
    "Swipe Ratio",
    "Long-press Ratio",
    "Swipe Velocity",
    "Swipe Velocity X",
    "Swipe Velocity Y",
    "Swipe Acceleration",
    "Swipe Acceleration X",
    "Swipe Acceleration Y",
    "Swipe Jerk",
    "Swipe Jerk X",
    "Swipe Jerk Y",
    "Swipe Direction",
    "Swipe Angular Velocity",
    "High Velocity Touch Count",
    "Swipe Path Length",
    "Swipe Displacement",
    "Swipe Path Curvature",
    "Range-based Swipe Dispersion",
    "RMS-based Swipe Dispersion",
    "Bbox Area-based Swipe Dispersion",
    "Swipe Direction Change Count",
    "Range-based Swipe Endpoint Dispersion",
    "RMS-based Swipe Endpoint Dispersion",
    "Bbox Area-based Swipe Endpoint Dispersion",
    "Swipe Bbox Width",
    "Swipe Bbox Height",
    "Swipe Bbox Area",
    "Swipe Bbox Aspect Ratio",
    "Horizontal Swipe Rate",
    "Vertical Swipe Rate",
    "Touch Area Slope",
    "Touch Convex Hull Area",
    "Touch Convex Hull Perimeter",
    "Swipe Path Fractal Dimension",
    "Touch Hold Duration",
    "Tap Duration",
    "Swipe Duration",
    "Long-press Duration",
    "Inter-touch Interval",
    "Active Touch Time",
    "Touch Stroke Pressure Slope",
    "Inter-tap Interval",
    "Inter-swipe Interval",
    "Inter-Long-press Interval",
    "Tap Pressure",
    "Swipe Pressure",
    "Long-press Pressure",
    "Touch Entropy",
    "Edge Touch Ratio",
    "Left-half Touch Ratio",
    "Right-half Touch Ratio",
    "Top-half Touch Ratio",
    "Bottom-half Touch Ratio",
    "Central Touch Ratio",
    "Touch Offset to Screen Center",
    "Central Bias",
    "Range-based Touch Dispersion",
    "RMS-based Touch Dispersion",
    "Bbox Area-based Touch Dispersion",
    "Touch Covariance",
    "Velocity Rhythm Rate",
    "Velocity Rhythm Spectral Entropy",
    "Velocity Rhythm Regularity",
    "Acceleration Rhythm Rate",
    "Acceleration Rhythm Spectral Entropy",
    "Acceleration Rhythm Regularity",
]

STATS = ("mean", "std", "min", "max")


def slugify_feature_name(name: str) -> str:
    s = name.strip().lower()
    for ch in [" ", "-", "/", "(", ")", ",", ":", "."]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


@dataclass
class PISpec:
    feature_name: str
    stat: str
    key: str


def load_touch_pi_specs() -> List[PISpec]:
    specs: List[PISpec] = []
    for fname in FEATURE_NAMES:
        base = slugify_feature_name(fname)
        for st in STATS:
            key = f"{base}_{st}"
            specs.append(PISpec(feature_name=fname, stat=st, key=key))
    return specs


@dataclass
class WindowParams:
    window_size_sec: float = WINDOW_SIZE_SEC
    stride_sec: float = STRIDE_SEC
    screen_width_px: int = SCREEN_WIDTH_PX
    screen_height_px: int = SCREEN_HEIGHT_PX


# =========================
# CSV load & windowing
# =========================

def load_stylus_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # iso timestamp column
    ts_col = None
    if "it" in df.columns:
        ts_col = "it"
    elif "timeStamp" in df.columns:
        ts_col = "timeStamp"
    if ts_col is None:
        raise ValueError(f"Cannot find ISO timestamp column in {csv_path}")

    df["timestamp"] = pd.to_datetime(df[ts_col])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # relative time in seconds
    t0 = df["timestamp"].iloc[0]
    df["t_rel"] = (df["timestamp"] - t0).dt.total_seconds()

    # make sure required columns exist
    for col in ["px", "py", "p", "h", "w", "et", "ws", "pi"]:
        if col not in df.columns:
            if col in ["px", "py", "p", "h", "w"]:
                df[col] = np.nan
            else:
                df[col] = 0
    return df


def segment_into_windows(
    df: pd.DataFrame,
    params: WindowParams,
) -> List[Tuple[int, float, float, pd.DataFrame]]:
    max_t = df["t_rel"].max()
    windows: List[Tuple[int, float, float, pd.DataFrame]] = []
    start = 0.0
    win_idx = 1

    while start <= max_t:
        end = start + params.window_size_sec
        mask = (df["t_rel"] >= start) & (df["t_rel"] < end)
        df_win = df[mask]
        if len(df_win) > 0:
            windows.append((win_idx, start, end, df_win.copy()))
            win_idx += 1
        start += params.stride_sec
        if params.stride_sec <= 0:
            break

    return windows


# =========================
# Stroke & kinematics
# =========================

def assign_stroke_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 pointerId(pi) 별로 DOWN~UP 구간을 stroke_id로 붙여주는 함수.
    (윈도우 슬라이스 때문에 인덱스 라벨이 뒤죽박죽일 수 있으므로
    반드시 reset_index(drop=True) 후, 위치 인덱스로만 접근해야 함.)
    """
    # 라벨 인덱스 초기화 + 정렬
    df = df.sort_values(["pi", "timestamp"]).reset_index(drop=True).copy()

    # 위치 인덱스 기준으로 stroke_ids 배열 생성
    stroke_ids = np.full(len(df), np.nan)
    current: Dict[int, Optional[int]] = {}
    next_id = 0

    # enumerate를 써서 i는 항상 0 ~ len(df)-1
    for i, (_, row) in enumerate(df.iterrows()):
        pointer = int(row["pi"])
        et = int(row["et"])  # 0=ENTER,1=DOWN,2=MOVE,3=UP,4=LEAVE

        if pointer not in current:
            current[pointer] = None
        cur_id = current[pointer]

        if et == 1:  # DOWN
            cur_id = next_id
            next_id += 1
            current[pointer] = cur_id
            stroke_ids[i] = cur_id

        elif et == 3:  # UP
            stroke_ids[i] = cur_id
            current[pointer] = None

        else:  # MOVE, ENTER, LEAVE 등
            if cur_id is not None:
                stroke_ids[i] = cur_id

    df["stroke_id"] = stroke_ids
    return df



def compute_velocity_and_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("t_rel").copy()
    x = df["px"].values.astype(float)
    y = df["py"].values.astype(float)
    t = df["t_rel"].values.astype(float)

    dt = np.diff(t, prepend=t[0])
    dt[dt == 0] = np.nan

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    vx = dx / dt
    vy = dy / dt
    v = np.sqrt(vx ** 2 + vy ** 2)

    dvx = np.diff(vx, prepend=vx[0])
    dvy = np.diff(vy, prepend=vy[0])
    dv = np.diff(v, prepend=v[0])

    ax = dvx / dt
    ay = dvy / dt
    a = dv / dt

    df["vx"] = vx
    df["vy"] = vy
    df["v"] = v
    df["ax"] = ax
    df["ay"] = ay
    df["a"] = a
    return df


def compute_stroke_metrics(df_win: pd.DataFrame) -> pd.DataFrame:
    df_s = df_win[df_win["stroke_id"].notna()].copy()
    if len(df_s) == 0:
        return pd.DataFrame(
            columns=[
                "stroke_id", "t_start", "t_end", "duration",
                "path_length", "displacement",
                "x_start", "y_start", "x_end", "y_end",
                "mean_pressure",
            ]
        )

    rows = []
    for sid, g in df_s.groupby("stroke_id"):
        g = g.sort_values("t_rel")
        t = g["t_rel"].values.astype(float)
        x = g["px"].values.astype(float)
        y = g["py"].values.astype(float)
        p = g["p"].values.astype(float)

        if len(t) == 0:
            continue

        duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.sqrt(dx ** 2 + dy ** 2)
        path_length = float(np.nansum(seg_len))
        displacement = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
        mean_p = float(np.nanmean(p)) if np.any(~np.isnan(p)) else np.nan

        rows.append(
            {
                "stroke_id": float(sid),
                "t_start": float(t[0]),
                "t_end": float(t[-1]),
                "duration": duration,
                "path_length": path_length,
                "displacement": displacement,
                "x_start": float(x[0]),
                "y_start": float(y[0]),
                "x_end": float(x[-1]),
                "y_end": float(y[-1]),
                "mean_pressure": mean_p,
            }
        )

    return pd.DataFrame(rows)


def classify_strokes(strokes: pd.DataFrame) -> pd.DataFrame:
    if strokes.empty:
        strokes["gesture_type"] = []
        return strokes

    gtypes = []
    for _, row in strokes.iterrows():
        duration = row["duration"]
        disp = row["displacement"]
        path_len = row["path_length"]

        if duration >= LONG_PRESS_DURATION_S and disp <= MAX_TAP_DISPLACEMENT_PX:
            gtypes.append("longpress")
        else:
            if disp >= MIN_SWIPE_DISPLACEMENT_PX or path_len >= MIN_SWIPE_DISPLACEMENT_PX:
                gtypes.append("swipe")
            else:
                gtypes.append("tap")
    strokes = strokes.copy()
    strokes["gesture_type"] = gtypes
    return strokes


# =========================
# 시퀀스 & helper
# =========================

def compute_basic_touch_sequences(
    df_win: pd.DataFrame,
    params: WindowParams,
) -> Dict[str, np.ndarray]:
    seq: Dict[str, np.ndarray] = {}

    x = df_win["px"].values.astype(float)
    y = df_win["py"].values.astype(float)
    p = df_win["p"].values.astype(float)
    h = df_win["h"].values.astype(float)
    w = df_win["w"].values.astype(float)

    seq["x"] = x
    seq["y"] = y
    seq["x_norm"] = x / float(params.screen_width_px)
    seq["y_norm"] = y / float(params.screen_height_px)
    seq["p"] = p
    seq["area"] = h * w
    seq["major_axis"] = np.maximum(h, w)
    seq["minor_axis"] = np.minimum(h, w)
    seq["t_rel"] = df_win["t_rel"].values.astype(float)

    for col in ["v", "vx", "vy", "a", "ax", "ay"]:
        if col in df_win.columns:
            seq[col] = df_win[col].values.astype(float)
        else:
            seq[col] = np.full(len(df_win), np.nan)

    return seq


def nan_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_fft_rhythm_features(series: np.ndarray, sampling_rate: float) -> Dict[str, Optional[float]]:
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    if len(series) < 4 or sampling_rate <= 0:
        return {"f_dom": None, "entropy": None, "regularity": None}

    series = series - np.mean(series)
    fft_vals = np.fft.rfft(series)
    freqs = np.fft.rfftfreq(len(series), d=1.0 / sampling_rate)
    power = np.abs(fft_vals) ** 2

    valid = freqs > 0
    freqs = freqs[valid]
    power = power[valid]

    if len(freqs) == 0 or np.all(power == 0):
        return {"f_dom": None, "entropy": None, "regularity": None}

    idx_max = int(np.argmax(power))
    f_dom = float(freqs[idx_max])

    p_norm = power / np.sum(power)
    entropy = float(-np.sum(p_norm * np.log2(p_norm + 1e-12)))
    regularity = float(power[idx_max] / (np.mean(power) + 1e-12))

    return {"f_dom": f_dom, "entropy": entropy, "regularity": regularity}


def convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def polygon_area_perimeter(points: np.ndarray) -> Tuple[float, float]:
    if len(points) < 3:
        return 0.0, 0.0
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    dx = np.diff(np.r_[x, x[0]])
    dy = np.diff(np.r_[y, y[0]])
    perim = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
    return float(area), perim


def estimate_fractal_dimension(points: np.ndarray, scales=(4, 8, 16)) -> Optional[float]:
    points = np.asarray(points, dtype=float)
    if len(points) < 10:
        return None
    x = points[:, 0]
    y = points[:, 1]
    if np.all(x == x[0]) and np.all(y == y[0]):
        return 0.0
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    if x_min == x_max or y_min == y_max:
        return 0.0

    log_eps_inv = []
    log_N = []
    for s in scales:
        xs = ((x - x_min) / (x_max - x_min) * s).astype(int)
        ys = ((y - y_min) / (y_max - y_min) * s).astype(int)
        xs = np.clip(xs, 0, s - 1)
        ys = np.clip(ys, 0, s - 1)
        cells = set(zip(xs, ys))
        N = len(cells)
        if N <= 0:
            continue
        eps = 1.0 / s
        log_eps_inv.append(np.log(1.0 / eps))
        log_N.append(np.log(N))

    if len(log_eps_inv) < 2:
        return None
    slope = np.polyfit(log_eps_inv, log_N, 1)[0]
    return float(slope)


# =========================
# Feature별 base series
# =========================

def compute_base_series_for_feature(
    feature_name: str,
    df_win: pd.DataFrame,
    seq: Dict[str, np.ndarray],
    strokes: pd.DataFrame,
    params: WindowParams,
    window_start: float,
    window_end: float,
    vel_rhythm: Dict[str, Optional[float]],
    acc_rhythm: Dict[str, Optional[float]],
) -> np.ndarray:
    t = seq["t_rel"]
    T = max(window_end - window_start, 0.0)
    x = seq["x"]
    y = seq["y"]
    p = seq["p"]
    area = seq["area"]
    v = seq["v"]
    a = seq["a"]
    df_g = df_win.copy()
    N_event = len(df_g)
    N_stroke = len(strokes)

    def stroke_type_mask(stype: str) -> np.ndarray:
        return (df_g.get("gesture_type", None) == stype).values if "gesture_type" in df_g.columns else np.zeros(
            len(df_g), dtype=bool
        )

    def strokes_of_type(stype: str) -> pd.DataFrame:
        if strokes.empty:
            return strokes
        return strokes[strokes["gesture_type"] == stype]

    # --- 1. 기본 위치/압력 ---
    if feature_name == "Touch Position X":
        return x
    if feature_name == "Touch Position Y":
        return y
    if feature_name == "Touch Pressure":
        return p

    # --- 2. Stroke & Touch 개수 ---
    if feature_name == "Total Touch Stroke Count":
        return np.array([N_stroke], dtype=float)
    if feature_name == "Touch Stroke Rate":
        return np.array([N_stroke / T if T > 0 else np.nan], dtype=float)
    if feature_name == "Total Touch Count":
        return np.array([N_event], dtype=float)
    if feature_name == "Touch Event Rate":
        return np.array([N_event / T if T > 0 else np.nan], dtype=float)
    if feature_name == "Touch Event-Touch Stroke Ratio":
        return np.array([N_event / N_stroke if N_stroke > 0 else np.nan], dtype=float)

    # --- 3. Pressure / Area ---
    if feature_name == "Touch Pressure Change Rate":
        if len(p) > 1 and len(t) > 1:
            dp = np.diff(p)
            dt = np.diff(t)
            valid = dt > 0
            rate = np.abs(dp[valid] / dt[valid]) if np.any(valid) else np.array([])
            return rate
        return np.array([], dtype=float)

    if feature_name == "High Pressure Touch Count":
        return np.array([np.sum(p > HIGH_PRESSURE_THRESHOLD)], dtype=float)

    if feature_name == "Touch Area":
        return area
    if feature_name == "Touch Major Axis":
        return seq["major_axis"]
    if feature_name == "Touch Minor Axis":
        return seq["minor_axis"]

    if feature_name == "Touch Area Change Rate":
        A = area
        if len(A) > 1 and len(t) > 1:
            dA = np.diff(A)
            dt = np.diff(t)
            valid = dt > 0
            rate = np.abs(dA[valid] / dt[valid]) if np.any(valid) else np.array([])
            return rate
        return np.array([], dtype=float)

    if feature_name == "Touch Pressure-Area Correlation":
        mask = ~np.isnan(p) & ~np.isnan(area)
        if np.sum(mask) >= 2:
            corr = np.corrcoef(p[mask], area[mask])[0, 1]
            return np.array([corr], dtype=float)
        return np.array([], dtype=float)

    # --- 4. 첫 터치 ---
    if feature_name == "First Touch Latency":
        if strokes.empty:
            return np.array([], dtype=float)
        first = strokes.sort_values("t_start").iloc[0]
        latency = float(first["t_start"] - window_start)
        return np.array([latency], dtype=float)

    if feature_name == "First Touch Duration":
        if strokes.empty:
            return np.array([], dtype=float)
        first = strokes.sort_values("t_start").iloc[0]
        return np.array([first["duration"]], dtype=float)

    # --- 5. Tap / Swipe / Long-press 비율 ---
    if feature_name == "Tap Ratio":
        if N_stroke == 0:
            return np.array([], dtype=float)
        n_tap = np.sum(strokes["gesture_type"] == "tap")
        return np.array([n_tap / N_stroke], dtype=float)
    if feature_name == "Swipe Ratio":
        if N_stroke == 0:
            return np.array([], dtype=float)
        n_swipe = np.sum(strokes["gesture_type"] == "swipe")
        return np.array([n_swipe / N_stroke], dtype=float)
    if feature_name == "Long-press Ratio":
        if N_stroke == 0:
            return np.array([], dtype=float)
        n_lp = np.sum(strokes["gesture_type"] == "longpress")
        return np.array([n_lp / N_stroke], dtype=float)

    # --- 6. Swipe Velocity / Acc / Jerk ---
    if feature_name in [
        "Swipe Velocity",
        "Swipe Velocity X",
        "Swipe Velocity Y",
        "Swipe Acceleration",
        "Swipe Acceleration X",
        "Swipe Acceleration Y",
        "Swipe Jerk",
        "Swipe Jerk X",
        "Swipe Jerk Y",
    ]:
        swipe_mask = stroke_type_mask("swipe")
        if not np.any(swipe_mask):
            return np.array([], dtype=float)
        df_sw = df_g[swipe_mask].sort_values(["stroke_id", "t_rel"])
        t_sw = df_sw["t_rel"].values.astype(float)
        vx_sw = df_sw["vx"].values.astype(float)
        vy_sw = df_sw["vy"].values.astype(float)
        v_sw = df_sw["v"].values.astype(float)
        ax_sw = df_sw["ax"].values.astype(float)
        ay_sw = df_sw["ay"].values.astype(float)
        a_sw = df_sw["a"].values.astype(float)

        if feature_name == "Swipe Velocity":
            return v_sw[~np.isnan(v_sw)]
        if feature_name == "Swipe Velocity X":
            return vx_sw[~np.isnan(vx_sw)]
        if feature_name == "Swipe Velocity Y":
            return vy_sw[~np.isnan(vy_sw)]
        if feature_name == "Swipe Acceleration":
            return a_sw[~np.isnan(a_sw)]
        if feature_name == "Swipe Acceleration X":
            return ax_sw[~np.isnan(ax_sw)]
        if feature_name == "Swipe Acceleration Y":
            return ay_sw[~np.isnan(ay_sw)]

        if len(t_sw) > 1:
            # 시간 차이 (N-1 길이)
            dt_sw = np.diff(t_sw)
            valid = dt_sw > 0

            if not np.any(valid):
                return np.array([], dtype=float)

            dt_valid = dt_sw[valid]

            # 가속도의 변화량 (N-1 길이)
            da = np.diff(a_sw)[valid]
            dax = np.diff(ax_sw)[valid]
            day = np.diff(ay_sw)[valid]

            # jerk = da/dt
            j  = da  / dt_valid
            jx = dax / dt_valid
            jy = day / dt_valid

            if feature_name == "Swipe Jerk":
                return j[~np.isnan(j)]
            if feature_name == "Swipe Jerk X":
                return jx[~np.isnan(jx)]
            if feature_name == "Swipe Jerk Y":
                return jy[~np.isnan(jy)]
        return np.array([], dtype=float)


    # --- 7. Swipe Direction / Angular Velocity / Direction Change Count ---
    if feature_name in ["Swipe Direction", "Swipe Angular Velocity", "Swipe Direction Change Count"]:
        swipe_mask = stroke_type_mask("swipe")
        if not np.any(swipe_mask):
            return np.array([], dtype=float)
        df_sw = df_g[swipe_mask].sort_values(["stroke_id", "t_rel"])
        x_sw = df_sw["px"].values.astype(float)
        y_sw = df_sw["py"].values.astype(float)
        t_sw = df_sw["t_rel"].values.astype(float)
        sid_sw = df_sw["stroke_id"].values

        dx = np.diff(x_sw, prepend=x_sw[0])
        dy = np.diff(y_sw, prepend=y_sw[0])
        theta = np.arctan2(dy, dx)

        if feature_name == "Swipe Direction":
            return theta

        dtheta = np.diff(theta, prepend=theta[0])
        dt = np.diff(t_sw, prepend=t_sw[0])
        sid_diff = np.diff(sid_sw, prepend=sid_sw[0])
        dt[sid_diff != 0] = np.nan

        omega = dtheta / dt
        if feature_name == "Swipe Angular Velocity":
            return omega[~np.isnan(omega)]

        high_change = (np.abs(dtheta) > DIRECTION_CHANGE_THRESHOLD_RAD) & (sid_diff == 0)
        count = int(np.sum(high_change))
        return np.array([count], dtype=float)

    # --- 8. High Velocity Touch Count ---
    if feature_name == "High Velocity Touch Count":
        return np.array([np.sum(v > HIGH_VELOCITY_THRESHOLD)], dtype=float)

    # --- 9. Swipe Path Length / Displacement / Curvature ---
    if feature_name in ["Swipe Path Length", "Swipe Displacement", "Swipe Path Curvature"]:
        sw = strokes_of_type("swipe")
        if sw.empty:
            return np.array([], dtype=float)
        if feature_name == "Swipe Path Length":
            return sw["path_length"].values.astype(float)
        if feature_name == "Swipe Displacement":
            return sw["displacement"].values.astype(float)
        if feature_name == "Swipe Path Curvature":
            L = sw["path_length"].values.astype(float)
            D = sw["displacement"].values.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                S = D / L
            return S

    # --- 10. Swipe Dispersion (전체 / Endpoint) ---
    if feature_name in [
        "Range-based Swipe Dispersion",
        "RMS-based Swipe Dispersion",
        "Bbox Area-based Swipe Dispersion",
        "Range-based Swipe Endpoint Dispersion",
        "RMS-based Swipe Endpoint Dispersion",
        "Bbox Area-based Swipe Endpoint Dispersion",
    ]:
        sw = strokes_of_type("swipe")
        if sw.empty:
            return np.array([], dtype=float)

        swipe_mask = stroke_type_mask("swipe")
        df_sw = df_g[swipe_mask]
        x_sw_all = df_sw["px"].values.astype(float)
        y_sw_all = df_sw["py"].values.astype(float)

        x_endpts = np.concatenate([sw["x_start"].values, sw["x_end"].values])
        y_endpts = np.concatenate([sw["y_start"].values, sw["y_end"].values])

        if feature_name == "Range-based Swipe Dispersion":
            rx = np.nanmax(x_sw_all) - np.nanmin(x_sw_all)
            ry = np.nanmax(y_sw_all) - np.nanmin(y_sw_all)
            return np.array([rx + ry], dtype=float)
        if feature_name == "RMS-based Swipe Dispersion":
            varx = np.nanvar(x_sw_all)
            vary = np.nanvar(y_sw_all)
            return np.array([np.sqrt(varx + vary)], dtype=float)
        if feature_name == "Bbox Area-based Swipe Dispersion":
            rx = np.nanmax(x_sw_all) - np.nanmin(x_sw_all)
            ry = np.nanmax(y_sw_all) - np.nanmin(y_sw_all)
            return np.array([rx * ry], dtype=float)

        if feature_name == "Range-based Swipe Endpoint Dispersion":
            rx = np.nanmax(x_endpts) - np.nanmin(x_endpts)
            ry = np.nanmax(y_endpts) - np.nanmin(y_endpts)
            return np.array([rx + ry], dtype=float)
        if feature_name == "RMS-based Swipe Endpoint Dispersion":
            varx = np.nanvar(x_endpts)
            vary = np.nanvar(y_endpts)
            return np.array([np.sqrt(varx + vary)], dtype=float)
        if feature_name == "Bbox Area-based Swipe Endpoint Dispersion":
            rx = np.nanmax(x_endpts) - np.nanmin(x_endpts)
            ry = np.nanmax(y_endpts) - np.nanmin(y_endpts)
            return np.array([rx * ry], dtype=float)

    # --- 11. Swipe Bbox Width/Height/Area/Aspect Ratio ---
    if feature_name in ["Swipe Bbox Width", "Swipe Bbox Height", "Swipe Bbox Area", "Swipe Bbox Aspect Ratio"]:
        sw_mask = stroke_type_mask("swipe")
        if not np.any(sw_mask):
            return np.array([], dtype=float)
        df_sw = df_g[sw_mask]
        xs = df_sw["px"].values.astype(float)
        ys = df_sw["py"].values.astype(float)
        rx = np.nanmax(xs) - np.nanmin(xs)
        ry = np.nanmax(ys) - np.nanmin(ys)
        if feature_name == "Swipe Bbox Width":
            return np.array([rx], dtype=float)
        if feature_name == "Swipe Bbox Height":
            return np.array([ry], dtype=float)
        if feature_name == "Swipe Bbox Area":
            return np.array([rx * ry], dtype=float)
        if feature_name == "Swipe Bbox Aspect Ratio":
            return np.array([rx / ry if ry > 0 else np.nan], dtype=float)

    # --- 12. Horizontal / Vertical Swipe Rate ---
    if feature_name in ["Horizontal Swipe Rate", "Vertical Swipe Rate"]:
        sw = strokes_of_type("swipe")
        if sw.empty or T <= 0:
            return np.array([], dtype=float)
        dx = sw["x_end"].values - sw["x_start"].values
        dy = sw["y_end"].values - sw["y_start"].values
        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)
        horizontal = abs_dx > abs_dy
        vertical = abs_dy > abs_dx
        n_h = int(np.sum(horizontal))
        n_v = int(np.sum(vertical))
        if feature_name == "Horizontal Swipe Rate":
            return np.array([n_h / T], dtype=float)
        else:
            return np.array([n_v / T], dtype=float)

    # --- 13. Area Slope / Convex Hull / Fractal Dimension ---
    if feature_name == "Touch Area Slope":
        # area(t)의 선형 기울기
        if len(area) < 2:
            return np.array([], dtype=float)

        # 유효한 값만 사용 (NaN/inf 제거)
        mask = np.isfinite(area) & np.isfinite(t)
        if np.count_nonzero(mask) < 2:
            return np.array([], dtype=float)

        t_valid = t[mask]
        a_valid = area[mask]

        # 시간축이 전부 같으면 기울기 정의 불가
        if np.unique(t_valid).size < 2:
            return np.array([], dtype=float)

        try:
            coeffs = np.polyfit(t_valid, a_valid, 1)
            return np.array([coeffs[0]], dtype=float)
        except np.linalg.LinAlgError:
            # SVD가 수렴하지 않는 병맛 윈도우는 그냥 스킵
            return np.array([], dtype=float)
        except FloatingPointError:
            # 혹시라도 다른 수치 에러가 나면 역시 스킵
            return np.array([], dtype=float)


    if feature_name in ["Touch Convex Hull Area", "Touch Convex Hull Perimeter"]:
        pts = np.column_stack([x, y])
        pts = pts[~np.isnan(pts).any(axis=1)]
        if len(pts) < 3:
            return np.array([], dtype=float)
        hull = convex_hull(pts)
        area_h, perim_h = polygon_area_perimeter(hull)
        if feature_name == "Touch Convex Hull Area":
            return np.array([area_h], dtype=float)
        else:
            return np.array([perim_h], dtype=float)

    if feature_name == "Swipe Path Fractal Dimension":
        sw = strokes_of_type("swipe")
        if sw.empty:
            return np.array([], dtype=float)
        df_sw_all = df_g[stroke_type_mask("swipe")]
        if df_sw_all.empty:
            return np.array([], dtype=float)
        pts = df_sw_all[["px", "py"]].values.astype(float)
        d = estimate_fractal_dimension(pts)
        if d is None:
            return np.array([], dtype=float)
        return np.array([d], dtype=float)

    # --- 14. Hold / Tap / Swipe / Long-press Duration ---
    if feature_name == "Touch Hold Duration":
        if strokes.empty:
            return np.array([], dtype=float)
        return strokes["duration"].values.astype(float)
    if feature_name == "Tap Duration":
        tap_st = strokes_of_type("tap")
        return tap_st["duration"].values.astype(float) if not tap_st.empty else np.array([], dtype=float)
    if feature_name == "Swipe Duration":
        sw = strokes_of_type("swipe")
        return sw["duration"].values.astype(float) if not sw.empty else np.array([], dtype=float)
    if feature_name == "Long-press Duration":
        lp = strokes_of_type("longpress")
        return lp["duration"].values.astype(float) if not lp.empty else np.array([], dtype=float)

    # --- 15. Inter- interval 들 ---
    def inter_intervals(sub: pd.DataFrame) -> np.ndarray:
        if sub.empty:
            return np.array([], dtype=float)
        sub = sub.sort_values("t_start")
        t_down = sub["t_start"].values.astype(float)
        t_up = sub["t_end"].values.astype(float)
        if len(t_down) < 2:
            return np.array([], dtype=float)
        return t_down[1:] - t_up[:-1]

    if feature_name == "Inter-touch Interval":
        return inter_intervals(strokes)
    if feature_name == "Inter-tap Interval":
        return inter_intervals(strokes_of_type("tap"))
    if feature_name == "Inter-swipe Interval":
        return inter_intervals(strokes_of_type("swipe"))
    if feature_name == "Inter-Long-press Interval":
        return inter_intervals(strokes_of_type("longpress"))

    # --- 16. Active Touch Time ---
    if feature_name == "Active Touch Time":
        if strokes.empty:
            return np.array([], dtype=float)
        total_active = float(strokes["duration"].sum())
        return np.array([total_active], dtype=float)

        # --- 17. Touch Stroke Pressure Slope ---
    if feature_name == "Touch Stroke Pressure Slope":
        # stroke_id 가 있는 row만 사용
        df_s = df_g[df_g["stroke_id"].notna()].copy()
        if df_s.empty:
            return np.array([], dtype=float)

        slopes = []
        for _, g in df_s.groupby("stroke_id"):
            # 시간, 압력 시퀀스 추출
            t_s = g["t_rel"].values.astype(float)
            p_s = g["p"].values.astype(float)

            # 1) NaN + inf 모두 제거
            mask = np.isfinite(t_s) & np.isfinite(p_s)
            if np.count_nonzero(mask) < 2:
                continue

            t_valid = t_s[mask]
            p_valid = p_s[mask]

            # 2) 시간값이 전부 같으면 기울기 정의 불가
            if np.unique(t_valid).size < 2:
                continue

            # 3) polyfit 자체도 방어적으로 감싸기
            try:
                coeffs = np.polyfit(t_valid, p_valid, 1)
            except np.linalg.LinAlgError:
                # SVD가 수렴 안 되는 stroke는 버림
                continue
            except FloatingPointError:
                continue

            slopes.append(coeffs[0])

        return np.array(slopes, dtype=float)



    # --- 18. Tap/Swipe/Long-press Pressure ---
    if feature_name in ["Tap Pressure", "Swipe Pressure", "Long-press Pressure"]:
        if "gesture_type" not in df_g.columns:
            return np.array([], dtype=float)
        if feature_name == "Tap Pressure":
            mask = df_g["gesture_type"] == "tap"
        elif feature_name == "Swipe Pressure":
            mask = df_g["gesture_type"] == "swipe"
        else:
            mask = df_g["gesture_type"] == "longpress"
        vals = df_g.loc[mask, "p"].values.astype(float)
        return vals[~np.isnan(vals)]

    # --- 19. Touch Entropy ---
    if feature_name == "Touch Entropy":
        pts = np.column_stack([x, y])
        pts = pts[~np.isnan(pts).any(axis=1)]
        if len(pts) == 0:
            return np.array([], dtype=float)
        xs = pts[:, 0]
        ys = pts[:, 1]
        x_min, x_max = np.nanmin(xs), np.nanmax(xs)
        y_min, y_max = np.nanmin(ys), np.nanmax(ys)
        if x_min == x_max or y_min == y_max:
            return np.array([], dtype=float)
        s = TOUCH_ENTROPY_GRID_SIZE
        gx = ((xs - x_min) / (x_max - x_min) * s).astype(int)
        gy = ((ys - y_min) / (y_max - y_min) * s).astype(int)
        gx = np.clip(gx, 0, s - 1)
        gy = np.clip(gy, 0, s - 1)
        grid = np.zeros((s, s), dtype=float)
        for i in range(len(gx)):
            grid[gy[i], gx[i]] += 1.0
        total = grid.sum()
        if total <= 0:
            return np.array([], dtype=float)
        p_b = grid.flatten() / total
        p_b = p_b[p_b > 0]
        H = -np.sum(p_b * np.log(p_b + 1e-12))
        return np.array([H], dtype=float)

    # --- 20. Screen 위치 비율 & Dispersion & Covariance ---
    if feature_name == "Edge Touch Ratio":
        if N_event == 0:
            return np.array([], dtype=float)
        margin_x = params.screen_width_px * 0.1
        margin_y = params.screen_height_px * 0.1
        in_edge = np.sum(
            (x < margin_x)
            | (x > params.screen_width_px - margin_x)
            | (y < margin_y)
            | (y > params.screen_height_px - margin_y)
        )
        return np.array([in_edge / N_event], dtype=float)

    if feature_name == "Left-half Touch Ratio":
        if N_event == 0:
            return np.array([], dtype=float)
        in_left = np.sum(x < params.screen_width_px / 2.0)
        return np.array([in_left / N_event], dtype=float)

    if feature_name == "Right-half Touch Ratio":
        if N_event == 0:
            return np.array([], dtype=float)
        in_right = np.sum(x >= params.screen_width_px / 2.0)
        return np.array([in_right / N_event], dtype=float)

    if feature_name == "Top-half Touch Ratio":
        if N_event == 0:
            return np.array([], dtype=float)
        in_top = np.sum(y < params.screen_height_px / 2.0)
        return np.array([in_top / N_event], dtype=float)

    if feature_name == "Bottom-half Touch Ratio":
        if N_event == 0:
            return np.array([], dtype=float)
        in_bottom = np.sum(y >= params.screen_height_px / 2.0)
        return np.array([in_bottom / N_event], dtype=float)

    if feature_name == "Central Touch Ratio":
        if N_event == 0:
            return np.array([], dtype=float)
        x_min_c = params.screen_width_px / 3.0
        x_max_c = 2 * params.screen_width_px / 3.0
        y_min_c = params.screen_height_px / 3.0
        y_max_c = 2 * params.screen_height_px / 3.0
        in_center = np.sum(
            (x >= x_min_c) & (x <= x_max_c) & (y >= y_min_c) & (y <= y_max_c)
        )
        return np.array([in_center / N_event], dtype=float)

    if feature_name == "Touch Offset to Screen Center":
        cx = params.screen_width_px / 2.0
        cy = params.screen_height_px / 2.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return dist

    if feature_name == "Central Bias":
        if len(x) == 0:
            return np.array([], dtype=float)
        cx = params.screen_width_px / 2.0
        cy = params.screen_height_px / 2.0
        mx = np.nanmean(x)
        my = np.nanmean(y)
        dist = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)
        return np.array([dist], dtype=float)

    if feature_name == "Range-based Touch Dispersion":
        if len(x) == 0:
            return np.array([], dtype=float)
        rx = np.nanmax(x) - np.nanmin(x)
        ry = np.nanmax(y) - np.nanmin(y)
        return np.array([rx + ry], dtype=float)

    if feature_name == "RMS-based Touch Dispersion":
        if len(x) == 0:
            return np.array([], dtype=float)
        varx = np.nanvar(x)
        vary = np.nanvar(y)
        return np.array([np.sqrt(varx + vary)], dtype=float)

    if feature_name == "Bbox Area-based Touch Dispersion":
        if len(x) == 0:
            return np.array([], dtype=float)
        rx = np.nanmax(x) - np.nanmin(x)
        ry = np.nanmax(y) - np.nanmin(y)
        return np.array([rx * ry], dtype=float)

    if feature_name == "Touch Covariance":
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) < 2:
            return np.array([], dtype=float)
        cov = np.cov(x[mask], y[mask])[0, 1]
        return np.array([cov], dtype=float)

    # --- 21. Rhythm (Velocity / Acceleration) ---
    if feature_name == "Velocity Rhythm Rate":
        f_dom = vel_rhythm["f_dom"]
        return np.array([f_dom], dtype=float) if f_dom is not None else np.array([], dtype=float)

    if feature_name == "Velocity Rhythm Spectral Entropy":
        val = vel_rhythm["entropy"]
        return np.array([val], dtype=float) if val is not None else np.array([], dtype=float)

    if feature_name == "Velocity Rhythm Regularity":
        val = vel_rhythm["regularity"]
        return np.array([val], dtype=float) if val is not None else np.array([], dtype=float)

    if feature_name == "Acceleration Rhythm Rate":
        f_dom = acc_rhythm["f_dom"]
        return np.array([f_dom], dtype=float) if f_dom is not None else np.array([], dtype=float)

    if feature_name == "Acceleration Rhythm Spectral Entropy":
        val = acc_rhythm["entropy"]
        return np.array([val], dtype=float) if val is not None else np.array([], dtype=float)

    if feature_name == "Acceleration Rhythm Regularity":
        val = acc_rhythm["regularity"]
        return np.array([val], dtype=float) if val is not None else np.array([], dtype=float)

    return np.array([], dtype=float)


# =========================
# Window 단위 PI 계산
# =========================

def compute_pi_for_window(
    df_win: pd.DataFrame,
    pi_specs: List[PISpec],
    params: WindowParams,
    window_start: float,
    window_end: float,
) -> Dict[str, Optional[float]]:
    df_win = assign_stroke_ids(df_win)
    df_win = compute_velocity_and_acceleration(df_win)
    strokes = compute_stroke_metrics(df_win)
    strokes = classify_strokes(strokes)

    gtype_map = {row["stroke_id"]: row["gesture_type"] for _, row in strokes.iterrows()}
    df_win["gesture_type"] = df_win["stroke_id"].map(gtype_map)

    seq = compute_basic_touch_sequences(df_win, params)

    t = seq["t_rel"]
    if len(t) > 1:
        dt = np.diff(t)
        dt = dt[dt > 0]
        fs = 1.0 / float(np.mean(dt)) if len(dt) > 0 else np.nan
    else:
        fs = np.nan

    if not np.isnan(fs) and fs > 0:
        vel_rhythm = compute_fft_rhythm_features(seq["v"], fs)
        acc_rhythm = compute_fft_rhythm_features(seq["a"], fs)
    else:
        vel_rhythm = {"f_dom": None, "entropy": None, "regularity": None}
        acc_rhythm = {"f_dom": None, "entropy": None, "regularity": None}

    base_stats_cache: Dict[str, Dict[str, Optional[float]]] = {}
    pi: Dict[str, Optional[float]] = {}

    for spec in pi_specs:
        fname = spec.feature_name
        if fname not in base_stats_cache:
            base_series = compute_base_series_for_feature(
                fname,
                df_win,
                seq,
                strokes,
                params=params,
                window_start=window_start,
                window_end=window_end,
                vel_rhythm=vel_rhythm,
                acc_rhythm=acc_rhythm,
            )
            base_stats_cache[fname] = nan_stats(base_series)
        stats = base_stats_cache[fname]
        pi[spec.key] = stats.get(spec.stat, None)

    return pi


# =========================
# JSON & path handling
# =========================

def format_ts_like_input(ts: pd.Timestamp) -> str:
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    base = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")
    off = ts.strftime("%z")  # +0900 형태
    if len(off) == 5 and off.endswith("00"):
        off = off[:-2]  # +09
    elif len(off) == 5:
        off = off[:3] + ":" + off[3:]
    return base + off

from pathlib import Path

def infer_hospital_root_from_csv(csv_path: str) -> Optional[str]:
    """
    /.../pdss/hospital_data/YYYYMMDD/.../stylus.csv
    이런 경로에서 /.../pdss/hospital_data 까지를 root_dir로 추론.
    hospital_data가 없으면 None 반환.
    """
    p = Path(csv_path).resolve()
    parts = p.parts  # ('/', 'data_248', 'pdss', 'hospital_data', '20250925', ...)

    if "hospital_data" not in parts:
        return None

    idx = parts.index("hospital_data")
    # /.../pdss/hospital_data
    root = Path(*parts[: idx + 1])
    return str(root)


def parse_path_components(root_dir: str, csv_path: str) -> Tuple[str, str, str, str]:
    """
    YYYYMMDD/<client_id>/<task>/stylus/<archive_id>/stylus.csv
    에서 (date, client_id, task, archive_id) 추출
    """
    rel = os.path.relpath(csv_path, root_dir)
    parts = rel.split(os.sep)
    lower = [p.lower() for p in parts]
    if "stylus" not in lower:
        raise ValueError(f"Path does not contain 'stylus': {csv_path}")
    sidx = lower.index("stylus")
    date = parts[sidx - 3] if sidx - 3 >= 0 else "unknown_date"
    client_id = parts[sidx - 2] if sidx - 2 >= 0 else "unknown_client"
    task = parts[sidx - 1] if sidx - 1 >= 0 else "unknown_task"
    archive_id = parts[sidx + 1] if sidx + 1 < len(parts) else "unknown_archive"
    return date, client_id, task, archive_id


def make_output_path(
    pdss_root: str,
    client_id: str,
    task: str,
) -> str:
    """
    Output path:
    <pdss_root>/primitive_indicator/<client_id>/<task>/touch_pi.json

    예:
      pdss_root = /data_248/pdss
      -> /data_248/pdss/primitive_indicator/<client_id>/<task>/touch_pi.json
    """
    out_dir = os.path.join(pdss_root, "primitive_indicator", client_id, task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUTPUT_FILENAME)
    return out_path


def build_output_json_for_stylus(
    client_id: str,
    context: str,
    df: pd.DataFrame,
    windows: List[Tuple[int, float, float, pd.DataFrame]],
    pi_specs: List[PISpec],
    params: WindowParams,
) -> dict:
    t0 = df["timestamp"].iloc[0]
    out = {
        "client_id": client_id,
        "context": context,
        "sensor_modality": "touch",
        "screen": {
            "width": float(params.screen_width_px),
            "height": float(params.screen_height_px),
        },
        "windows": [],
    }

    for window_number, start_sec, end_sec, df_win in windows:
        start_ts = t0 + pd.to_timedelta(start_sec, unit="s")
        end_ts = t0 + pd.to_timedelta(end_sec, unit="s")
        pi = compute_pi_for_window(
            df_win,
            pi_specs=pi_specs,
            params=params,
            window_start=start_sec,
            window_end=end_sec,
        )
        out["windows"].append(
            {
                "window_number": window_number,
                "start_timestamp": format_ts_like_input(start_ts),
                "end_timestamp": format_ts_like_input(end_ts),
                "pi": pi,
            }
        )
    return out


def iter_stylus_csv_paths(root_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower() == "stylus.csv":
                out.append(os.path.join(dirpath, fn))
    return out


def process_one_stylus_csv(
    csv_path: str,
    root_dir: Optional[str],
    pi_specs: List[PISpec],
    params: WindowParams,
    output_base_path: Optional[str] = None,
) -> Optional[str]:
    p = Path(csv_path)
    try:
        if root_dir is not None:
            date, client_id, task, archive_id = parse_path_components(root_dir, csv_path)
        else:
            # root_dir 없이 단일 파일 테스트할 때
            date, client_id, task, archive_id = "unknown_date", "unknown_client", p.parent.name, p.parent.parent.name
    except Exception as e:
        print(f"[SKIP] {csv_path} (path parse error: {e})")
        return None

    print(f"[INFO] Processing stylus: {csv_path} (date={date}, client_id={client_id}, task={task})")

    try:
        df = load_stylus_csv(p)
    except Exception as e:
        print(f"[SKIP] {csv_path} (load error: {e})")
        return None

    windows = segment_into_windows(df, params)
    if not windows:
        print(f"[SKIP] {csv_path} (no windows)")
        return None

    out_obj = build_output_json_for_stylus(
        client_id=client_id,
        context=task,
        df=df,
        windows=windows,
        pi_specs=pi_specs,
        params=params,
    )

    # ---- 출력 경로 결정 ----
    if output_base_path is not None:
        # 쉘에서 넘겨준 output_base_path 기준으로 저장
        # 예: /data_248/pdss/primitive_indicator/<client_id>/<task>/touch_pi.json
        out_dir = os.path.join(output_base_path, client_id, task)
        out_path = os.path.join(out_dir, OUTPUT_FILENAME)
        
    elif root_dir is not None:
        # root_dir = /data_248/pdss/hospital_data
        # -> pdss_root = /data_248/pdss
        pdss_root = os.path.dirname(os.path.abspath(root_dir))
        out_path = make_output_path(
            pdss_root=pdss_root,
            client_id=client_id,
            task=task,
        )
    else:
        # hospital_data를 못 찾은 정말 예외적인 경우에만 fallback
        # (이 경우에만 stylus.csv 옆에 저장)
        out_path = str(p.with_name(OUTPUT_FILENAME))


    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[OK]  {csv_path} -> {out_path}")
    return out_path


# =========================
# main
# =========================

def main():
    ap = argparse.ArgumentParser(
        description="Touch primitive indicator extractor"
    )
    ap.add_argument(
        "--base_path",
        help="hospital_data root (e.g., /data_248/pdss/hospital_data). "
             "csv_path 미지정 시 이 경로 아래 모든 stylus.csv 순회.",
    )
    ap.add_argument(
        "--output_base_path",
        help="생성된 touch_pi.json을 저장할 루트 (예: /data_248/pdss/primitive_indicator)",
    )
    ap.add_argument(
        "--csv_path",
        help="단일 stylus.csv만 실험적으로 처리하고 싶을 때 해당 파일 경로 지정",
    )

    ap.add_argument("--window_size_sec", type=float, default=WINDOW_SIZE_SEC,
                    help="window size in seconds (기본 10)")
    ap.add_argument("--stride_sec", type=float, default=STRIDE_SEC,
                    help="window stride in seconds (기본 5)")

    ap.add_argument("--screen_width_px", type=int, default=SCREEN_WIDTH_PX,
                    help="화면 너비(px), 기본 1440")
    ap.add_argument("--screen_height_px", type=int, default=SCREEN_HEIGHT_PX,
                    help="화면 높이(px), 기본 900")

    ap.add_argument("--only_client_id", default=None,
                    help="지정 시 해당 client_id 데이터만 처리")

    args = ap.parse_args()

    if args.output_base_path is None and args.base_path is None:
        raise ValueError("--base_path 와 --output_base_path 는 반드시 지정해야 합니다.")

    params = WindowParams(
        window_size_sec=float(args.window_size_sec),
        stride_sec=float(args.stride_sec),
        screen_width_px=int(args.screen_width_px),
        screen_height_px=int(args.screen_height_px),
    )

    pi_specs = load_touch_pi_specs()

    print(f"[INFO] SCREEN: {params.screen_width_px}x{params.screen_height_px} px")
    print(f"[INFO] window_size={params.window_size_sec}s stride={params.stride_sec}s")

    if args.csv_path is not None:
        # csv_path만 줬을 때도 /pdss/hospital_data 를 자동 추론
        root_dir = args.base_path
        if root_dir is None:
            root_dir = infer_hospital_root_from_csv(args.csv_path)

        process_one_stylus_csv(
            csv_path=args.csv_path,
            root_dir=root_dir,
            pi_specs=pi_specs,
            params=params,
            output_base_path=args.output_base_path,
        )
        return

    # base_path 전체 순회
    root_dir = args.base_path
    csv_paths = iter_stylus_csv_paths(root_dir)
    if not csv_paths:
        print("[DONE] No stylus.csv found.")
        return

    print(f"[INFO] Found {len(csv_paths)} stylus.csv files")

    ok = 0
    for path in csv_paths:
        if args.only_client_id is not None:
            try:
                _, client_id, _, _ = parse_path_components(root_dir, path)
            except Exception:
                continue
            if client_id != args.only_client_id:
                continue

        out = process_one_stylus_csv(
            csv_path=path,
            root_dir=root_dir,
            pi_specs=pi_specs,
            params=params,
            output_base_path=args.output_base_path,
        )
        if out:
            ok += 1

    print(f"[DONE] processed={ok} / found={len(csv_paths)}")


if __name__ == "__main__":
    main()
