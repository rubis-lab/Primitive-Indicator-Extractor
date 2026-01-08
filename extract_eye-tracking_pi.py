"""
extract_eye_pi.py

Compute eye-tracking Primitive Indicators (PIs) from a CSV of gaze samples and
export them to a JSON windowed format.

- Input CSV (required columns):
    timestamp : int/float ms since epoch (or ISO8601 string)
    x         : gaze x in pixels
    y         : gaze y in pixels

- Windowing:
    Sliding windows with user-defined window_size (default 60s) and stride (default 30s).

- PI naming:
    Each PI key is the Excel "Name" lowercased + snake_cased, with "_{mean|std|min|max}" suffix.

Notes / assumptions:
- Fixation vs saccade segmentation uses a simple I-VT velocity threshold on on-screen points.
  Threshold is adaptive: max(BASE_SACCADE_THR_PX_S, median(speed) + MAD_MULT * MAD(speed)).
- Off-screen = (x,y) outside the screen bounds or NaN.
  Most spatial/kinematic features are computed on on-screen samples only;
  on/off-screen ratios are computed using dwell-time.
- Edge / center region definitions are parameterized (EDGE_MARGIN_PCT, CENTER_REGION_PCT).
- "Backtrack saccade" is interpreted as direction reversal: |Δθ - π| < BACKTRACK_TAU_RAD.
- Saccade Direction stats use circular mean/std (angles are periodic).
- IMPORTANT (your request): if a PI yields only ONE value in a window,
  mean/min/max are filled with that value and std is set to None.

Dependencies:
  numpy, pandas
  (optional) scipy for convex hull. If scipy is missing, hull area/perimeter become 0.

Example:
  python extract_eye_pi.py \
    --input_csv sample.csv \
    --output_json out.json \
    --pi_excel "Primitive Indicator.xlsx" \
    --window_size 60 --stride 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from scipy.spatial import ConvexHull  # type: ignore
except Exception:
    ConvexHull = None  # type: ignore

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore


# =========================
# Parameters (tune as needed)
# =========================
DEFAULT_SCREEN_WIDTH = 1440
DEFAULT_SCREEN_HEIGHT = 900

# Windowing
DEFAULT_WINDOW_SIZE_S = 60.0
DEFAULT_STRIDE_S = 30.0

# Gaps / dwell time handling
MAX_GAP_MS = 200.0       # for velocity / segmentation (ignore intervals larger than this)
MAX_DWELL_MS = 200.0     # for region duration accumulation (ignore intervals larger than this)

# Fixation / saccade segmentation (I-VT)
MIN_FIX_MS = 100.0
MIN_SAC_MS = 15.0
BASE_SACCADE_THR_PX_S = 3000.0
MAD_MULT = 6.0

# Entropy / RQA discretization
ENTROPY_BIN_PX = 50.0
RQA_BIN_PX = 50.0
RQA_LMIN = 2
MAX_RQA_POINTS = 2000

# Regions (edge/center)
EDGE_MARGIN_PCT = 0.10
CENTER_REGION_PCT = 0.50  # center rectangle width/height is this fraction of screen

# Direction-related
SACCADE_AXIS_THR_RAD = math.radians(15.0)  # horizontal/vertical classification
BACKTRACK_TAU_RAD = math.radians(20.0)     # for "backtrack" definition


# =========================
# Helpers
# =========================
def keyify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute angular difference (radians) between angles a and b."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return np.abs(d)


def circular_mean(angles: np.ndarray) -> float:
    s = float(np.mean(np.sin(angles)))
    c = float(np.mean(np.cos(angles)))
    return float(math.atan2(s, c))


def circular_std(angles: np.ndarray) -> Optional[float]:
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size <= 1:
        return None
    s = float(np.mean(np.sin(angles)))
    c = float(np.mean(np.cos(angles)))
    R = math.sqrt(s * s + c * c)
    if R <= 0:
        return float(np.pi / math.sqrt(3))
    return float(math.sqrt(-2.0 * math.log(R)))


def stats_from_vec(vec: np.ndarray, circular: bool = False) -> Dict[str, Optional[float]]:
    """
    If vec has only ONE value:
      mean=min=max=value, std=None  (per your request)
    """
    vec = np.asarray(vec, dtype=float)
    vec = vec[np.isfinite(vec)]
    if vec.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}

    if circular:
        # normalize to [-pi, pi]
        v = ((vec + np.pi) % (2 * np.pi)) - np.pi
        if v.size == 1:
            val = float(v[0])
            return {"mean": val, "std": None, "min": val, "max": val}
        m = circular_mean(v)
        sd = circular_std(v)
        return {"mean": float(m), "std": (None if sd is None else float(sd)),
                "min": float(np.min(v)), "max": float(np.max(v))}

    if vec.size == 1:
        val = float(vec[0])
        return {"mean": val, "std": None, "min": val, "max": val}

    m = float(np.mean(vec))
    sd = float(np.std(vec, ddof=1))
    return {"mean": m, "std": sd, "min": float(np.min(vec)), "max": float(np.max(vec))}


def entropy_2d(x: np.ndarray, y: np.ndarray, bin_px: float = ENTROPY_BIN_PX) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return float("nan")
    bx = np.floor(x / bin_px).astype(int)
    by = np.floor(y / bin_px).astype(int)
    pairs = np.stack([bx, by], axis=1)
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def convex_hull_area_perimeter(points_xy: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(points_xy, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3 or ConvexHull is None:
        return 0.0, 0.0
    try:
        hull = ConvexHull(pts)
        # In 2D: hull.volume == area, hull.area == perimeter
        return float(hull.volume), float(hull.area)
    except Exception:
        return 0.0, 0.0


def box_count_fractal_dimension(
    x: np.ndarray,
    y: np.ndarray,
    box_sizes: Tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256),
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return float("nan")

    Ns: List[int] = []
    Es: List[int] = []
    for e in box_sizes:
        bx = np.floor(x / e).astype(int)
        by = np.floor(y / e).astype(int)
        pairs = np.stack([bx, by], axis=1)
        N = np.unique(pairs, axis=0).shape[0]
        if N > 0:
            Ns.append(int(N))
            Es.append(int(e))

    if len(Ns) < 2:
        return float("nan")

    logN = np.log(Ns)
    logInv = np.log([1.0 / e for e in Es])
    slope = float(np.polyfit(logInv, logN, 1)[0])
    return slope


def rqa_revisit_determinism(
    cell_ids: np.ndarray,
    l_min: int = RQA_LMIN,
    max_points: int = MAX_RQA_POINTS,
) -> Tuple[float, float]:
    g = np.asarray(cell_ids)
    if g.size == 0:
        return float("nan"), float("nan")
    if g.size > max_points:
        idx = np.linspace(0, g.size - 1, max_points).astype(int)
        g = g[idx]

    N = int(g.size)
    M = (g[:, None] == g[None, :])
    ones = int(M.sum())
    rr = ones / float(N * N)

    # Determinism: fraction of recurrence points forming diagonal lines (>= l_min)
    num = 0
    for k in range(-(N - 1), N):
        diag = np.diagonal(M, offset=k)
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run >= l_min:
                    num += run
                run = 0
        if run >= l_min:
            num += run

    det = (num / ones) if ones > 0 else float("nan")
    return float(rr), float(det)


def compute_derivatives(v: np.ndarray, dt_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    v and dt_s are interval-level arrays length M (between samples).
    Returns:
      a: acceleration (length M-1) computed as dv / dt_{i+1}
      j: jerk (length M-2) computed as da / dt_{i+2}
    """
    v = np.asarray(v, dtype=float)
    dt_s = np.asarray(dt_s, dtype=float)
    if v.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    dv = np.diff(v)
    dt2 = dt_s[1:]
    a = np.full_like(dv, np.nan, dtype=float)
    m = np.isfinite(dv) & np.isfinite(dt2) & (dt2 > 0)
    a[m] = dv[m] / dt2[m]

    if a.size < 2:
        return a[np.isfinite(a)], np.array([], dtype=float)

    da = np.diff(a)
    dt3 = dt_s[2:]
    j = np.full_like(da, np.nan, dtype=float)
    m2 = np.isfinite(da) & np.isfinite(dt3) & (dt3 > 0)
    j[m2] = da[m2] / dt3[m2]
    return a[np.isfinite(a)], j[np.isfinite(j)]


def in_center_region(x: np.ndarray, y: np.ndarray, screen_w: float, screen_h: float) -> np.ndarray:
    cx = screen_w / 2.0
    cy = screen_h / 2.0
    w = CENTER_REGION_PCT * screen_w
    h = CENTER_REGION_PCT * screen_h
    x0 = cx - w / 2.0
    x1 = cx + w / 2.0
    y0 = cy - h / 2.0
    y1 = cy + h / 2.0
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def in_edge_region(x: np.ndarray, y: np.ndarray, screen_w: float, screen_h: float) -> np.ndarray:
    mx = EDGE_MARGIN_PCT * screen_w
    my = EDGE_MARGIN_PCT * screen_h
    return (x <= mx) | (x >= (screen_w - mx)) | (y <= my) | (y >= (screen_h - my))


@dataclass
class Event:
    kind: str              # 'fixation' or 'saccade'
    start_idx: int         # sample index (on-screen filtered series)
    end_idx: int           # sample index inclusive
    start_ms: int
    end_ms: int
    xs: np.ndarray
    ys: np.ndarray


def segment_fix_sac(
    x: np.ndarray,
    y: np.ndarray,
    t_ms: np.ndarray,
    max_gap_ms: float = MAX_GAP_MS,
    min_fix_ms: float = MIN_FIX_MS,
    min_sac_ms: float = MIN_SAC_MS,
    base_thr_px_s: float = BASE_SACCADE_THR_PX_S,
    mad_multiplier: float = MAD_MULT,
) -> Tuple[List[Event], List[Event], Dict[str, Any]]:
    """
    I-VT segmentation on on-screen samples:
      - build interval velocities
      - mark intervals as saccade if speed > threshold
      - group consecutive intervals into events (break on invalid/gap intervals)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t_ms = np.asarray(t_ms, dtype=np.int64)

    n = int(x.size)
    if n < 2:
        return [], [], {
            "threshold_px_s": float("nan"),
            "speed": np.array([], dtype=float),
            "vx": np.array([], dtype=float),
            "vy": np.array([], dtype=float),
            "dt_s": np.array([], dtype=float),
            "theta": np.array([], dtype=float),
        }

    dt_ms = np.diff(t_ms).astype(float)  # length M
    valid_interval = (dt_ms > 0) & (dt_ms <= max_gap_ms)
    dt_s = dt_ms / 1000.0

    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx * dx + dy * dy)

    M = int(dt_ms.size)
    vx = np.full(M, np.nan, dtype=float)
    vy = np.full(M, np.nan, dtype=float)
    speed = np.full(M, np.nan, dtype=float)
    theta = np.full(M, np.nan, dtype=float)

    vx[valid_interval] = dx[valid_interval] / dt_s[valid_interval]
    vy[valid_interval] = dy[valid_interval] / dt_s[valid_interval]
    speed[valid_interval] = dist[valid_interval] / dt_s[valid_interval]
    theta[valid_interval] = np.arctan2(dy[valid_interval], dx[valid_interval])

    sp_valid = speed[np.isfinite(speed)]
    if sp_valid.size >= 10:
        med = float(np.median(sp_valid))
        mad = float(np.median(np.abs(sp_valid - med)))
        thr = max(base_thr_px_s, med + mad_multiplier * mad)
    elif sp_valid.size > 0:
        thr = max(base_thr_px_s, float(np.median(sp_valid)))
    else:
        thr = base_thr_px_s

    is_saccade_interval = valid_interval & (speed > thr)

    # Segment intervals into events
    segments: List[Tuple[str, int, int]] = []
    cur_label: Optional[str] = None
    cur_start: Optional[int] = None
    for i in range(M):
        if not valid_interval[i]:
            if cur_label is not None:
                segments.append((cur_label, int(cur_start), i - 1))  # close
                cur_label, cur_start = None, None
            continue

        label = "saccade" if bool(is_saccade_interval[i]) else "fixation"
        if cur_label is None:
            cur_label, cur_start = label, i
        elif label != cur_label:
            segments.append((cur_label, int(cur_start), i - 1))
            cur_label, cur_start = label, i

    if cur_label is not None:
        segments.append((cur_label, int(cur_start), M - 1))

    fixations: List[Event] = []
    saccades: List[Event] = []
    for label, i0, i1 in segments:
        s0 = i0
        s1 = i1 + 1  # convert interval indices to sample end index
        start = int(t_ms[s0])
        end = int(t_ms[s1])
        dur = float(end - start)
        if label == "fixation" and dur >= min_fix_ms:
            fixations.append(Event("fixation", s0, s1, start, end, x[s0:s1 + 1], y[s0:s1 + 1]))
        elif label == "saccade" and dur >= min_sac_ms:
            saccades.append(Event("saccade", s0, s1, start, end, x[s0:s1 + 1], y[s0:s1 + 1]))

    meta = {"threshold_px_s": float(thr), "speed": speed, "vx": vx, "vy": vy, "dt_s": dt_s, "theta": theta}
    return fixations, saccades, meta


def parse_timestamp_column(ts: pd.Series) -> np.ndarray:
    """Return timestamps in ms as int64."""
    if np.issubdtype(ts.dtype, np.number):
        return ts.astype("int64").to_numpy()
    dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if dt.isna().any():
        raise ValueError("timestamp column contains non-numeric values that could not be parsed as datetime.")
    # ns -> ms
    return (dt.view("int64") // 1_000_000).astype("int64").to_numpy()


def ms_to_iso(ms: int, tz_name: str = "Asia/Seoul") -> str:
    """
    Format like: 2025-09-18T12:26:20.660000+09
    (matches template style: +09 without ':00')
    """
    if ZoneInfo is not None:
        tz = ZoneInfo(tz_name)
        dt = datetime.fromtimestamp(ms / 1000.0, tz=tz)
        s = dt.isoformat(timespec="microseconds")
        return s.replace("+09:00", "+09")
    tz = timezone(timedelta(hours=9))
    dt = datetime.fromtimestamp(ms / 1000.0, tz=tz)
    s = dt.isoformat(timespec="microseconds")
    return s.replace("+09:00", "+09")


def load_pi_names_from_excel(pi_excel_path: str, sheet_name: str = "eye-tracking") -> List[str]:
    df = pd.read_excel(pi_excel_path, sheet_name=sheet_name)
    names = [str(n) for n in df["Name"].dropna().tolist()]
    # defensively drop if present
    names = [n for n in names if n.strip().lower() != "scanmatch similarity"]
    return names


def compute_window_base_vectors(
    dfw: pd.DataFrame,
    w_start_ms: int,
    w_end_ms: int,
    screen_w: float,
    screen_h: float,
) -> Dict[str, np.ndarray]:
    """
    Compute base vectors (one per PI base name; each is either:
      - a vector of observations, or
      - a length-1 vector for window-level scalar PIs)
    """
    out: Dict[str, np.ndarray] = {}

    if dfw is None or dfw.shape[0] == 0:
        return out

    dfw = dfw.sort_values("timestamp")
    x_all = dfw["x"].to_numpy(dtype=float)
    y_all = dfw["y"].to_numpy(dtype=float)
    t_all = dfw["timestamp"].to_numpy(dtype=np.int64)

    cx = screen_w / 2.0
    cy = screen_h / 2.0

    finite = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(t_all.astype(float))
    on_screen = finite & (x_all >= 0) & (x_all <= screen_w) & (y_all >= 0) & (y_all <= screen_h)

    # Region / on-off durations use dwell time on the ORIGINAL timeline
    if t_all.size >= 2:
        dt_ms = np.diff(t_all).astype(float)
        dwell = np.where((dt_ms > 0) & (dt_ms <= MAX_DWELL_MS), dt_ms, 0.0)

        on_dur = float(np.sum(dwell[on_screen[:-1]]))
        off_dur = float(np.sum(dwell[~on_screen[:-1]]))

        x0 = x_all[:-1]
        y0 = y_all[:-1]
        on_mask = on_screen[:-1]

        edge_dur = float(np.sum(dwell[on_mask & in_edge_region(x0, y0, screen_w, screen_h)]))
        center_dur = float(np.sum(dwell[on_mask & in_center_region(x0, y0, screen_w, screen_h)]))
        left_dur = float(np.sum(dwell[on_mask & (x0 < cx)]))
        right_dur = float(np.sum(dwell[on_mask & (x0 >= cx)]))
        top_dur = float(np.sum(dwell[on_mask & (y0 < cy)]))
        bottom_dur = float(np.sum(dwell[on_mask & (y0 >= cy)]))
    else:
        on_dur = off_dur = edge_dur = center_dur = left_dur = right_dur = top_dur = bottom_dur = float("nan")

    window_dur_ms = float(w_end_ms - w_start_ms)
    window_dur_s = window_dur_ms / 1000.0 if window_dur_ms > 0 else float("nan")

    def scalar(v: float) -> np.ndarray:
        return np.array([v], dtype=float)

    out["on_screen_duration"] = scalar(on_dur)
    out["on_screen_gaze_ratio"] = scalar(on_dur / window_dur_ms if (np.isfinite(on_dur) and window_dur_ms > 0) else float("nan"))
    out["off_screen_duration"] = scalar(off_dur)
    out["off_screen_gaze_ratio"] = scalar(off_dur / window_dur_ms if (np.isfinite(off_dur) and window_dur_ms > 0) else float("nan"))
    out["edge_gaze_ratio"] = scalar(edge_dur / window_dur_ms if (np.isfinite(edge_dur) and window_dur_ms > 0) else float("nan"))
    out["left_half_gaze_ratio"] = scalar(left_dur / window_dur_ms if (np.isfinite(left_dur) and window_dur_ms > 0) else float("nan"))
    out["right_half_gaze_ratio"] = scalar(right_dur / window_dur_ms if (np.isfinite(right_dur) and window_dur_ms > 0) else float("nan"))
    out["top_half_gaze_ratio"] = scalar(top_dur / window_dur_ms if (np.isfinite(top_dur) and window_dur_ms > 0) else float("nan"))
    out["bottom_half_gaze_ratio"] = scalar(bottom_dur / window_dur_ms if (np.isfinite(bottom_dur) and window_dur_ms > 0) else float("nan"))
    out["central_gaze_ratio"] = scalar(center_dur / window_dur_ms if (np.isfinite(center_dur) and window_dur_ms > 0) else float("nan"))

    # Most PIs use on-screen samples only
    xv = x_all[on_screen]
    yv = y_all[on_screen]
    tv = t_all[on_screen]
    if xv.size == 0:
        return out

    out["gaze_position_x"] = xv
    out["gaze_position_y"] = yv

    # Interval-level kinematics between on-screen samples
    if tv.size >= 2:
        dt_ms_v = np.diff(tv).astype(float)
        dx = np.diff(xv)
        dy = np.diff(yv)
        dist = np.sqrt(dx * dx + dy * dy)

        valid_int = (dt_ms_v > 0) & (dt_ms_v <= MAX_GAP_MS)
        dt_s_v = dt_ms_v / 1000.0

        M = int(dt_ms_v.size)
        vx = np.full(M, np.nan, dtype=float)
        vy = np.full(M, np.nan, dtype=float)
        speed = np.full(M, np.nan, dtype=float)
        theta = np.full(M, np.nan, dtype=float)

        vx[valid_int] = dx[valid_int] / dt_s_v[valid_int]
        vy[valid_int] = dy[valid_int] / dt_s_v[valid_int]
        speed[valid_int] = dist[valid_int] / dt_s_v[valid_int]
        theta[valid_int] = np.arctan2(dy[valid_int], dx[valid_int])
    else:
        vx = vy = speed = theta = np.array([], dtype=float)
        dist = np.array([], dtype=float)
        dt_s_v = np.array([], dtype=float)
        valid_int = np.array([], dtype=bool)

    out["gaze_velocity"] = speed[np.isfinite(speed)]
    out["gaze_velocity_x"] = vx[np.isfinite(vx)]
    out["gaze_velocity_y"] = vy[np.isfinite(vy)]

    # Derivatives
    a, j = compute_derivatives(speed, dt_s_v) if speed.size else (np.array([], dtype=float), np.array([], dtype=float))
    ax, jx = compute_derivatives(vx, dt_s_v) if vx.size else (np.array([], dtype=float), np.array([], dtype=float))
    ay, jy = compute_derivatives(vy, dt_s_v) if vy.size else (np.array([], dtype=float), np.array([], dtype=float))

    out["gaze_acceleration"] = a
    out["gaze_acceleration_x"] = ax
    out["gaze_acceleration_y"] = ay
    out["gaze_jerk"] = j
    out["gaze_jerk_x"] = jx
    out["gaze_jerk_y"] = jy

    # Gaze path metrics (scalar per window)
    path_length = float(np.nansum(dist[valid_int])) if dist.size else 0.0
    out["gaze_path_length"] = np.array([path_length], dtype=float)

    disp = float(np.sqrt((xv[-1] - xv[0]) ** 2 + (yv[-1] - yv[0]) ** 2)) if xv.size >= 2 else 0.0
    out["gaze_path_displacement"] = np.array([disp], dtype=float)
    out["gaze_path_curvature"] = np.array([path_length / disp if disp > 0 else float("nan")], dtype=float)

    # Bbox + dispersion (scalar per window)
    xr = float(np.nanmax(xv) - np.nanmin(xv))
    yr = float(np.nanmax(yv) - np.nanmin(yv))
    out["gaze_bbox_width"] = np.array([xr], dtype=float)
    out["gaze_bbox_height"] = np.array([yr], dtype=float)
    out["gaze_bbox_aspect_ratio"] = np.array([xr / yr if yr != 0 else float("nan")], dtype=float)
    out["range_based_gaze_dispersion"] = np.array([xr + yr], dtype=float)
    out["rms_based_gaze_dispersion"] = np.array([float(np.sqrt(np.nanvar(xv, ddof=1) + np.nanvar(yv, ddof=1))) if xv.size > 1 else 0.0], dtype=float)
    out["bbox_area_based_gaze_dispersion"] = np.array([xr * yr], dtype=float)
    out["gaze_covariance"] = np.array([float(np.cov(xv, yv, ddof=1)[0, 1]) if xv.size > 1 else 0.0], dtype=float)
    out["gaze_entropy"] = np.array([entropy_2d(xv, yv)], dtype=float)

    d_center = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)
    out["gaze_offset_to_screen_center"] = d_center
    dmax = float(np.nanmax(d_center)) if d_center.size else float("nan")
    dmean = float(np.nanmean(d_center)) if d_center.size else float("nan")
    out["central_bias"] = np.array([1.0 - dmean / dmax if (np.isfinite(dmean) and np.isfinite(dmax) and dmax > 0) else float("nan")], dtype=float)

    # Fixation/saccade segmentation on on-screen samples
    fixations, saccades, meta = segment_fix_sac(xv, yv, tv)
    out["fixation_rate"] = np.array([len(fixations) / window_dur_s if window_dur_s > 0 else float("nan")], dtype=float)
    out["saccade_rate"] = np.array([len(saccades) / window_dur_s if window_dur_s > 0 else float("nan")], dtype=float)
    out["fixation_saccade_ratio"] = np.array([len(fixations) / len(saccades) if len(saccades) > 0 else float("nan")], dtype=float)

    # Fixation metrics (event-level vectors)
    fix_durs: List[float] = []
    fix_bbox_w: List[float] = []
    fix_bbox_h: List[float] = []
    fix_bbox_ar: List[float] = []
    fix_disp_range: List[float] = []
    fix_disp_rms: List[float] = []
    fix_disp_area: List[float] = []
    fix_ent: List[float] = []
    fix_centroids: List[Tuple[float, float]] = []
    drift_v: List[float] = []
    drift_vx: List[float] = []
    drift_vy: List[float] = []

    for ev in fixations:
        fix_durs.append(float(ev.end_ms - ev.start_ms))
        w = float(np.nanmax(ev.xs) - np.nanmin(ev.xs))
        h = float(np.nanmax(ev.ys) - np.nanmin(ev.ys))
        fix_bbox_w.append(w)
        fix_bbox_h.append(h)
        fix_bbox_ar.append(w / h if h != 0 else float("nan"))
        fix_disp_range.append(w + h)
        fix_disp_rms.append(float(np.sqrt(np.nanvar(ev.xs, ddof=1) + np.nanvar(ev.ys, ddof=1))) if ev.xs.size > 1 else 0.0)
        fix_disp_area.append(w * h)
        fix_ent.append(entropy_2d(ev.xs, ev.ys))
        fix_centroids.append((float(np.nanmean(ev.xs)), float(np.nanmean(ev.ys))))

        i0 = ev.start_idx
        i1 = ev.end_idx - 1
        sp = meta["speed"][i0:i1 + 1]
        vx_ev = meta["vx"][i0:i1 + 1]
        vy_ev = meta["vy"][i0:i1 + 1]
        drift_v.extend(sp[np.isfinite(sp)].tolist())
        drift_vx.extend(vx_ev[np.isfinite(vx_ev)].tolist())
        drift_vy.extend(vy_ev[np.isfinite(vy_ev)].tolist())

    out["fixation_duration"] = np.asarray(fix_durs, dtype=float)
    out["fixation_bbox_width"] = np.asarray(fix_bbox_w, dtype=float)
    out["fixation_bbox_height"] = np.asarray(fix_bbox_h, dtype=float)
    out["fixation_bbox_aspect_ratio"] = np.asarray(fix_bbox_ar, dtype=float)
    out["range_based_fixation_dispersion"] = np.asarray(fix_disp_range, dtype=float)
    out["rms_based_fixation_dispersion"] = np.asarray(fix_disp_rms, dtype=float)
    out["bbox_area_based_fixation_dispersion"] = np.asarray(fix_disp_area, dtype=float)
    out["fixation_entropy"] = np.asarray(fix_ent, dtype=float)
    out["fixational_drift_velocity"] = np.asarray(drift_v, dtype=float)
    out["fixational_drift_velocity_x"] = np.asarray(drift_vx, dtype=float)
    out["fixational_drift_velocity_y"] = np.asarray(drift_vy, dtype=float)

    if fixations:
        first = fixations[0]
        out["first_fixation_latency"] = np.array([float(first.start_ms - w_start_ms)], dtype=float)
        out["first_fixation_duration"] = np.array([float(first.end_ms - first.start_ms)], dtype=float)
    else:
        out["first_fixation_latency"] = np.array([float("nan")], dtype=float)
        out["first_fixation_duration"] = np.array([float("nan")], dtype=float)

    # Fixation centroid dispersion + covariance + convex hull (scalar per window)
    if fix_centroids:
        cxs = np.asarray([p[0] for p in fix_centroids], dtype=float)
        cys = np.asarray([p[1] for p in fix_centroids], dtype=float)
        cxr = float(np.nanmax(cxs) - np.nanmin(cxs))
        cyr = float(np.nanmax(cys) - np.nanmin(cys))
        out["range_based_fixation_centroid_dispersion"] = np.array([cxr + cyr], dtype=float)
        out["rms_based_fixation_centroid_dispersion"] = np.array([float(np.sqrt(np.nanvar(cxs, ddof=1) + np.nanvar(cys, ddof=1))) if cxs.size > 1 else 0.0], dtype=float)
        out["bbox_area_based_fixation_centroid_dispersion"] = np.array([cxr * cyr], dtype=float)
        out["fixation_centroid_covariance"] = np.array([float(np.cov(cxs, cys, ddof=1)[0, 1]) if cxs.size > 1 else 0.0], dtype=float)
        area, per = convex_hull_area_perimeter(np.stack([cxs, cys], axis=1))
        out["fixation_convex_hull_area"] = np.array([area], dtype=float)
        out["fixation_convex_hull_perimeter"] = np.array([per], dtype=float)
    else:
        out["range_based_fixation_centroid_dispersion"] = np.array([float("nan")], dtype=float)
        out["rms_based_fixation_centroid_dispersion"] = np.array([float("nan")], dtype=float)
        out["bbox_area_based_fixation_centroid_dispersion"] = np.array([float("nan")], dtype=float)
        out["fixation_centroid_covariance"] = np.array([float("nan")], dtype=float)
        out["fixation_convex_hull_area"] = np.array([float("nan")], dtype=float)
        out["fixation_convex_hull_perimeter"] = np.array([float("nan")], dtype=float)

    # Inter-fixation interval
    if len(fixations) >= 2:
        ifi = [float(fixations[i + 1].start_ms - fixations[i].end_ms) for i in range(len(fixations) - 1)]
        out["inter_fixation_interval"] = np.asarray(ifi, dtype=float)
    else:
        out["inter_fixation_interval"] = np.array([], dtype=float)

    # Saccade metrics (event-level vectors)
    sac_durs: List[float] = []
    sac_bbox_w: List[float] = []
    sac_bbox_h: List[float] = []
    sac_bbox_ar: List[float] = []
    sac_disp_range: List[float] = []
    sac_disp_rms: List[float] = []
    sac_disp_area: List[float] = []
    sac_ent: List[float] = []
    sac_displacement: List[float] = []
    sac_dir: List[float] = []
    sac_path_len: List[float] = []
    sac_path_curv: List[float] = []
    sac_v: List[float] = []
    sac_vx: List[float] = []
    sac_vy: List[float] = []
    sac_a: List[float] = []
    sac_ax: List[float] = []
    sac_ay: List[float] = []
    sac_j: List[float] = []
    sac_jx: List[float] = []
    sac_jy: List[float] = []
    sac_angvel: List[float] = []
    endpoints: List[Tuple[float, float]] = []

    for ev in saccades:
        sac_durs.append(float(ev.end_ms - ev.start_ms))
        w = float(np.nanmax(ev.xs) - np.nanmin(ev.xs))
        h = float(np.nanmax(ev.ys) - np.nanmin(ev.ys))
        sac_bbox_w.append(w)
        sac_bbox_h.append(h)
        sac_bbox_ar.append(w / h if h != 0 else float("nan"))
        sac_disp_range.append(w + h)
        sac_disp_rms.append(float(np.sqrt(np.nanvar(ev.xs, ddof=1) + np.nanvar(ev.ys, ddof=1))) if ev.xs.size > 1 else 0.0)
        sac_disp_area.append(w * h)
        sac_ent.append(entropy_2d(ev.xs, ev.ys))

        dx_e = float(ev.xs[-1] - ev.xs[0])
        dy_e = float(ev.ys[-1] - ev.ys[0])
        disp_e = float(np.sqrt(dx_e * dx_e + dy_e * dy_e))
        sac_displacement.append(disp_e)
        sac_dir.append(float(np.arctan2(dy_e, dx_e)))

        if ev.xs.size >= 2:
            dxe = np.diff(ev.xs)
            dye = np.diff(ev.ys)
            sac_path_len.append(float(np.nansum(np.sqrt(dxe * dxe + dye * dye))))
        else:
            sac_path_len.append(0.0)
        sac_path_curv.append(sac_path_len[-1] / disp_e if disp_e > 0 else float("nan"))

        endpoints.append((float(ev.xs[-1]), float(ev.ys[-1])))

        i0 = ev.start_idx
        i1 = ev.end_idx - 1
        sp = meta["speed"][i0:i1 + 1]
        vx_ev = meta["vx"][i0:i1 + 1]
        vy_ev = meta["vy"][i0:i1 + 1]
        dt_ev = meta["dt_s"][i0:i1 + 1]

        sac_v.extend(sp[np.isfinite(sp)].tolist())
        sac_vx.extend(vx_ev[np.isfinite(vx_ev)].tolist())
        sac_vy.extend(vy_ev[np.isfinite(vy_ev)].tolist())

        acc, jer = compute_derivatives(sp, dt_ev)
        accx, jerx = compute_derivatives(vx_ev, dt_ev)
        accy, jery = compute_derivatives(vy_ev, dt_ev)
        sac_a.extend(acc.tolist())
        sac_j.extend(jer.tolist())
        sac_ax.extend(accx.tolist())
        sac_jx.extend(jerx.tolist())
        sac_ay.extend(accy.tolist())
        sac_jy.extend(jery.tolist())

        # Angular velocity
        th = meta["theta"][i0:i1 + 1]
        m_th = np.isfinite(th)
        if np.count_nonzero(m_th) >= 2:
            th2 = th[m_th]
            th_un = np.unwrap(th2)
            dt_th = dt_ev[m_th][1:]
            dth = np.diff(th_un)
            m_ok = np.isfinite(dt_th) & (dt_th > 0)
            sac_angvel.extend((dth[m_ok] / dt_th[m_ok]).tolist())

    out["saccade_duration"] = np.asarray(sac_durs, dtype=float)
    out["saccade_bbox_width"] = np.asarray(sac_bbox_w, dtype=float)
    out["saccade_bbox_height"] = np.asarray(sac_bbox_h, dtype=float)
    out["saccade_bbox_aspect_ratio"] = np.asarray(sac_bbox_ar, dtype=float)
    out["range_based_saccade_dispersion"] = np.asarray(sac_disp_range, dtype=float)
    out["rms_based_saccade_dispersion"] = np.asarray(sac_disp_rms, dtype=float)
    out["bbox_area_based_saccade_dispersion"] = np.asarray(sac_disp_area, dtype=float)
    out["saccade_entropy"] = np.asarray(sac_ent, dtype=float)
    out["saccade_displacement"] = np.asarray(sac_displacement, dtype=float)
    out["saccade_velocity"] = np.asarray(sac_v, dtype=float)
    out["saccade_velocity_x"] = np.asarray(sac_vx, dtype=float)
    out["saccade_velocity_y"] = np.asarray(sac_vy, dtype=float)
    out["saccade_acceleration"] = np.asarray(sac_a, dtype=float)
    out["saccade_acceleration_x"] = np.asarray(sac_ax, dtype=float)
    out["saccade_acceleration_y"] = np.asarray(sac_ay, dtype=float)
    out["saccade_jerk"] = np.asarray(sac_j, dtype=float)
    out["saccade_jerk_x"] = np.asarray(sac_jx, dtype=float)
    out["saccade_jerk_y"] = np.asarray(sac_jy, dtype=float)
    out["saccade_direction"] = np.asarray(sac_dir, dtype=float)
    out["saccade_angular_velocity"] = np.asarray(sac_angvel, dtype=float)
    out["saccade_path_length"] = np.asarray(sac_path_len, dtype=float)
    out["saccade_path_curvature"] = np.asarray(sac_path_curv, dtype=float)

    if saccades:
        first = saccades[0]
        out["first_saccade_latency"] = np.array([float(first.start_ms - w_start_ms)], dtype=float)
        out["first_saccade_duration"] = np.array([float(first.end_ms - first.start_ms)], dtype=float)
    else:
        out["first_saccade_latency"] = np.array([float("nan")], dtype=float)
        out["first_saccade_duration"] = np.array([float("nan")], dtype=float)

    if len(saccades) >= 2:
        isi = [float(saccades[i + 1].start_ms - saccades[i].end_ms) for i in range(len(saccades) - 1)]
        out["inter_saccade_interval"] = np.asarray(isi, dtype=float)
    else:
        out["inter_saccade_interval"] = np.array([], dtype=float)

    # Horizontal/vertical saccade rates (scalar per window)
    if sac_dir:
        dirs = np.asarray(sac_dir, dtype=float)
        hmask = (angular_diff(dirs, 0.0) < SACCADE_AXIS_THR_RAD) | (angular_diff(dirs, np.pi) < SACCADE_AXIS_THR_RAD)
        vmask = (angular_diff(dirs, np.pi / 2) < SACCADE_AXIS_THR_RAD) | (angular_diff(dirs, -np.pi / 2) < SACCADE_AXIS_THR_RAD)
        out["horizontal_saccade_rate"] = np.array([int(hmask.sum()) / window_dur_s if window_dur_s > 0 else float("nan")], dtype=float)
        out["vertical_saccade_rate"] = np.array([int(vmask.sum()) / window_dur_s if window_dur_s > 0 else float("nan")], dtype=float)
    else:
        out["horizontal_saccade_rate"] = np.array([float("nan")], dtype=float)
        out["vertical_saccade_rate"] = np.array([float("nan")], dtype=float)

    # Endpoint dispersion + covariance (scalar per window)
    if endpoints:
        ex = np.asarray([p[0] for p in endpoints], dtype=float)
        ey = np.asarray([p[1] for p in endpoints], dtype=float)
        exr = float(np.nanmax(ex) - np.nanmin(ex))
        eyr = float(np.nanmax(ey) - np.nanmin(ey))
        out["range_based_saccade_endpoint_dispersion"] = np.array([exr + eyr], dtype=float)
        out["rms_based_saccade_endpoint_dispersion"] = np.array([float(np.sqrt(np.nanvar(ex, ddof=1) + np.nanvar(ey, ddof=1))) if ex.size > 1 else 0.0], dtype=float)
        out["bbox_area_based_saccade_endpoint_dispersion"] = np.array([exr * eyr], dtype=float)
        out["saccade_endpoint_covariance"] = np.array([float(np.cov(ex, ey, ddof=1)[0, 1]) if ex.size > 1 else 0.0], dtype=float)
    else:
        out["range_based_saccade_endpoint_dispersion"] = np.array([float("nan")], dtype=float)
        out["rms_based_saccade_endpoint_dispersion"] = np.array([float("nan")], dtype=float)
        out["bbox_area_based_saccade_endpoint_dispersion"] = np.array([float("nan")], dtype=float)
        out["saccade_endpoint_covariance"] = np.array([float("nan")], dtype=float)

    # Scanpath metrics (scalar per window)
    out["scanpath_fractal_dimension"] = np.array([box_count_fractal_dimension(xv, yv)], dtype=float)

    bx = np.floor(xv / RQA_BIN_PX).astype(int)
    by = np.floor(yv / RQA_BIN_PX).astype(int)
    cell_ids = bx * 100000 + by
    rr, det = rqa_revisit_determinism(cell_ids)
    out["scanpath_revisit_ratio"] = np.array([rr], dtype=float)
    out["scanpath_determinism_ratio"] = np.array([det], dtype=float)

    # Backtrack saccades (scalar per window)
    if len(sac_dir) >= 2:
        dirs = np.asarray(sac_dir, dtype=float)
        diffs = angular_diff(dirs[1:], dirs[:-1])
        back = np.abs(diffs - np.pi) < BACKTRACK_TAU_RAD
        n_back = int(back.sum())
        total_sac_dur_s = float(np.sum(np.asarray(sac_durs, dtype=float)) / 1000.0) if sac_durs else 0.0
        out["backtrack_saccade_ratio"] = np.array([n_back / len(sac_dir) if len(sac_dir) > 0 else float("nan")], dtype=float)
        out["backtrack_saccade_rate"] = np.array([n_back / total_sac_dur_s if total_sac_dur_s > 0 else float("nan")], dtype=float)
    else:
        out["backtrack_saccade_ratio"] = np.array([float("nan")], dtype=float)
        out["backtrack_saccade_rate"] = np.array([float("nan")], dtype=float)

    return out


def build_pi_stats(base_vecs: Dict[str, np.ndarray], pi_names: List[str]) -> Dict[str, Any]:
    """
    Convert base vectors into the final pi dict with 4 stats per PI.
    """
    pi: Dict[str, Any] = {}
    for name in pi_names:
        base_key = keyify(name)
        vec = base_vecs.get(base_key, np.array([], dtype=float))
        circ = (base_key == "saccade_direction")
        st = stats_from_vec(vec, circular=circ)
        for stat in ("mean", "std", "min", "max"):
            v = st[stat]
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                pi[f"{base_key}_{stat}"] = None
            else:
                pi[f"{base_key}_{stat}"] = float(v)
    return pi


def iter_windows_partial(t_min_ms: int, t_max_ms: int, window_ms: int, stride_ms: int) -> List[Tuple[int, int, int]]:
    """
    stride로 start를 이동시키면서,
    end = min(start + window_ms, t_max_ms) 로 partial window를 허용.

    단, 여기서는 "구간에 샘플이 있는지"는 체크하지 않는다.
    (샘플 체크는 main/process_one_file에서 dfw로 확인하는 게 정확함)
    """
    windows: List[Tuple[int, int, int]] = []
    w = 1
    start = int(t_min_ms)

    # start가 t_max보다 작기만 하면 윈도우 후보 생성
    while start < t_max_ms:
        end = min(start + window_ms, int(t_max_ms))
        if end > start:  # 안전장치
            windows.append((w, start, end))
            w += 1
        start += stride_ms

    return windows


def process_one_file(
    input_csv: str,
    output_json: str,
    client_id: str,
    context: str,
    sensor_modality: str,
    timezone_name: str,
    screen_w: float,
    screen_h: float,
    pi_excel: str,
    pi_sheet: str,
    window_size_s: float,
    stride_s: float,
) -> None:
    df = pd.read_csv(input_csv)
    for col in ("timestamp", "x", "y"):
        if col not in df.columns:
            raise ValueError(f"[{input_csv}] Missing required column: {col}")

    # Normalize timestamp column to ms int
    df["timestamp"] = parse_timestamp_column(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Load PI names from Excel (fallback: if excel missing -> use computed base keys)
    if pi_excel and os.path.exists(pi_excel):
        pi_names = load_pi_names_from_excel(pi_excel, sheet_name=pi_sheet)
    else:
        pi_names = []
        print(f"[WARN] PI Excel not found at '{pi_excel}'. Will auto-use computed base keys per window. (file={input_csv})")

    window_ms = int(round(window_size_s * 1000.0))
    stride_ms = int(round(stride_s * 1000.0))

    t_min = int(df["timestamp"].min())
    t_max = int(df["timestamp"].max())
    windows = iter_windows_partial(t_min, t_max, window_ms, stride_ms)

    out_obj: Dict[str, Any] = {
        "client_id": client_id,
        "context": context,
        "sensor_modality": sensor_modality,
        "screen": {"width": screen_w, "height": screen_h},
        "windows": [],
    }

    for wnum, w_start, w_end in windows:
        dfw = df[(df["timestamp"] >= w_start) & (df["timestamp"] < w_end)]
        if dfw.shape[0] == 0:
            continue
        base_vecs = compute_window_base_vectors(
            dfw=dfw,
            w_start_ms=w_start,
            w_end_ms=w_end,
            screen_w=screen_w,
            screen_h=screen_h,
        )

        # If Excel PI list missing, derive keys from this window's base_vecs
        names_this = pi_names or sorted(base_vecs.keys())
        pi = build_pi_stats(base_vecs, names_this)

        out_obj["windows"].append(
            {
                "window_number": int(wnum),
                "start_timestamp": ms_to_iso(int(w_start), tz_name=timezone_name),
                "end_timestamp": ms_to_iso(int(w_end), tz_name=timezone_name),
                "pi": pi,
            }
        )

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[OK] {client_id}/{context} -> {output_json}  (windows={len(windows)})")



def run_batch_mode(args) -> None:
    """
    base_path 아래를 순회하면서
    {base_path}/{date}/{client_id}/{task}/eye-tracking/{archive_id}/eye-tracking.csv
    를 찾아서, 각각을
    {output_base_path}/{client_id}/{task}/eye-tracking_pi.json
    으로 저장.
    """
    base_path = os.path.abspath(args.base_path)
    out_base = os.path.abspath(args.output_base_path)
    client_filter = set(args.clients) if args.clients else None

    n_files = 0

    for root, dirs, files in os.walk(base_path):
        if "eye-tracking.csv" not in files:
            continue

        csv_path = os.path.join(root, "eye-tracking.csv")
        # base_path 기준 상대 경로
        rel = os.path.relpath(csv_path, base_path)
        parts = rel.split(os.sep)

        # 기대 구조: date/client_id/task/eye-tracking/archive_id/eye-tracking.csv
        # parts = [date, client_id, task, "eye-tracking", archive_id, "eye-tracking.csv"]
        if len(parts) < 6:
            print(f"[WARN] Unexpected path structure, skip: {csv_path}")
            continue

        date_str = parts[0]
        client_id = parts[1]
        task = parts[2]
        modality_dir = parts[3]

        if modality_dir != "eye-tracking":
            print(f"[WARN] modality dir is not 'eye-tracking' (got '{modality_dir}'), skip: {csv_path}")
            continue

        # 클라이언트 필터링 (테스트용)
        if client_filter is not None and client_id not in client_filter:
            # print(f"[SKIP] client_id {client_id} not in filter")
            continue

        # 출력 경로: {output_base_path}/{client_id}/{task}/eye-tracking_pi.json
        out_dir = os.path.join(out_base, client_id, task)
        output_json = os.path.join(out_dir, "eye-tracking_pi.json")

        print(f"[INFO] Processing: csv={csv_path} -> json={output_json} (client={client_id}, task={task}, date={date_str})")

        process_one_file(
            input_csv=csv_path,
            output_json=output_json,
            client_id=client_id,
            context=task,  # task를 곧바로 context로 사용
            sensor_modality="eye-tracking",
            timezone_name=args.timezone,
            screen_w=float(args.screen_width),
            screen_h=float(args.screen_height),
            pi_excel=args.pi_excel,
            pi_sheet=args.pi_sheet,
            window_size_s=args.window_size,
            stride_s=args.stride,
        )

        n_files += 1

    print(f"[DONE] Batch processed {n_files} eye-tracking.csv files under {base_path}")



def main() -> None:
    ap = argparse.ArgumentParser()

    # (A) 단일 파일 모드용
    ap.add_argument("--input_csv", help="Single input CSV (eye-tracking)")
    ap.add_argument("--output_json", help="Single output JSON path")

    # (B) 배치 모드용
    ap.add_argument("--base_path", help="Base path containing date/client/task/.../eye-tracking.csv")
    ap.add_argument("--output_base_path", help="Base path to write JSONs into")

    # 공통 옵션
    ap.add_argument("--pi_excel", default="Primitive Indicator_eye.xlsx")
    ap.add_argument("--pi_sheet", default="eye-tracking")

    ap.add_argument("--window_size", type=float, default=DEFAULT_WINDOW_SIZE_S, help="seconds")
    ap.add_argument("--stride", type=float, default=DEFAULT_STRIDE_S, help="seconds")

    # 단일 모드에서만 직접 주입하는 메타데이터 (배치 모드에서는 경로에서 뽑음)
    ap.add_argument("--client_id", default="client_001")
    ap.add_argument("--context", default="TASK_X")
    ap.add_argument("--sensor_modality", default="eye-tracking")
    ap.add_argument("--timezone", default="Asia/Seoul")

    ap.add_argument("--screen_width", type=float, default=DEFAULT_SCREEN_WIDTH)
    ap.add_argument("--screen_height", type=float, default=DEFAULT_SCREEN_HEIGHT)

    # ★ 특정 client만 돌리고 싶을 때: --clients clientA clientB ...
    ap.add_argument(
        "--clients",
        nargs="*",
        help="If given, only process these client_ids in batch mode",
    )

    args = ap.parse_args()

    # --- 모드 결정 ---

    # 1) 단일 파일 모드: input_csv + output_json 이 둘 다 주어지면 이 모드
    if args.input_csv and args.output_json:
        process_one_file(
            input_csv=args.input_csv,
            output_json=args.output_json,
            client_id=args.client_id,
            context=args.context,
            sensor_modality=args.sensor_modality,
            timezone_name=args.timezone,
            screen_w=float(args.screen_width),
            screen_h=float(args.screen_height),
            pi_excel=args.pi_excel,
            pi_sheet=args.pi_sheet,
            window_size_s=args.window_size,
            stride_s=args.stride,
        )
        return

    # 2) 배치 모드: base_path + output_base_path 가 있어야 함
    if args.base_path and args.output_base_path:
        run_batch_mode(args)
        return

    # 둘 다 아니면 에러
    raise SystemExit(
        "You must either:\n"
        "  (a) provide --input_csv and --output_json for single-file mode, OR\n"
        "  (b) provide --base_path and --output_base_path for batch mode."
    )


if __name__ == "__main__":
    main()
