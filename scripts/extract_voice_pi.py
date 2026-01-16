"""
Voice Primitive Indicator Extractor

Input directory layout example:
  /data_248/pdss/hospital_data/YYYYMMDD/<client_id>/<task>/voice/<uuid>/voice.wav

Output directory:
  <output_base_path>/<client_id>/<task>/voice_pi.json
  default output_base_path = <parent_of_hospital_data>/primitive_indicator

Notes:
- If multiple voice.wav exist for the same (client_id, task), this path will be overwritten.
  If you need to avoid overwrites, include date/uuid in the filename or create deeper dirs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.signal import find_peaks

try:
    import soundfile as sf
except ImportError as e:
    raise ImportError(
        "soundfile is required for voice PI extraction. "
        "Install it via: pip install soundfile  (or conda install -c conda-forge pysoundfile)"
    ) from e

try:
    import librosa
except ImportError as e:
    raise ImportError(
        "librosa is required for voice PI extraction (pitch-related indicators). "
        "Install it via: pip install librosa  (or conda install -c conda-forge librosa)"
    ) from e

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================
# Parameters
# =========================

# Screen size (default)
SCREEN_WIDTH_PX = 1440
SCREEN_HEIGHT_PX = 900

# Windowing parameters
WINDOW_SIZE_SEC = 60.0
STRIDE_SEC = 30.0

# Voice PI-specific parameters
FRAME_LEN_SEC = 0.025
HOP_SEC = 0.010

# Simple energy VAD thresholding (dynamic)
VAD_RMS_FLOOR = 1e-6
VAD_SPEECH_RMS_PERCENTILE = 75.0
VAD_SPEECH_RMS_MULT = 0.15

# Pause segmentation
PAUSE_MIN_DUR_SEC = 0.20

# Syllable nuclei estimation (for Speech Velocity)
SYLLABLE_MIN_GAP_SEC = 0.08

# Tremor / rhythm analysis
RHYTHM_MIN_HZ = 0.5
RHYTHM_MAX_HZ = 12.0
TREMOR_MIN_HZ = 3.0
TREMOR_MAX_HZ = 12.0

# Pitch extraction bounds
PITCH_FMIN = 50.0
PITCH_FMAX = 500.0
PITCH_REF_HZ = 55.0  # for semitone conversion

OUTPUT_FILENAME = "voice_pi.json"
LOCAL_TZ_NAME = "Asia/Seoul"


# =========================
# Indicator list
# =========================

BASE_INDICATOR_NAMES = [
    "Total Speech Time",
    "Speech Ratio",
    "Total Nonspeech Time",
    "Nonspeech Ratio",
    "Speech Velocity",
    "Pause Duration",
    "Pause Ratio",
    "Pause Rate",
    "Utterance Duration",
    "Utterance Ratio",
    "Utterance Rate",
    "Utterance-Pause Ratio",
    "Loudness Level",
    "Loudness Range",
    "Loudness Slope",
    "Pitch Level (F0)",
    "Pitch Variability",
    "Pitch Range",
    "Pitch Tremor Extent",
    "Pitch Tremor Rate",
    "Voiced Ratio",
    "Unvoiced Ratio",
    "Voicing Breaks Rate",
    "Envelope Rhythm Rate",
    "Envelope Rhythm Spectral Entropy",
    "Envelope Rhythm Regularity",
    "Amplitude Tremor Extent",
    "Amplitude Tremor Rate",
    "Dynamic Time Warping Similarity",
    "Frequency Domain Similarity",
    "ROUGE-L Similarity",
]
STATS = ("mean", "std", "min", "max")


def to_snake(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("rouge-l", "rouge_l")
    s = s.replace("(f0)", "f0")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


SNAKE_BASES = [to_snake(n) for n in BASE_INDICATOR_NAMES]


# =========================
# Utilities
# =========================

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
    pi: Dict[str, Optional[float]] = {}
    for base in SNAKE_BASES:
        for st in STATS:
            pi[f"{base}_{st}"] = None
    return pi


def fill_pi_stats(pi: Dict[str, Optional[float]], base_snake: str, values: List[float]) -> None:
    st = safe_stats(values)
    for k in STATS:
        pi[f"{base_snake}_{k}"] = st[k]


def format_ts_like_input(ts: pd.Timestamp) -> str:
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    base = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")
    off = ts.strftime("%z")  # +0900
    if len(off) == 5 and off.endswith("00"):
        off = off[:-2]
    elif len(off) == 5:
        off = off[:3] + ":" + off[3:]
    return base + off


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def default_output_root_from_hospital(base_path: str) -> str:
    parent = os.path.dirname(os.path.abspath(base_path))
    return os.path.join(parent, "primitive_indicator")


def make_output_path(output_root: str, client_id: str, task: str) -> str:
    out_dir = os.path.join(output_root, client_id, task)
    ensure_dir(out_dir)
    return os.path.join(out_dir, OUTPUT_FILENAME)


def iter_voice_wav_paths(base_path: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(base_path):
        for fn in filenames:
            if fn.lower() == "voice.wav":
                out.append(os.path.join(dirpath, fn))
    return out


def parse_path_components(base_path: str, wav_path: str) -> Tuple[str, str, str, str]:
    """
    Return (date, client_id, task, uuid) inferred from:
      YYYYMMDD/<client_id>/<task>/voice/<uuid>/voice.wav
    """
    rel = os.path.relpath(wav_path, base_path)
    parts = rel.split(os.sep)
    lower = [p.lower() for p in parts]

    if "voice" not in lower:
        raise ValueError(f"Path does not contain 'voice': {wav_path}")
    vidx = lower.index("voice")

    date = parts[vidx - 3] if vidx - 3 >= 0 else "unknown_date"
    client_id = parts[vidx - 2] if vidx - 2 >= 0 else "unknown_client"
    task = parts[vidx - 1] if vidx - 1 >= 0 else "unknown_task"
    uuid = parts[vidx + 1] if vidx + 1 < len(parts) else "unknown_uuid"
    return date, client_id, task, uuid


# =========================
# Audio IO + timing
# =========================

def read_wav_mono_float32(wav_path: str) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(wav_path, always_2d=False)
    y = np.asarray(y)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.dtype.kind in ("i", "u"):
        maxv = np.iinfo(y.dtype).max
        y = y.astype(np.float32) / float(maxv)
    else:
        y = y.astype(np.float32)
    return y, int(sr)


def infer_file_start_end_timestamp(wav_path: str, duration_sec: float) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Use mtime as approximate file end timestamp, and back-calculate start = end - duration.
    """
    mtime = os.path.getmtime(wav_path)
    if ZoneInfo is not None:
        tz = ZoneInfo(LOCAL_TZ_NAME)
        end_dt = datetime.fromtimestamp(mtime, tz=tz)
    else:
        end_dt = datetime.fromtimestamp(mtime)

    start_dt = end_dt - timedelta(seconds=float(duration_sec))
    end_ts = pd.Timestamp(end_dt)
    start_ts = pd.Timestamp(start_dt)

    if end_ts.tz is None:
        end_ts = end_ts.tz_localize(LOCAL_TZ_NAME)
        start_ts = start_ts.tz_localize(LOCAL_TZ_NAME)

    return start_ts, end_ts


@dataclass
class WindowParams:
    window_size_sec: float = WINDOW_SIZE_SEC
    stride_sec: float = STRIDE_SEC


def build_windows(start_ts: pd.Timestamp, end_ts: pd.Timestamp, params: WindowParams) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    wsize = pd.Timedelta(seconds=float(params.window_size_sec))
    stride = pd.Timedelta(seconds=float(params.stride_sec))

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    w_start = start_ts
    while w_start < end_ts:
        w_end = w_start + wsize
        windows.append((w_start, w_end))
        w_start = w_start + stride
        if stride <= pd.Timedelta(0):
            break
        if w_start > end_ts and len(windows) > 0:
            break
    return windows


def slice_audio_with_padding(
    y: np.ndarray,
    sr: int,
    file_start_ts: pd.Timestamp,
    w_start: pd.Timestamp,
    window_size_sec: float,
) -> np.ndarray:
    offset_sec = (w_start - file_start_ts).total_seconds()
    start_sample = int(round(offset_sec * sr))
    win_samples = int(round(window_size_sec * sr))

    seg = np.zeros((win_samples,), dtype=np.float32)

    src_start = max(0, start_sample)
    src_end = min(len(y), start_sample + win_samples)

    dst_start = max(0, -start_sample)
    dst_end = dst_start + max(0, src_end - src_start)

    if src_end > src_start and dst_end > dst_start:
        seg[dst_start:dst_end] = y[src_start:src_end].astype(np.float32, copy=False)

    return seg


# =========================
# Frame features + segmentation
# =========================

def frame_signal(y: np.ndarray, sr: int, frame_len_sec: float, hop_sec: float) -> np.ndarray:
    frame_len = int(round(frame_len_sec * sr))
    hop = int(round(hop_sec * sr))
    if frame_len <= 0 or hop <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    n = len(y)
    if n < frame_len:
        pad = np.zeros((frame_len - n,), dtype=np.float32)
        y = np.concatenate([y, pad], axis=0)
        n = len(y)

    num = 1 + (n - frame_len) // hop
    frames = np.zeros((num, frame_len), dtype=np.float32)
    for i in range(num):
        s = i * hop
        frames[i, :] = y[s:s + frame_len]
    return frames


def rms_per_frame(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1) + 1e-12).astype(np.float32)


def vad_speech_mask(rms: np.ndarray) -> np.ndarray:
    if rms.size == 0:
        return np.zeros((0,), dtype=bool)
    ref = np.percentile(rms, VAD_SPEECH_RMS_PERCENTILE)
    thr = max(VAD_RMS_FLOOR, float(ref) * float(VAD_SPEECH_RMS_MULT))
    return rms >= thr


def segments_from_mask(mask: np.ndarray, hop_sec: float) -> List[Tuple[int, int, float]]:
    segs: List[Tuple[int, int, float]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        dur = (j - i) * hop_sec
        segs.append((i, j, dur))
        i = j
    return segs


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if x.size == 0:
        return x
    if win <= 1:
        return x
    win = min(win, x.size)
    k = np.ones((win,), dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)


def dominant_freq_fft(x: np.ndarray, fs: float, fmin: float, fmax: float) -> Optional[float]:
    if x.size < 4:
        return None
    x = x.astype(np.float64)
    x = x - np.mean(x)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)

    valid = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    if not np.any(valid):
        return None
    idx = int(np.argmax(mag[valid]))
    return float(freqs[valid][idx])


def spectral_entropy_fft(x: np.ndarray, fs: float, fmin: float, fmax: float) -> Optional[float]:
    if x.size < 4:
        return None
    x = x.astype(np.float64)
    x = x - np.mean(x)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)

    valid = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    if not np.any(valid):
        return None
    m = mag[valid]
    s = float(np.sum(m))
    if s <= 0.0:
        return None
    p = m / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def rhythm_regularity_fft(x: np.ndarray, fs: float, fmin: float, fmax: float) -> Optional[float]:
    if x.size < 4:
        return None
    x = x.astype(np.float64)
    x = x - np.mean(x)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)

    valid = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    if not np.any(valid):
        return None
    m = mag[valid]
    peak = float(np.max(m))
    base = float(np.mean(m)) + 1e-12
    return float(peak / base)


def semitone(f0_hz: np.ndarray, ref_hz: float) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    f0 = np.clip(f0, 1e-9, None)
    return (12.0 * np.log2(f0 / float(ref_hz))).astype(np.float64)


# =========================
# PI computation per window
# =========================

def compute_voice_pi_for_window(y_win: np.ndarray, sr: int) -> Dict[str, Optional[float]]:
    pi = empty_pi_dict()

    T = float(len(y_win)) / float(sr) if sr > 0 else 0.0
    if T <= 0.0:
        return pi

    frames = frame_signal(y_win, sr, FRAME_LEN_SEC, HOP_SEC)
    rms = rms_per_frame(frames)
    hop_fs = 1.0 / HOP_SEC

    speech_mask = vad_speech_mask(rms)
    nonspeech_mask = ~speech_mask

    T_speech = float(np.sum(speech_mask)) * HOP_SEC
    T_nonspeech = float(np.sum(nonspeech_mask)) * HOP_SEC
    R_speech = T_speech / T
    R_nonspeech = T_nonspeech / T

    fill_pi_stats(pi, to_snake("Total Speech Time"), [T_speech])
    fill_pi_stats(pi, to_snake("Speech Ratio"), [R_speech])
    fill_pi_stats(pi, to_snake("Total Nonspeech Time"), [T_nonspeech])
    fill_pi_stats(pi, to_snake("Nonspeech Ratio"), [R_nonspeech])

    nonspeech_segs = segments_from_mask(nonspeech_mask, HOP_SEC)
    pause_segs = [(s, e, d) for (s, e, d) in nonspeech_segs if d >= PAUSE_MIN_DUR_SEC]
    pause_durations = [d for (_, _, d) in pause_segs]
    T_pause = float(np.sum(pause_durations)) if pause_durations else 0.0
    R_pause = (T_pause / T) if T > 0 else None
    r_pause = (float(len(pause_segs)) / T) if T > 0 else None

    fill_pi_stats(pi, to_snake("Pause Duration"), pause_durations)
    fill_pi_stats(pi, to_snake("Pause Ratio"), [R_pause] if R_pause is not None else [])
    fill_pi_stats(pi, to_snake("Pause Rate"), [r_pause] if r_pause is not None else [])

    pause_frame_mask = np.zeros_like(speech_mask, dtype=bool)
    for s, e, _ in pause_segs:
        pause_frame_mask[s:e] = True

    utterance_durations: List[float] = []
    i = 0
    nF = len(speech_mask)
    while i < nF:
        if pause_frame_mask[i]:
            i += 1
            continue
        j = i + 1
        while j < nF and (not pause_frame_mask[j]):
            j += 1

        region = speech_mask[i:j]
        if np.any(region):
            idxs = np.where(region)[0] + i
            u_start = float(idxs[0]) * HOP_SEC
            u_end = float(idxs[-1] + 1) * HOP_SEC
            utterance_durations.append(max(0.0, u_end - u_start))

        i = j

    T_utt = float(np.sum(utterance_durations)) if utterance_durations else 0.0
    R_utt = (T_utt / T) if T > 0 else None
    r_utt = (float(len(utterance_durations)) / T) if T > 0 else None
    R_utt_pause = (T_utt / T_pause) if T_pause > 0 else None

    fill_pi_stats(pi, to_snake("Utterance Duration"), utterance_durations)
    fill_pi_stats(pi, to_snake("Utterance Ratio"), [R_utt] if R_utt is not None else [])
    fill_pi_stats(pi, to_snake("Utterance Rate"), [r_utt] if r_utt is not None else [])
    fill_pi_stats(pi, to_snake("Utterance-Pause Ratio"), [R_utt_pause] if R_utt_pause is not None else [])

    loud_db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    loud_vals = loud_db[speech_mask].astype(np.float64)
    fill_pi_stats(pi, to_snake("Loudness Level"), loud_vals.tolist())

    if loud_vals.size > 0:
        L_range = float(np.percentile(loud_vals, 95) - np.percentile(loud_vals, 5))
        fill_pi_stats(pi, to_snake("Loudness Range"), [L_range])

        t = (np.arange(loud_vals.size, dtype=np.float64) * HOP_SEC)
        if loud_vals.size >= 2:
            a = np.polyfit(t, loud_vals, 1)[0]
            fill_pi_stats(pi, to_snake("Loudness Slope"), [float(a)])
        else:
            fill_pi_stats(pi, to_snake("Loudness Slope"), [])
    else:
        fill_pi_stats(pi, to_snake("Loudness Range"), [])
        fill_pi_stats(pi, to_snake("Loudness Slope"), [])

    env = rms.astype(np.float64)
    env_sm = moving_average(env, win=max(1, int(round(0.05 / HOP_SEC))))
    env_speech = env_sm.copy()
    env_speech[~speech_mask] = 0.0
    min_dist = max(1, int(round(SYLLABLE_MIN_GAP_SEC / HOP_SEC)))
    peaks, _ = find_peaks(
        env_speech,
        distance=min_dist,
        height=np.percentile(env_speech, 75) if np.any(env_speech > 0) else None,
    )
    N_syll = int(len(peaks))
    S_speech = (float(N_syll) / T_speech) if T_speech > 0 else None
    fill_pi_stats(pi, to_snake("Speech Velocity"), [S_speech] if S_speech is not None else [])

    if sr > 0:
        frame_len = int(round(FRAME_LEN_SEC * sr))
        hop_len = int(round(HOP_SEC * sr))
        try:
            f0 = librosa.yin(
                y_win.astype(np.float32),
                fmin=PITCH_FMIN,
                fmax=PITCH_FMAX,
                sr=sr,
                frame_length=frame_len,
                hop_length=hop_len,
            )
            f0 = np.asarray(f0, dtype=np.float64)
            m = min(len(f0), len(speech_mask))
            f0 = f0[:m]
            sm = speech_mask[:m]
            voiced = np.isfinite(f0) & (f0 > 0) & sm

            f0_level = f0[voiced]
            fill_pi_stats(pi, to_snake("Pitch Level (F0)"), f0_level.tolist())

            if f0_level.size > 0:
                st = semitone(f0_level, PITCH_REF_HZ)
                P_var = float(np.std(st, ddof=0)) if st.size > 0 else None
                P_rng = float(np.percentile(st, 95) - np.percentile(st, 5)) if st.size > 0 else None
                fill_pi_stats(pi, to_snake("Pitch Variability"), [P_var] if P_var is not None else [])
                fill_pi_stats(pi, to_snake("Pitch Range"), [P_rng] if P_rng is not None else [])

                detr_win = max(3, int(round(0.5 / HOP_SEC)))
                trend = moving_average(st.astype(np.float64), detr_win).astype(np.float64)
                resid = st.astype(np.float64) - trend
                PTE = float(np.sqrt(np.mean(resid ** 2))) if resid.size > 0 else None
                PTR = dominant_freq_fft(resid, fs=hop_fs, fmin=TREMOR_MIN_HZ, fmax=TREMOR_MAX_HZ)
                fill_pi_stats(pi, to_snake("Pitch Tremor Extent"), [PTE] if PTE is not None else [])
                fill_pi_stats(pi, to_snake("Pitch Tremor Rate"), [PTR] if PTR is not None else [])

                speech_frames = int(np.sum(sm))
                voiced_frames = int(np.sum(voiced))
                R_voiced = (float(voiced_frames) / float(speech_frames)) if speech_frames > 0 else None
                R_unvoiced = (1.0 - R_voiced) if R_voiced is not None else None
                fill_pi_stats(pi, to_snake("Voiced Ratio"), [R_voiced] if R_voiced is not None else [])
                fill_pi_stats(pi, to_snake("Unvoiced Ratio"), [R_unvoiced] if R_unvoiced is not None else [])

                v_in_speech = voiced[sm]
                if v_in_speech.size >= 2 and T_speech > 0:
                    breaks = int(np.sum((v_in_speech[:-1] == True) & (v_in_speech[1:] == False)))
                    vbr = float(breaks) / float(T_speech)
                    fill_pi_stats(pi, to_snake("Voicing Breaks Rate"), [vbr])
                else:
                    fill_pi_stats(pi, to_snake("Voicing Breaks Rate"), [])
            else:
                fill_pi_stats(pi, to_snake("Pitch Variability"), [])
                fill_pi_stats(pi, to_snake("Pitch Range"), [])
                fill_pi_stats(pi, to_snake("Pitch Tremor Extent"), [])
                fill_pi_stats(pi, to_snake("Pitch Tremor Rate"), [])
                fill_pi_stats(pi, to_snake("Voiced Ratio"), [])
                fill_pi_stats(pi, to_snake("Unvoiced Ratio"), [])
                fill_pi_stats(pi, to_snake("Voicing Breaks Rate"), [])
        except Exception:
            pass

    env2 = env_sm.astype(np.float64)
    env2[~speech_mask] = 0.0

    ER_rate = dominant_freq_fft(env2, fs=hop_fs, fmin=RHYTHM_MIN_HZ, fmax=RHYTHM_MAX_HZ)
    ER_ent = spectral_entropy_fft(env2, fs=hop_fs, fmin=RHYTHM_MIN_HZ, fmax=RHYTHM_MAX_HZ)
    ER_reg = rhythm_regularity_fft(env2, fs=hop_fs, fmin=RHYTHM_MIN_HZ, fmax=RHYTHM_MAX_HZ)

    fill_pi_stats(pi, to_snake("Envelope Rhythm Rate"), [ER_rate] if ER_rate is not None else [])
    fill_pi_stats(pi, to_snake("Envelope Rhythm Spectral Entropy"), [ER_ent] if ER_ent is not None else [])
    fill_pi_stats(pi, to_snake("Envelope Rhythm Regularity"), [ER_reg] if ER_reg is not None else [])

    detr_win = max(3, int(round(0.5 / HOP_SEC)))
    env_trend = moving_average(env2, detr_win).astype(np.float64)
    env_resid = env2 - env_trend

    ATE = float(np.sqrt(np.mean(env_resid ** 2))) if env_resid.size > 0 else None
    ATR = dominant_freq_fft(env_resid, fs=hop_fs, fmin=TREMOR_MIN_HZ, fmax=TREMOR_MAX_HZ)

    fill_pi_stats(pi, to_snake("Amplitude Tremor Extent"), [ATE] if ATE is not None else [])
    fill_pi_stats(pi, to_snake("Amplitude Tremor Rate"), [ATR] if ATR is not None else [])

    fill_pi_stats(pi, to_snake("Dynamic Time Warping Similarity"), [])
    fill_pi_stats(pi, to_snake("Frequency Domain Similarity"), [])
    fill_pi_stats(pi, to_snake("ROUGE-L Similarity"), [])

    return pi


# =========================
# File processing
# =========================

def process_one_voice_wav(
    wav_path: str,
    base_path: str,
    output_root: str,
    params: WindowParams,
    screen_width_px: int,
    screen_height_px: int,
) -> Optional[str]:
    try:
        _, client_id, task, _ = parse_path_components(base_path, wav_path)
        y, sr = read_wav_mono_float32(wav_path)
    except Exception:
        return None

    duration_sec = float(len(y)) / float(sr) if sr > 0 else 0.0
    if duration_sec <= 0.0:
        return None

    file_start_ts, file_end_ts = infer_file_start_end_timestamp(wav_path, duration_sec)
    windows = build_windows(file_start_ts, file_end_ts, params)
    if not windows:
        return None

    out_obj = {
        "client_id": client_id,
        "context": task,
        "sensor_modality": "voice",
        "screen": {"width": float(screen_width_px), "height": float(screen_height_px)},
        "windows": [],
    }

    for idx, (w_start, w_end) in enumerate(windows, start=1):
        y_win = slice_audio_with_padding(
            y=y,
            sr=sr,
            file_start_ts=file_start_ts,
            w_start=w_start,
            window_size_sec=params.window_size_sec,
        )
        pi = compute_voice_pi_for_window(y_win, sr)
        out_obj["windows"].append(
            {
                "window_number": idx,
                "start_timestamp": format_ts_like_input(w_start),
                "end_timestamp": format_ts_like_input(w_end),
                "pi": pi,
            }
        )

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

    ap.add_argument("--screen_width_px", type=int, default=SCREEN_WIDTH_PX)
    ap.add_argument("--screen_height_px", type=int, default=SCREEN_HEIGHT_PX)

    ap.add_argument("--only_client_id", default=None, help="If set, process only this client_id")

    args = ap.parse_args()

    base_path = args.base_path or args.root_dir
    output_root = args.output_base_path or default_output_root_from_hospital(base_path)

    params = WindowParams(
        window_size_sec=args.window_size_sec,
        stride_sec=args.stride_sec,
    )

    wav_paths_all = iter_voice_wav_paths(base_path)
    if not wav_paths_all:
        print("[DONE] No voice.wav found.")
        return

    wav_paths: List[str] = []
    client_ids = set()
    for p in wav_paths_all:
        try:
            _, client_id, _, _ = parse_path_components(base_path, p)
        except Exception:
            continue
        if args.only_client_id is not None and client_id != args.only_client_id:
            continue
        wav_paths.append(p)
        client_ids.add(client_id)

    print(f"[INFO] clients={len(client_ids)} wav_files={len(wav_paths)}")
    print(f"[INFO] SCREEN: {args.screen_width_px}x{args.screen_height_px} (px)")
    print(f"[INFO] window_size={params.window_size_sec}s stride={params.stride_sec}s output_root={output_root}")

    ok = 0
    for p in tqdm(wav_paths, desc="voice.wav -> voice_pi.json", unit="file"):
        out_path = process_one_voice_wav(
            wav_path=p,
            base_path=base_path,
            output_root=output_root,
            params=params,
            screen_width_px=args.screen_width_px,
            screen_height_px=args.screen_height_px,
        )
        if out_path:
            ok += 1

    print(f"[DONE] processed={ok} / total={len(wav_paths)}")


if __name__ == "__main__":
    main()
