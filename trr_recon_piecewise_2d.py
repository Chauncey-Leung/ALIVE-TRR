# -*- coding: utf-8 -*-
"""
Piecewise 2D contrast-max TDI reconstruction using event-camera data.

The script:
1. Loads event data from RAW, NPY, NPZ, or CSV files.
2. Preprocesses events.
3. Reconstructs an expanded-canvas TDI-like image.
4. Saves the reconstructed image as a 16-bit TIFF.

Time input is in microseconds.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, laplace
from tqdm import tqdm


# =============================================================================
# USER CONFIGURATION
# Edit only this section for routine use.
# =============================================================================

# Input event file. Place your RAW file in the repository's data/ folder.
FILE_PATH = r'E:\recording.raw'

# Output folder. None creates an output folder next to the input file.
OUTPUT_DIR = r'E:\res'

# Batch intervals: (start_us, end_us, expected_signed_vy_px_per_ms).
# Replace these example intervals with the time ranges of your own recording.
# Time is in microseconds; velocity is in pixels per millisecond.
TIME_RANGES = [
    (0.240752e6, 2.572065e6, -4.84),
    (3.628286e6, 5.959600e6, 4.84),
    (6.804014e6, 9.135330e6, -4.84),
    (10.110030e6, 12.441340e6, 4.84),
    (13.335170e6, 15.666480e6, -4.84),
    (16.508550e6, 18.839870e6, 4.84),
    (19.704150e6, 22.035460e6, -4.84),
    (22.877820e6, 25.209140e6, 4.84),
    (25.845730e6, 28.177040e6, -4.84),
    (28.772510e6, 31.103820e6, 4.84),
    (31.985210e6, 34.316530e6, -4.84),
    (35.092080e6, 37.423390e6, 4.84),
]

# Motion-compensation search bounds (px/ms).
# VX_RANGE is shared by all intervals; vy is searched around each interval's
# expected signed vy using VY_HALF_WIDTH.
VX_RANGE = (-0.06, 0.08)
VY_HALF_WIDTH = 0.11

# Local processing window and time step (microseconds).
LOCAL_WINDOW_US = 80_000.0
LOCAL_STEP_US = 80_000.0

# Event selection and optional sensor-coordinate ROI [x0, y0, x1, y1].
POLARITY_MODE = "positive"  # "positive", "negative", "count"/"abs", or "signed"
XY_LIMITS = None            # None uses the full sensor area.

# Reconstruction settings.
FINAL_SPLAT_MODE = "bilinear"  # "nearest" or "bilinear"
USE_RAW_LOCAL_VELOCITY_FOR_RECON = True  # Keep raw local estimates; no smoothing.


# =============================================================================
# Event loading and preprocessing
# =============================================================================

EVENT_DTYPE = np.dtype([("x", np.uint16), ("y", np.uint16), ("p", np.int16), ("t", np.int64)])


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_output_dir(file_path: Union[str, Path], output_dir: Optional[Union[str, Path]]) -> Path:
    if output_dir is None:
        path = Path(file_path)
        return ensure_dir(path.parent / f"{path.stem}_time_named_two_outputs")
    return ensure_dir(output_dir)


def numeric_xytep_to_structured(x, y, t, p) -> np.ndarray:
    n = len(x)
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["x"] = np.asarray(x).astype(np.uint16)
    events["y"] = np.asarray(y).astype(np.uint16)
    events["p"] = np.asarray(p).astype(np.int16)
    events["t"] = np.asarray(t).astype(np.int64)
    return events


def filter_time_structured(
    events: np.ndarray,
    t_start_us: Optional[float],
    t_end_us: Optional[float],
) -> np.ndarray:
    if t_start_us is None and t_end_us is None:
        return events
    keep = np.ones(events.size, dtype=bool)
    if t_start_us is not None:
        keep &= events["t"] >= int(t_start_us)
    if t_end_us is not None:
        keep &= events["t"] < int(t_end_us)
    return events[keep]


def load_events_from_raw(
    file_path: Union[str, Path],
    t_start_us: float,
    t_end_us: float,
    chunk_len_us: float = 1_000.0,
) -> np.ndarray:
    """Load a RAW time interval with EventsIterator time slices.

    Important implementation detail:
    `RawReader` uses a preallocated circular event buffer and its default decoder
    time slice is 50 ms in the installed OpenEB implementation. Calling
    `load_delta_t(782)` on that reader does NOT shrink the decoder's internal
    50-ms production batch, so repeatedly splitting the requested interval can
    still overflow the same fixed buffer. `EventsIterator` constructs a
    `RawReaderBase` with the requested `delta_t`, so the decoder emits shorter
    batches rather than using the fixed-size RawReader circular buffer.

    The displayed event/data rates are used as planning limits for the slice
    duration. Events are filtered to the exact half-open interval [start, end).
    """
    try:
        from metavision_core.event_io import EventsIterator
    except Exception:
        try:
            from evt3_raw_reader import read_evt3_raw_events_and_triggers
        except Exception as exc:
            raise ImportError(
                "RAW input requires metavision_core.event_io.EventsIterator or "
                "local evt3_raw_reader.py."
            ) from exc

        print("    metavision_core EventsIterator unavailable; using local EVT3 RAW reader fallback.")
        events, _triggers = read_evt3_raw_events_and_triggers(
            file_path,
            t_start_us=t_start_us,
            t_end_us=t_end_us,
            do_time_shifting=True,
        )
        if events.size == 0:
            raise RuntimeError("No events loaded in selected RAW time range.")
        events = events.astype(EVENT_DTYPE, copy=False)
        return filter_time_structured(events, t_start_us, t_end_us)

    if t_end_us <= t_start_us:
        raise ValueError("t_end_us must be larger than t_start_us.")
    if chunk_len_us <= 0:
        raise ValueError("chunk_len_us must be positive.")

    event_rate_cap = 330.0 * 1e6
    data_rate_cap_mb_s = 180.0
    if event_rate_cap <= 0 or data_rate_cap_mb_s <= 0:
        raise ValueError("RAW event-rate and data-rate upper limits must be positive.")

    # Respect the fixed rate-based budgets and the requested time-slice
    # target. A 1-ms target gives about 0.330 M events / 0.180 MB at the stated
    # average-rate limits, while EventsIterator avoids RawReader's fixed ring buffer.
    event_budget_chunk_us = 20_000_000 / event_rate_cap * 1e6
    data_budget_chunk_us = 16.0 / data_rate_cap_mb_s * 1e6
    effective_chunk_us = min(
        float(chunk_len_us), event_budget_chunk_us, data_budget_chunk_us
    )
    chunk_us = max(1, int(math.floor(effective_chunk_us)))

    start_int = int(round(t_start_us))
    end_int = int(round(t_end_us))
    if end_int <= start_int:
        raise ValueError("Rounded t_end_us must be larger than rounded t_start_us.")

    # EventsIterator requires start_ts to be a multiple of delta_t. Start at the
    # preceding slice boundary, then filter returned events back to the exact
    # requested start timestamp.
    iterator_start_us = (start_int // chunk_us) * chunk_us
    iterator_duration_us = end_int - iterator_start_us
    expected_events = event_rate_cap * (chunk_us / 1e6)
    expected_data_mb = data_rate_cap_mb_s * (chunk_us / 1e6)
    n_slices = int(math.ceil(iterator_duration_us / chunk_us))

    print(
        "    [RAW rate caps] "
        f"event_rate<={event_rate_cap / 1e6:.1f} MEv/s; "
        f"data_rate<={data_rate_cap_mb_s:.1f} MB/s"
    )
    print(
        "    [RAW iterator plan] "
        f"slice={chunk_us / 1000:.3f} ms; "
        f"events_at_event_rate_cap={expected_events / 1e6:.3f} M/slice; "
        f"estimated_input_at_data_rate_cap={expected_data_mb:.3f} MB/slice; "
        "reader=EventsIterator/RawReaderBase (time-sliced decoder; no fixed RawReader ring buffer)"
    )
    print(
        "    [RAW iterator range] "
        f"aligned_start={iterator_start_us} us; "
        f"requested=[{start_int}, {end_int}) us; "
        f"slices~{n_slices}"
    )

    chunks = []
    n_loaded_before_filter = 0
    n_kept = 0
    iterator = EventsIterator(
        input_path=str(file_path),
        start_ts=iterator_start_us,
        mode="delta_t",
        delta_t=chunk_us,
        max_duration=iterator_duration_us,
        relative_timestamps=False,
    )

    for ev in tqdm(iterator, total=n_slices, desc="load RAW time slices"):
        if ev.size == 0:
            continue
        n_loaded_before_filter += int(ev.size)
        ts = ev["t"]
        keep = (ts >= start_int) & (ts < end_int)
        if np.any(keep):
            selected = ev[keep].astype(EVENT_DTYPE, copy=False).copy()
            chunks.append(selected)
            n_kept += int(selected.size)

    if not chunks:
        raise RuntimeError(
            f"No events loaded in selected RAW time range [{start_int}, {end_int}) us. "
            f"Iterator inspected {n_loaded_before_filter:,} events before exact-range filtering."
        )

    print(
        f"    [RAW iterator done] returned={n_loaded_before_filter:,} events; "
        f"kept_in_requested_range={n_kept:,} events"
    )
    return np.concatenate(chunks).astype(EVENT_DTYPE, copy=False)


def load_events_generic(
    file_path: Union[str, Path],
    t_start_us: Optional[float],
    t_end_us: Optional[float],
) -> np.ndarray:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".raw":
        if t_start_us is None or t_end_us is None:
            raise ValueError(".raw input requires t_start_us and t_end_us.")
        return load_events_from_raw(path, t_start_us, t_end_us)

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.dtype.names is not None:
            events = arr.astype(EVENT_DTYPE, copy=False)
        else:
            if arr.ndim != 2 or arr.shape[1] < 4:
                raise ValueError(".npy numeric array must have at least columns x,y,t,p.")
            events = numeric_xytep_to_structured(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
        return filter_time_structured(events, t_start_us, t_end_us)

    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        if "events" in data:
            arr = data["events"]
            if arr.dtype.names is not None:
                events = arr.astype(EVENT_DTYPE, copy=False)
            else:
                events = numeric_xytep_to_structured(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
        else:
            events = numeric_xytep_to_structured(data["x"], data["y"], data["t"], data["p"])
        return filter_time_structured(events, t_start_us, t_end_us)

    if suffix == ".csv":
        df = pd.read_csv(path)
        for col in ["x", "y", "t", "p"]:
            if col not in df.columns:
                raise ValueError(".csv must contain x,y,t,p columns.")
        events = numeric_xytep_to_structured(df["x"], df["y"], df["t"], df["p"])
        return filter_time_structured(events, t_start_us, t_end_us)

    raise ValueError(f"Unsupported file suffix: {suffix}")


def normalize_polarity_to_signed(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    sample = p[: min(p.size, 100_000)]
    unique = np.unique(sample)
    if np.all(np.isin(unique, [0, 1])):
        return (2 * p.astype(np.int16) - 1).astype(np.int16)
    return np.where(p > 0, 1, -1).astype(np.int16)


def select_by_roi(
    events: np.ndarray,
    xy_limits: Optional[Union[str, Sequence[int]]],
) -> Tuple[np.ndarray, Dict[str, int]]:
    if events.size == 0:
        raise RuntimeError("No events before ROI filtering.")

    if xy_limits is None or xy_limits == "auto":
        x0 = int(events["x"].min())
        y0 = int(events["y"].min())
        x1 = int(events["x"].max()) + 1
        y1 = int(events["y"].max()) + 1
    else:
        x0, y0, x1, y1 = map(int, xy_limits)

    keep = (
        (events["x"] >= x0) & (events["x"] < x1) &
        (events["y"] >= y0) & (events["y"] < y1)
    )
    ev = events[keep].copy()
    if ev.size == 0:
        raise RuntimeError("No events left after ROI filtering.")
    ev["x"] = ev["x"] - x0
    ev["y"] = ev["y"] - y0
    return ev, dict(x0=x0, y0=y0, x1=x1, y1=y1, W=x1 - x0, H=y1 - y0)


def filter_hot_pixels(
    events: np.ndarray,
    H: int,
    W: int,
    percentile: float,
    abs_threshold: Optional[float],
) -> Tuple[np.ndarray, Dict[str, Union[int, float, None]]]:
    if events.size == 0:
        return events, dict(threshold=0.0, n_hot=0, removed=0)

    flat = events["y"].astype(np.int64) * int(W) + events["x"].astype(np.int64)
    valid = (flat >= 0) & (flat < int(H) * int(W))
    counts = np.bincount(flat[valid], minlength=int(H) * int(W))
    nonzero = counts[counts > 0]
    thr_p = float(np.percentile(nonzero, percentile)) if nonzero.size else np.inf
    thr = thr_p if abs_threshold is None else max(thr_p, float(abs_threshold))

    hot_flat = np.flatnonzero(counts > thr)
    if hot_flat.size == 0:
        return events, dict(threshold=float(thr), n_hot=0, removed=0)

    is_hot = np.zeros(int(H) * int(W), dtype=bool)
    is_hot[hot_flat] = True
    keep = valid & (~is_hot[flat])
    return events[keep].copy(), dict(
        threshold=float(thr),
        n_hot=int(hot_flat.size),
        removed=int(events.size - keep.sum()),
    )


def apply_polarity_mode(events: np.ndarray, polarity_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    mode = polarity_mode.lower()
    p_signed = normalize_polarity_to_signed(events["p"])

    if mode in ("count", "abs"):
        return events, np.ones(events.size, dtype=np.float32)
    if mode == "positive":
        keep = p_signed > 0
        return events[keep].copy(), np.ones(int(keep.sum()), dtype=np.float32)
    if mode == "negative":
        keep = p_signed < 0
        return events[keep].copy(), np.ones(int(keep.sum()), dtype=np.float32)
    if mode == "signed":
        return events, p_signed.astype(np.float32)
    raise ValueError(f"Unknown polarity mode: {polarity_mode}")


def sort_events_and_weights_by_time(
    events: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if events.size <= 1 or np.all(events["t"][:-1] <= events["t"][1:]):
        return events, weights
    order = np.argsort(events["t"], kind="mergesort")
    return events[order].copy(), weights[order].copy()


def prepare_events(
    file_path: Union[str, Path],
    t_start_us: float,
    t_end_us: float,
    xy_limits,
    polarity_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    print(f"[1] Load events: {t_start_us:.0f} us to {t_end_us:.0f} us")
    events = load_events_generic(file_path, t_start_us, t_end_us)
    print(f"    loaded events: {events.size}")

    print("[2] Apply ROI")
    events, roi_meta = select_by_roi(events, xy_limits)
    H, W = int(roi_meta["H"]), int(roi_meta["W"])
    print(f"    ROI canvas: H={H}, W={W}, events={events.size}")

    print("[3] Hot-pixel filtering")
    events, hp_meta = filter_hot_pixels(
        events,
        H,
        W,
        percentile=99.95,
        abs_threshold=None,
    )
    print(
        f"    hot pixels={hp_meta['n_hot']}, removed={hp_meta['removed']}, "
        f"threshold={hp_meta['threshold']:.2f}"
    )

    print("[4] Apply polarity")
    events, weights = apply_polarity_mode(events, polarity_mode)
    events, weights = sort_events_and_weights_by_time(events, weights)
    print(f"    polarity_mode={polarity_mode}, events={events.size}")
    if events.size == 0:
        raise RuntimeError("No events left after preprocessing.")

    meta = dict(
        file_path=str(file_path),
        start_us=float(t_start_us),
        end_us=float(t_end_us),
        roi=roi_meta,
        hot_pixel=hp_meta,
        H=H,
        W=W,
        n_events=int(events.size),
        polarity_mode=polarity_mode,
    )
    return events, weights, meta


def infer_active_estimation_roi(
    events: np.ndarray,
    H: int,
    W: int,
    t_start_us: float,
    t_end_us: float,
) -> Dict[str, Union[int, float]]:
    if events.size == 0:
        raise RuntimeError("Cannot infer estimation ROI from empty events.")

    t = events["t"].astype(np.float64)
    window_us = 200_000.0
    step_us = 200_000.0
    if window_us <= 0 or step_us <= 0:
        raise ValueError("The internal estimation ROI window and step must be positive.")

    if float(t_end_us) - float(t_start_us) <= window_us + 1e-9:
        return dict(
            x0=0,
            y0=0,
            x1=int(W),
            y1=int(H),
            W=int(W),
            H=int(H),
            active_start_us=float(t_start_us),
            active_end_us=float(t_end_us),
            active_events=int(events.size),
            active_events_window_sum=int(events.size),
            active_best_events=int(events.size),
            active_top_windows=[
                dict(start_us=float(t_start_us), end_us=float(t_end_us), events=int(events.size))
            ],
            mode="active_window_full_range",
        )

    rows: List[Tuple[int, int, float, float]] = []
    cur = float(t_start_us)
    last_start = max(float(t_start_us), float(t_end_us) - window_us)
    idx = 0
    while cur <= last_start + 1e-9:
        end = min(cur + window_us, float(t_end_us))
        i0 = int(np.searchsorted(t, cur, side="left"))
        i1 = int(np.searchsorted(t, end, side="left"))
        n = i1 - i0
        rows.append((int(n), int(idx), float(cur), float(end)))
        idx += 1
        cur += step_us

    if not rows:
        rows.append((int(events.size), 0, float(t_start_us), float(t_end_us)))

    rows_sorted = sorted(rows, key=lambda item: item[0], reverse=True)
    top_n = max(1, min(int(8), len(rows_sorted)))
    selected = sorted(rows_sorted[:top_n], key=lambda item: item[1])
    best_count, _best_idx, best_start, best_end = rows_sorted[0]

    active_mask = np.zeros(events.size, dtype=bool)
    active_events_sum = 0
    for count, _idx, start, end in selected:
        i0 = int(np.searchsorted(t, start, side="left"))
        i1 = int(np.searchsorted(t, end, side="left"))
        if i1 > i0:
            active_mask[i0:i1] = True
            active_events_sum += int(count)

    active = events[active_mask]
    if active.size == 0:
        return dict(
            x0=0,
            y0=0,
            x1=int(W),
            y1=int(H),
            W=int(W),
            H=int(H),
            active_start_us=float(best_start),
            active_end_us=float(best_end),
            active_events=0,
            active_best_events=int(best_count),
            active_top_windows=[],
            mode="fallback_full_reconstruction_roi",
        )

    margin = int(0)
    x_low_q = float(np.clip(0.0, 0.0, 100.0))
    x_high_q = float(np.clip(99.85, 0.0, 100.0))
    if x_high_q < x_low_q:
        x_low_q, x_high_q = x_high_q, x_low_q

    x0 = max(0, int(math.floor(np.percentile(active["x"], x_low_q))) - margin)
    x1 = min(int(W), int(math.ceil(np.percentile(active["x"], x_high_q))) + 1 + margin)
    y0 = 0
    y1 = int(H)
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, int(W), int(H)

    return dict(
        x0=int(x0),
        y0=int(y0),
        x1=int(x1),
        y1=int(y1),
        W=int(x1 - x0),
        H=int(y1 - y0),
        active_start_us=float(best_start),
        active_end_us=float(best_end),
        active_events=int(active.size),
        active_events_window_sum=int(active_events_sum),
        active_best_events=int(best_count),
        active_top_windows=[
            dict(start_us=float(start), end_us=float(end), events=int(count))
            for count, _idx, start, end in selected
        ],
        x_low_percentile=float(x_low_q),
        x_high_percentile=float(x_high_q),
        use_full_height=bool(True),
        mode="active_window",
    )


def make_estimation_events(
    events: np.ndarray,
    weights: np.ndarray,
    reconstruction_meta: Dict,
    t_start_us: float,
    t_end_us: float,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    mode = "active_window"
    recon_H = int(reconstruction_meta["H"])
    recon_W = int(reconstruction_meta["W"])

    if mode == "reconstruction":
        roi = dict(
            x0=0,
            y0=0,
            x1=recon_W,
            y1=recon_H,
            W=recon_W,
            H=recon_H,
            active_start_us=None,
            active_end_us=None,
            active_events=int(events.size),
            mode="reconstruction",
        )
    elif mode == "active_window":
        roi = infer_active_estimation_roi(events, recon_H, recon_W, t_start_us, t_end_us)
    else:
        raise ValueError("Unknown internal estimation ROI mode.")

    x0, y0, x1, y1 = int(roi["x0"]), int(roi["y0"]), int(roi["x1"]), int(roi["y1"])
    keep = (
        (events["x"] >= x0) & (events["x"] < x1) &
        (events["y"] >= y0) & (events["y"] < y1)
    )
    ev = events[keep].copy()
    wt = weights[keep].copy()
    if ev.size == 0:
        raise RuntimeError("No events left in estimation ROI.")
    ev["x"] = ev["x"] - x0
    ev["y"] = ev["y"] - y0

    # Include original sensor coordinates for easier comparison with earlier runs.
    recon_roi = reconstruction_meta.get("roi", {})
    roi["original_x0"] = int(recon_roi.get("x0", 0)) + x0
    roi["original_y0"] = int(recon_roi.get("y0", 0)) + y0
    roi["original_x1"] = int(recon_roi.get("x0", 0)) + x1
    roi["original_y1"] = int(recon_roi.get("y0", 0)) + y1
    roi["n_events"] = int(ev.size)
    return ev, wt, roi


# =============================================================================
# Fast IWE scoring
# =============================================================================

def make_velocity_grid_range(v_range: Tuple[float, float], step: float) -> np.ndarray:
    lo, hi = map(float, v_range)
    if hi < lo:
        lo, hi = hi, lo
    if step <= 0:
        raise ValueError("step must be positive.")
    n = int(np.floor((hi - lo) / step)) + 1
    grid = lo + np.arange(n + 1, dtype=np.float64) * float(step)
    grid = grid[(grid >= lo - 1e-12) & (grid <= hi + 1e-12)]
    grid = np.r_[grid, lo, hi]
    return np.unique(np.round(grid, 9))


def make_velocity_grid_centered(
    center: float,
    radius: float,
    step: float,
    bounds: Tuple[float, float],
) -> np.ndarray:
    lo_b, hi_b = sorted(map(float, bounds))
    lo = max(lo_b, float(center) - float(radius))
    hi = min(hi_b, float(center) + float(radius))
    if hi < lo:
        lo, hi = lo_b, hi_b
    if radius <= 0:
        return np.asarray([float(np.clip(center, lo_b, hi_b))], dtype=np.float64)
    grid = make_velocity_grid_range((lo, hi), step)
    grid = np.r_[grid, np.clip(float(center), lo_b, hi_b)]
    return np.unique(np.round(grid, 9))


def sample_packet_arrays(
    events: np.ndarray,
    weights: np.ndarray,
    t_ref_us: float,
    max_events: Optional[int],
) -> Dict[str, np.ndarray]:
    if max_events is not None and events.size > int(max_events):
        idx = np.linspace(0, events.size - 1, int(max_events)).astype(np.int64)
        ev = events[idx]
        wt = weights[idx]
    else:
        ev = events
        wt = weights
    return dict(
        x=ev["x"].astype(np.float32),
        y=ev["y"].astype(np.float32),
        dt_ms=((ev["t"].astype(np.float64) - float(t_ref_us)) / 1000.0).astype(np.float32),
        weights=wt.astype(np.float32, copy=False),
        n_total=int(ev.size),
    )


def common_overlap_bbox(
    H: int,
    W: int,
    t_start_us: float,
    t_end_us: float,
    t_ref_us: float,
    vx_values: np.ndarray,
    vy_values: np.ndarray,
    margin_px: float,
) -> Optional[Tuple[int, int, int, int]]:
    dt0_ms = (float(t_start_us) - float(t_ref_us)) / 1000.0
    dt1_ms = (float(t_end_us) - float(t_ref_us)) / 1000.0
    vx0, vx1 = float(np.min(vx_values)), float(np.max(vx_values))
    vy0, vy1 = float(np.min(vy_values)), float(np.max(vy_values))

    sx_vals = [vx0 * dt0_ms, vx0 * dt1_ms, vx1 * dt0_ms, vx1 * dt1_ms]
    sy_vals = [vy0 * dt0_ms, vy0 * dt1_ms, vy1 * dt0_ms, vy1 * dt1_ms]
    sx_min, sx_max = min(sx_vals), max(sx_vals)
    sy_min, sy_max = min(sy_vals), max(sy_vals)

    x0 = int(math.ceil(max(0.0, sx_max) + margin_px))
    x1 = int(math.floor(min(float(W), float(W) + sx_min) - margin_px))
    y0 = int(math.ceil(max(0.0, sy_max) + margin_px))
    y1 = int(math.floor(min(float(H), float(H) + sy_min) - margin_px))
    if x1 <= x0 or y1 <= y0:
        return None
    area_ratio = ((x1 - x0) * (y1 - y0)) / float(max(1, H * W))
    if area_ratio < 0.05:
        return None
    return x0, y0, x1, y1


def accumulate_nearest_bincount(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    H: int,
    W: int,
) -> Tuple[np.ndarray, int]:
    xi = np.rint(xs).astype(np.int64)
    yi = np.rint(ys).astype(np.int64)
    valid = (xi >= 0) & (xi < int(W)) & (yi >= 0) & (yi < int(H))
    if not np.any(valid):
        return np.zeros((int(H), int(W)), dtype=np.float32), 0
    flat = yi[valid] * int(W) + xi[valid]
    img = np.bincount(flat, weights=weights[valid], minlength=int(H) * int(W))
    return img.reshape(int(H), int(W)).astype(np.float32, copy=False), int(valid.sum())


def focus_score_from_iwe(
    img: np.ndarray,
    n_valid: int,
    n_total: int,
    score_bbox: Optional[Tuple[int, int, int, int]],
    gaussian_sigma: float = 0.0,
    clip_percentile: Optional[float] = None,
) -> float:
    if n_total <= 0 or n_valid <= 0:
        return -np.inf
    if n_valid / float(n_total) < 0.70:
        return -np.inf

    I = img.astype(np.float32, copy=False)
    if score_bbox is not None:
        x0, y0, x1, y1 = map(int, score_bbox)
        x0 = max(0, min(x0, I.shape[1]))
        x1 = max(0, min(x1, I.shape[1]))
        y0 = max(0, min(y0, I.shape[0]))
        y1 = max(0, min(y1, I.shape[0]))
        if x1 <= x0 or y1 <= y0:
            return -np.inf
        I = I[y0:y1, x0:x1]

    if I.size == 0:
        return -np.inf

    if gaussian_sigma is not None and gaussian_sigma > 0:
        I = gaussian_filter(I, sigma=float(gaussian_sigma))

    if clip_percentile is not None:
        I = I.copy()
        I[~np.isfinite(I)] = 0
        if np.min(I) < 0:
            hi = np.percentile(np.abs(I), float(clip_percentile))
            if np.isfinite(hi) and hi > 0:
                I = np.clip(I, -hi, hi)
        else:
            hi = np.percentile(I, float(clip_percentile))
            if np.isfinite(hi) and hi > 0:
                I = np.clip(I, None, hi)

    mode = "variance"
    if mode == "variance":
        mu = float(np.mean(I))
        return float(np.mean(I * I) - mu * mu)
    if mode == "normalized_variance":
        mu_abs = float(np.mean(np.abs(I))) if np.min(I) < 0 else float(np.mean(I))
        var = float(np.var(I))
        return float(var / (mu_abs * mu_abs + 1e-12))
    if mode in ("gradient_magnitude", "gradient", "tenengrad"):
        gy, gx = np.gradient(I)
        return float(np.mean(gx * gx + gy * gy))
    if mode in ("laplacian_magnitude", "laplacian"):
        L = laplace(I)
        return float(np.mean(L * L))
    if mode == "variance_of_laplacian":
        L = laplace(I)
        return float(np.var(L))
    raise ValueError(f"Unknown focus-score mode: {mode}")


def evaluate_packet_velocity(
    packet: Dict[str, np.ndarray],
    vx: float,
    vy: float,
    H: int,
    W: int,
    score_bbox: Optional[Tuple[int, int, int, int]],
    gaussian_sigma: float = 0.0,
    clip_percentile: Optional[float] = None,
) -> Tuple[float, int]:
    xs = packet["x"] + float(vx) * packet["dt_ms"]
    ys = packet["y"] + float(vy) * packet["dt_ms"]
    img, n_valid = accumulate_nearest_bincount(xs, ys, packet["weights"], H, W)
    score = focus_score_from_iwe(
        img, n_valid, int(packet["n_total"]), score_bbox,
        gaussian_sigma=gaussian_sigma, clip_percentile=clip_percentile,
    )
    return score, n_valid


def score_velocity_over_packets(
    packets: List[Dict],
    vx: float,
    vy: float,
    H: int,
    W: int,
    gaussian_sigma: float = 0.0,
    clip_percentile: Optional[float] = None,
) -> Tuple[float, int]:
    scores = []
    n_valid_total = 0
    for pkt in packets:
        score, n_valid = evaluate_packet_velocity(
            pkt["packet"], vx, vy, H, W, pkt.get("score_bbox"),
            gaussian_sigma=gaussian_sigma, clip_percentile=clip_percentile,
        )
        n_valid_total += int(n_valid)
        if np.isfinite(score):
            scores.append(float(score))
    if not scores:
        return -np.inf, n_valid_total
    return float(np.median(scores)), n_valid_total


def grid_search_packets(
    packets: List[Dict],
    H: int,
    W: int,
    vx_grid: np.ndarray,
    vy_grid: np.ndarray,
    stage: str,
    keep_rows: bool,
    gaussian_sigma: float = 0.0,
    clip_percentile: Optional[float] = None,
) -> Tuple[Dict, List[Dict]]:
    best = dict(vx=np.nan, vy=np.nan, score=-np.inf, n_valid=0, stage=stage)
    rows: List[Dict] = []
    for vx in vx_grid:
        for vy in vy_grid:
            score, n_valid = score_velocity_over_packets(
                packets, float(vx), float(vy), H, W,
                gaussian_sigma=gaussian_sigma, clip_percentile=clip_percentile,
            )
            if keep_rows:
                rows.append(dict(stage=stage, vx=float(vx), vy=float(vy), score=float(score), n_valid=int(n_valid)))
            if score > best["score"]:
                best.update(vx=float(vx), vy=float(vy), score=float(score), n_valid=int(n_valid))
    return best, rows


def peak_plateau_representative(
    rows: List[Dict],
    best: Dict,
    stage: str,
    relative_threshold: float,
) -> Dict:
    if not rows or not np.isfinite(best.get("score", -np.inf)):
        return best

    df = pd.DataFrame(rows)
    if df.empty or "score" not in df:
        return best
    sub = df[df["stage"] == stage].copy()
    sub = sub[np.isfinite(sub["score"].astype(float))]
    if sub.empty:
        return best

    best_score = float(sub["score"].max())
    if not np.isfinite(best_score):
        return best
    rel = float(np.clip(relative_threshold, 0.0, 1.0))
    plateau = sub[sub["score"].astype(float) >= best_score * rel]
    if plateau.empty:
        return best

    out = dict(best)
    out["vx_argmax"] = float(best["vx"])
    out["vy_argmax"] = float(best["vy"])
    out["score_argmax"] = float(best["score"])
    out["vx"] = float(np.median(plateau["vx"].astype(float)))
    out["vy"] = float(np.median(plateau["vy"].astype(float)))
    out["score"] = best_score
    out["stage"] = f"{stage}_plateau"
    out["peak_plateau_relative_threshold"] = rel
    out["peak_plateau_count"] = int(plateau.shape[0])
    return out


def slice_events_by_time(
    events: np.ndarray,
    weights: np.ndarray,
    start_us: float,
    end_us: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t = events["t"].astype(np.float64)
    i0 = int(np.searchsorted(t, start_us, side="left"))
    i1 = int(np.searchsorted(t, end_us, side="left"))
    return events[i0:i1], weights[i0:i1]


def build_scoring_packets_for_windows(
    events: np.ndarray,
    weights: np.ndarray,
    H: int,
    W: int,
    windows: Sequence[Tuple[float, float]],
    vx_grid: np.ndarray,
    vy_grid: np.ndarray,
    min_events: int,
) -> List[Dict]:
    packets = []
    for start_us, end_us in windows:
        ev, wt = slice_events_by_time(events, weights, start_us, end_us)
        if ev.size < int(min_events):
            continue
        t_ref_us = 0.5 * (float(start_us) + float(end_us))
        bbox = common_overlap_bbox(
            H,
            W,
            start_us,
            end_us,
            t_ref_us,
            vx_grid,
            vy_grid,
            4,
        )
        packets.append(
            dict(
                start_us=float(start_us),
                end_us=float(end_us),
                center_us=float(t_ref_us),
                n_events=int(ev.size),
                score_bbox=bbox,
                packet=sample_packet_arrays(ev, wt, t_ref_us, None),
            )
        )
    return packets


def make_sliding_windows(start_us: float, end_us: float, window_us: float, step_us: float) -> List[Tuple[float, float]]:
    if window_us <= 0 or step_us <= 0:
        raise ValueError("window_us and step_us must be positive.")
    if end_us <= start_us:
        raise ValueError("end_us must be larger than start_us.")

    windows: List[Tuple[float, float]] = []
    cur = float(start_us)
    last_start = max(float(start_us), float(end_us) - float(window_us))
    while cur <= last_start + 1e-9:
        windows.append((cur, min(cur + float(window_us), float(end_us))))
        cur += float(step_us)
    if not windows:
        windows.append((float(start_us), float(end_us)))
    elif windows[-1][0] < last_start - 1e-9:
        windows.append((last_start, float(end_us)))
    return windows


def sample_init_windows_evenly(windows: Sequence[Tuple[float, float]], max_windows: int) -> List[Tuple[float, float]]:
    if len(windows) <= max_windows:
        return list(windows)
    idx = np.linspace(0, len(windows) - 1, int(max_windows)).round().astype(int)
    idx = np.unique(idx)
    return [windows[int(i)] for i in idx]


def sample_init_windows_by_event_count(
    events: np.ndarray,
    windows: Sequence[Tuple[float, float]],
    max_windows: int,
) -> List[Tuple[float, float]]:
    if len(windows) <= max_windows:
        return list(windows)
    t = events["t"].astype(np.float64)
    rows = []
    for idx, (start, end) in enumerate(windows):
        i0 = int(np.searchsorted(t, start, side="left"))
        i1 = int(np.searchsorted(t, end, side="left"))
        rows.append((i1 - i0, idx, start, end))
    rows.sort(key=lambda item: item[0], reverse=True)
    chosen = sorted(rows[: int(max_windows)], key=lambda item: item[1])
    return [(float(start), float(end)) for _count, _idx, start, end in chosen]


def automatic_global_velocity_init(
    events: np.ndarray,
    weights: np.ndarray,
    H: int,
    W: int,
    start_us: float,
    end_us: float,
    vx_range: Tuple[float, float],
    vy_range: Tuple[float, float],
    keep_rows: bool,
) -> Tuple[Dict, pd.DataFrame]:
    """Coarse-to-fine global velocity initialization using fixed internal defaults."""
    print("[5] Automatic global 2D velocity initialization")
    init_window_us = 80_000.0
    max_init_windows = 4
    coarse_step = 0.10
    fine_radius = 0.08
    fine_step = 0.01
    min_events = 2000
    global_gaussian_sigma = 0.8
    global_clip_percentile = 99.9
    plateau_relative_threshold = 0.995

    print(
        "    global-init score config: "
        f"gaussian_sigma={global_gaussian_sigma}, "
        f"clip_percentile={global_clip_percentile}"
    )
    init_windows_all = make_sliding_windows(
        start_us, end_us, init_window_us, init_window_us
    )
    init_windows = sample_init_windows_by_event_count(
        events, init_windows_all, max_init_windows
    )
    print(
        f"    init windows: {len(init_windows)} / {len(init_windows_all)} "
        "selected by event count"
    )

    vx_coarse = make_velocity_grid_range(vx_range, coarse_step)
    vy_coarse = make_velocity_grid_range(vy_range, coarse_step)
    packets = build_scoring_packets_for_windows(
        events, weights, H, W, init_windows,
        vx_coarse, vy_coarse, min_events,
    )
    if not packets:
        raise RuntimeError("No valid packets for global initialization.")

    print(
        f"    coarse grid: Nx={vx_coarse.size}, "
        f"Ny={vy_coarse.size}, packets={len(packets)}"
    )
    best, rows = grid_search_packets(
        packets, H, W, vx_coarse, vy_coarse, "global_coarse", keep_rows,
        gaussian_sigma=global_gaussian_sigma,
        clip_percentile=global_clip_percentile,
    )

    vx_fine = make_velocity_grid_centered(
        best["vx"], fine_radius, fine_step, vx_range
    )
    vy_fine = make_velocity_grid_centered(
        best["vy"], fine_radius, fine_step, vy_range
    )
    packets = build_scoring_packets_for_windows(
        events, weights, H, W, init_windows,
        vx_fine, vy_fine, min_events,
    )
    print(
        f"    fine grid around vx={best['vx']:+.6f}, "
        f"vy={best['vy']:+.6f}: Nx={vx_fine.size}, Ny={vy_fine.size}"
    )
    best_fine, rows_fine = grid_search_packets(
        packets, H, W, vx_fine, vy_fine, "global_fine", keep_rows,
        gaussian_sigma=global_gaussian_sigma,
        clip_percentile=global_clip_percentile,
    )
    rows.extend(rows_fine)
    if best_fine["score"] > best["score"]:
        best = best_fine

    best = peak_plateau_representative(
        rows,
        best,
        stage="global_fine",
        relative_threshold=plateau_relative_threshold,
    )
    print(
        f"    global init best: vx={best['vx']:+.6f}, "
        f"vy={best['vy']:+.6f}, score={best['score']:.6g}"
    )
    return best, pd.DataFrame(rows)


def search_one_local_window(
    events: np.ndarray,
    weights: np.ndarray,
    H: int,
    W: int,
    start_us: float,
    end_us: float,
    center_vx: float,
    center_vy: float,
    vx_range: Tuple[float, float],
    vy_range: Tuple[float, float],
    force_full_range: bool,
    keep_rows: bool,
) -> Tuple[Dict, List[Dict]]:
    ev, wt = slice_events_by_time(events, weights, start_us, end_us)
    t_ref_us = 0.5 * (float(start_us) + float(end_us))

    if ev.size < 2000:
        return dict(
            start_us=float(start_us),
            end_us=float(end_us),
            center_us=float(t_ref_us),
            n_events=int(ev.size),
            vx=np.nan,
            vy=np.nan,
            score=-np.inf,
            n_valid=0,
            stage="skipped",
        ), []

    if force_full_range:
        vx_grid = make_velocity_grid_range(vx_range, 0.08)
        vy_grid = make_velocity_grid_range(vy_range, 0.08)
        stage0 = "local_full_range"
    else:
        vx_grid = make_velocity_grid_centered(center_vx, 0.06, 0.02, vx_range)
        vy_grid = make_velocity_grid_centered(center_vy, 0.06, 0.02, vy_range)
        stage0 = "local_track"

    rows: List[Dict] = []
    packets = build_scoring_packets_for_windows(
        events,
        weights,
        H,
        W,
        [(start_us, end_us)],
        vx_grid,
        vy_grid,
        2000,
    )
    if not packets:
        return dict(
            start_us=float(start_us),
            end_us=float(end_us),
            center_us=float(t_ref_us),
            n_events=int(ev.size),
            vx=np.nan,
            vy=np.nan,
            score=-np.inf,
            n_valid=0,
            stage="skipped",
        ), []

    best, stage_rows = grid_search_packets(packets, H, W, vx_grid, vy_grid, stage0, keep_rows)
    for row in stage_rows:
        row.update(start_us=float(start_us), end_us=float(end_us), center_us=float(t_ref_us), n_events=int(ev.size))
    rows.extend(stage_rows)

    vx_fine = make_velocity_grid_centered(best["vx"], 0.015, 0.005, vx_range)
    vy_fine = make_velocity_grid_centered(best["vy"], 0.015, 0.005, vy_range)
    packets = build_scoring_packets_for_windows(
        events,
        weights,
        H,
        W,
        [(start_us, end_us)],
        vx_fine,
        vy_fine,
        2000,
    )
    best_fine, fine_rows = grid_search_packets(packets, H, W, vx_fine, vy_fine, "local_fine", keep_rows)
    for row in fine_rows:
        row.update(start_us=float(start_us), end_us=float(end_us), center_us=float(t_ref_us), n_events=int(ev.size))
    rows.extend(fine_rows)
    if best_fine["score"] > best["score"]:
        best = best_fine

    return dict(
        start_us=float(start_us),
        end_us=float(end_us),
        center_us=float(t_ref_us),
        n_events=int(ev.size),
        vx=float(best["vx"]),
        vy=float(best["vy"]),
        score=float(best["score"]),
        n_valid=int(best["n_valid"]),
        stage=str(best["stage"]),
    ), rows


def estimate_local_velocity_curve(
    events: np.ndarray,
    weights: np.ndarray,
    H: int,
    W: int,
    start_us: float,
    end_us: float,
    vx_range: Tuple[float, float],
    vy_range: Tuple[float, float],
    init_vx: float,
    init_vy: float,
    keep_rows: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate local velocities using the fixed global-anchor tracking strategy."""
    print("[6] Estimate local piecewise 2D velocities")
    windows = make_sliding_windows(
        start_us, end_us, LOCAL_WINDOW_US, LOCAL_STEP_US
    )
    print(
        f"    local windows={len(windows)}, "
        f"window_us={LOCAL_WINDOW_US:g}, step_us={LOCAL_STEP_US:g}"
    )

    rows: List[Dict] = []
    grid_rows: List[Dict] = []

    for i, (start, end) in enumerate(tqdm(windows, desc="local windows")):
        # Original defaults: full-range scan for the first window; global anchor thereafter.
        force_full = (i == 0)
        row, stage_rows = search_one_local_window(
            events,
            weights,
            H,
            W,
            start,
            end,
            float(init_vx),
            float(init_vy),
            vx_range,
            vy_range,
            force_full_range=force_full,
            keep_rows=keep_rows,
        )
        rows.append(row)
        grid_rows.extend(stage_rows)

    return pd.DataFrame(rows), pd.DataFrame(grid_rows)


def mark_local_velocity_outliers(
    df: pd.DataFrame,
    anchor_vx: float,
    anchor_vy: float,
) -> pd.DataFrame:
    """Optional fixed-threshold outlier rejection used only when raw mode is disabled."""
    out = df.copy()
    if out.empty:
        return out

    vx = out["vx"].to_numpy(dtype=np.float64)
    vy = out["vy"].to_numpy(dtype=np.float64)
    score = out["score"].to_numpy(dtype=np.float64)
    finite = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(score)

    dist_global = np.sqrt((vx - float(anchor_vx)) ** 2 + (vy - float(anchor_vy)) ** 2)
    out["dist_from_global"] = dist_global
    out["step_change"] = np.nan
    out["score_gate_threshold"] = np.nan

    prev_vx = np.nan
    prev_vy = np.nan
    step_change = np.full(out.shape[0], np.nan, dtype=np.float64)
    for i in range(out.shape[0]):
        if not finite[i]:
            continue
        if np.isfinite(prev_vx) and np.isfinite(prev_vy):
            step_change[i] = math.hypot(float(vx[i] - prev_vx), float(vy[i] - prev_vy))
        prev_vx = float(vx[i])
        prev_vy = float(vy[i])
    out["step_change"] = step_change

    valid = finite.copy()
    reasons = np.full(out.shape[0], "", dtype=object)
    reasons[~finite] = "nonfinite_or_skipped"

    far_global = np.zeros(out.shape[0], dtype=bool)
    far_global[finite] = dist_global[finite] > 0.12
    valid &= ~far_global
    reasons[far_global] = "far_from_global"

    big_jump = np.zeros(out.shape[0], dtype=bool)
    jump_finite = finite & np.isfinite(step_change)
    big_jump[jump_finite] = step_change[jump_finite] > 0.08
    valid &= ~big_jump
    reasons[big_jump & (reasons == "")] = "large_step_change"
    reasons[big_jump & (reasons != "")] = (
        reasons[big_jump & (reasons != "")] + ";large_step_change"
    )

    score_finite = finite & np.isfinite(score)
    if np.any(score_finite):
        score_ref = float(np.quantile(score[score_finite], 0.80))
        score_threshold = float(score_ref * 0.50)
        out["score_gate_threshold"] = score_threshold
        low_score = score_finite & (score < score_threshold)
        valid &= ~low_score
        reasons[low_score & (reasons == "")] = "low_score"
        reasons[low_score & (reasons != "")] = (
            reasons[low_score & (reasons != "")] + ";low_score"
        )

    out["is_valid_for_smooth"] = valid
    out["outlier_reason"] = reasons
    out["vx_filtered"] = np.where(valid, vx, np.nan)
    out["vy_filtered"] = np.where(valid, vy, np.nan)
    return out


def use_raw_local_velocity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use local velocity estimates exactly as measured by the local grid search.

    This function intentionally does not:
        - reject outliers,
        - apply score gating,
        - interpolate missing velocity windows,
        - median-smooth or low-pass-filter the velocity curve.

    For compatibility with downstream reconstruction code, it copies:
        vx -> vx_smooth
        vy -> vy_smooth

    Windows with non-finite vx/vy or score remain non-finite and are ignored by
    warp_events_piecewise(), which then interpolates displacement between the
    remaining finite velocity centers.
    """
    out = df.copy()
    if out.empty:
        return out

    vx = out["vx"].to_numpy(dtype=np.float64)
    vy = out["vy"].to_numpy(dtype=np.float64)
    if "score" in out.columns:
        score = out["score"].to_numpy(dtype=np.float64)
        finite = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(score)
    else:
        finite = np.isfinite(vx) & np.isfinite(vy)

    out["dist_from_global"] = np.nan
    out["step_change"] = np.nan
    out["score_gate_threshold"] = np.nan
    out["is_valid_for_smooth"] = finite
    out["outlier_reason"] = np.where(finite, "raw_used_no_filter_no_smooth", "nonfinite_or_skipped")
    out["vx_filtered"] = vx
    out["vy_filtered"] = vy
    out["vx_interp"] = vx
    out["vy_interp"] = vy
    out["vx_smooth"] = vx
    out["vy_smooth"] = vy
    out["velocity_usage_mode"] = "raw_local_velocity_no_filter_no_smooth"
    return out


def smooth_velocity_curve(df: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    vx_source = "vx_filtered" if "vx_filtered" in out.columns else "vx"
    vy_source = "vy_filtered" if "vy_filtered" in out.columns else "vy"
    out["vx_interp"] = out[vx_source].interpolate(limit_direction="both")
    out["vy_interp"] = out[vy_source].interpolate(limit_direction="both")
    if smooth_window <= 1:
        out["vx_smooth"] = out["vx_interp"]
        out["vy_smooth"] = out["vy_interp"]
        return out

    win = int(smooth_window)
    if win % 2 == 0:
        win += 1
    out["vx_smooth"] = out["vx_interp"].rolling(win, center=True, min_periods=1).median()
    out["vy_smooth"] = out["vy_interp"].rolling(win, center=True, min_periods=1).median()
    return out


def save_velocity_stability_report(df: pd.DataFrame, out_dir: Path) -> Dict:
    if df.empty or "vx_smooth" not in df or "vy_smooth" not in df:
        return dict(is_constant_like=False, reason="no_velocity_curve")

    out = df.copy()
    out["time_bin_s"] = np.floor(out["center_us"].to_numpy(dtype=np.float64) / 1e6).astype(int)
    out["speed_smooth"] = np.sqrt(out["vx_smooth"] ** 2 + out["vy_smooth"] ** 2)
    if "is_valid_for_smooth" in out:
        valid = out["is_valid_for_smooth"].astype(bool)
    else:
        valid = np.isfinite(out["vx_smooth"]) & np.isfinite(out["vy_smooth"])
    out["valid_for_report"] = valid

    finite_valid = out[
        valid &
        np.isfinite(out["vx_smooth"]) &
        np.isfinite(out["vy_smooth"]) &
        np.isfinite(out["speed_smooth"])
    ].copy()

    bin_rows = []
    for bin_s, sub in out.groupby("time_bin_s"):
        sub_valid = sub[sub["valid_for_report"]]
        finite = sub_valid[
            np.isfinite(sub_valid["vx_smooth"]) &
            np.isfinite(sub_valid["vy_smooth"]) &
            np.isfinite(sub_valid["speed_smooth"])
        ]
        score_values = sub["score"].replace([np.inf, -np.inf], np.nan) if "score" in sub else pd.Series(dtype=float)
        score_values = score_values[np.isfinite(score_values)]
        bin_rows.append(
            dict(
                time_bin_s=int(bin_s),
                n_windows=int(sub.shape[0]),
                n_valid=int(finite.shape[0]),
                score_median=float(score_values.median()) if score_values.size else np.nan,
                vx_mean=float(finite["vx_smooth"].mean()) if not finite.empty else np.nan,
                vy_mean=float(finite["vy_smooth"].mean()) if not finite.empty else np.nan,
                speed_mean=float(finite["speed_smooth"].mean()) if not finite.empty else np.nan,
                vx_std=float(finite["vx_smooth"].std(ddof=0)) if not finite.empty else np.nan,
                vy_std=float(finite["vy_smooth"].std(ddof=0)) if not finite.empty else np.nan,
                speed_std=float(finite["speed_smooth"].std(ddof=0)) if not finite.empty else np.nan,
                vx_min=float(finite["vx_smooth"].min()) if not finite.empty else np.nan,
                vx_max=float(finite["vx_smooth"].max()) if not finite.empty else np.nan,
                vy_min=float(finite["vy_smooth"].min()) if not finite.empty else np.nan,
                vy_max=float(finite["vy_smooth"].max()) if not finite.empty else np.nan,
            )
        )

    bins_df = pd.DataFrame(bin_rows)
    bins_df.to_csv(out_dir / "velocity_stability_by_1s.csv", index=False)

    if finite_valid.empty:
        return dict(is_constant_like=False, reason="no_valid_velocity_points")

    vx_std = float(finite_valid["vx_smooth"].std(ddof=0))
    vy_std = float(finite_valid["vy_smooth"].std(ddof=0))
    speed_std = float(finite_valid["speed_smooth"].std(ddof=0))
    vx_range = float(finite_valid["vx_smooth"].max() - finite_valid["vx_smooth"].min())
    vy_range = float(finite_valid["vy_smooth"].max() - finite_valid["vy_smooth"].min())
    speed_range = float(finite_valid["speed_smooth"].max() - finite_valid["speed_smooth"].min())
    is_constant_like = (
        vx_std <= float(0.005) and
        vy_std <= float(0.005) and
        vx_range <= float(0.02) and
        vy_range <= float(0.02)
    )

    return dict(
        is_constant_like=bool(is_constant_like),
        n_valid_points=int(finite_valid.shape[0]),
        vx_mean=float(finite_valid["vx_smooth"].mean()),
        vy_mean=float(finite_valid["vy_smooth"].mean()),
        speed_mean=float(finite_valid["speed_smooth"].mean()),
        vx_std=vx_std,
        vy_std=vy_std,
        speed_std=speed_std,
        vx_range=vx_range,
        vy_range=vy_range,
        speed_range=speed_range,
        std_threshold=float(0.005),
        range_threshold=float(0.02),
    )


# =============================================================================
# Piecewise reconstruction
# =============================================================================

def cumulative_displacement_from_velocity(
    centers_us: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    disp = np.zeros_like(centers_us, dtype=np.float64)
    if centers_us.size <= 1:
        return disp
    dt_ms = np.diff(centers_us.astype(np.float64)) / 1000.0
    disp[1:] = np.cumsum(0.5 * (v[:-1] + v[1:]) * dt_ms)
    return disp


def interpolate_displacement(
    t_us: np.ndarray,
    centers_us: np.ndarray,
    v: np.ndarray,
    disp_centers: np.ndarray,
) -> np.ndarray:
    t = t_us.astype(np.float64)
    if centers_us.size == 1:
        return disp_centers[0] + v[0] * (t - centers_us[0]) / 1000.0

    disp = np.interp(t, centers_us, disp_centers)
    left = t < centers_us[0]
    right = t > centers_us[-1]
    if np.any(left):
        disp[left] = disp_centers[0] + v[0] * (t[left] - centers_us[0]) / 1000.0
    if np.any(right):
        disp[right] = disp_centers[-1] + v[-1] * (t[right] - centers_us[-1]) / 1000.0
    return disp


def warp_events_piecewise(
    events: np.ndarray,
    velocity_df: pd.DataFrame,
    t_ref_us: float,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    curve = velocity_df[
        np.isfinite(velocity_df["vx_smooth"]) &
        np.isfinite(velocity_df["vy_smooth"])
    ].copy()
    if curve.empty:
        raise RuntimeError("No finite local velocities available for reconstruction.")

    curve = curve.sort_values("center_us")
    centers_us = curve["center_us"].to_numpy(dtype=np.float64)
    vx = curve["vx_smooth"].to_numpy(dtype=np.float64)
    vy = curve["vy_smooth"].to_numpy(dtype=np.float64)

    dx_centers = cumulative_displacement_from_velocity(centers_us, vx)
    dy_centers = cumulative_displacement_from_velocity(centers_us, vy)

    t_us = events["t"].astype(np.float64)
    disp_x = interpolate_displacement(t_us, centers_us, vx, dx_centers)
    disp_y = interpolate_displacement(t_us, centers_us, vy, dy_centers)
    disp_ref_x = float(interpolate_displacement(np.asarray([t_ref_us], dtype=np.float64), centers_us, vx, dx_centers)[0])
    disp_ref_y = float(interpolate_displacement(np.asarray([t_ref_us], dtype=np.float64), centers_us, vy, dy_centers)[0])

    xs = events["x"].astype(np.float64) + (disp_x - disp_ref_x)
    ys = events["y"].astype(np.float64) + (disp_y - disp_ref_y)
    meta = dict(
        t_ref_us=float(t_ref_us),
        n_velocity_points=int(curve.shape[0]),
        vx_mean=float(np.mean(vx)),
        vy_mean=float(np.mean(vy)),
        vx_std=float(np.std(vx)),
        vy_std=float(np.std(vy)),
        vx_min=float(np.min(vx)),
        vx_max=float(np.max(vx)),
        vy_min=float(np.min(vy)),
        vy_max=float(np.max(vy)),
        displacement_ref_x=float(disp_ref_x),
        displacement_ref_y=float(disp_ref_y),
        displacement_start_x=float(interpolate_displacement(np.asarray([t_us.min()]), centers_us, vx, dx_centers)[0]),
        displacement_end_x=float(interpolate_displacement(np.asarray([t_us.max()]), centers_us, vx, dx_centers)[0]),
        displacement_start_y=float(interpolate_displacement(np.asarray([t_us.min()]), centers_us, vy, dy_centers)[0]),
        displacement_end_y=float(interpolate_displacement(np.asarray([t_us.max()]), centers_us, vy, dy_centers)[0]),
    )
    return xs, ys, meta


def expanded_canvas_from_shifted(
    xs: np.ndarray,
    ys: np.ndarray,
    sensor_H: int,
    sensor_W: int,
    include_sensor_bounds: bool,
) -> Dict[str, Union[int, float, bool]]:
    if xs.size == 0:
        raise RuntimeError("Cannot infer canvas from empty coordinates.")

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    if include_sensor_bounds:
        x_min = min(0.0, x_min)
        y_min = min(0.0, y_min)
        x_max = max(float(sensor_W - 1), x_max)
        y_max = max(float(sensor_H - 1), y_max)

    x_origin = float(np.floor(x_min))
    y_origin = float(np.floor(y_min))
    W = int(np.floor(x_max - x_origin) + 2)
    H = int(np.floor(y_max - y_origin) + 2)
    return dict(
        x_shifted_min=x_min,
        x_shifted_max=x_max,
        y_shifted_min=y_min,
        y_shifted_max=y_max,
        x_origin=x_origin,
        y_origin=y_origin,
        W=W,
        H=H,
        width_before=int(sensor_W),
        height_before=int(sensor_H),
        width_after=W,
        height_after=H,
        include_sensor_bounds=bool(include_sensor_bounds),
    )


def accumulate_iwe(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    H: int,
    W: int,
    mode: str,
) -> Tuple[np.ndarray, int]:
    img = np.zeros((int(H), int(W)), dtype=np.float32)
    if xs.size == 0:
        return img, 0

    mode = mode.lower()
    if mode == "nearest":
        return accumulate_nearest_bincount(xs, ys, weights, H, W)

    if mode != "bilinear":
        raise ValueError("FINAL_SPLAT_MODE must be 'nearest' or 'bilinear'.")

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    dx = xs - x0
    dy = ys - y0
    valid = (x0 >= 0) & (x0 + 1 < int(W)) & (y0 >= 0) & (y0 + 1 < int(H))
    if not np.any(valid):
        return img, 0

    x0m = x0[valid]
    y0m = y0[valid]
    dxm = dx[valid].astype(np.float32)
    dym = dy[valid].astype(np.float32)
    wm = weights[valid].astype(np.float32)

    np.add.at(img, (y0m, x0m), (1.0 - dxm) * (1.0 - dym) * wm)
    np.add.at(img, (y0m, x0m + 1), dxm * (1.0 - dym) * wm)
    np.add.at(img, (y0m + 1, x0m), (1.0 - dxm) * dym * wm)
    np.add.at(img, (y0m + 1, x0m + 1), dxm * dym * wm)
    return img, int(valid.sum())


def build_piecewise_iwe_expanded(
    events: np.ndarray,
    weights: np.ndarray,
    velocity_df: pd.DataFrame,
    H: int,
    W: int,
    t_ref_us: float,
) -> Tuple[np.ndarray, int, Dict]:
    xs, ys, warp_meta = warp_events_piecewise(events, velocity_df, t_ref_us)
    canvas = expanded_canvas_from_shifted(
        xs,
        ys,
        sensor_H=H,
        sensor_W=W,
        include_sensor_bounds=True,
    )
    local_x = xs - float(canvas["x_origin"])
    local_y = ys - float(canvas["y_origin"])
    img, n_valid = accumulate_iwe(local_x, local_y, weights, int(canvas["H"]), int(canvas["W"]), FINAL_SPLAT_MODE)
    canvas.update(warp_meta)
    canvas["n_valid"] = int(n_valid)
    canvas["n_total"] = int(events.size)
    canvas["valid_ratio"] = 0.0 if events.size == 0 else float(n_valid / events.size)
    return img, n_valid, canvas


# =============================================================================
# Output
# =============================================================================

def normalize_to_uint16(img: np.ndarray) -> np.ndarray:
    clip_percentile = 99.995
    gamma = 1.0
    I = img.astype(np.float32, copy=True)
    I[~np.isfinite(I)] = 0
    lo = float(np.min(I))
    hi = float(np.max(I)) if clip_percentile is None else float(np.percentile(I, clip_percentile))
    if not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(I, dtype=np.uint16)
    J = np.clip((I - lo) / (hi - lo), 0, 1)
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("The fixed output gamma must be positive.")
    J = np.power(J, gamma)
    return np.round(J * 65535).astype(np.uint16)


def format_time_seconds(t_us: float) -> str:
    """Format microseconds as a compact filename-safe seconds label."""
    value = f"{float(t_us) / 1e6:.3f}".rstrip("0").rstrip(".")
    return value.replace("-", "m").replace(".", "p")


def make_time_output_prefix(start_us: float, end_us: float) -> str:
    return f"{format_time_seconds(start_us)}s-{format_time_seconds(end_us)}s"


def save_tif_output(img: np.ndarray, output_path: Path) -> None:
    """Save only the normalized 16-bit TIFF reconstruction."""
    imageio.imwrite(output_path, normalize_to_uint16(img))




def save_global_score_map(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    sub = df[df["stage"] == "global_fine"].copy()
    if sub.empty:
        sub = df.copy()
    pivot = sub.pivot_table(index="vy", columns="vx", values="score", aggfunc="max")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    if pivot.empty:
        return
    vx_vals = pivot.columns.to_numpy(dtype=float)
    vy_vals = pivot.index.to_numpy(dtype=float)
    Z = pivot.to_numpy(dtype=float)
    finite = np.isfinite(Z)
    if not np.any(finite):
        return
    fill = float(np.nanmin(Z[finite]))
    Z = np.nan_to_num(Z, nan=fill, posinf=fill, neginf=fill)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[vx_vals.min(), vx_vals.max(), vy_vals.min(), vy_vals.max()],
    )
    ax.set_xlabel("vx (px/ms)")
    ax.set_ylabel("vy (px/ms)")
    ax.set_title("Automatic global velocity score map")
    fig.colorbar(im, ax=ax, label="score (variance)")
    fig.tight_layout()
    fig.savefig(out_dir / "global_velocity_score_map.png", dpi=200)
    plt.close(fig)


# =============================================================================
# CLI and main
# =============================================================================

def parse_range(values: Optional[List[float]], default: Tuple[float, float]) -> Tuple[float, float]:
    if values is None:
        return tuple(map(float, default))
    if len(values) != 2:
        raise ValueError("range arguments need exactly two values.")
    return float(values[0]), float(values[1])


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-range",
        action="store_true",
        help="Process one interval using the built-in single-range defaults instead of TIME_RANGES.",
    )
    parser.set_defaults(
        file=Path(FILE_PATH),
        start_us=180000,
        end_us=2580000,
        vx_range=None,
        vy_range=None,
        output_dir=Path(OUTPUT_DIR) if OUTPUT_DIR is not None else None,
        skip_reconstruction=False,
    )
    return parser


def run_single_range(args) -> Dict:
    file_path = Path(args.file)
    vx_range = parse_range(args.vx_range, VX_RANGE)
    vy_range = parse_range(args.vy_range, (-4.87, -4.79))
    out_dir = infer_output_dir(file_path, args.output_dir)
    time_prefix = make_time_output_prefix(float(args.start_us), float(args.end_us))
    tif_path = out_dir / f"{time_prefix}_piecewise_2d_tdi.tif"

    print(f"[Output] {out_dir}")
    print(f"[Config] vx_range={vx_range}, vy_range={vy_range}")
    print(f"[Time] {float(args.start_us) / 1e6:.3f}s - {float(args.end_us) / 1e6:.3f}s")

    t0 = time.perf_counter()
    events, weights, prep_meta = prepare_events(
        file_path,
        float(args.start_us),
        float(args.end_us),
        XY_LIMITS,
        POLARITY_MODE,
    )
    H, W = int(prep_meta["H"]), int(prep_meta["W"])

    est_events, est_weights, estimation_roi = make_estimation_events(
        events,
        weights,
        prep_meta,
        float(args.start_us),
        float(args.end_us),
    )
    est_H, est_W = int(estimation_roi["H"]), int(estimation_roi["W"])
    print(
        "[4b] Velocity estimation ROI: "
        f"mode={estimation_roi['mode']}, "
        f"local=({estimation_roi['x0']},{estimation_roi['y0']})-"
        f"({estimation_roi['x1']},{estimation_roi['y1']}), "
        f"original=({estimation_roi['original_x0']},{estimation_roi['original_y0']})-"
        f"({estimation_roi['original_x1']},{estimation_roi['original_y1']}), "
        f"H={est_H}, W={est_W}, events={est_events.size}"
    )

    global_best, _global_df = automatic_global_velocity_init(
        est_events,
        est_weights,
        est_H,
        est_W,
        float(args.start_us),
        float(args.end_us),
        vx_range,
        vy_range,
        keep_rows=False,
    )

    local_df, _local_grid_df = estimate_local_velocity_curve(
        est_events,
        est_weights,
        est_H,
        est_W,
        float(args.start_us),
        float(args.end_us),
        vx_range,
        vy_range,
        init_vx=float(global_best["vx"]),
        init_vy=float(global_best["vy"]),
        keep_rows=False,
    )
    if USE_RAW_LOCAL_VELOCITY_FOR_RECON:
        print("    velocity usage: raw local estimates, no outlier rejection, no interpolation, no smoothing")
        local_df = use_raw_local_velocity_curve(local_df)
    else:
        local_df = mark_local_velocity_outliers(
            local_df,
            anchor_vx=float(global_best["vx"]),
            anchor_vy=float(global_best["vy"]),
        )
        local_df = smooth_velocity_curve(local_df, 1)

    # Local motion estimates are used internally for image reconstruction only.
    finite = local_df[
        np.isfinite(local_df["vx_smooth"]) &
        np.isfinite(local_df["vy_smooth"])
    ] if not local_df.empty else pd.DataFrame()

    if finite.empty and not args.skip_reconstruction:
        raise RuntimeError(
            "No valid local motion estimates; image reconstruction cannot proceed."
        )

    n_valid = None
    final_canvas = None
    if not args.skip_reconstruction:
        print("[7] Build final expanded-canvas piecewise IWE")
        t_ref_us = 0.5 * (float(args.start_us) + float(args.end_us))
        img, n_valid, final_canvas = build_piecewise_iwe_expanded(
            events,
            weights,
            local_df,
            H,
            W,
            t_ref_us=t_ref_us,
        )
        # Output 2/2: time-named 16-bit reconstruction TIFF.
        save_tif_output(img, tif_path)
        print(
            "    final canvas: "
            f"{final_canvas['width_before']}x{final_canvas['height_before']} -> "
            f"{final_canvas['width_after']}x{final_canvas['height_after']}, "
            f"origin=({final_canvas['x_origin']:.3f}, {final_canvas['y_origin']:.3f}), "
            f"valid={n_valid}/{events.size}"
        )

    elapsed_s = time.perf_counter() - t0
    print("[DONE]")
    print(f"    global init best: vx={global_best['vx']:+.6f}, vy={global_best['vy']:+.6f}")
    print(f"    elapsed: {elapsed_s:.1f} s")
    if not args.skip_reconstruction:
        print(f"    saved: {tif_path.name}")

    return dict(
        start_us=float(args.start_us),
        end_us=float(args.end_us),
        elapsed_s=float(elapsed_s),
        global_best=global_best,
        tif_path=str(tif_path) if not args.skip_reconstruction else None,
    )


def main() -> None:
    args = make_parser().parse_args()

    use_batch = bool(TIME_RANGES and not args.single_range)
    if not use_batch:
        run_single_range(args)
        return

    file_path = Path(args.file)
    base_out_dir = infer_output_dir(file_path, args.output_dir)
    print(f"[Batch] {len(TIME_RANGES)} time ranges")
    print(f"[Batch output] {base_out_dir}")
    print(
        "[Display] "
        f"clip_percentile={99.995:g}, gamma={1.0:g}"
    )

    n_ok = 0
    n_failed = 0
    batch_t0 = time.perf_counter()
    for index, (start_us, end_us, vy_center) in enumerate(TIME_RANGES, start=1):
        if float(end_us) <= float(start_us):
            raise ValueError(f"Invalid TIME_RANGES[{index - 1}]: end must be greater than start.")
        if not np.isfinite(vy_center):
            raise ValueError(f"Invalid TIME_RANGES[{index - 1}]: vy_center must be finite.")

        # Set the velocity bounds separately for each segment.
        # Preserve -/+ signs: -5.34 -> (-5.44, -5.24),
        #                       +5.34 -> (+5.24, +5.44).
        range_args = argparse.Namespace(**vars(args))
        range_args.start_us = float(start_us)
        range_args.end_us = float(end_us)
        range_args.vx_range = tuple(VX_RANGE)
        range_args.vy_range = (
            round(float(vy_center) - VY_HALF_WIDTH, 6),
            round(float(vy_center) + VY_HALF_WIDTH, 6),
        )
        range_args.output_dir = base_out_dir

        print("\n" + "=" * 80)
        print(
            f"[Batch {index:02d}/{len(TIME_RANGES):02d}] "
            f"{float(start_us) / 1e6:.3f}s - {float(end_us) / 1e6:.3f}s, "
            f"vy_center={float(vy_center):+.2f} px/ms"
        )
        try:
            run_single_range(range_args)
            n_ok += 1
        except Exception as exc:
            n_failed += 1
            print(f"[ERROR] Batch range {index} failed: {type(exc).__name__}: {exc}")

    elapsed_s = time.perf_counter() - batch_t0
    print("\n" + "=" * 80)
    print(f"[BATCH DONE] ok={n_ok}, failed={n_failed}, elapsed={elapsed_s:.1f}s")
    print(f"    TIFF reconstructions were saved to: {base_out_dir}")


if __name__ == "__main__":
    main()
