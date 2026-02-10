"""
trr_recon_stream.py

Description:
This script implements image reconstruction from event-based sensor data using
Time-Reversal Reassignment (TRR) - analogue to Time Delay Integration (TDI) principles.
Supports motion vectors in both X and Y directions.

Author: Qianxi Liang (梁谦禧)
Date: 2026-2-10
"""

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from tqdm import tqdm
from metavision_core.event_io.raw_reader import RawReader


def trr_propress(
        file_path,
        events,
        chunk_idx,
        v_px_per_ms,
        axis,
        t_ref
):
    """
    Applies TRR motion compensation to a chunk of events and saves the intermediate result to disk.

    Motion Compensation Logic:
        - If axis = 'y': y' = y + v * (t - t_ref); x' = x
        - If axis = 'x': x' = x + v * (t - t_ref); y' = y
        - If axis = 'xy':
            x' = x + vx * (t - t_ref)
            y' = y + vy * (t - t_ref)

    Args:
        file_path (str): Path to the source raw file. Used to determine the output
                         directory for intermediate .npy files.
        events (ndarray): Structured array containing event data.
                          Expected fields: 'x', 'y', 't', 'p'.
                          Time 't' unit must match 'v_px_per_ms' (usually ms).
        chunk_idx (int): The index of the current time chunk (used for file naming).
        v_px_per_ms (float or tuple): Sample velocity in pixels/ms.
                                      - Float: For 1D motion (along 'x' or 'y').
                                      - Tuple/List: [vx, vy] for 2D motion.
        axis (str): Direction of motion compensation. Options: 'x', 'y', 'xy'.
        t_ref (float): Reference time (t0) used as the zero-point for coordinate shifting.

    Returns:
        tuple: A tuple containing:
            - int: Number of events processed in this chunk.
            - str: Path to the saved .npy file containing shifted events [x, y, t, p].
            - list: Bounding box of shifted coordinates [x_min, x_max, y_min, y_max].
                    Returns (0, None, None) if the chunk is empty.

    """

    if events.size == 0:
        return 0, None, None, None, None, None

    x = events['x']  # uint16
    y = events['y']  # uint16
    t = events['t'] / 1000.  # int64->float64 (us->ms)
    p = events['p']

    # Set reference time t_ref
    if t_ref is None:
        t_ref = t.min()

    # TRR coordinate shift
    if axis.lower() == 'y':
        y_shifted = y + v_px_per_ms * (t - t_ref)
        x_shifted = x.copy()
    elif axis.lower() == 'x':
        x_shifted = x + v_px_per_ms * (t - t_ref)
        y_shifted = y.copy()
    elif axis.lower() == 'xy':
        x_shifted = x + v_px_per_ms[0] * (t - t_ref)
        y_shifted = y + v_px_per_ms[1] * (t - t_ref)
    else:
        raise ValueError("axis 必须为 'x' 或 'y'")


    root_dir = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    chunk_dir = os.path.join(root_dir, f"{base}_stream_chunks_v_{v_px_per_ms}")
    os.makedirs(chunk_dir, exist_ok=True)
    if axis.lower() == 'y':
        npy_path = os.path.join(
            chunk_dir,
            f"{base}_chunk_{chunk_idx:04d}_vy_{v_px_per_ms:+.5f}.npy"
        )
    elif axis.lower() == 'x':
        npy_path = os.path.join(
            chunk_dir,
            f"{base}_chunk_{chunk_idx:04d}_vx_{v_px_per_ms:+.5f}.npy"
        )
    elif axis.lower() == 'xy':
        npy_path = os.path.join(
            chunk_dir,
            f"{base}_chunk_{chunk_idx:04d}_vx_{v_px_per_ms[0]:+.5f}_vy_{v_px_per_ms[1]:+.5f}.npy"
        )

    np.save(npy_path, np.column_stack([x_shifted, y_shifted, t * 1000, p]).astype(np.float32))

    x_min_c = float(x_shifted.min())
    x_max_c = float(x_shifted.max())
    y_min_c = float(y_shifted.min())
    y_max_c = float(y_shifted.max())

    return p.shape[0], npy_path, [x_min_c, x_max_c, y_min_c, y_max_c]


def trr_reconstruct_file_by_time_chunks(
        file_path,
        t_global_start_us,
        t_global_end_us,
        v_px_per_ms,
        axis='y',
        xy_limits=None,
        chunk_len_us=2e6,  # 每块长度，默认 2 s
        p_pos_neg=[0, 1],
        splat_mode='bilinear'
):
    """
    Reconstructs an image from a large event file using a stream-processing approach.

    Args:
        file_path (str): Path to the input .raw file.
        t_global_start_us (float): Global start time for reconstruction in microseconds.
        t_global_end_us (float): Global end time for reconstruction in microseconds.
        v_px_per_ms (float or list): Velocity in px/ms. Float for 1D, [vx, vy] for 2D.
        axis (str): Motion axis ('x', 'y', or 'xy').
        xy_limits (list, optional): Crop ROI [x_min, y_min, x_max, y_max].
                                    Events outside this box are discarded before processing.
        chunk_len_us (float): Duration of each processing chunk in microseconds.
                              Smaller chunks use less RAM but may increase I/O overhead.
        p_pos_neg (list): Weights for negative (0) and positive (1) polarity events.
                          Example: [0, 1] maps pol=0 to 0.0 and pol=1 to 1.0.
        splat_mode (str): Interpolation method for accumulation ('nearest' or 'bilinear').
    Returns:
        tuple: (img_u16, metadata)
            - img_u16 (ndarray): The reconstructed image.
            - metadata (dict): Contains 'H', 'W', and 'bbox' of the final image.
    """

    assert t_global_end_us > t_global_start_us
    assert chunk_len_us > 0

    t_ref_global_ms = t_global_start_us / 1000.0

    global_x_min = np.inf
    global_x_max = -np.inf
    global_y_min = np.inf
    global_y_max = -np.inf

    total_chunks = int(np.ceil((t_global_end_us - t_global_start_us) / chunk_len_us))
    all_npy_paths = []
    total_events = 0

    buffer_size = 1e8
    record_raw = RawReader(file_path, max_events=int(buffer_size))
    for i in tqdm(range(total_chunks), desc="TRR Streaming [1/2] coord shift"):
        cur_start_us = t_global_start_us + i * chunk_len_us
        cur_end_us = min(cur_start_us + chunk_len_us, t_global_end_us)

        record_raw.seek_time(cur_start_us)
        events = record_raw.load_delta_t(cur_end_us - cur_start_us)
        if not xy_limits is None:
            if xy_limits == 'auto':
                xy_limits = [np.min(events['x']),
                             np.min(events['y']),
                             np.max(events['x']) + 1,
                             np.max(events['y']) + 1]
            events = events[(events['x'] >= xy_limits[0]) *
                            (events['x'] < xy_limits[2]) *
                            (events['y'] >= xy_limits[1]) *
                            (events['y'] < xy_limits[3])]
            events['x'] -= xy_limits[0]
            events['y'] -= xy_limits[1]
        if events.size == 0:
            continue

        # bbox = [x_min_c, x_max_c, y_min_c, y_max_c]
        n_evt, npy_path, bbox = trr_propress(file_path,
                                             events,
                                             i,
                                             v_px_per_ms=v_px_per_ms,
                                             axis=axis,
                                             t_ref=t_ref_global_ms)

        if n_evt == 0:
            continue

        total_events += n_evt
        all_npy_paths.append(npy_path)
        global_x_min = min(global_x_min, bbox[0])
        global_x_max = max(global_x_max, bbox[1])
        global_y_min = min(global_y_min, bbox[2])
        global_y_max = max(global_y_max, bbox[3])

    x_min = global_x_min
    y_min = global_y_min
    W = int(np.ceil(global_x_max - global_x_min) + 1)
    H = int(np.ceil(global_y_max - global_y_min) + 1)

    img = np.zeros((H, W), dtype=np.float32)

    for npy_path in tqdm(all_npy_paths, desc="TRR Streaming [2/2] splat image"):
        arr = np.load(npy_path)
        if arr.size == 0:
            continue
        img = trr_accumulate(img, arr[:, 0], arr[:, 1], arr[:, 3],
                             x_ref=x_min,
                             y_ref=y_min,
                             p_pos_neg=p_pos_neg,
                             splat_mode=splat_mode)

    meta = dict(
        H=H,
        W=W,
        global_x_min=global_x_min,
        global_y_min=global_y_min,
        v_px_per_ms=v_px_per_ms,
        axis=axis,
        t_global_start_us=t_global_start_us,
        t_global_end_us=t_global_end_us,
        chunk_len_us=chunk_len_us,
        splat_mode=splat_mode,
    )


    mask_valid = np.isfinite(img)
    if np.any(mask_valid):
        v_eff = img[mask_valid]
        lo = v_eff.min()
        hi = v_eff.max()
        if hi > lo:
            t_norm = (img - lo) / (hi - lo)
        else:
            t_norm = np.zeros_like(img, dtype=np.float32)
        t_norm[~mask_valid] = 0.0
    else:
        t_norm = np.zeros_like(img, dtype=np.float32)

    t_norm = np.clip(t_norm, 0.0, 1.0)
    img_u16 = np.round(t_norm * 65535.0).astype(np.uint16)

    root_dir = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    if axis.lower() == 'y':
        out_tif = os.path.join(
            root_dir,
            f"{base}_StreamRecon_vy_{v_px_per_ms:+.5f}_"
            f"p{p_pos_neg[0]}_{p_pos_neg[1]}_{splat_mode}.tif"
        )
    elif axis.lower() == 'x':
        out_tif = os.path.join(
            root_dir,
            f"{base}_StreamRecon_vx_{v_px_per_ms:+.5f}_"
            f"p{p_pos_neg[0]}_{p_pos_neg[1]}_{splat_mode}.tif"
        )
    elif axis.lower() == 'xy':
        out_tif = os.path.join(
            root_dir,
            f"{base}_StreamRecon_vx_{v_px_per_ms[0]:+.5f}_vy_{v_px_per_ms[1]:+.5f}_"
            f"p{p_pos_neg[0]}_{p_pos_neg[1]}_{splat_mode}.tif"
        )

    imageio.imwrite(out_tif, img_u16)
    print(f"[INFO] 16-bit TIFF is saved: {out_tif}  | size: {img_u16.shape[0]}x{img_u16.shape[1]}")
    meta["out_tif"] = out_tif

    return img_u16, meta


def trr_accumulate(img, x_shifted, y_shifted, p,
                   x_ref=None,
                   y_ref=None,
                   p_pos_neg=None,
                   splat_mode='nearest'):
    """
    Accumulates shifted events onto the image grid.

    Args:
        img (ndarray): The target image canvas (H, W).
        x_shifted (ndarray): Shifted x-coordinates of events.
        y_shifted (ndarray): Shifted y-coordinates of events.
        p (ndarray): Event polarity array (0 or 1).
        x_ref (float, optional): Global x-offset to map shifted coordinates to image indices.
                                 Usually the minimum x value of the global scene.
        y_ref (float, optional): Global y-offset to map shifted coordinates to image indices.
        p_pos_neg (list, optional): Mapping for polarity weights [weight_for_0, weight_for_1].
                                    If None, raw polarity values are used directly.
        splat_mode (str): Splatting method.
                          - 'nearest': Fastest. Snaps events to the nearest pixel center.
                          - 'bilinear': Distributes event energy to 4 neighboring pixels
                                        based on sub-pixel position. Smoother results.

    Returns:
        ndarray: The updated image canvas.
    """

    if x_ref is None:
        x_ref = x_shifted.min()
    if y_ref is None:
        y_ref = y_shifted.min()

    # weight handling
    if p_pos_neg is None:
        w = p
    else:
        w = (p_pos_neg[1] - p_pos_neg[0]) * p + p_pos_neg[0]

    xs = x_shifted - x_ref
    ys = y_shifted - y_ref

    if splat_mode == 'nearest':
        x_idx = np.round(xs).astype(np.int64)
        y_idx = np.round(ys).astype(np.int64)
        np.add.at(img, (y_idx, x_idx), w)
    elif splat_mode == 'bilinear':
        H, W = img.shape
        img_padding = np.zeros((H + 1, W + 1))
        img_padding[:H, :W] = img

        x_base = np.floor(xs).astype(np.int64)
        y_base = np.floor(ys).astype(np.int64)
        dx = xs - x_base  # in [0,1)
        dy = ys - y_base  # in [0,1)

        x0_idx = x_base
        y0_idx = y_base
        x1_idx = x_base + 1
        y1_idx = y_base + 1

        np.add.at(img_padding, (y0_idx, x0_idx), (1.0 - dx) * (1.0 - dy) * w)
        np.add.at(img_padding, (y0_idx, x1_idx), dx * (1.0 - dy) * w)
        np.add.at(img_padding, (y1_idx, x0_idx), (1.0 - dx) * dy * w)
        np.add.at(img_padding, (y1_idx, x1_idx), dx * dy * w)
        img = img_padding[:H, :W]

    return img


if __name__ == "__main__":
    """.raw文件路径"""
    file_path = 'data/rawdata.raw'

    # Define time window (us)
    # t_start does not need offset adjustment here as we are not syncing frames yet
    t_global_start = 0 * 1e6  # us
    t_global_end = 3 * 1e6  # us

    v = -0.124
    xy_limits = None
    # splat_mode = 'nearest'
    splat_mode = 'bilinear'

    # 每块 2 s = 2e6 us
    img_global, meta = trr_reconstruct_file_by_time_chunks(
        file_path=file_path,
        t_global_start_us=t_global_start,
        t_global_end_us=t_global_end,
        v_px_per_ms=v,
        axis='y',
        xy_limits=xy_limits,
        chunk_len_us=5e5,
        p_pos_neg=[0, 1],
        splat_mode=splat_mode
    )

    plt.imshow(img_global, cmap='gray')
    plt.title(f"STREAM TRR recon (v={v:.5f} px/ms)")
    plt.colorbar()
    plt.show()
