"""Load all events from given raw file"""


def load_event(file_path):
    import numpy as np
    print('Loading data...')
    if file_path[-4:] == '.raw':
        from metavision_core.event_io.raw_reader import RawReader
        buffer_size = 1e9
        record_raw = RawReader(file_path, max_events=int(buffer_size))
        sums = 0
        while not record_raw.is_done() and record_raw.current_event_index() < buffer_size:
            events = record_raw.load_delta_t(50000)
            sums += events.size

        record_raw.reset()
        events = record_raw.load_n_events(sums)

    elif file_path[-4:] == '.npy':
        events = np.load(file_path)
        nb_ev_init = len(events)
    print('Data loaded')
    return events


"""Load events from given point and delta_t"""


def load_event_t(file_path, t_start, t_end, buffer_size=1e9):
    """
    :param t_start:  in [us]
    :param t_end:  in [us]
    """
    from metavision_core.event_io.raw_reader import RawReader
    record_raw = RawReader(file_path, max_events=int(buffer_size))
    record_raw.seek_time(t_start)
    if record_raw.is_done():
        print("Start timestamp is out of events times range")
    delta_t = t_end - t_start
    events = record_raw.load_delta_t(delta_t)
    if record_raw.is_done():
        print("End timestamp is out of events times range")
    return events


def load_event_t_with_roi_and_trange(file_path, t_start, t_end, xy_limits, chunk_duration_us=50000):
    import numpy as np
    from tqdm import tqdm
    from metavision_core.event_io.raw_reader import RawReader
    """
    An efficient event loading function that follows the 'load in chunks, filter in chunks' principle.

    :param file_path: Path to the .raw file
    :param t_start: Start time [us]
    :param t_end: End time [us]
    :param xy_limits: Spatial cropping region [x_min, y_min, x_max, y_max]
    :param chunk_duration_us: Duration of each chunk to load [us], for the progress bar
    :return: A NumPy array of events filtered both spatially and temporally
    """
    record_raw = RawReader(file_path)

    record_raw.seek_time(t_start)
    if record_raw.is_done():
        print("Warning: Start timestamp is outside the event data range")
        return np.empty(0, dtype=[('x', '<u2'), ('y', '<u2'), ('p', '<i1'), ('t', '<i8')])

    total_duration_us = t_end - t_start
    all_events_filtered = []

    x_min, y_min, x_max, y_max = xy_limits
    print(f"Data loading with chunked NumPy filter: x=[{x_min}-{x_max}], y=[{y_min}-{y_max}]")

    # --- PROGRESS BAR FIX: Use seconds for a more readable display ---
    with tqdm(total=total_duration_us / 1e6, desc="Loading & Filtering", unit="s") as pbar:
        current_time = t_start
        while current_time < t_end and not record_raw.is_done():
            remaining_time = t_end - current_time
            load_duration_us = min(chunk_duration_us, remaining_time)

            events_chunk_unfiltered = record_raw.load_delta_t(load_duration_us)

            if events_chunk_unfiltered.size > 0:
                chunk = events_chunk_unfiltered
                mask = (chunk['x'] >= x_min) & (chunk['x'] < x_max) & \
                       (chunk['y'] >= y_min) & (chunk['y'] < y_max)
                filtered_chunk = chunk[mask]
                if filtered_chunk.size > 0:
                    all_events_filtered.append(filtered_chunk)

            # Update the progress bar by the duration in seconds
            pbar.update(load_duration_us / 1e6)

            current_time += load_duration_us

    if not all_events_filtered:
        return np.empty(0, dtype=[('x', '<u2'), ('y', '<u2'), ('p', '<i1'), ('t', '<i8')])
    final_events = np.concatenate(all_events_filtered)

    frame_size = [np.max(final_events['y']) + 1, np.max(final_events['x']) + 1]
    t_range = [final_events['t'].min(), final_events['t'].max()]
    print(f"Data loaded. Total events in ROI: {len(final_events)}")

    return final_events, frame_size, t_range


def load_events_new(file_path, t_start, t_end, xy_limits, chunk_duration_us=50000, to_gpu=True):
    """
    An efficient and flexible event loading function. It loads data in chunks,
    filters them on the CPU, and then returns either a consolidated NumPy structured
    array or a dictionary of CuPy column arrays for GPU processing.

    :param file_path: Path to the .raw file.
    :param t_start: Start time [us].
    :param t_end: End time [us].
    :param xy_limits: Spatial cropping region [x_min, y_min, x_max, y_max].
    :param chunk_duration_us: Duration of each chunk to load [us], for the progress bar.
    :param to_gpu: If False (default), returns a NumPy structured array.
                   If True, returns a dictionary of CuPy arrays {'x', 'y', 'p', 't'}.
    :return: A NumPy array (CPU) or a dictionary of CuPy arrays (GPU).
    """
    import numpy as np
    from tqdm import tqdm
    from metavision_core.event_io.raw_reader import RawReader

    # --- Step 1: Load and filter chunks on CPU (code is unchanged) ---
    record_raw = RawReader(file_path)
    record_raw.seek_time(t_start)
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('p', '<i1'), ('t', '<i8')])

    if record_raw.is_done():
        print("Warning: Start timestamp is outside the event data range")
        return np.empty(0, dtype=dtype)

    total_duration_us = t_end - t_start
    all_events_filtered = []
    x_min, y_min, x_max, y_max = xy_limits
    print(f"Data loading with chunked NumPy filter: x=[{x_min}-{x_max}], y=[{y_min}-{y_max}]")

    with tqdm(total=total_duration_us / 1e6, desc="[IO] Loading & Filtering", unit="s") as pbar:
        current_time = t_start
        while current_time < t_end and not record_raw.is_done():
            remaining_time = t_end - current_time
            load_duration_us = min(chunk_duration_us, remaining_time)
            events_chunk_unfiltered = record_raw.load_delta_t(load_duration_us)

            if events_chunk_unfiltered.size > 0:
                chunk = events_chunk_unfiltered
                mask = (chunk['x'] >= x_min) & (chunk['x'] < x_max) & \
                       (chunk['y'] >= y_min) & (chunk['y'] < y_max)
                filtered_chunk = chunk[mask]
                if filtered_chunk.size > 0:
                    all_events_filtered.append(filtered_chunk)

            pbar.update(load_duration_us / 1e6)
            current_time += load_duration_us

    if not all_events_filtered:
        print("No events found in the specified time range and ROI.")
        # Return the correct empty format based on the target device
        if to_gpu:
            import cupy as cp
            return {'x': cp.empty(0, dtype=cp.int32), 'y': cp.empty(0, dtype=cp.int32),
                    'p': cp.empty(0, dtype=cp.uint8), 't': cp.empty(0, dtype=cp.int64)}
        else:
            return np.empty(0, dtype=dtype)

    # --- Step 2: Consolidate into a single NumPy array on CPU ---
    final_events_cpu = np.concatenate(all_events_filtered)
    print(f"Data loaded to CPU. Total events in ROI: {len(final_events_cpu)}")
    frame_size = [np.max(final_events_cpu['y']) + 1, np.max(final_events_cpu['x']) + 1]
    t_range = [final_events_cpu['t'].min(), final_events_cpu['t'].max()]
    # --- Step 3: Based on to_gpu, either return the CPU array or convert to GPU columns ---
    if to_gpu:
        print("Transferring data to GPU and converting to columnar format...")
        import cupy as cp

        # Convert to the specified GPU format (dictionary of CuPy arrays)
        events_gpu_dict = {
            'x': cp.asarray(final_events_cpu['x'], dtype=cp.int32),
            'y': cp.asarray(final_events_cpu['y'], dtype=cp.int32),
            'p': cp.asarray(final_events_cpu['p'], dtype=cp.uint8),
            't': cp.asarray(final_events_cpu['t'], dtype=cp.int64)
        }
        print("Transfer complete.")
        return events_gpu_dict, frame_size, t_range
    else:
        # Return the NumPy structured array directly
        return final_events_cpu, frame_size, t_range
