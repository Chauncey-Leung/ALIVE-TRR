"""Small, SDK-free reader for the little-endian EVT3 RAW used by this project.

Only CD events and the RawReader time-window interface are supported. NumPy is
the only dependency. See docs/raw_reader.md for the format and limitations.
"""

import math
import struct

import numpy as np


# Match Metavision EventCD, including its two padding bytes before t.
EVENT_DTYPE = np.dtype({
    'names': ['x', 'y', 'p', 't'],
    'formats': ['<u2', '<u2', '<i2', '<i8'],
    'offsets': [0, 2, 4, 8], 'itemsize': 16,
})


def _microseconds(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError('Time must be finite and non-negative (microseconds)')
    return int(value)  # RawReader also truncates float microseconds.


class RawReader:
    """Read [current_time, current_time + delta_t) in integer microseconds.

    max_events is accepted for call-site compatibility; it does not preallocate
    an SDK-style ring buffer. Memory is bounded by one decoded block plus the
    requested output window. Backward seek is rejected; reset() rewinds.
    """

    def __init__(self, record_base, max_events=10_000_000,
                 do_time_shifting=True, *, chunk_words=65536):
        if int(max_events) <= 0 or int(chunk_words) <= 0:
            raise ValueError('max_events and chunk_words must be positive')
        self.path = str(record_base)
        self.do_time_shifting = bool(do_time_shifting)
        self._chunk_words = int(chunk_words)
        self._file = open(self.path, 'rb')
        try:
            self.header = self._read_header()
            fmt = self.header.get('format', '').split(';')
            if fmt[0] != 'EVT3' or self.header.get('evt', '3.0') != '3.0':
                raise ValueError('Only explicitly declared EVT3 / evt 3.0 RAW is supported')
            params = dict(item.split('=', 1) for item in fmt[1:] if '=' in item)
            # This implementation is deliberately limited to the verified capture
            # profile; other sensor byte orders must not be guessed from payload.
            if self.header.get('sensor_name') != 'IMX636':
                raise ValueError('Only the verified IMX636 little-endian EVT3 profile is supported')
            self.width, self.height = int(params['width']), int(params['height'])
            if not (0 < self.width <= 2048 and 0 < self.height <= 2048):
                raise ValueError('Invalid EVT3 geometry')
            if self.header.get('geometry', f'{self.width}x{self.height}') != f'{self.width}x{self.height}':
                raise ValueError('Conflicting RAW geometry declarations')
            self._data_offset = self._file.tell()
            self._file.seek(0, 2)
            if (self._file.tell() - self._data_offset) % 2:
                raise ValueError('Truncated EVT3 word (odd payload size)')
            self.reset()
        except BaseException:
            self.close()
            raise

    def _read_header(self):
        header = {}
        for _ in range(1024):
            line = self._file.readline(4097)
            if not line.startswith(b'% ') or len(line) > 4096:
                raise ValueError('Missing or invalid Prophesee RAW header')
            text = line[2:].decode('ascii').strip()
            if text == 'end':
                return header
            key, _, value = text.partition(' ')
            header[key] = value
        raise ValueError('RAW header is too long or missing % end')

    def reset(self):
        self._file.seek(self._data_offset)
        self.current_time = 0
        self._event_index = 0
        self._high = None
        self._low = self._loops = 0
        self.timestamp_shift = None
        self._y = self._base_x = self._base_p = None
        self._last_cd_time = None
        self._pending = np.empty(0, dtype=EVENT_DTYPE)
        self._eof = False

    def _decode_block(self):
        data = self._file.read(2 * self._chunk_words)
        if not data:
            self._eof = True
            if self._high is None:
                raise ValueError('No EVT_TIME_HIGH found; cannot establish time base')
            return np.empty(0, dtype=EVENT_DTYPE)
        xs, ys, ps, ts = [], [], [], []
        high, low, loops = self._high, self._low, self._loops
        y, base_x, base_p = self._y, self._base_x, self._base_p
        shift = self.timestamp_shift
        last_t = self._last_cd_time
        for (word,) in struct.iter_unpack('<H', data):
            kind, value = word >> 12, word & 0xFFF
            if high is None:
                if kind != 8:
                    continue  # SDK discards the incomplete prefix before TIME_HIGH.
                high = max(value - 1, 0)
                shift = high << 12
            if kind == 8:
                if high >= value + 2048:
                    loops += 1
                elif value < high:
                    raise ValueError('Non-monotonic EVT_TIME_HIGH; unsupported/corrupt stream')
                if high != value:
                    low = 0
                high = value
            elif kind == 6:
                low = value
            elif kind == 0:
                if value & 0x800:
                    raise ValueError('Slave-camera EVT3 streams are not supported')
                y = value & 0x7FF
                if y >= self.height:
                    raise ValueError('EVT3 Y coordinate outside declared geometry')
            elif kind == 3:
                base_x, base_p = value & 0x7FF, value >> 11
            elif kind in (2, 4, 5):
                if y is None:
                    raise ValueError('CD event without EVT_ADDR_Y')
                if kind == 2:
                    coords, polarity = (value & 0x7FF,), value >> 11
                else:
                    if base_x is None:
                        raise ValueError('Vector event without VECT_BASE_X')
                    bits = 12 if kind == 4 else 8
                    mask = value & ((1 << bits) - 1)
                    coords = []
                    while mask:
                        bit = mask & -mask
                        coords.append(base_x + bit.bit_length() - 1)
                        mask ^= bit
                    polarity = base_p
                    base_x += bits
                t = (loops << 24) + (high << 12) + low
                if self.do_time_shifting:
                    t -= shift
                if coords:
                    if coords[-1] >= self.width:
                        raise ValueError('EVT3 X coordinate outside declared geometry')
                    if last_t is not None and t < last_t:
                        raise ValueError('Non-monotonic CD timestamps are not supported')
                    last_t = t
                    xs.extend(coords)
                    ys.extend([y] * len(coords))
                    ps.extend([polarity] * len(coords))
                    ts.extend([t] * len(coords))
            elif kind not in (7, 10, 14, 15):
                raise ValueError(f'Unsupported/reserved EVT3 word type: {kind:#x}')
            # External triggers and monitoring/continued words are not CD events.
        self._high, self._low, self._loops = high, low, loops
        self._y, self._base_x, self._base_p = y, base_x, base_p
        self.timestamp_shift, self._last_cd_time = shift, last_t
        events = np.zeros(len(xs), dtype=EVENT_DTYPE)
        events['x'], events['y'], events['p'], events['t'] = xs, ys, ps, ts
        return events

    def _consume_until(self, end, collect):
        parts = []
        while True:
            if not self._pending.size:
                if self._eof:
                    break
                self._pending = self._decode_block()
                continue
            n = int(np.searchsorted(self._pending['t'], end, side='left'))
            if n:
                if collect:
                    parts.append(self._pending[:n])
                self._event_index += n
                self._pending = self._pending[n:]
            if self._pending.size:
                break
        if collect:
            return np.concatenate(parts) if parts else np.empty(0, dtype=EVENT_DTYPE)

    def seek_time(self, final_time):
        final_time = _microseconds(final_time)
        if final_time < self.current_time:
            raise RuntimeError('Cannot seek backward in RAW file; call reset() first')
        self._consume_until(final_time, collect=False)
        self.current_time = final_time

    def load_delta_t(self, delta_t):
        end = self.current_time + _microseconds(delta_t)
        events = self._consume_until(end, collect=True)
        self.current_time = end
        return events

    def get_size(self):
        return self.height, self.width

    def current_event_index(self):
        return self._event_index

    def is_done(self):
        # Look ahead one block so consuming the last event reports EOF promptly.
        while not self._pending.size and not self._eof:
            self._pending = self._decode_block()
        return self._eof and not self._pending.size

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if hasattr(self, '_file'):
            self.close()
