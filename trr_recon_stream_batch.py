
"""
ALIVE-TRR batch reconstruction wrapper

This script extends the public TRR reconstruction implementation
(trr_recon_stream.py) to process multiple scan ranges sequentially.

The reconstruction core is intentionally imported from the public script
to preserve identical TRR coordinate transformation, bbox calculation,
splatting, and TIFF generation behavior.

"""

from pathlib import Path
from trr_recon_stream import trr_reconstruct_file_by_time_chunks


# ============================================================
# User configuration
# ============================================================

raw_file = r'E:\stage_ebs_38s.raw'

output_dir = Path(
    r"E:\batch_output"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# Raw event stream time windows for the large-scale renal F-actin imaging
# dataset available at:
# https://doi.org/10.6084/m9.figshare.33472165
#
# Format:
# (start_timestamp_us, end_timestamp_us, stage_velocity_px_ms)

time_ranges = [

    (180000, 2580000, -5.34),
    (3360000, 5760000, +5.34),
    (6440000, 8840000, -5.34),
    (9820000, 12220000, +5.34),
    (13060000, 15460000, -5.34),
    (16300000, 18700000, +5.34),
    (19570000, 21970000, -5.34),
    (22680000, 25080000, +5.34),
    (25860000, 28260000, -5.34),
    (29090000, 31490000, +5.34),
    (32230000, 34630000, -5.34),
    (35330000, 37730000, +5.34),
]


axis = "y"

# Keep the same reconstruction mode as the public code
chunk_len_us = 500000

splat_mode = "nearest"

p_pos_neg = [0, 1]


# ============================================================
# Batch reconstruction
# ============================================================

if __name__ == "__main__":

    all_metadata = []

    for idx, (start_us, end_us, velocity) in enumerate(time_ranges):

        print(
            f"[BATCH] {idx:02d}/{len(time_ranges)-1:02d}: "
            f"{start_us} -> {end_us} us, "
            f"v={velocity:+.2f} px/ms"
        )

        output_path = (
            output_dir /
            f"TRR_t{start_us}_{end_us}_v{velocity:+.2f}.tif"
        )

        _, meta = trr_reconstruct_file_by_time_chunks(
            file_path=raw_file,
            t_global_start_us=start_us,
            t_global_end_us=end_us,
            v_px_per_ms=velocity,
            axis=axis,
            chunk_len_us=chunk_len_us,
            p_pos_neg=p_pos_neg,
            splat_mode=splat_mode,
            output_path=str(output_path),
        )

        all_metadata.append(meta)

        print(
            f"[DONE] {output_path} "
            f"| size={meta['H']}x{meta['W']}"
        )

    print("[BATCH] All ranges completed.")
