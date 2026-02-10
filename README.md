# ALIVE-TRR

`ALIVE-TRR` is a Python-based image reconstruction framework designed for event-based sensors. It implements **Time-Reversal Reassignment (TRR)**, a principle analogue to **Time Delay Integration (TDI)**, to reconstruct high-resolution, motion-compensated intensity images from raw event streams.

---

## Key Features

* **High-Fidelity Reconstruction**: Uses TRR motion compensation to eliminate motion blur from event data.
* **Multi-Axis Motion Support**: Capable of handling motion vectors in X, Y, or combined XY directions.
* **Memory-Efficient Streaming**: Implements a chunk-based processing approach, allowing reconstruction of large `.raw` files by processing them in time-defined segments.
* **Advanced Splatting Modes**:
    * **Nearest**: Fast accumulation by snapping events to the nearest pixel center.
    * **Bilinear**: Distributes event energy to 4 neighboring pixels for smoother, sub-pixel accuracy.
* **Automated Output**: Generates 16-bit TIFF images with normalized intensity mapping.

---

## Project Structure

* `trr_recon_stream.py`: Core reconstruction engine containing the streaming logic and TRR algorithms.
* `data/`: Recommended directory to store input `.raw` event files.

---

## Quick Start

### 1. Requirements
Ensure you have the following dependencies installed:
* `numpy`
* `matplotlib`
* `imageio`
* `tqdm`
* `metavision_core` (Prophesee SDK for reading `.raw` files)

### 2. Basic Usage
Configure your parameters in the `if __name__ == "__main__":` block of `trr_recon_stream.py` and run the script:

```python
from trr_recon_stream import trr_reconstruct_file_by_time_chunks

# Define parameters
file_path = 'data/your_data.raw'
velocity = -0.124  # px/ms
t_start = 0        # Start time in microseconds
t_end = 3e6        # End time (e.g., 3 seconds)

# Start reconstruction
img, meta = trr_reconstruct_file_by_time_chunks(
    file_path=file_path,
    t_global_start_us=t_start,
    t_global_end_us=t_end,
    v_px_per_ms=velocity,
    axis='y',            # Movement axis ('x', 'y', or 'xy')
    splat_mode='bilinear'
)
```

---

## Technical Logic

### Motion Compensation
The algorithm shifts event coordinates based on their timestamps ($t$) relative to a reference time ($t_{ref}$):
* **Y-Axis**: $y' = y + v \cdot (t - t_\text{ref})$
* **X-Axis**: $x' = x + v \cdot (t - t_\text{ref})$
* **XY-Axis**: $x' = x + v_x \cdot (t - t_\text{ref})$ and $y' = y + v_y \cdot (t - t_\text{ref})$

### Accumulation & Polarity
Events are accumulated onto a 2D grid. The contribution of each event is weighted by its polarity ($p$) using the `p_pos_neg` parameter. The weight $w$ is calculated as:
$$w = (p\_pos\_neg[1] - p\_pos\_neg[0]) \cdot p + p\_pos\_neg[0]$$

---

## Output Handling

* **16-bit TIFF**: The final reconstructed image is normalized and saved as a 16-bit TIFF file using `imageio`.
* **Intermediate Chunks**: To maintain memory efficiency, shifted event data is temporarily saved as `.npy` files in a dedicated sub-folder.

---

## Contact

For any questions, please contact:, please contact:

**Qianxi Liang (梁谦禧)**  
Peking University  
Email: [chaunceyl@stu.pku.edu.cn](mailto:chaunceyl@stu.pku.edu.cn)
