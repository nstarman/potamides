# Getting Started

This guide demonstrates the basic workflow for stellar stream analysis using
data from Jake's paper.

The workflow consists of the following steps:

1. Stream visualization
2. Extraction of representative points along the stream
3. Stream fitting with splines

## Stream Visualization

The stream data is 3-dimensional, but we work with a 2D projection (x-z plane).
For visualization purposes, we refer to this projection as the (x-y) plane.

First, load and visualize the stream data:

```python
import numpy as np
import matplotlib.pyplot as plt

name = "fig5_streamB.npy"
file_dir = "enter_your_file_dir"  # Replace with your actual file directory
data = np.load(file_dir + name, allow_pickle="True")

x = data.item()["x"]
y = data.item()["z"]

plt.figure(figsize=(5, 5))
plt.plot(x, y, ".")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Stream Data Visualization")
plt.show()
```

You can also generate this plot directly using the `plot` directive:

```{plot}
:context: close-figs

# This is an example - replace with your actual data
import numpy as np
import matplotlib.pyplot as plt

# Placeholder for demonstration
x = np.random.randn(1000) * 20
y = np.random.randn(1000) * 20 + x * 0.3

plt.figure(figsize=(5, 5))
plt.plot(x, y, '.', alpha=0.5)
plt.xlabel('X (pixel)')
plt.ylabel('Y (pixel)')
plt.title('Stream Data Example')
plt.grid(True, alpha=0.3)
```

## Extract Representative Points Along the Stream

In this step, we compute a **median trajectory** for the stream by binning stars
in angular sectors and calculating the median position in each bin.

From the visualization, we observe that the stream exhibits:

- A long, gradually narrowing tail extending towards the right
- A distinct U-shaped structure on the left side
- A pronounced turn at the U-shaped region with denser stellar concentration

**Note on straight segments:** Straight segments have near-zero curvature and
contribute minimally to the inference. Therefore, we exclude these portions from
the analysis.

### Angular Binning Algorithm

```python
X, Y = x, y

# Original angles in the range (-π, π]
angles = np.arctan2(Y, X)

# ===== 1. Define starting angle =====
angle_start = -np.pi / 2  # Bin #1 lower boundary set to -π/2
total_span = 2 * np.pi  # Full circular coverage

# ===== 2. Shift angles into [0, 2π) =====
angles_shifted = (angles - angle_start) % total_span

# ===== 3. Construct equally spaced bins =====
n_bins_angle = 100
angle_bins = np.linspace(0, total_span, n_bins_angle + 1)

# ===== 4. Assign points to bins =====
angle_idx = np.digitize(
    angles_shifted, angle_bins, right=False
)  # Range: 1…n_bins_angle
radius = np.hypot(X, Y)
radius_idx = np.ones_like(angle_idx)  # Only one radial bin used

# ===== 5. Group points by (angle_bin, radius_bin) =====
bins = list(zip(angle_idx, radius_idx))

binned = {}
for (ab, rb), x_val, y_val in zip(bins, X, Y):
    binned.setdefault((ab, rb), []).append((x_val, y_val))

# ===== 6. Compute medians and sort by angle bin =====
median_x_ls, median_y_ls = [], []
sorted_bins = sorted(binned.keys(), key=lambda k: k[0])  # Sort by angular order
for b in sorted_bins:
    pts = np.asarray(binned[b])
    median_x_ls.append(np.median(pts[:, 0]))
    median_y_ls.append(np.median(pts[:, 1]))
```

### Selecting the Region of Interest

The computed median trajectory includes all angular bins. However, we select
only a subset corresponding to the curved portion of the stream. The operation
`[42:-4]` extracts this region based on visual inspection and physical
considerations.

```python
x_cent = np.array(median_x_ls[42:-4])
y_cent = np.array(median_y_ls[42:-4])
plt.figure(figsize=(5, 5))
plt.plot(X, Y, ".", alpha=0.3, label="All stars")
plt.plot(0, 0, "r*", markersize=10, label="Center")
plt.plot(x_cent, y_cent, "o", color="orange", markersize=6, label="Median trajectory")
plt.xlim(-40, 40)
plt.ylim(-40, 40)
plt.xlabel("X (pixel)")
plt.ylabel("Y (pixel)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

Visualization of the median trajectory selection:

```{plot}
:context: close-figs

import numpy as np
import matplotlib.pyplot as plt

# Example visualization
np.random.seed(42)
theta = np.linspace(0, 1.5*np.pi, 1000)
r = 20 + theta * 3 + np.random.randn(1000) * 2
X = r * np.cos(theta)
Y = r * np.sin(theta)

# Simulated median points
theta_med = np.linspace(0.4*np.pi, 1.3*np.pi, 20)
r_med = 20 + theta_med * 3
x_cent = r_med * np.cos(theta_med)
y_cent = r_med * np.sin(theta_med)

plt.figure(figsize=(5, 5))
plt.plot(X, Y, '.', alpha=0.3, label='All stars')
plt.plot(0, 0, 'r*', markersize=10, label='Center')
plt.plot(x_cent, y_cent, 'o', color='orange', markersize=6, label='Median trajectory')
plt.xlim(-40, 40)
plt.ylim(-40, 40)
plt.xlabel('X (pixel)')
plt.ylabel('Y (pixel)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Median Trajectory Extraction')
```
