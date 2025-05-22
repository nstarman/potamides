# Stream Fitting with Splines

This guide demonstrates how to fit a cubic spline to the median trajectory
points extracted from the stellar stream.

## Data Preparation

First, convert the data from NumPy to JAX arrays for compatibility with the
spline fitting routines:

```python
import jax

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision
import jax.numpy as jnp

x_jax = jnp.array(x_cent, dtype=jnp.float64)
y_jax = jnp.array(y_cent, dtype=jnp.float64)
xy_centered = jnp.stack([x_jax, y_jax], axis=1)
```

## Initial Spline Fitting

The key parameter `num_knots` controls the spline fit quality, specifying how
many control points (knots) the spline uses. More knots provide more flexibility
but may overfit the data.

```python
from potamides import splinelib as splib
import interpax

fid_gamma, fid_knots = splib.make_increasing_gamma_from_data(xy_centered)
fiducial_spline = interpax.Interpolator1D(fid_gamma, fid_knots, method="cubic2")
ref_gamma = jnp.linspace(fid_gamma.min(), fid_gamma.max(), num=128)
ref_points = fiducial_spline(ref_gamma)
```

### Visualize the Preliminary Fit

```python
plt.figure(figsize=(5, 5))
plt.plot(X, Y, ".", alpha=0.3, label="All stars")
plt.plot(0, 0, "r*", markersize=10, label="Center")
plt.plot(
    ref_points[:, 0],
    ref_points[:, 1],
    "-",
    color="blue",
    linewidth=2,
    label="Initial spline",
)
plt.xlim(-40, 40)
plt.ylim(-40, 40)
plt.xlabel("X (pixel)")
plt.ylabel("Y (pixel)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Spline Optimization

In this step, we optimize the spline knots and reparameterize γ to be
proportional to arc length. This ensures that:

- The spline smoothly follows the stream
- γ advances uniformly along the physical path
- Curvature calculations are more accurate

```python
from xmmutablemap import ImmutableMap

knots = splib.optimize_spline_knots(
    splib.default_cost_fn,
    fid_knots,
    fid_gamma,
    cost_args=(ref_gamma, ref_points),
    cost_kwargs=ImmutableMap({"concavity_weight": 1e12}),
)

# Create a spline from the optimized knots
spline = interpax.Interpolator1D(fid_gamma, knots, method="cubic2")

# Create a new gamma, proportional to the arc-length from the spline
opt_gamma, opt_knots = splib.new_gamma_knots_from_spline(spline, nknots=num_knots)

track = ptd.Track(opt_gamma, opt_knots)
```

## Visualize the Final Result

```python
axlim = 50
figsize = 5
fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=150)
plt.plot(X, Y, "c.", zorder=0, alpha=0.3, label="All stars")
plt.plot(0, 0, "r*", markersize=10, label="Center")
plt.plot(x_cent, y_cent, "o", color="orange", markersize=6, label="Median points")
plot_sparse_gamma = jnp.linspace(track.gamma.min(), track.gamma.max(), num=8)
track.plot_all(plot_sparse_gamma, ax=ax, show_tangents=False)
ax.set_xlabel("X (pixel)")
ax.set_ylabel("Y (pixel)")
ax.set_xlim(-axlim, axlim)
ax.set_ylim(-axlim, axlim)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
```

This creates a `Track` object that represents the optimized spline fit to the
stellar stream. The track can be used for:

- Computing curvature along the stream
- Calculating accelerations and likelihood functions
- Performing dynamical inference on the underlying gravitational potential
