# 2D Parameter Inference

This guide demonstrates how to perform inference on two parameters
simultaneously, visualizing the joint likelihood using corner plots.

## Define Parameter Ranges

Specify the ranges for two parameters to explore:

```python
import corner
import numpy as np

ranges = {
    "q1": (0.1, 1),  # Axis ratio q1
    "phi": (-np.pi / 2, np.pi / 2),  # Orientation angle
}
```

## Sample the 2D Parameter Space

Generate a large number of random samples across the parameter space:

```python
key = jr.key(0)
skeys = jr.split(key, num=len(ranges))
nsamples = 1_000_000  # Use many samples for smooth 2D visualization
params = {
    k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=nsamples)
    for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
}
```

## Compute Likelihood

Evaluate the likelihood for all parameter combinations:

```python
gamma = jnp.linspace(-0.95, 0.95, 128)

lnlik_seg = compute_ln_likelihood(params, track(gamma), track.curvature(gamma), None)
```

## Visualize with Corner Plot

Create a 2D histogram showing the likelihood contours in parameter space:

```python
hist2d_kw = {
    "bins": 20,
    "color": "purple",
    "levels": [0.68, 0.95, 0.997],  # 1σ, 2σ, 3σ confidence levels
    "plot_density": True,
    "plot_contours": True,
    "plot_datapoints": False,
}

fig, ax = plt.subplots(figsize=(4, 4))
corner.hist2d(
    params["q1"],
    params["phi"] * 180 / jnp.pi,  # Convert to degrees for display
    weights=np.exp(lnlik_seg - lnlik_seg.max()),  # Normalize weights
    **hist2d_kw,
)
ax.set_xlabel("q1 (axis ratio)")
ax.set_ylabel("φ (degrees)")
ax.set_title("2D Likelihood Contours")
plt.tight_layout()
plt.show()
```

The resulting plot shows:

- **Contour levels**: Represent confidence regions (68%, 95%, 99.7%)
- **Color density**: Indicates the relative likelihood
- **Best-fit region**: Where contours are most concentrated

This visualization helps identify:

- Parameter degeneracies (elongated contours)
- Preferred parameter values (peak of the distribution)
- Confidence intervals in both parameters simultaneously
