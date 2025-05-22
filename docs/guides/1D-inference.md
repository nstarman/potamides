# 1D Parameter Inference

This guide demonstrates how to perform inference on a single parameter of the
gravitational potential using stellar stream curvature.

## Setup: Acceleration and Likelihood Functions

First, define functions to compute accelerations from the potential model and
calculate the likelihood:

```python
import potamides as ptd
import unxt as u
import jax
import jax.numpy as jnp
import jax.random as jr


@jax.jit
def compute_acc_hat(params, pos2d):
    """Compute accelerations from 2D positions."""
    # Positions: 2D -> 3D
    pos3d = jnp.zeros((len(pos2d), 3))
    pos3d = pos3d.at[:, :2].set(pos2d)

    # Accelerations from the potential model
    params = params_defaults | params_statics | params
    params["origin"] = jnp.array(
        [
            params.pop("origin_x", 0),
            params.pop("origin_y", 0),
            params.pop("origin_z", 0),
        ]
    )
    return ptd.compute_accelerations(pos3d, **params)


@jax.jit
def compute_ln_likelihood_scalar(params, pos2d, unit_curvature, where_straight=None):
    """Compute log-likelihood for a single parameter set."""
    unit_acc_xy = compute_acc_hat(params, pos2d)
    where_straight = (
        where_straight
        if where_straight is not None
        else jnp.zeros(len(unit_curvature), dtype=bool)
    )
    return ptd.compute_ln_likelihood(
        unit_curvature, unit_acc_xy, where_straight=where_straight
    ) / len(unit_curvature)
```

## Define Halo Parameters

Set default parameters for the dark matter halo and disk:

```python
# Parameters for the halo
params_defaults = {
    "rs_halo": 16,  # Scale radius [kpc]
    "vc_halo": u.Quantity(250, "km/s").ustrip("kpc/Myr"),  # Circular velocity
    "q1": 1.0,  # Axis ratio 1
    "q2": 1.0,  # Axis ratio 2
    "q3": 1.0,  # Axis ratio 3
    "phi": 0.0,  # Orientation angle
    "Mdisk": 1.2e10,  # Disk mass [Msun]
    "origin_x": 0,
    "origin_y": 0,
    "origin_z": 0,
    "rot_z": 0.0,  # Rotation around z-axis
    "rot_x": 0.0,  # Rotation around x-axis
}
params_statics = {"withdisk": False}  # Exclude disk component
```

## Vectorize for Multiple Parameter Sets

Use `jax.vmap` to evaluate the likelihood over many parameter values
simultaneously:

```python
compute_ln_likelihood = jax.vmap(
    compute_ln_likelihood_scalar, in_axes=(0, None, None, None)
)
```

## Sample Parameter Space

Define the parameter range to explore and generate random samples:

```python
ranges = {
    "q1": (0.1, 2),  # Explore axis ratio q1 from 0.1 to 2
}

key = jr.key(0)
skeys = jr.split(key, num=len(ranges))
nsamples = 1_000
params = {
    k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=nsamples)
    for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
}
```

## Compute Likelihood

Evaluate the likelihood along the stream track:

```python
gamma = jnp.linspace(-0.95, 0.95, 128)

lnlik_seg = compute_ln_likelihood(params, track(gamma), track.curvature(gamma), None)
```

The result `lnlik_seg` contains the log-likelihood values for each sampled
parameter set. You can use this to:

- Identify the best-fit parameter value (maximum likelihood)
- Compute confidence intervals
- Visualize the likelihood as a function of the parameter
