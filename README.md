# potamides

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![JOSS][joss-badge]][joss-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

## About

**potamides** is a Python package for constraining gravitational potentials
using stellar stream curvature analysis. The name is inspired by Greek ποταμίδες
("potamídes", meaning "river streams"), with the initial "P" representing
$\Phi$, the conventional symbol for gravitational potential in astronomy.

### Key Features

- 🌊 **Spline-based stream modeling**: Smooth parametric representation of
  stellar streams with cubic spline interpolation
- 📐 **Curvature analysis**: Compute geometric properties including tangent
  vectors, curvature, principal normals, and arc-length
- 🌌 **Gravitational field fitting**: Match stream curvature to potential models
  with customizable halo and disk components
- ⚡ **JAX-accelerated**: Fast, GPU-compatible computations with automatic
  differentiation and JIT compilation
- 📊 **Likelihood framework**: Bayesian inference for potential parameters using
  curvature-acceleration alignment
- 📈 **Visualization tools**: Built-in plotting methods for tracks, geometry
  vectors, and gravitational fields

## Installation

### Using pip (recommended)

```bash
pip install potamides
```

### From source

```bash
git clone https://github.com/xggs-dev/potamides.git
cd potamides
uv pip install -e .
```

### Requirements

- Python >= 3.11
- JAX >= 0.5.3
- For GPU support, install JAX with CUDA support separately
- See `pyproject.toml` for full dependency list

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Citation

If you use this software in your research, please cite our
[JOSS paper](https://doi.org/10.21105/joss.10712):

```bibtex
@article{Wu2026,
  author = {Wu, Sirui and Starkman, Nathaniel and Nibauer, Jacob and Pearson, Sarah},
  title = {Potamides: JAX tools for curvature-based inference from stellar streams},
  journal = {Journal of Open Source Software},
  year = {2026},
  doi = {10.21105/joss.10712},
  url = {https://doi.org/10.21105/joss.10712}
}
```

## Acknowledgments

This package builds upon excellent open-source scientific software:

- **[JAX](https://github.com/google/jax)**: High-performance numerical computing
  with automatic differentiation
- **[galax](https://github.com/GalacticDynamics/galax)**: Galactic dynamics in
  JAX
- **[interpax](https://github.com/f0uriest/interpax)**: Interpolation library
  for JAX
- **[Astropy](https://www.astropy.org/)**: Community Python library for
  astronomy
- **[unxt](https://github.com/GalacticDynamics/unxt)**: Unitful quantities for
  JAX

## AI Usage Disclosure

Portions of this codebase (including tests and documentation) were refactored
and generated with the assistance of Language Models. All AI contributions have
been and will continue to be reviewed and verified by the human maintainers.

---

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/xggs-dev/potamides/workflows/CI/badge.svg
[actions-link]:             https://github.com/xggs-dev/potamides/actions
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/xggs-dev/potamides/discussions
[joss-badge]:               https://joss.theoj.org/papers/10.21105/joss.10712/status.svg
[joss-link]:                https://doi.org/10.21105/joss.10712
[pypi-link]:                https://pypi.org/project/potamides/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/potamides
[pypi-version]:             https://img.shields.io/pypi/v/potamides
[rtd-badge]:                https://readthedocs.org/projects/potamides/badge/?version=latest
[rtd-link]:                 https://potamides.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
