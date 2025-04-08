# JaxLayerLumos: A JAX-based Efficient Transfer-Matrix Method Framework for Optical Simulations


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12602789.svg)](https://doi.org/10.5281/zenodo.12602789)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jaxlayerlumos)](https://pypi.org/project/jaxlayerlumos/)
![GitHub Release](https://img.shields.io/github/v/release/JaxLayerLumos/jaxlayerlumos)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<p align="center">
<img src="https://raw.githubusercontent.com/JaxLayerLumos/JaxLayerLumos/main/assets/jaxlayerlumos.jpg" width="400" />
</p>


## Overview

**JaxLayerLumos** is open-source transfer-matrix method (TMM) software designed for scientists, engineers, and researchers in optics and photonics. It provides a powerful yet intuitive interface for calculating the reflection and transmission (RT) of light through multi-layer optical structures. By inputting the refractive index, thickness of each layer, and the frequency vector, users can analyze how light interacts with layered materials, including the option to adjust for incidence angles.

Our mission is to offer a lightweight, flexible, and fast alternative to commercial software, enabling users to perform complex optical simulations with ease. JaxLayerLumos is built with performance and usability in mind, facilitating the exploration of optical phenomena in research and development settings.


## Features

- **Lightweight and Efficient**: Optimized for performance, JaxLayerLumos ensures rapid calculations without the overhead of large-scale commercial software.
- **Gradient Calculation**: Calculates the gradients over any variables involved in RT, powered by JAX.
- **Flexibility**: Accommodates a wide range of materials and structures by allowing users to specify complex refractive indices, layer thicknesses, and frequency vectors.
- **Angle of Incidence Support**: Expands simulation capabilities to include angled light incidence, providing more detailed analysis for advanced optical designs.
- **Open Source and Community-Driven**: Encourages contributions and feedback from the community, ensuring continuous improvement and innovation.
- **Comprehensive Material Database**: Includes a growing database of materials with their optical properties, streamlining the simulation setup process.


## Installation

JaxLayerLumos can be easily installed by the following command using the [PyPI repository](https://pypi.org/project/jaxlayerlumos/).

```bash
pip install jaxlayerlumos
```

Alternatively, JaxLayerLumos can be installed from source.

```bash
pip install .
```

In addition, we support three installation modes, `dev`, `benchmarking`, and `examples`, where `dev` is defined for installing the packages required for development, `benchmarking` is for installing the packages required for benchmarking against differnt TMM software programs, and `examples` is needed for running the examples included in the `examples` directory.
One of these modes can be used by commanding `pip install .[dev]`, `pip install .[benchmarking]`, or `pip install .[examples]`.


## Examples

A collection of examples in the `examples` directory exhibits various use cases and capabilities of our software.
We provide the following examples:

1. [Reflection Spectra over Wavelengths Varying Incidence Angles](examples/angle-variation.ipynb)
2. [Color Conversion](examples/color-conversion.ipynb)
3. [Color Exploration with Thin-Film Structures](examples/color-exploration.ipynb)
4. [Gradient Computation](examples/gradient-computation.ipynb)
5. [Visualization of Light Sources](examples/light-source-visualization.ipynb)
6. [Plotting of Optical Constants](examples/n-k-extrapolation.ipynb)
7. [Thin-Film Structure Optimization with Bayesian Optimization](examples/optimization-bayeso.ipynb)
8. [Thin-Film Structure Optimization with DoG Optimizer](examples/optimization-dog.ipynb)
9. [Reflection Spectra over Frequencies for Radar Design](examples/radar-design.ipynb)
10. [Analysis of Solar Cells](examples/solar-cell-analysis.ipynb)
11. [Transmission Spectra over Wavelengths Varying Thicknesses](examples/thickness-variation.ipynb)
12. [Triple Junction Solar Cells](examples/triple-junction-solar-cells.ipynb)


## Comparison of TMM Packages

We compare [Ansys Optics](https://www.ansys.com/products/optics), [TMM-Fast](https://github.com/MLResearchAtOSRAM/tmm_fast), and [tmm](https://github.com/sbyrnes321/tmm) to our software.

| Feature | Ansys Optics (stackrt) | TMM-Fast | tmm (sbyrnes) | JaxLayerLumos |
|-----|-----|-----|-----|-----|
| **Lightweight** | ❌ (Commercial, bulky) | ✅ (PyTorch/NumPy) | ✅ (Pure Python) | ✅ (JAX) |
| **Speed** | Moderate | ✅ Fast (PyTorch) | Slow (CPU-bound) | ✅ Fast (JAX) |
| **Gradient Support** | ❌ | ✅ (PyTorch) | ❌ | ✅ (JAX) |
| **GPU Support** | ❌ | ✅ (PyTorch) | ❌ | ✅ (JAX) |  
| **TPU Support** | ❌                               | ❌                        | ❌                  | ✅ (JAX)         |  
| **Position-Dependent Poynting** | ❌                  | ❌                        | ❌                  | ✅                          
| **Optical Simulation** | ✅ Full-spectrum                 | ✅ Optimized              | ✅ Basic            | ✅ User-defined          |  
| **Infrared Simulation** | ❌ Limited                       | ✅ Limited                | ❌                 | ✅ User-defined          |  
| **Radar (HF) Simulation** | ❌ Limited                       | ❌                        | ❌                 | ✅ Magnetic materials covered |  
| **Material Database** | ✅ Extensive (Commercial)        | ❌ User-defined           | ❌ User-defined     | ✅ Growing library       |  
| **Open Source** | ❌                               | ✅ MIT                    | ✅ BSD-3-Clause     | ✅ MIT                   |  


## Benchmarking against Other Software

We benchmark JaxLayerLumos against other software.
Detailed benchmarking results can be found in [this file](markdowns/COMPARISONS.md).
These comparisons include the results of [Ansys Optics](https://www.ansys.com/products/optics), [TMM-Fast](https://github.com/MLResearchAtOSRAM/tmm_fast), and [tmm](https://github.com/sbyrnes321/tmm).

To obtain these results, you should install additional required packages.
Before installing the packages, you should install PyTorch first.
In particular, if you need the CPU version of PyTorch, you can install it using the following command.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For details, you can refer to [the official instruction of PyTorch](https://pytorch.org).
Then, the required packages can be installed by the following command.

```bash
pip install .[benchmarking]
```

Finally, you can run the benchmarking code `compare_methods.py` in the `benchmarking` directory.


## Supported Materials

Materials supported by our software are described in [this file](markdowns/MATERIALS.md).


## License

JaxLayerLumos is released under the [MIT License](LICENSE), promoting open and unrestricted access to software for academic and commercial use.


## Acknowledgments

- We sincerely thank all contributors and users for your support and feedback.
- This work is supported by the [Center for Materials Data Science for Reliability and Degradation (MDS-Rely)](https://mds-rely.org), which is the [Industry-University Cooperative Research Center (IUCRC)](https://iucrc.nsf.gov) of [National Science Foundation](https://www.nsf.gov).
- The University of Pittsburgh, Case Western Reserve University, and Carnegie Mellon University are participating institutions in MDS-Rely.
