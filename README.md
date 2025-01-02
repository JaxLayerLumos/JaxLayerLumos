# JaxLayerLumos: A JAX-based Efficient Transfer-Matrix Method Framework for Optical Simulations

<p align="center">
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/layerlumos.jpg" width="400" />
</p>


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12602789.svg)](https://doi.org/10.5281/zenodo.12602789)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jaxlayerlumos)](https://pypi.org/project/jaxlayerlumos/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Overview

**JaxLayerLumos** is an open-source software designed for scientists, engineers, and researchers in optics and photonics. It provides a powerful yet intuitive interface for calculating the reflection and transmission (RT) of light through multi-layer optical structures. By inputting the refractive index, thickness of each layer, and the frequency vector, users can analyze how light interacts with layered materials, including the option to adjust for incidence angles.

Our mission is to offer a lightweight, flexible, and fast alternative to commercial software, enabling users to perform complex optical simulations with ease. JaxLayerLumos is built with performance and usability in mind, facilitating the exploration of optical phenomena in research and development settings.

## Features

- **Lightweight and Efficient**: Optimized for performance, JaxLayerLumos ensures rapid calculations without the overhead of large-scale commercial software.
- **Gradient Calculation**: Calculates the gradients over any variables involved in RT, powered by Jax.
- **Flexibility**: Accommodates a wide range of materials and structures by allowing users to specify complex refractive indices, layer thicknesses, and frequency vectors.
- **Angle of Incidence Support**: Expands simulation capabilities to include angled light incidence, providing more detailed analysis for advanced optical designs.
- **Open Source and Community-Driven**: Encourages contributions and feedback from the community, ensuring continuous improvement and innovation.
- **Comprehensive Material Database**: Includes a growing database of materials with their optical properties, streamlining the simulation setup process.

## Getting Started

### Installation

JaxLayerLumos can be easily installed by the following command.

```bash
pip install jaxlayerlumos
```

Alternatively, JaxLayerLumos can be installed from source.

```bash
pip install .
```

### Examples

A collection of examples in the `examples` directory exhibits various use cases and capabilities of JaxLayerLumos.

## Benchmarking

You should install the packages required for benchmarking our software against existing methods.
Before installing the packages, you should install PyTorch first.
In particular, you can install it using the following command.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For details, you can refer to [the official instruction of PyTorch](https://pytorch.org).
Then, the required packages can be installed by the following command.

```bash
pip install .[benchmarking]
```

### Comparisons to Ansys Optics

Simulation results of JaxLayerLumos are compared to the results of [stackrt](https://optics.ansys.com/hc/en-us/articles/360034406254-stackrt-Script-command), which is included in [Ansys Optics](https://www.ansys.com/products/optics).
Our results are matched to the Ansys Optics results with sufficiently small errors.

<p align="center">
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_100.0nm_angle_0.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_100.0nm_angle_45.0_deg.png" width="400" />
<br>
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_100.0nm_angle_75.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_100.0nm_angle_89.0_deg.png" width="400" />
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_10.0nm_Al_11.0nm_Ag_12.0nm_angle_0.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_10.0nm_Al_11.0nm_Ag_12.0nm_angle_45.0_deg.png" width="400" />
<br>
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_10.0nm_Al_11.0nm_Ag_12.0nm_angle_75.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_Ag_10.0nm_Al_11.0nm_Ag_12.0nm_angle_89.0_deg.png" width="400" />
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_TiO2_20.0nm_Ag_5.0nm_TiO2_30.0nm_angle_0.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_TiO2_20.0nm_Ag_5.0nm_TiO2_30.0nm_angle_45.0_deg.png" width="400" />
<br>
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_TiO2_20.0nm_Ag_5.0nm_TiO2_30.0nm_angle_75.0_deg.png" width="400" />
<img src="https://raw.githubusercontent.com/mil152/JaxLayerLumos/main/assets/comparisons/tmm_TiO2_20.0nm_Ag_5.0nm_TiO2_30.0nm_angle_89.0_deg.png" width="400" />
</p>

## Supported Materials

Materials supported by JaxLayerLumos are described in [this file](MATERIALS.md).

## License

JaxLayerLumos is released under the [MIT License](LICENSE), promoting open and unrestricted access to software for academic and commercial use.

## Acknowledgments

- Thanks to all contributors and users for your support and feedback.
