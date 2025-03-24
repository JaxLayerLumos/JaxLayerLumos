# JaxLayerLumos: A JAX-based Efficient Transfer-Matrix Method Framework for Optical Simulations

<p align="center">
<img src="https://raw.githubusercontent.com/JaxLayerLumos/JaxLayerLumos/main/assets/layerlumos.jpg" width="400" />
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

### **Comparison of TMM Packages**  
| Feature                | Ansys Optics (stackrt)          | TMM-Fast                  | tmm (sbyrnes)       | JaxLayerLumos            |  
|------------------------|----------------------------------|---------------------------|---------------------|--------------------------|  
| **Lightweight**        | ❌ (Commercial, bulky)           | ✅ (PyTorch/NumPy)        | ✅ (Pure Python)     | ✅ (Minimal dependencies)|  
| **Speed**              | Moderate                         | ✅ Fast (PyTorch Cuda)    | Slow (CPU-bound)    | ✅ Fast (JAX JIT)        |  
| **Gradient Support**   | ❌                               | ✅ (PyTorch Autograd)     | ❌                  | ✅ (JAX autodiff)        |  
| **GPU Support**        | ❌                               | ✅ (PyTorch GPU)          | ❌                  | ✅ (JAX backend)         |  
| **TPU Support**        | ❌                               | ❌                        | ❌                  | ✅ (JAX backend)         |  
| **Position-Dependent Poynting** | ❌                  | ❌                        | ❌                  | ✅                          
|  **Optical Simulation**          | ✅ Full-spectrum                 | ✅ Optimized              | ✅ Basic            | ✅ User-defined          |  
|  **Infrared Simulation**         | ❌ Limited                       | ✅ Limited                | ❌                 | ✅ User-defined          |  
|  **Radar (HF) Simulation**       | ❌ Limited                       | ❌                        | ❌                 | ✅ **Cover Magnetic Materials**|  
| **Material Database**  | ✅ Extensive (Commercial)        | ❌ User-defined           | ❌ User-defined     | ✅ Growing library       |  
| **Open Source**        | ❌                               | ✅ MIT                    | ✅ BSD-3-Clause     | ✅ MIT                   |  

---

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

### Comparisons to Other Software

We benchmark our software against other software; please refer to [this file](markdowns/COMPARISONS.md).
These comparisons include the results by [Ansys Optics](https://www.ansys.com/products/optics), [TMM-Fast](https://github.com/MLResearchAtOSRAM/tmm_fast), and [tmm](https://github.com/sbyrnes321/tmm).

## Supported Materials

Materials supported by JaxLayerLumos are described in [this file](markdowns/MATERIALS.md).

## License

JaxLayerLumos is released under the [MIT License](LICENSE), promoting open and unrestricted access to software for academic and commercial use.

## Acknowledgments

- Thanks to all contributors and users for your support and feedback.
