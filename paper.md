---
title: 'JaxLayerLumos: A JAX-based Differentiable Simulator for Multilayer Optical and Radio Frequency Structures'
tags:
  - Python
  - JAX
  - optics
  - photonics
  - radio frequency
  - transfer-matrix method
  - simulation
  - differentiable programming
  - inverse design
  - machine learning
authors:
  - name: Mingxuan Li
    orcid: 0000-0001-6217-9382
    affiliation: "1"
    corresponding: true
  - name: Jungtaek Kim
    orcid: 0000-0002-1905-1399
    affiliation: "2"
    corresponding: true
  - name: Paul W. Leu 
    orcid: 0000-0002-1599-7144
    affiliation: "1, 3"
    corresponding: true
affiliations:
 - name: Department of Chemical Engineering, University of Pittsburgh, Pittsburgh, PA 15261, USA
   index: 1
 - name: Department of Electrical and Computer Engineering, University of Wisconsin--Madison, Madison, WI 53706, USA
   index: 2
 - name: Department of Industrial Engineering, University of Pittsburgh, Pittsburgh, PA 15261, USA
   index: 3
date: 19 May 2025 # Use format: %e %B %Y, e.g., 9 October 2024
bibliography: paper.bib
---

# Summary
JaxLayerLumos is an open-source Python package for simulating electromagnetic wave interactions with multilayer structures using the transfer-matrix method (TMM). It is designed for researchers and engineers working with applications in optics, photonics, and related fields. The software efficiently computes reflection, transmission, and absorption across a broad spectral range, including ultraviolet (UV), visible, infrared, microwave, and radio frequencies (RF), with support for magnetic effects in the microwave and radio regimes. A key feature of JaxLayerLumos is its implementation in JAX, which enables automatic differentiation with respect to any input parameter (e.g., layer thicknesses and refractive indices) and supports fast execution on GPUs and TPUs. In particular, this differentiability is valuable for gradient-based optimization and for integrating simulations into machine learning pipelines, accelerating the discovery of novel devices and materials.

# Statement of need

Multilayer structures are essential in a wide range of technologies, including structural color coatings, 
next-generation solar cells, radar-absorbing materials, and electromagnetic interference (EMI) shielding, as presented in Figure 1. 
They are also key components in optical filters, antireflection coatings, and other photonic devices.

![Applications of JaxLayerLumos](assets/applications.png)

The transfer-matrix method (TMM) [@BornWolf1999]  is a foundational analytical technique for modeling wave interactions in these systems. 
Table 1 compares several TMM implementations, including
[Ansys Optics](https://www.ansys.com/products/optics), [TMM-Fast](https://github.com/MLResearchAtOSRAM/tmm_fast), [tmm](https://github.com/sbyrnes321/tmm), and our open-source package. Most TMM tools, such as [@tmmSbyrnes] and [@tmm_fast]), 
using the complex refractive index formulation and lack support for magnetic materials or frequencies relevant to radio frequency (RF) and microwave applications.
<!-- focus primarily on optical wavelengths (UV-Vis-IR) and lack support for magnetic materials or frequencies relevant to radio frequency (RF) and microwave applications.  -->
There is a growing need for simulation tools that

* Operate efficiently across a broader spectral range--including optical, RF, and microwave frequencies,
* Handle magnetic and lossy materials with complex permittivities and permeability,
* Support modern workflows that integrate machine learning and large-scale optimization.

| **Feature** | **Ansys Optics** (stackrt) | **TMM-Fast** (PyTorch/NumPy) | **tmm** (sbyrnes) (Pure Python) | **JaxLayerLumos** (Jax) |
|-----|-----|-----|-----|-----|
| **Lightweight** | $\times$ Commercial, bulky | $\checkmark$ Lightweight | $\checkmark$ Lightweight | $\checkmark$ Lightweight |
| **Speed** | $\sim$ Moderate | $\checkmark$ Fast  | $\times$ Slow (CPU-bound) | $\checkmark$ Fast |
| **Gradient Support** | $\times$ | $\checkmark$ Yes | $\times$ | $\checkmark$ Yes |
| **GPU Support** | $\times$ | $\checkmark$ Yes | $\times$ | $\checkmark$ Yes |
| **TPU Support** | $\times$ | $\times$ | $\times$ | $\checkmark$ Yes |
| **Position-Dependent Absorption** | $\times$ | $\times$ | $\checkmark$ Supported | $\checkmark$ Supported |                   
| **Optical Simulations** | $\checkmark$ Full-spectrum | $\checkmark$ Optimized | $\checkmark$ Basic | $\checkmark$ User-defined |
| **Infrared Simulations** | $\sim$ Limited | $\sim$ Limited | $\times$ | $\checkmark$ User-defined |
| **Radio Wave Simulations** | $\sim$ Limited | $\times$ | $\times$ | $\checkmark$ Handles magnetic materials |
| **Material Database** | $\checkmark$ Extensive (Commercial) | $\times$ User-defined | $\times$ User-defined | $\checkmark$ Growing library |
| **Open Source** | $\times$ | $\checkmark$ MIT | $\checkmark$ BSD-3-Clause | $\checkmark$ MIT |
Table: Comparison of other TMM packages with JaxLayerLumos

JaxLayerLumos addresses this need by offering a JAX-based TMM framework. Its core advantages include:

* **Differentiability**: Automatically computes gradients with respect to any simulation parameter (e.g., layer thickness, refractive index), enabling inverse design, sensitivity analysis, and seamless integration with gradient-based optimization and machine learning pipelines.

* **High Performance**: Utilizes JAX’s just-in-time (JIT) compilation and hardware acceleration (CPU, GPU, TPU) for fast computation—ideal for large parameter sweeps or model training.

* **Broad Spectral and Material Support**: Accommodates complex permittivities and permeabilities (necessary for magnetic and RF materials), customizable layer structures, oblique incidence, and both TE and TM polarizations—enabling simulations across optical, RF, and microwave regimes.

* **Ecosystem Integration**: Easily integrates with Python’s scientific computing stack, including optimization libraries and ML frameworks like JAX and TensorFlow.

These capabilities make JaxLayerLumos particularly valuable for researchers working at the intersection of computational electromagnetics and machine learning. It is well-suited for tasks such as training neural networks for inverse design (predicting layer structures from target spectra) and performing large-scale device optimization across broad frequency ranges. As an open-source, lightweight alternative to commercial tools, it offers speed, flexibility, and ease of use for advanced research.

# Methodology

![Schematic of the transfer-matrix method showing a multilayer structure with incident, reflected, and transmitted waves. Each layer is characterized by its thickness $d_j$, permittivity $\varepsilon_{r,j}$, and permeability $\mu_{r,j}$.](assets/TMM.png)

The core of JaxLayerLumos implements the TMM method, which calculates the propagation of electromagnetic waves through a stack of $L$ planar layers.  It calculates key optical properties, such as reflection $R(f)$, transmission $T(f)$, and absorption $A(f)$, as functions of frequency $f$ or wavelength $\lambda$.  The software also supports position-resolved absorption and per-layer absorption calculations. Each layer $j$ is defined by 

* thickness $d_j$,
* complex relative permittivity $\varepsilon_{r,j}$, and
* complex relative permeability $\mu_{r,j}$.
  
For a given frequency $f$ and incidence angle $\theta_0$, the propagation of light is described by interface matrices $\mathbf{D}_j$ 
that capture Fresnel coefficients at the boundary between layer $j$ and its following layer and propagation matrices $\mathbf{P}_j$ representing full wave propagation within each layer and captures both phase shift and attenuation due to absorption in lossy media.  The total transfer matrix $\mathbf{M}$ for the entire stack is the product of these individual matrices:
$$\mathbf{M}=(\mathbf{P}_0\mathbf{D}_0)(\mathbf{P}_1\mathbf{D}_1)\cdots(\mathbf{P}_L\mathbf{D}_L)\mathbf{P}_{L+1}$$

JaxLayerLumos includes a growing library of materials, which are specified using either complex refractive indices or complex permittivities and permeabilities.
When only complex refractive indices are given, magnetic effects are assumed to be negligible, and the relative permeability is set to unity:
$\mu_{r,j} = 1$. This assumption is typically valid at optical frequencies. Users can also define their own custom materials.


<!-- From the elements of $\mathbf{M}$, the complex reflection $r$ and transmission $t$ amplitudes are calculated, from which $R = |r|^2$ and $T = |t|^2 \times \text{factor}$ (where factor accounts for impedance and angles of incident/exit media) are derived. -->
<!-- JaxLayerLumos uses `lax.associative_scan` in JAX for efficient parallel computation of the matrix product. Is this that important?-->

# Mention of use

Jupyter notebook examples are available in the [examples folder](./examples/).

JaxLayerLumos is built for a wide range of applications in optical and RF science and engineering. Example use cases provided with the software demonstrate its versatility:

* **Radar-absorbing materials and frequency-selective surfaces**: Simulate spectral responses in the microwave and RF ranges, with full support for magnetic materials.
* **Thin-film structural optimization**: Use Bayesian optimization or gradient-based methods to tailor spectral responses across both optical and RF domains.
* **Solar cell design**: Model and analyze single- and multi-junction solar cell architectures.
* **Structural color**: Explore engineered structural coloration for novel material design.
* **Inverse design with machine learning**: Train Transformer-based models using datasets generated by JaxLayerLumos to design optical coatings and RF devices.

Due to its differentiability and high-performance execution, JaxLayerLumos is well-suited for both advanced research in complex electromagnetic systems and educational use in computational photonics and applied electromagnetics.

# Acknowledgements

We acknowledge the developers of JAX [@jax2018github] and other open-source libraries that JaxLayerLumos builds upon. This work was supported by the Center for Materials Data Science for Reliability and Degradation ([MDS-Rely](https://mds-rely.org/)), an Industry--University Cooperative Research Center of the National Science Foundation. We also thank the open-source community for contributions and feedback.

# References
