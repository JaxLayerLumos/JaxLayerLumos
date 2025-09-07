---
title: 'JaxLayerLumos: A JAX-based Differentiable Optical and Radio Frequency Simulator for Multilayer Structures'
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
    equal-contrib: true
  - name: Jungtaek Kim
    orcid: 0000-0002-1905-1399
    affiliation: "2"
    equal-contrib: true
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

JaxLayerLumos is an open-source Python package for simulating electromagnetic wave interactions with multilayer structures using the transfer-matrix method (TMM). It is designed for researchers and engineers working with applications in optics, photonics, and related fields. The software efficiently computes reflection, transmission, and absorption across a broad spectral range, including ultraviolet, visible, infrared, microwave, and radio frequencies (RF), with support for magnetic effects in the microwave and radio regimes. A key feature of JaxLayerLumos is its implementation in JAX, which enables automatic differentiation with respect to any input parameter (e.g., layer thicknesses and refractive indices) and supports seamless execution on graphics processing units (GPUs) and tensor processing units (TPUs). In particular, this differentiability is valuable for gradient-based optimization and for integrating simulations into machine learning pipelines, accelerating the discovery of novel devices and materials.

# Statement of need

Multilayer structures are essential in a wide range of technologies, including structural color coatings [@sun2013structural;@elkabbash2023fano], next-generation solar cells [@Gao:14aug;@Gao:16;@Wang:15apr;@bati2023next], radar-absorbing materials [@michielssen1993design;@vinoy1996radar], and electromagnetic interference (EMI) shielding [@Li:22;@KimJ2023neurips;@KimJ2024dd;@zhao2024electromagnetic], as presented in Figure 1. 
They are also key components in optical filters, antireflection coatings [@Haghanifar:20], and other photonic devices.

![Applications of JaxLayerLumos](../assets/applications.png)

TMM [@BornWolf1999] is a foundational analytical technique for modeling wave interactions in these systems. 
Table 1 compares several TMM implementations, including
[Ansys Optics](https://www.ansys.com/products/optics), [TMM-Fast](https://github.com/MLResearchAtOSRAM/tmm_fast), [tmm](https://github.com/sbyrnes321/tmm), and our package. Most TMM tools, such as [@tmmSbyrnes] and [@luce2022tmm], 
use the complex refractive index formulation and lack support for magnetic materials or frequencies relevant to RF and microwave applications.
There is a growing need for simulation tools that

* Operate efficiently across a broader spectral range including optical, RF, and microwave frequencies,
* Handle magnetic and lossy materials with complex permittivities and complex permeabilities,
* Support modern workflows that integrate machine learning and large-scale optimization.

| **Feature** | **Ansys Optics** (stackrt) | **TMM-Fast** (PyTorch/NumPy) | **tmm** (Pure Python) | **JaxLayerLumos** (JAX) |
|-----|-----|-----|-----|-----|
| **Lightweight** | $\times$ Bulky | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **Speed on CPUs** | $\times$ Slow | $\checkmark$ Fast  | $\times$ Slow | $\sim$ Moderate |
| **Gradient Support** | $\times$ | $\checkmark$ | $\times$ | $\checkmark$ |
| **GPU Support** | $\times$ | $\checkmark$ | $\times$ | $\checkmark$ |
| **TPU Support**[^1] | $\times$ | $\times$ | $\times$ | $\checkmark$ |
| **Position-Dependent Absorption** | $\times$ | $\times$ | $\checkmark$ | $\checkmark$ |                   
| **Optical Simulations** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **Infrared Simulations** | $\sim$ Limited | $\sim$ Limited | $\times$ | $\checkmark$ |
| **Radio Wave Simulations** | $\sim$ Limited | $\times$ | $\times$ | $\checkmark$ Handles magnetic materials |
| **Open Source** | $\times$ Commercial | $\checkmark$ MIT | $\checkmark$ BSD-3-Clause | $\checkmark$ MIT |
Table: Comparison of other TMM packages with JaxLayerLumos

JaxLayerLumos addresses this need by offering a JAX-based TMM framework. Its core advantages include:

* **Differentiability**: Automatically computes gradients with respect to any simulation parameters (e.g., layer thicknesses and refractive indices).

* **Seamless Execution**: Utilizes JAX’s just-in-time compilation and hardware acceleration with CPUs, GPUs, or TPUs for efficient computation.

* **Broad Spectral and Material Support**: Accommodates complex permittivities and permeabilities (necessary for magnetic and RF materials), customizable layer structures, oblique incidence, and both TE and TM polarizations.

* **Ecosystem Integration**: Easily integrates with Python’s scientific computing stack, including optimization libraries and machine learning frameworks such as JAX [@jax2018github] and Scikit-learn [@pedregosa2011scikit].

These capabilities make JaxLayerLumos particularly valuable for researchers working at the intersection of computational electromagnetics and machine learning. As an open-source, lightweight alternative to commercial tools, it offers speed, flexibility, and ease of use for advanced research.

[^1]: Because TPUs are optimized for low-precision computation, their numerical precision may be reduced.

# Methodology

![Schematic of TMM showing a multilayer structure with incident, reflected, and transmitted waves. Each layer is characterized by its thickness $d_j$, permittivity $\varepsilon_{r,j}$, and permeability $\mu_{r,j}$.](../assets/TMM.png)

The core of JaxLayerLumos implements TMM, which calculates the propagation of electromagnetic waves through a stack of $L$ planar layers [@BornWolf1999]. It calculates key optical properties, such as reflection $R(f)$, transmission $T(f)$, and absorption $A(f)$, as functions of frequency $f$ or wavelength $\lambda$. The software also supports position-resolved absorption and per-layer absorption calculations. Each layer $j$ is defined by thickness $d_j$, complex relative permittivity $\varepsilon_{r,j}$, and complex relative permeability $\mu_{r,j}$.
  
For a given frequency $f$ and incidence angle $\theta_0$, the propagation of light is described by interface matrices $\mathbf{D}_j$ 
that capture Fresnel coefficients at the boundary between layer $j$ and its following layer and propagation matrices $\mathbf{P}_j$ representing full wave propagation within each layer and captures both phase shift and attenuation due to absorption in lossy media.  The total transfer matrix $\mathbf{M}$ for the entire stack is the product of these individual matrices:
$$\mathbf{M}=(\mathbf{P}_0\mathbf{D}_0)(\mathbf{P}_1\mathbf{D}_1)\cdots(\mathbf{P}_L\mathbf{D}_L)\mathbf{P}_{L+1}$$

JaxLayerLumos includes a growing library of materials, which are specified using either complex refractive indices or complex permittivities and permeabilities, which can be sourced from the literature or 
specified by users based on experimental data.
When only complex refractive indices are provided, magnetic effects are assumed to be negligible, and the relative permeability is set to unity
($\mu_{r,j} = 1$), an assumption typically valid at optical frequencies.
In the RF and microwave regimes, the electromagnetic properties of metals are derived from their electrical conductivity and magnetic susceptibility, while dielectrics are generally modeled with constant permittivity and negligible loss.

# Mention of use

Diverse use cases demonstrate the versatility of JaxLayerLumos:

* **Radar-absorbing materials and frequency-selective surfaces**: Simulate spectral responses in the microwave and RF ranges [@michielssen1993design], with full support for magnetic materials.
* **Thin-film structural optimization**: Use Bayesian optimization [@garnett2023bayesian] or gradient-based methods [@boyd2004convex] to tailor spectral responses across both optical and RF domains.
* **Solar cell design**: Model and analyze single- and multi-junction solar cell architectures [@Gao:14aug;@Gao:16;@Wang:15apr].
* **Structural color**: Explore engineered structural coloration for novel material design [@sun2013structural;@elkabbash2023fano].
* **Inverse design with machine learning**: Train Transformer-based models [@vaswani2017attention] to design optical coatings and RF devices.

Jupyter notebook examples are available in the [examples directory](https://github.com/JaxLayerLumos/JaxLayerLumos/tree/main/examples).

# Acknowledgements

We acknowledge the developers of JAX and other open-source libraries. This work was supported by the Center for Materials Data Science for Reliability and Degradation ([MDS-Rely](https://mds-rely.org/)), an Industry--University Cooperative Research Center of the National Science Foundation, through awards EEC-2052662, EEC-2052776, and EEC-2435402. We also thank the open-source community for contributions and feedback.

# References
