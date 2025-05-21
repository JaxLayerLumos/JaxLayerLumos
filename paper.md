---
title: 'JaxLayerLumos: A JAX-based Differentiable Simulator for Multilayer Optical/RF Structures'
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
  - name: Mingxuan Li # Or use given-names/surname structure if 
    orcid: 0000-0001-6217-9382 # Replace with your ORCID
    affiliation: "1" # Corresponds to an index in the affiliations 
    corresponding: true # if you are the corresponding author
  - name: Jungtaek Kim
    orcid: 0000-0002-1905-1399
    affiliation: "2" # Can list multiple affiliations
  - name: Paul W. Leu # Or use given-names/surname structure if 
    orcid: 0000-0002-1599-7144 # Replace with your ORCID
    affiliation: "1, 3" # Corresponds to an index in the
affiliations:
 - name: Department of Chemical Engineering, University of Pittsburgh, Pittsburgh, PA 15261, USA
   index: 1
 - name: Department of Electrical and Computer Engineering, University of Wisconsin–Madison, Madison, WI 53706, USA
   index: 2
 - name: Department of Industrial Engineering, University of Pittsburgh, Pittsburgh, PA 15261, USA
   index: 3
date: 19 May 2025 # Use format: %e %B %Y, e.g., 9 October 2024
bibliography: paper.bib
---

# Summary
JaxLayerLumos is an open-source Python software package for simulating electromagnetic wave interactions with multilayer structures using the transfer-matrix method (TMM). It is designed for researchers and engineers working with applications in optics, photonics, and radio frequencies.  The software efficiently computes  reflection, transmission, and absorption across a broad spectral range. A key feature of JaxLayerLumos is its implementation in JAX [@jax2018github], which enables automatic differentiation with respect to any input parameter (e.g., layer thickness, refractive index, permeability) and supports fast execution on GPUs and TPUs. This differentiability is especially valuable for gradient-based optimization and for integrating simulations into machine learning pipelines, accelerating the discovery and design of novel devices and materials.

# Statement of Need

The design of multilayer structures is fundamental to numerous applications, including optical filters, advanced solar cells, engineered structural coloration, and radar-absorbing materials. The transfer-matrix method [@BornWolf1999] is a cornerstone analytical technique for modeling such systems. While several TMM implementations exist (e.g., [@tmmSbyrnes], [@tmm_fast]), many primarily focus on the optical (UV-Vis-IR) spectrum and may not readily support simulations involving magnetic materials or the lower frequencies typical of radio frequency (RF) and microwave applications. There is a growing need for tools that are not only fast and flexible across traditional optical ranges but also explicitly cater to these lower frequencies while seamlessly integrating with modern computational paradigms.

JaxLayerLumos addresses this need by offering a JAX-based TMM framework. Its core advantages include:

**Differentiability**: Automatic computation of gradients for any simulation parameter, crucial for inverse design and sensitivity analysis.

**Performance**: Leveraging JAX's JIT compilation and hardware acceleration (CPU, GPU, TPU) for rapid calculations, especially beneficial for large parameter sweeps or training machine learning models.

**Broad Spectral Applicability & Flexibility**: Support for complex refractive indices *and permeabilities* (essential for RF/microwave and magnetic materials), customizable layer stacks, varying incidence angles, and both TE and TM polarizations, enabling simulations across a wide range of frequencies from optical to RF.

**Integration**: Easy integration with the Python scientific computing ecosystem, including optimization libraries and machine learning frameworks.

These features make JaxLayerLumos particularly suited for research at the interface of computational electromagnetics and machine learning. This includes training neural networks for inverse design problems (i.e., predicting layer configurations from target spectral responses) across diverse spectral regions or performing large-scale optimization of device performance for both optical and RF applications. The software provides a lightweight, open-source alternative to some commercial packages, with a focus on speed, differentiability, and ease of use for advanced research.

# Mathematics
The core of JaxLayerLumos implements the transfer-matrix method (TMM) for calculating the reflection $R$ and transmission $T$ of electromagnetic waves through a stack of $L$ planar layers. Each layer $j$ is defined by its thickness $d_j$, complex relative permittivity $\varepsilon_{r,j}$, and complex relative permeability $\mu_{r,j}$. The complex refractive index is $n_j = \sqrt{\varepsilon_{r,j}\mu_{r,j}}$.

For a given wavelength $\lambda$ (or angular frequency $\omega$) and incidence angle $\theta_0$, the propagation of light is described by interface matrices $\mathbf{D}_j$ (capturing Fresnel coefficients at the boundary between layer $j$ and $j+1$) and propagation matrices $\mathbf{P}_j$ (capturing phase accumulation within layer $j$). The total transfer matrix $\mathbf{M}$ for the stack is the product of these individual matrices:
$$ \mathbf{M} = (\mathbf{P}_0\mathbf{D}_0) (\mathbf{P}_1\mathbf{D}_1) \cdots (\mathbf{P}_{L}\mathbf{D}_{L}) \mathbf{P}_{L+1} $$
From the elements of $\mathbf{M}$, the complex reflection $r$ and transmission $t$ amplitudes are calculated, from which $R = |r|^2$ and $T = |t|^2 \times \text{factor}$ (where factor accounts for impedance and angles of incident/exit media) are derived. JaxLayerLumos uses `lax.associative_scan` in JAX for efficient parallel computation of the matrix product.

# Mention of Use
JaxLayerLumos is designed for a range of applications in optical and RF science and engineering. Example use cases provided with the software include:

* Design of radar-absorbing materials and frequency-selective surfaces by simulating spectra in microwave and radio frequency ranges, leveraging its support for magnetic material properties.
* Optimization of thin-film structures for targeted spectral responses using Bayesian optimization or gradient-based methods across optical and RF spectra.
* Analysis and design of solar cells, including multi-junction configurations.
* Exploration of structural coloration in nature and for novel material design.
* Inverse design of optical coatings and RF devices by training Transformer-based models on datasets generated by JaxLayerLumos.

The software's differentiability and performance make it a valuable tool for researchers exploring complex electromagnetic systems and for educators teaching computational photonics and applied electromagnetics.

# Acknowledgements

We acknowledge the developers of JAX [@jax2018github] and other open-source libraries that JaxLayerLumos builds upon. This work was supported by the Center for Materials Data Science for Reliability and Degradation (MDS-Rely), an Industry-University Cooperative Research Center (IUCRC) of the National Science Foundation (NSF). The University of Pittsburgh, Case Western Reserve University, and Carnegie Mellon University are participating institutions in MDS-Rely. We also thank the open-source community for contributions and feedback.

# References
