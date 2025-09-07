About JaxLayerLumos
###################

Overview
========

JaxLayerLumos is open-source transfer-matrix method (TMM) software designed for scientists, engineers, and researchers in optics and photonics. It provides a powerful yet intuitive interface for calculating the reflection and transmission (RT) of light through multi-layer optical structures. By inputting the refractive index, thickness of each layer, and the frequency vector, users can analyze how light interacts with layered materials, including the option to adjust for incidence angles.

Our mission is to offer a lightweight, flexible, and fast alternative to commercial software, enabling users to perform complex optical simulations with ease. JaxLayerLumos is built with performance and usability in mind, facilitating the exploration of optical phenomena in research and development settings.

Features
========

- Gradient Calculation: Calculates the gradients over any variables involved in RT, powered by JAX.
- Flexibility: Accommodates a wide range of materials and structures by allowing users to specify complex refractive indices, layer thicknesses, and frequency vectors.
- Angle of Incidence Support: Expands simulation capabilities to include angled light incidence, providing more detailed analysis for advanced optical designs.
- Open Source and Community-Driven: Encourages contributions and feedback from the community, ensuring continuous improvement and innovation.
- Comprehensive Material Database: Includes a growing database of materials with their optical properties, streamlining the simulation setup process.

Installation
============

JaxLayerLumos can be easily installed by the following command using the `PyPI repository <https://pypi.org/project/jaxlayerlumos/>`_.

.. code-block:: bash

    pip install jaxlayerlumos

Alternatively, JaxLayerLumos can be installed from source.

.. code-block:: bash

    pip install .

In addition, we support three installation modes, ``dev``, ``benchmarking``, and ``examples``, where ``dev`` is defined for installing the packages required for development and software testing, ``benchmarking`` is for installing the packages required for benchmarking against differnt TMM software programs, and ``examples`` is needed for running the examples included in the ``examples`` directory. One of these modes can be used by commanding ``pip install .[dev]``, ``pip install .[benchmarking]``, or ``pip install .[examples]``.

Examples
========

A collection of examples in the ``examples`` directory exhibits various use cases and capabilities of our software. We provide the following examples:

1. `Reflection Spectra over Wavelengths Varying Incidence Angles <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/angle-variation.ipynb>`_
2. `Color Conversion <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/color-conversion.ipynb>`_
3. `Color Exploration with Thin-Film Structures <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/color-exploration.ipynb>`_
4. `Gradient Computation <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/gradient-computation.ipynb>`_
5. `Visualization of Light Sources <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/light-source-visualization.ipynb>`_

6. `Plotting of Optical Constants <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/n-k-extrapolation.ipynb>`_
7. `Thin-Film Structure Optimization with Bayesian Optimization <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/optimization-bayeso.ipynb>`_
8. `Thin-Film Structure Optimization with DoG Optimizer <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/optimization-dog.ipynb>`_
9. `Reflection Spectra over Frequencies for Radar Design <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/radar-design.ipynb>`_
10. `Analysis of Solar Cells <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/solar-cell-analysis.ipynb>`_

11. `Transmission Spectra over Wavelengths Varying Thicknesses <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/thickness-variation.ipynb>`_
12. `Triple Junction Solar Cells <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/examples/triple-junction-solar-cells.ipynb>`_

Software Testing and Test Automation
====================================

We provide a variety of test files in the ``tests`` directory. Before running the test files, the required packages should be installed by using ``pip install .[dev]``. They can be run by commanding ``pytest tests/``. Moreover, these test files are automatically tested via GitHub Actions, of which the configuration is defined in ``.github/workflows/pytest.yml``.

Supported Materials
===================

Materials supported by our software are described in `MATERIALS.md <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/markdowns/MATERIALS.md>`_. For example, Ag, Air, Al2O3, Al, aSi, Au, BK7, Cr, cSi, Cu, Fe, FusedSilica, GaAs, GaInP, GaP, Ge, InP, ITO, Mg, Mn, Ni, Pb, Pd, Pt, Sapphire, Si3N4, SiO2, TiN, TiO2, Ti, W, ZnO, Zn are included in JaxLayerLumos.

JaxLayerLumos includes a growing library of materials, which are specified using either complex refractive indices or complex permittivities and permeabilities, which can be sourced from the literature or specified by users based on experimental data. When only complex refractive indices are provided, magnetic effects are assumed to be negligible, and the relative permeability is set to unity, an assumption typically valid at optical frequencies. In the RF and microwave regimes, the electromagnetic properties of metals are derived from their electrical conductivity and magnetic susceptibility, while dielectrics are generally modeled with constant permittivity and negligible loss.

Contributing Guidelines
=======================

To contribute, please read `CONTRIBUTING.md <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/JOSS/markdowns/CONTRIBUTING.md>`_ for our guidelines on issues, enhancements, and pull requests. Follow the outlined standards to keep the project consistent and collaborative.

License
=======

JaxLayerLumos is released under the `MIT License <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/LICENSE>`_, promoting open and unrestricted access to software for academic and commercial use.

Citation
========

.. code-block:: latex

    @misc{LiM2024jaxlayerlumos,
        title={{JaxLayerLumos}: A {JAX}-based Differentiable Optical and Radio Frequency Simulator for Multilayer Structures},
        author={Li, Mingxuan and Kim, Jungtaek and Leu, Paul W.},
        howpublished={\url{https://doi.org/10.5281/zenodo.12602789}},
        year={2024}
    }
