Supported Materials
###################

.. image:: _static/img/Ag.jpg
    :width: 400
    :align: center
    :alt: Ag

.. image:: _static/img/TiO2.jpg
    :width: 400
    :align: center
    :alt: TiO2

Materials supported by our software are described in `MATERIALS.md <https://github.com/JaxLayerLumos/JaxLayerLumos/blob/main/markdowns/MATERIALS.md>`_. For example, Ag, Air, Al2O3, Al, aSi, Au, BK7, Cr, cSi, Cu, Fe, FusedSilica, GaAs, GaInP, GaP, Ge, InP, ITO, Mg, Mn, Ni, Pb, Pd, Pt, Sapphire, Si3N4, SiO2, TiN, TiO2, Ti, W, ZnO, Zn are included in JaxLayerLumos.

JaxLayerLumos includes a growing library of materials, which are specified using either complex refractive indices or complex permittivities and permeabilities, which can be sourced from the literature or specified by users based on experimental data. When only complex refractive indices are provided, magnetic effects are assumed to be negligible, and the relative permeability is set to unity, an assumption typically valid at optical frequencies. In the RF and microwave regimes, the electromagnetic properties of metals are derived from their electrical conductivity and magnetic susceptibility, while dielectrics are generally modeled with constant permittivity and negligible loss.
