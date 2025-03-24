from jaxlayerlumos.colors import transform


def spectrum_to_xyY(
    wavelengths, values, str_color_space="cie1931", str_illuminant="d65"
):
    XYZ = transform.spectrum_to_XYZ(
        wavelengths,
        values,
        str_color_space=str_color_space,
        str_illuminant=str_illuminant,
    )
    xyY = transform.XYZ_to_xyY(XYZ)

    return xyY


def spectrum_to_sRGB(
    wavelengths,
    values,
    str_color_space="cie1931",
    str_illuminant="d65",
    use_clipping=False,
):
    XYZ = transform.spectrum_to_XYZ(
        wavelengths,
        values,
        str_color_space=str_color_space,
        str_illuminant=str_illuminant,
    )
    sRGB = transform.XYZ_to_sRGB(XYZ, use_clipping=use_clipping)

    return sRGB


def spectrum_to_Lab(
    wavelengths, values, str_color_space="cie1931", str_illuminant="d65"
):
    XYZ = transform.spectrum_to_XYZ(
        wavelengths,
        values,
        str_color_space=str_color_space,
        str_illuminant=str_illuminant,
    )
    Lab = transform.XYZ_to_Lab(XYZ, str_illuminant=str_illuminant)

    return Lab
