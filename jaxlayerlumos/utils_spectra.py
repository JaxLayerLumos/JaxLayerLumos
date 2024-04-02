import jax.numpy as jnp
import scipy.constants as scic

def convert_frequencies_to_wavelengths(f):
    """
    Convert frequency to wavelength in a JAX-compatible manner.

    Parameters:
    - f: Frequency in Hertz. Can be a float or a JAX array of floats.

    Returns:
    - Wavelength in meters. Has the same shape as input f.
    """
    # Note: In JAX, explicit type and dimension checks like those done in NumPy 
    # might not be as straightforward due to JAX's lazy evaluation model. 
    # If you still need type checks, consider doing them at a higher level 
    # or using JAX's custom JIT-compiled function behaviors for runtime assertions.

    wvl = scic.c / f
    return wvl

def convert_wavelengths_to_frequencies(wavelengths):
    """
    Convert wavelength to frequency in a JAX-compatible manner.

    Parameters:
    - wavelengths: Wavelength in meters. Can be a float or a JAX array of floats.

    Returns:
    - Frequency in Hertz. Has the same shape as input wavelengths.
    """
    frequencies = scic.c / wavelengths
    return frequencies