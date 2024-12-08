"""
Module for Handling LSF/lsf Curves
"""

# Typing
from typing import Callable

# Import packages
import astropy.units as u
from astropy.table import Table

# Bayesian Inference
from numpyro import sample, deterministic as determ
from simul_specfit import priors

# JAX packages
from jax import jit, numpy as jnp

# Generic Calibration
class Calibration:
    """
    Generic Calibration
    """

    def __init__(self, names: list, fixed: list) -> None:
        """
        Initialize the calibration

        Parameters
        ----------
        spectra : Spectra
            Spectra Object

        Returns
        -------
        None
        """

        # Create LSF scale
        self.lsf_scale = sample('lsf_scale', priors.lsf_scale_prior())

        # Create Pixel Offsets, Flux Scales
        self.pixel_offsets, self.flux_scales = {}, {}

        # Iteraete over the spectra
        for name, fix in zip(names, fixed):
            # If fixed
            if fix:
                # Fixed offset
                self.pixel_offsets[name] = determ(f'{name}_offset', 0)
                self.flux_scales[name] = determ(f'{name}_flux', 1)

            else:
                # Sample offset
                self.pixel_offsets[name] = sample(
                    f'{name}_offset', priors.pixel_offset_prior()
                )
                self.flux_scales[name] = sample(
                    f'{name}_flux', priors.flux_scale_prior()
                )

    def __call__(self, name: str) -> tuple[float, float, float]:
        return (
            self.lsf_scale,
            self.pixel_offsets[name],
            self.flux_scales[name],
        )


# Interpolated lsf Curve
def InterpLSFCurve(lsf_file: str, λ_unit: u.Unit) -> Callable:
    """
    Initialize the lsf curve

    Parameters
    ----------
    lsf_file : str
        File containing the lsf curve
    λ_unit : u.Unit
        Wavelength target unit

    Returns
    -------
    None
    """

    # Load the lsf from file
    lsf_tab = Table.read(lsf_file)

    # Convert to JAX arrays in the correct units
    wave = jnp.array((lsf_tab['wave']).to(λ_unit))
    sigma = jnp.array((lsf_tab['sigma']).to(λ_unit))

    # Compute Interpolated lsf Curve
    @jit
    def lsf(λ, scale):
        lsf_interp = scale * jnp.interp(
            λ, wave, sigma, left='extrapolate', right='extrapolate'
        )
        return lsf_interp

    return lsf


def PixelOffset(dispersion_file: str, λ_unit: u.Unit) -> Callable:
    """
    Return the pixel offset calibration function

    Parameters
    ----------
    dispersion_file : str
        File containing the dispersion curve
    λ_unit : u.Unit
        Wavelength target unit

    Returns
    -------
    Callable
        Pixel Offset Calibration Function
    """

    # Load the dispersion curve
    u.set_enabled_aliases(
        {'MICRONS': u.micron, 'PIXEL': u.pix, 'RESOLUTION': u.Angstrom / u.micron}
    )
    disp_tab = Table.read(dispersion_file)

    # Convert to JAX arrays in the correct units
    wave = jnp.array((disp_tab['WAVELENGTH']).to(λ_unit))
    disp = jnp.array((disp_tab['DLDS']).to(λ_unit / u.pix))

    # Compute Interpolated offset Curve
    @jit
    def pxoff(λ, offset):
        return offset * jnp.interp(
            λ, wave, disp, left='extrapolate', right='extrapolate'
        )

    return pxoff
