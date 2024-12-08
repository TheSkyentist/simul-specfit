"""Multi-Spectrum Model"""

# Standard Imports
from typing import Final

# Astropy
from astropy import units as u, constants as consts

# JAX
from jax import numpy as jnp
from jax.experimental.sparse import BCOO

# Bayesian Inference
from numpyro import plate, sample, deterministic as determ, distributions as dist

# Simul-SpecFit
from simul_specfit import priors, optimized
from simul_specfit.spectra import Spectra
from simul_specfit.calibration import Calibration

# Speed of light
C: Final[float] = consts.c.to(u.km / u.s).value


# Define the model
def multiSpecModel(
    spectra: Spectra,
    Z: BCOO,
    Σ: BCOO,
    F: BCOO,
    line_centers: jnp.ndarray,
    line_guesses: jnp.ndarray,
    cont_regs: jnp.ndarray,
    cont_guesses: jnp.ndarray,
) -> None:
    """
    Multi-Spectrum Model

    Parameters
    ----------
    spectra : Spectra
        Spectra to fit
    Z : BCOO
        Redshift matrix
    Σ : BCOO
        Width matrix
    F : BCOO
        Flux matrix
    line_centers : jnp.ndarray
        Line centers
    line_guesses : jnp.ndarray
        Line guesses
    cont_regs : jnp.ndarray
        Continuum regions
    cont_guesses : jnp.ndarray
        Continuum guesses

    Returns
    -------
    None
    """

    # Redshift the continuum regions
    continuum_regions = cont_regs * (1 + spectra.redshift_initial)

    # Plate over the continua
    Nc = len(continuum_regions)  # Number of continuum regions
    with plate(f'Nc = {Nc}', Nc):
        # Continuum centers
        cont_centers = determ('cont_center', continuum_regions.mean(axis=1))

        # Continuum angles
        angles = sample('cont_angle', priors.angle_prior())
        # angles = determ('cont_angle', 0)

        # Continuum offsets
        offsets = sample('cont_offset', priors.height_prior(cont_guesses))

    # Build Spectrum Calibratsion
    calib = Calibration(spectra.names, spectra.fixed)

    # Plate for redshifts
    Nz = Z.shape[0]  # Number of independent redshifts
    with plate(f'Nz = {Nz}', Nz):
        # Sample redshifts
        redshift = sample('z', priors.redshift_prior(spectra.redshift_initial))
        centers = line_centers * (1 + redshift @ Z)

        # Keep track of redshifts
        determ('z_all', centers)

    # Plate for widths
    Nσ = Σ.shape[0]  # Number of independent widths
    with plate(f'Nσ = {Nσ}', Nσ):
        # Sample widths
        widths = sample('σ', priors.sigma_prior()) 
        widths = centers * determ('σ_all', widths @ Σ) / C

    # Plate for fluxes
    Nf = F.shape[0]  # Number of independent fluxes
    with plate(f'Nf = {Nf}', Nf):
        # Sample fluxes
        fluxes = sample('f', priors.flux_prior(F @ line_guesses))
        fluxes = fluxes @ F

        # Keep track of fluxes
        determ('f_all', fluxes)

    # Compute equivalent widths
    linecont = optimized.linearContinua(
        centers, cont_centers, angles, offsets, continuum_regions
    ).sum(1)
    determ('ew_all', fluxes / linecont)

    # Loop over spectra
    for spectrum in spectra.spectra:
        # Get the spectrum
        low, wave, high, flux, err = spectrum()

        # Get the calibration
        lsf_scale, pixel_offset, flux_scale = calib(spectrum.name)

        # Apply pixel offset
        low = low + spectrum.offset(low, pixel_offset)
        high = high + spectrum.offset(high, pixel_offset)

        # Broaden the lines
        lsf = spectrum.lsf(centers, lsf_scale)
        tot_widths = jnp.sqrt(jnp.square(widths) + jnp.square(lsf))

        # Compute lines
        lines = determ(
            f'{spectrum.name}_lines',
            optimized.integrate(low, high, centers, tot_widths, fluxes),
        )

        # Compute continuum
        continuum = determ(
            f'{spectrum.name}_cont',
            optimized.linearContinua(
                wave, cont_centers, angles, offsets, continuum_regions
            ).sum(1),
        )

        # Compute model
        model = determ(
            f'{spectrum.name}_model', flux_scale * (lines.sum(1) + continuum)
        )

        # Compute likelihood
        sample(f'{spectrum.name}', dist.Normal(model, err), obs=flux)
