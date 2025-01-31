"""
Multi-Spectrum Model
"""

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
from simul_specfit.calibration import RubiesCalibration

# Speed of light
C: Final[float] = consts.c.to(u.km / u.s).value


# Define the model
def multiSpecModel(
    spectra: Spectra,
    F: BCOO,
    Z: BCOO,
    Σs: tuple[BCOO, BCOO, BCOO],
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
    F : BCOO
        Flux matrix
    Z : BCOO
        Redshift matrix
    Σs : BCOO, BCOO, BCOO
        Width matrices
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
    cont_regs = cont_regs * (1 + spectra.redshift_initial)

    # Plate over the continua
    Nc = len(cont_regs)  # Number of continuum regions
    with plate(f'Nc = {Nc}', Nc):
        # Continuum centers
        cont_centers = determ('cont_center', cont_regs.mean(axis=1))

        # Continuum angles
        angles = sample('cont_angle', priors.angle_prior())

        # Continuum offsets
        offsets = sample('cont_offset', priors.height_prior(cont_guesses))

    # Build Spectrum Calibratsion
    calib = RubiesCalibration(spectra.names, spectra.fixed)

    # Plate for fluxes
    Nf = F.shape[0]  # Number of independent fluxes
    with plate(f'Nf = {Nf}', Nf):
        # Sample fluxes
        fluxes = sample('f', priors.flux_prior(jnp.ones_like(F) @ line_guesses))

        # Broadcast fluxes
        fluxes = determ('f_all', fluxes @ F)

    # Plate for redshifts
    Nz = Z.shape[0]  # Number of independent redshifts
    with plate(f'Nz = {Nz}', Nz):
        # Sample redshifts
        redshift = sample('z', priors.redshift_prior(spectra.redshift_initial))
        oneplusz = 1 + determ('z_all', redshift @ Z)

        # Broadcast redshifts and compute centers
        centers = line_centers * oneplusz

    # Unpack the width matrices
    Σ, Σadd, Σuadd = Σs

    # Plate for fwhms
    Nw = Σ.shape[0]  # Number of independent narrow fwhms
    with plate(f'Nw = {Nw}', Nw):
        # Sample fwhms
        fwhms = sample('fwhm', priors.fwhm_prior())

        # Broadcast fwhms
        all_fwhms = fwhms @ Σ

    # If there are additional fwhms, plate over them
    if Σadd.shape[0]:
        # Plate for additional fwhms
        Nwadd = Σadd.shape[0]  # Number of independent additional fwhms
        with plate(f'Nw_add = {Nwadd}', Nwadd):
            # Create lower bounds for initial fwhms
            fwhms_add_lower = fwhms @ Σuadd

            # Sample additional fwhms
            # Ideally encapsulate this to be dependent on the line type later!
            fwhms_add = sample(
                'fwhm_add', priors.fwhm_prior(low=fwhms_add_lower + 100, high=5000)
            )
            all_fwhms_add = fwhms_add @ Σadd

            # Combine the fwhms
            all_fwhms = all_fwhms + all_fwhms_add

    # Transform fwhms into wavelength units
    fwhms = centers * determ('fwhm_all', all_fwhms) / C

    # Compute equivalent widths
    linecont = optimized.linearContinua(
        centers, cont_centers, angles, offsets, cont_regs
    ).sum(1)
    determ('ew_all', fluxes / (linecont * oneplusz))

    # Loop over spectra
    for spectrum in spectra.spectra:
        # Get the spectrum
        low, wave, high, flux, err = (jnp.array(x) for x in spectrum())

        # Get the calibration
        lsf_scale, pixel_offset, flux_scale = calib[spectrum.name]

        # Apply pixel offset
        low = low - spectrum.offset(low, pixel_offset)
        wave = wave - spectrum.offset(wave, pixel_offset)
        high = high - spectrum.offset(high, pixel_offset)
        cont_regs_shift = cont_regs - spectrum.offset(cont_regs, pixel_offset)

        # Compute effective redshift after shift
        # centers_shift = centers - spectrum.offset(centers, pixel_offset)
        # determ(f'{spectrum.name}_z_all', (centers_shift / line_centers) - 1)

        # Get the LSF of the lines
        fwhms_lsf = spectrum.lsf(centers, lsf_scale)

        # Broad lines
        broad = jnp.any(Σadd.todense(), 0)

        # Integrate pixels (note, this is total integral, not a density)
        pixints = jnp.where(
            broad,
            optimized.integrateVoigt(low, high, centers, fwhms_lsf, fwhms),
            optimized.integrateGaussian(
                low, high, centers, jnp.sqrt(jnp.square(fwhms_lsf) + jnp.square(fwhms))
            ),
        )

        # Divide by bin width to compute flux density
        fλ = pixints / (high - low)[:, jnp.newaxis]

        # Multiply by line fluxes
        lines = determ(f'{spectrum.name}_lines', fluxes * fλ)

        # Compute continuum
        continuum = determ(
            f'{spectrum.name}_cont',
            optimized.linearContinua(
                wave, cont_centers, angles, offsets, cont_regs_shift
            ).sum(1),
        )

        # Compute model
        model = determ(
            f'{spectrum.name}_model', flux_scale * (lines.sum(1) + continuum)
        )

        # Compute likelihood
        sample(f'{spectrum.name}', dist.Normal(model, err), obs=flux)


# Define the model
def plotMultiSpecModel(
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
    with plate(f'Continua (N = {Nc})', Nc):
        # Continuum Parameters
        angles = sample('θ', priors.angle_prior())
        offsets = sample('Fλ', priors.height_prior(cont_guesses))
        cont = determ('Linear Continuum', angles + offsets).mean()

    cont = determ('Total Continuum', cont)

    # Plate for redshifts
    Nz = Z.shape[0]  # Number of independent redshifts
    with plate(f'Redshifts (N = {Nz})', Nz):
        # Sample redshifts
        redshift = sample('z', priors.redshift_prior(spectra.redshift_initial))
        redshift = redshift @ Z

    # Plate for fwhms
    Nw = Σ.shape[0]  # Number of independent fwhms
    with plate(f'Dispersions (N = {Nw})', Nw):
        # Sample fwhms
        fwhms = sample('fwhm', priors.fwhm_prior())
        fwhms = fwhms @ Σ

    # Plate for fluxes
    Nf = F.shape[0]  # Number of independent fluxes
    with plate(f'Fluxes (N = {Nf})', Nf):
        # Sample fluxes
        fluxes = sample('f', priors.flux_prior(F @ line_guesses))
        fluxes = fluxes @ F

    # Plate over the lines
    Nl = len(line_centers)  # Number of lines
    with plate(f'Lines (N = {Nl})', Nl):
        lines = determ('Lines', redshift + fwhms + fluxes).mean()
        determ('Equivalent Width', fluxes + cont)

    # LSF Scale
    lsf_scale = sample('LSF Scale', priors.lsf_scale_prior()).mean()

    # Plate for spectra
    Nspec = len(spectra.spectra)  # Number of spectra
    with plate(f'Spectra (N = {Nspec})', Nspec, dim=-1):
        # Get the calbrations
        flux_scale = sample('Flux Scale', priors.flux_scale_prior()).mean()
        pixel_offset = sample('Pixel Offsets', priors.pixel_offset_prior()).mean()
        λ = jnp.ones(1000)  # Faux wavelength grid

        with plate('λ', len(λ), dim=-2):
            # Compute the continuum model
            c = determ('Continuum Model', cont + pixel_offset)

            # Compute the line model
            li = determ('Lines Model', lines + lsf_scale + pixel_offset)

            # Compute the model
            model = determ('Total Model', flux_scale + c + li)

            # Compute the likelihood
            sample('Observations', dist.Normal(model, 1), obs=1)
