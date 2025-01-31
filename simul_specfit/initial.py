"""
Functions for initial parameter estimation
"""

# Astronomy packages
from astropy.stats import SigmaClippedStats
from astropy import units as u, constants as consts

# Numerical packages
import numpy as np
import jax.numpy as jnp

# Spectra class
from simul_specfit import defaults
from simul_specfit.spectra import Spectra, Spectrum


def linesFluxesGuess(
    config: list,
    spectra: Spectra,
    inner: u.Quantity = defaults.LINEPAD,
    outer: u.Quantity = defaults.CONTINUUM,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Guess the line fluxes for a given configuration

    Parameters
    ----------
    spectra : Spectra
        Spectra
    config : dict
        Configuration of emission lines
    inner : u.Quantity, optional
        Inner region to compute the flux
    outer : u.Quantity, optional
        Outer region to compute the median

    Returns
    -------
    tuple(jnp.ndarray, jnp.ndarray)
        Line centers and line flux guesses

    """
    centers = jnp.array(
        u.Quantity(
            [
                li['Wavelength']
                for g in config['Groups'].values()
                for s in g['Species']
                for li in s['Lines']
            ],
            config['Unit'],
        ).to(spectra.λ_unit)
    )

    # Get the guesses
    guesses = [
        max(
            [
                lineFluxGuess(spectrum, line, u.Unit(config['Unit']), inner, outer)
                for spectrum in spectra.spectra
            ]
        )
        / (line['RelStrength'] if line['RelStrength'] is not None else 1)
        for group in config['Groups'].values()
        for species in group['Species']
        for line in species['Lines']
    ]

    # For all lines that are tied, guess to the max value divided by number of tied lines
    i = 0
    for group in config['Groups'].values():
        for species in group['Species']:
            species_guesses, species_inds = [], []
            for line in species['Lines']:
                if line['RelStrength'] is not None:
                    species_guesses.append(guesses[i])
                    species_inds.append(i)
                i += 1
            if species_guesses:
                species_guess = max(species_guesses) / len(species_guesses)
                for ind in species_inds:
                    guesses[ind] = species_guess

    return centers, jnp.array(guesses)


# Line Flux Guess
def lineFluxGuess(
    spectrum: Spectrum, line: dict, unit: u.Unit, inner: u.Quantity, outer: u.Quantity
) -> float:
    """
    Compute the line flux guess as the sum of the flux in the inner region minus the median of the outer region

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum
    line : dict
        Line configuration
    inner : u.Quantity, optional
        Inner region to compute the flux
    outer : u.Quantity, optional
        Outer region to compute the median

    Returns
    -------
    float
        Line flux guess
    """

    # Convert to resolution
    inner = (inner / consts.c).to(u.dimensionless_unscaled).value
    outer = (outer / consts.c).to(u.dimensionless_unscaled).value

    # Compute the line wavelength
    linewav = (line['Wavelength'] * unit).to(spectrum.λ_unit)

    # Redshift the line
    linewav = linewav * (1 + spectrum.redshift_initial)
    innerwidth, outerwidth = linewav * inner, linewav * outer

    # Compute the boundaries
    ilow, ihigh = (linewav - innerwidth).value, (linewav + innerwidth).value
    olow, ohigh = (linewav - outerwidth).value, (linewav + outerwidth).value

    # Compute the mask
    imask, lmask = spectrum.coverage(ilow, ihigh), spectrum.coverage(olow, ohigh)

    # Check if the mask is empty
    if lmask.sum() == 0 or imask.sum() == 0:
        return -jnp.inf

    # Compute the median background
    background = SigmaClippedStats(spectrum.flux[lmask], sigma=3).median()

    # Get the spectruml flux in the line region
    flux = (
        (spectrum.flux[imask] - background) * (spectrum.high - spectrum.low)[imask]
    ).sum()

    # What to do if this is negative?
    if flux < 0:
        return jnp.abs(flux)

    # Get 1σ max flux
    return flux


def computeContinuumRegions(
    config: list, spectra: Spectra, pad: u.Quantity = defaults.CONTINUUM
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the continuum regions from the configuration

    Parameters
    ----------
    config : dict
        Configuration of emission lines
    spectra : Spectra
        Spectra
    pad : u.Quantity, optional
        Region width around the lines

    Returns
    -------
    (np.ndarray, np.ndarray)
        Continuum regions and continuum height guesses
    """

    # Get lines from config
    lines = np.sort(
        [
            line['Wavelength']
            for group in config['Groups'].values()
            for species in group['Species']
            for line in species['Lines']
        ]
    ) * u.Unit(config['Unit'])

    # Compute pad in correct units
    pad = (pad / consts.c).to(u.dimensionless_unscaled).value

    # Generate continuum regions
    allregs = lines[:, np.newaxis] + np.array([-1, 1]) * (pad * lines)[:, np.newaxis]
    cont_regs = [allregs[0]]
    for region in allregs[1:]:
        if region[0] < cont_regs[-1][1]:
            cont_regs[-1][1] = region[1]
        else:
            cont_regs.append(region)

    # Convert to correct units and redshift
    cont_regs = jnp.array(
        [cont_regs.to(spectra.λ_unit).value for cont_regs in cont_regs]
    ) * (1 + spectra.redshift_initial)

    return cont_regs, continuumHeightGuesses(cont_regs, config, spectra)


def continuumHeightGuesses(
    continuum_regions: jnp.ndarray,
    config: list,
    spectra: Spectra,
    linepad: u.Quantity = defaults.LINEPAD,
    sigma: float = 0,
) -> jnp.ndarray:
    """
    Guess the continuum height for different

    Parameters
    ----------
    spectra : Spectra
        Spectra
    continuum_regions : list
        List of continuum regions
    config : dict
        Configuration of emission lines
    linepad : u.Quantity, optional
        Padding to mask line
    sigma : float, optional


    Returns
    -------
    jnp.ndarray
        Array of continuum height guesses
    """

    # Return the updated config
    return jnp.array(
        [
            max(
                [
                    continuumHeightGuess(
                        config, continuum_regions, spectrum, linepad, sigma
                    )
                    for spectrum in spectra.spectra
                ]
            )
            for continuum_regions in continuum_regions
        ]
    )


# Continuum Height Guess
def continuumHeightGuess(
    config: dict,
    continuum_region: jnp.ndarray,
    spectrum: Spectrum,
    linepad: u.Quantity,
    sigma: float,
) -> jnp.ndarray:
    """
    Guess the continuum height for a spectrum

    Parameters
    ----------
    continuum_region : jnp.ndarray
        Boundary of the continuum region
    config : dict
        Configuration of emission lines
    linepad : u.Quantity
        Padding to mask line
    sigma : float
        Upper bound for median calculation

    Returns
    -------
    float
        Continuum Height Estimate
    """

    # Mask the lines
    mask = spectrum.maskLines(config, continuum_region, linepad)

    # If no coverage, return -∞
    if mask.sum() == 0:
        return -jnp.inf

    # Compute the median Nsigma upper bound
    return jnp.median(spectrum.flux[mask] + sigma * spectrum.err[mask])
