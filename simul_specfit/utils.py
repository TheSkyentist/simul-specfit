"""Utility functions for the simulation and fitting of spectra"""

# Import packages
import copy

# Astronomy packages
from astropy.stats import SigmaClippedStats
from astropy import units as u, constants as consts

# Numerical packages
import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

# Spectra class
from simul_specfit.spectra import Spectra, Spectrum

# Hard coded for now
LINEDETECT = 1_000 * (u.km / u.s)
LINEPAD = 4_000 * (u.km / u.s)
CONTINUUM = 10_000 * (u.km / u.s)


def configToMatrices(config: dict) -> tuple[BCOO, BCOO, BCOO]:
    """
    Convert the configuration to sparse matrices for the model

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    tuple[BCOO, BCOO, BCOO]
        Redshift, dispersion, and flux transformation matrice
    """

    # Keep track of indices
    i = 0
    i_f, f_inds, fluxes = 0, [], []
    i_z, z_inds = 0, []
    i_σ, σ_inds = 0, []
    # Iterate over groups, species, and lines
    for group in config['Groups']:
        for species in group['Species']:
            # Check if any fluxes in species are tied
            if not all([line['RelStrength'] is None for line in species['Lines']]):
                # Assign special index for tied fluxes
                species_f = i_f
                i_f += 1  # Increment for not tied fluxes
            # Iterate over lines
            for line in species['Lines']:
                # Keep track of nonzero matrix elements
                z_inds.append([i_z, i])
                σ_inds.append([i_σ, i])

                # If the flux is not tied, increment
                if line['RelStrength'] is None:
                    fluxes.append(1)
                    f_inds.append([i_f, i])
                    i_f += 1
                else:
                    fluxes.append(line['RelStrength'])
                    f_inds.append([species_f, i])

                # Increment line index
                i += 1

            # If Group is not tied, increment
            if not group['TieRedshift']:
                i_z += 1
            if not group['TieDispersion']:
                i_σ += 1

        # Increment between groups if necessary
        if group['TieRedshift']:
            i_z += 1
        if group['TieDispersion']:
            i_σ += 1

    # Create sparse transformaton matrices
    Z = BCOO((jnp.ones(i), z_inds), shape=(i_z, i))
    Σ = BCOO((jnp.ones(i), σ_inds), shape=(i_σ, i))
    F = BCOO((fluxes, f_inds), shape=(i_f, i))

    return Z, Σ, F


def restrictConfig(
    config: list, spectra: Spectra, linepad: u.Quantity = LINEDETECT
) -> list:
    """
    Restrict the configuration to only include lines that are covered by the spectra

    Parameters
    ----------
    config : list
        Configuration of emission lines
    linepad : u.Quantity, optional
        Padding around the lines necessary to cover the line
        In velocity space

    Returns
    -------
    list
        Updated configuration
    """

    # Initialize dictionary
    config = copy.deepcopy(config)

    # Effective resolution
    lineres = (linepad / consts.c).to(u.dimensionless_unscaled).value

    # Loop over config
    new_groups = []
    for group in config['Groups']:
        new_species = []
        for species in group['Species']:
            new_lines = []
            for line in species['Lines']:
                # Compute line wavelength
                linewav = (line['Wavelength'] * u.Unit(config['Unit'])).to(
                    spectra.λ_unit
                )

                # Redshift the line
                linewav = linewav * (1 + spectra.redshift_initial)
                linewidth = linewav * lineres

                # Compute boundaries
                low, high = (linewav - linewidth).value, (linewav + linewidth).value

                # Check coverage
                if jnp.logical_or.reduce(
                    jnp.array([s.coverage(low, high).any() for s in spectra.spectra])
                ):
                    new_lines.append(line)

            # Add species only if it has remaining lines
            if new_lines:
                species['Lines'] = new_lines
                new_species.append(species)

        # Add group only if it has remaining species
        if new_species:
            group['Species'] = new_species
            new_groups.append(group)

    # Update config with filtered groups
    config['Groups'] = new_groups

    # Return the updated config
    return config


def linesFluxesGuess(
    config: list,
    spectra: Spectra,
    inner: u.Quantity = LINEPAD,
    outer: u.Quantity = CONTINUUM,
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
                for g in config['Groups']
                for s in g['Species']
                for li in s['Lines']
            ],
            config['Unit'],
        ).to(spectra.λ_unit)
    )

    guesses = jnp.array(
        [
            max(
                [
                    lineFluxGuess(
                        spectrum,
                        line,
                        u.Unit(config['Unit']),
                        inner,
                        outer,
                    )
                    for spectrum in spectra.spectra
                ]
            )
            for group in config['Groups']
            for species in group['Species']
            for line in species['Lines']
        ]
    )

    return centers, guesses

# Line Flux Guess
def lineFluxGuess(
    spectrum: Spectrum,
    line: dict,
    unit: u.Unit,
    inner: u.Quantity,
    outer: u.Quantity,
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
    config: list, spectra: Spectra, pad: u.Quantity = CONTINUUM
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
            for group in config['Groups']
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

    # Convert to correct units
    cont_regs = jnp.array(
        [cont_regs.to(spectra.λ_unit).value for cont_regs in cont_regs]
    )

    return cont_regs, continuumHeightGuesses(cont_regs, config, spectra)


def continuumHeightGuesses(
    continuum_regions: list,
    config: list,
    spectra: Spectra,
    linepad: u.Quantity = LINEPAD,
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
        Padding around the lines necessary to cover the line
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
    config: list,
    continuum_region: list,
    spectrum: Spectrum,
    velpad: u.Quantity,
    sigma: float,
) -> jnp.ndarray:
    """
    Guess the continuum height for a spectrum

    Parameters
    ----------
    continuum_region : list
        Boundary of the continuum region
    config : dict
        Configuration of emission lines
    velpad : u.Quantity
        Padding around the lines
    sigma : float
        Upper bound for median calculation

    Returns
    -------
    float
        Continuum Height Estimate
    """

    # Grow by redshift
    opz = 1 + spectrum.redshift_initial
    continuum_region = continuum_region * opz
    pad = (velpad / consts.c).to(u.dimensionless_unscaled).value

    # Extract the region
    low, high = continuum_region
    mask = jnp.logical_and(low < spectrum.wave, spectrum.wave < high)

    # Mask each line
    λ_unit = u.Unit(config['Unit'])
    for group in config['Groups']:
        for species in group['Species']:
            for line in species['Lines']:
                # Compute the line wavelength
                linewav = (line['Wavelength'] * λ_unit).to(spectrum.λ_unit).value * opz

                # Get the effective padding
                linepad = linewav * pad

                # Compute the boundaries
                low, high = linewav - linepad, linewav + linepad

                # Mask the line
                linemask = jnp.logical_and(low < spectrum.wave, spectrum.wave < high)
                mask = jnp.logical_and(mask, jnp.invert(linemask))

    # If no coverage, return -∞
    if mask.sum() == 0:
        return -jnp.inf

    # Compute the median Nsigma upper bound
    return jnp.median(spectrum.flux[mask] + sigma * spectrum.err[mask])
