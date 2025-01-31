"""
Fitting functions for spectral data
"""

# Astropy packages
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack

# Numpyro
from numpyro import infer
from numpyro.handlers import trace, seed
from numpyro.contrib.nested_sampling import NestedSampler

# JAX
import numpy as np
from jax import random

# Simul-SpecFit
from simul_specfit import initial, parameters
from simul_specfit.spectra import RubiesSpectra
from simul_specfit.model import multiSpecModel
from simul_specfit.plotting import plotResults


def RubiesFit(config: dict, rows: Table, backend: str = 'MCMC'):
    # Get the model arguments
    config, model_args = RUBIESModelArgs(config, rows)

    # Get the random key
    rng_key = random.PRNGKey(0)

    # Fit the data
    match backend:
        case 'MCMC':
            samples, extras = MCMCFit(model_args, rng_key)
        case 'NS':
            samples, extras = NSFit(model_args, rng_key)

    # Plot the results
    plotResults(config, rows, model_args, samples)

    # Save the results
    saveResults(config, rows, model_args, samples, extras)


def RUBIESModelArgs(config: dict, rows: Table) -> tuple:
    """
    Get the model arguments for the RUBIES data.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    rows : Table
        Table of the rows

    Returns
    -------
    tuple
        Model arguments
    """

    # Load the spectra
    spectra = RubiesSpectra(rows, 'RUBIES/Spectra')

    # Restrict config to what we have coverage of
    config = parameters.restrictConfig(config, spectra)

    # If the config is empty, skip
    if len(config['Groups']) == 0:
        raise ValueError('No Line Coverage')

    # Get what we need for fitting
    F, Z, Σs = parameters.configToMatrices(config)
    line_centers, line_guesses = initial.linesFluxesGuess(config, spectra)

    # Get continuum regions
    cont_regs, cont_guesses = initial.computeContinuumRegions(config, spectra)

    # Restrict spectra to continuum regions and rescale errorbars in each region
    spectra.restrictAndRescale(config, cont_regs)

    # Skip if no data
    if len(spectra.spectra) == 0:
        raise ValueError('No Valid Data')

    # Model Args
    return config, (
        spectra,
        F,
        Z,
        Σs,
        line_centers,
        line_guesses,
        cont_regs,
        cont_guesses,
    )


def MCMCFit(
    model_args: tuple, rng_key: random.PRNGKey, N: int = 500
) -> tuple[dict, dict]:
    """
    Fit the RUBIES data with MCMC.

    Parameters
    ----------
    model_args : tuple
        Model Arguements

    Returns
    -------
    infer.MCMC
        MCMC object
    """

    # MCMC
    kernel = infer.NUTS(multiSpecModel)
    mcmc = infer.MCMC(kernel, num_samples=N, num_warmup=250)
    mcmc.run(rng_key, *model_args)

    # Get the samples
    samples = mcmc.get_samples()

    # Compute the WAIC
    logLs = infer.util.log_likelihood(multiSpecModel, samples, *model_args)
    logL = np.hstack([p for p in logLs.values()])  # Likelihood Matrix
    waic = -2 * (np.log(np.exp(logL).mean(axis=0)).sum() - logL.var(axis=0).sum())
    extras = {'WAIC': [waic]}

    return samples, extras


def NSFit(
    model_args: tuple, rng_key: random.PRNGKey, N: int = 1000
) -> tuple[dict, dict]:
    """
    Fit the RUBIES data with Nested Sampling.

    Parameters
    ----------
    model_args : tuple

    Returns
    -------
    NestedSampler
    """

    # Get number of variables
    with trace() as tr:
        with seed(multiSpecModel, rng_seed=rng_key):
            multiSpecModel(*model_args)
    nv = sum(
        [
            v['value'].size
            for v in tr.values()
            if v['type'] == 'sample' and not v['is_observed']
        ]
    )

    # Nested Sampling
    constructor_kwargs = {'num_live_points': 50 * (nv + 1), 'max_samples': 50000}
    termination_kwargs = {'dlogZ': 0.01}
    NS = NestedSampler(
        model=multiSpecModel,
        constructor_kwargs=constructor_kwargs,
        termination_kwargs=termination_kwargs,
    )
    NS.run(rng_key, *model_args)

    # Get the sample
    samples = NS.get_samples(rng_key, N)

    # Add log evidence to samples
    extras = {
        'log_Z': [NS._results.log_Z_mean],
        'log_Z_err': [NS._results.log_Z_uncert],
    }

    return samples, extras


def saveResults(config, rows, model_args, samples, extras) -> None:
    # Get config name
    cname = '_' + config['Name'] if config['Name'] else ''

    # Unpack model args
    spectra, _, _, _, _, _, _, _ = model_args

    # Correct sample units
    samples['f_all'] = samples['f_all'] * (spectra.fλ_unit * spectra.λ_unit).to(
        u.Unit(1e-20 * u.erg / (u.cm * u.cm * u.s))
    )
    samples['ew_all'] = samples['ew_all'] * spectra.λ_unit.to(u.AA)

    # Add spectra wavelength to samples
    for spectrum in spectra.spectra:
        samples[f'{spectrum.name}_wavelength'] = spectrum.wave

    # Create outputs
    colnames = [
        n for n in ['lsf_scale', 'PRISM_flux', 'PRISM_offset'] if n in samples.keys()
    ]
    out = Table([samples[name] for name in colnames], names=colnames)

    # Save all samples as npz
    np.savez(
        f'RUBIES/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}_full.npz',
        **samples,
    )

    # Get names of the lines
    line_names = [
        f'{species["Name"]}-{species["LineType"]}-{line["Wavelength"]}'
        for _, group in config['Groups'].items()
        for species in group['Species']
        for line in species['Lines']
    ]
    for sampname, colname, unit in zip(
        ['z', 'f', 'fwhm', 'ew'],
        ['redshift', 'flux', 'fwhm', 'ew'],
        [
            u.dimensionless_unscaled,
            u.Unit(1e-20 * u.erg / u.cm**2 / u.s),
            u.km / u.s,
            u.AA,
        ],
    ):
        data = np.array(samples[f'{sampname}_all'].T.tolist()) * unit
        out_part = Table(data.T, names=[f'{line}_{colname}' for line in line_names])
        out = hstack([out, out_part])

    # Create table from extras
    extra = Table(extras)

    # Create HDUList
    hdul = fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(out), fits.BinTableHDU(extra)]
    )
    hdul.writeto(
        f'RUBIES/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}_summary.fits',
        overwrite=True,
    )
