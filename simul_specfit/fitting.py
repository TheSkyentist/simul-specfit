"""
Fitting functions for spectral data
"""

# Standard library
import re

# Typing
from typing import Dict, Tuple

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
from jax import random, vmap

# Simul-SpecFit
from simul_specfit.model import multiSpecModel
from simul_specfit.plotting import plotResults
from simul_specfit.spectra import RubiesSpectra
from simul_specfit import utils, initial, parameters


def RubiesFit(config: dict, rows: Table, backend: str = 'MCMC') -> None:
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


def RUBIESModelArgs(config: dict, rows: Table) -> Tuple:
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
    config = utils.restrictConfig(config, spectra)

    # If the config is empty, skip
    if len(config['Groups']) == 0:
        raise ValueError('No Line Coverage')

    # Generate Parameter Matrices
    matrices, linetypes_all = parameters.configToMatrices(config)

    # Compute Continuum Regions and Initial Guesses
    cont_regs, cont_guesses = initial.computeContinuumRegions(config, spectra)

    # Compute Line Centers and Equalized estimates
    line_centers, line_estimates_eq = initial.linesFluxesGuess(
        config, spectra, cont_regs, cont_guesses
    )

    # Restrict spectra to continuum regions and rescale errorbars in each region
    spectra.restrictAndRescale(config, cont_regs)

    # Skip if no data
    if len(spectra.spectra) == 0:
        raise ValueError('No Valid Data')

    # Model Args
    return config, (
        spectra,
        matrices,
        linetypes_all,
        line_centers,
        line_estimates_eq,
        cont_regs,
        cont_guesses,
    )


def MCMCFit(
    model_args: tuple, rng_key: random.PRNGKey, N: int = 500
) -> Tuple[Dict, Dict]:
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

    # Compute relevant probabilities
    logL = computeProbs(samples, model_args)

    # Compute the WAIC
    waic = -2 * (
        np.log(np.exp(logL).mean(axis=0)).sum() - logL.var(axis=0, ddof=1).sum()
    )
    extras = {'WAIC': waic}

    return samples, extras


def NSFit(
    model_args: tuple, rng_key: random.PRNGKey, N: int = 1000
) -> Tuple[Dict, Dict]:
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

    # Compute relevant probabilities
    _ = computeProbs(samples, model_args)

    # Add log evidence to samples
    extras = {
        'logZ': float(NS._results.log_Z_mean),
        'logZ_err': float(NS._results.log_Z_uncert),
    }

    return samples, extras


def computeProbs(samples: dict, model_args: tuple) -> np.ndarray:
    # Compute the log likelihood
    logLs = infer.util.log_likelihood(multiSpecModel, samples, *model_args)
    for k, v in logLs.items():
        samples[k] = v
    logL = np.hstack([p for p in logLs.values()])  # Likelihood Matrix
    samples['logL'] = logL.sum(1)

    # Compute the log density
    logP = vmap(lambda s: infer.util.log_density(multiSpecModel, model_args, {}, s)[0])(
        samples
    )
    samples['logP'] = np.array(logP)

    return logL


def saveResults(config, rows, model_args, samples, extras) -> None:
    # Get config name
    cname = '_' + config['Name'] if config['Name'] else ''

    # Unpack model args
    spectra, _, _, _, _, cont_regs, _ = model_args

    # Correct sample units
    samples['flux_all'] = samples['flux_all'] * (spectra.fλ_unit * spectra.λ_unit).to(
        u.Unit(1e-20 * u.erg / (u.cm * u.cm * u.s))
    )
    samples['ew_all'] = samples['ew_all'] * spectra.λ_unit.to(u.AA)

    # Add spectra wavelength to samples
    for spectrum in spectra.spectra:
        samples[f'{spectrum.name}_wavelength'] = spectrum.wave

    # Create outputs
    colnames = [
        n
        for n in ['lsf_scale', 'PRISM_flux', 'PRISM_offset', 'logL', 'logP']
        if n in samples.keys()
    ]
    out = Table([samples[name] for name in colnames], names=colnames)

    # Add continuum regions and error scales to samples
    samples['cont_regs'] = np.array(cont_regs)
    samples.update(
        {
            f'{spectrum.name}_errscales': np.array(spectrum.errscales)
            for spectrum in spectra.spectra
        }
    )

    # Save all samples as npz
    np.savez(
        f'RUBIES/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}_full.npz',
        **samples,
    )

    # Get names of the lines
    # TODO: Better sanitization of line names?
    line_names = [
        re.sub(
            r'[\[\]]',
            '',
            f'{species["Name"]}_{species["LineType"]}_{line["Wavelength"]}',
        )
        for _, group in config['Groups'].items()
        for species in group['Species']
        for line in species['Lines']
    ]

    # Append line parameter samples
    for colname, unit in zip(
        ['redshift', 'flux', 'fwhm', 'ew'],
        [
            u.dimensionless_unscaled,
            u.Unit(1e-20 * u.erg / u.cm**2 / u.s),
            u.km / u.s,
            u.AA,
        ],
    ):
        data = np.array(samples[f'{colname}_all'].T.tolist()) * unit
        out_part = Table(data.T, names=[f'{line}_{colname}' for line in line_names])
        out = hstack([out, out_part])

    # Append LSF samples
    for spectrum in spectra.spectra:
        data = np.array(samples[f'{spectrum.name}_lsf'].T.tolist()) * spectra.λ_unit
        out_part = Table(
            data.T, names=[f'{spectrum.name}_{line}_lsf' for line in line_names]
        )
        out = hstack([out, out_part])

    # Create extra table
    extra = Table([[v] for v in extras.values()], names=extras.keys())

    # Create HDUList
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(out, name='PARAMS'),
            fits.BinTableHDU(extra, name='EXTRAS'),
        ]
    )

    # Save the summary
    hdul.writeto(
        f'RUBIES/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}_summary.fits',
        overwrite=True,
    )
