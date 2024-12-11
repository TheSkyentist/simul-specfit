"""
Fitting functions for spectral data
"""

# Astropy packages
import astropy.units as u
from astropy.table import Table, hstack

# Numpyro
from numpyro import infer

# JAX
import numpy as np
from jax import random

# Simul-SpecFit
from simul_specfit import utils
from simul_specfit.spectra import RubiesSpectra
from simul_specfit.model import multiSpecModel
from simul_specfit.plotting import plotResults


def RubiesMCMCFit(config: dict, rows: Table) -> infer.MCMC:
    """
    Fit the RUBIES data with MCMC.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    rows : Table
        Table of the rows

    Returns
    -------
    infer.MCMC
        MCMC object
    """

    spectra = RubiesSpectra(rows, 'RUBIES/Spectra')

    # Restrict config to what we have coverage of
    config = utils.restrictConfig(config, spectra)

    # If the config is empty, skip
    if len(config['Groups']) == 0:
        raise ValueError('No Line Coverage')

    # Get whacpft we need for fitting
    Z, Σ, F = utils.configToMatrices(config)
    line_centers, line_guesses = utils.linesFluxesGuess(config, spectra)

    # Get continuum regions
    cont_regs, cont_guesses = utils.computeContinuumRegions(config, spectra)

    # Restrict spectra to continuum regions and rescale errorbars in each region
    spectra.restrictAndRescale(config, cont_regs)

    # Skip if no data
    if len(spectra.spectra) == 0:
        raise ValueError('No Valid Data')

    # Model Args
    model_args = (spectra, Z, Σ, F, line_centers, line_guesses, cont_regs, cont_guesses)

    # MCMC
    rng = random.PRNGKey(0)
    kernel = infer.NUTS(multiSpecModel)
    mcmc = infer.MCMC(kernel, num_samples=1000, num_warmup=1000)
    mcmc.run(rng, *model_args)
    samples = mcmc.get_samples()

    # Plot results
    plotResults('RUBIES/Plots', config, rows, samples, model_args)

    # Correct sample units
    samples['f_all'] = samples['f_all'] * (spectra.fλ_unit * spectra.λ_unit).to(
        u.Unit(1e-20 * u.erg / (u.cm * u.cm * u.s))
    )
    samples['ew_all'] = samples['ew_all'] * spectra.λ_unit.to(u.AA)

    # Create outputs
    colnames = [
        n for n in ['lsf_scale', 'PRISM_flux', 'PRISM_offset'] if n in samples.keys()
    ]
    out = Table([samples[name] for name in colnames], names=colnames)

    # Get names of the lines
    line_names = [
        f'{group['Name']}-{species['Name']}-{line['Wavelength']}'
        for group in config['Groups']
        for species in group['Species']
        for line in species['Lines']
    ]
    for sampname, colname, unit in zip(
        ['z', 'f', 'σ', 'ew'],
        ['redshift', 'flux', 'width', 'ew'],
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

    # Save the output
    out.write(
        f'RUBIES/Results/{rows[0]['root']}-{rows[0]['srcid']}_fit.fits', overwrite=True
    )
