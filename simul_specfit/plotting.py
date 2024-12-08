"""Plotting functions for the results of the sampling."""

# Standard library
import os

# Plotting packages
from matplotlib import pyplot

# Astropy packages
from astropy.table import Table

# JAX packages
from jax import numpy as jnp

# Simul-SpecFit
from simul_specfit.spectra import Spectra


def plotResults(
    savedir: str, cont_regs: jnp.ndarray, spectra: Spectra, samples: dict, rows: Table
) -> None:
    """
    Plot the results of the sampling.

    Parameters
    ----------
    savedir : str
        Directory to save the plots
    cont_regs : jnp.ndarray
        Continuum regions
    spectra : Spectra
        Spectra to fit
    samples : dict
        Samples from the MCMC
    rows : Table
        Table of the rows
    """

    # Get the number of spectra and regions
    Nspec, Nregs = len(spectra.spectra), len(cont_regs)

    # Plotting
    fig, axes = pyplot.subplots(
        Nspec, Nregs, figsize=(7.5 * Nregs, 6 * Nspec), sharex='col', sharey='row'
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Plot the spectra
    for i, spectrum in enumerate(spectra.spectra):
        # Get the spectrum
        _, wave, _, flux, err = spectrum()

        for j, ax in enumerate(axes[i]):
            # Get the continuum region
            cont_reg = cont_regs[j] * (1 + spectra.redshift_initial)
            mask = jnp.logical_and(wave > cont_reg[0], wave < cont_reg[1])

            # Plot the spectrum
            ax.plot(wave[mask], flux[mask], color='k', ds='steps-mid')

            # Plot errorbars on the spectrum
            ax.errorbar(wave[mask], flux[mask], yerr=err[mask], fmt='none', color='k')

            # Plot the models
            model = samples[f'{spectrum.name}_model']
            for k in range(model.shape[0]):
                ax.plot(
                    wave[mask], model[k][mask], color='r', alpha=1 / 100, ds='steps-mid'
                )

            # Label the axes
            if j == 0:
                ax.set(ylabel=f'{spectrum.name}')
            # ax.set(xlim= * (1 + spectsra.redshift_initial))

            # Add rest frame axis
            rest_ax = ax.secondary_xaxis(
                'top',
                functions=(
                    lambda x: x / (1 + spectra.redshift_initial),
                    lambda x: x * (1 + spectra.redshift_initial),
                ),
            )

            # Turn off top xticklabels in the middle
            if i > 0:
                rest_ax.set(xticklabels=[])

            # Turn off top xticks
            ax.tick_params(axis='x', which='both', top=False)

    # Set superlabels
    fig.supylabel(
        f'Flux [{spectrum.fλ_unit.to_string(format="latex", fraction=False)}]'
    )
    fig.supxlabel(
        f'Wavelength (Observed) [{spectrum.λ_unit.to_string(format="latex", fraction=False)}]',
        y=0.06,
        va='center',
        fontsize='medium',
    )
    fig.suptitle(
        f'Wavelength (Rest) [{spectrum.λ_unit.to_string(format="latex", fraction=False)}]',
        y=0.93,
        va='center',
        fontsize='medium',
    )
    fig.text(
        0.5,
        0.97,
        f'{rows[0]['srcid']} ({rows[0]['root']}): $z = {spectrum.redshift_initial}$',
        ha='center',
        va='center',
        fontsize='large',
    )

    # Show the plot
    fig.savefig(os.path.join(savedir, f'{rows[0]['root']}-{rows[0]['srcid']}_fit.pdf'))
    pyplot.close(fig)
