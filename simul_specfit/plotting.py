"""
Plotting functions for the results of the sampling
"""

# Standard library
import os

# Plotting packages
from matplotlib import pyplot

# Astropy packages
from astropy import units as u
from astropy.table import Table

# Numerical packages
import numpy as np
from jax import numpy as jnp
from scipy.optimize import minimize


def plotResults(config: list, rows: Table, model_args: tuple, samples: dict) -> None:
    """
    Plot the results of the sampling.

    Parameters
    ----------
    savedir : str
        Directory to save the plots
    config: list
        Configuration list
    rows : Table
        Table of the rows
    model_args : tuple
        Arguments for the model
    samples : dict
        Samples from the MCMC


    Returns
    -------
    None

    """
    # Get config name
    cname = '_' + config['Name'] if config['Name'] else ''

    # Unpack model arguements
    spectra, _, _, _, line_centers, _, cont_regs, _ = model_args

    # Get the number of spectra and regions
    Nspec, Nregs = len(spectra.spectra), len(cont_regs)

    # Plotting
    figsize = (15, 12)  # (7.5 * Nregs, 6 * Nspec)
    fig, axes = pyplot.subplots(
        Nspec, Nregs, figsize=figsize, sharex='col', sharey='row'
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Ensure axes is always a 2D array
    if Nspec == 1 and Nregs == 1:
        axes = np.array([[axes]])  # Convert single Axes object to a 2D array
    elif Nspec == 1 or Nregs == 1:
        axes = np.atleast_2d(axes).reshape(Nspec, Nregs)  # Convert 1D array to 2D array

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
                    wave[mask],
                    model[k][mask],
                    color='#E20134',
                    alpha=1 / 100,
                    ds='steps-mid',
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

            # Line Labels
            for line in line_centers:
                line = line * (1 + spectra.redshift_initial)
                if line < cont_reg[0] or line > cont_reg[1]:
                    continue
                ax.axvline(line, color='k', linestyle='--', alpha=0.5)

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
        f"{rows[0]['srcid']} ({rows[0]['root']}): $z = {spectrum.redshift_initial:.3f}$",
        ha='center',
        va='center',
        fontsize='large',
    )

    # Show the plot
    fig.savefig(
        os.path.join('RUBIES/Plots', f'{rows[0]['root']}-{rows[0]['srcid']}{cname}_fit.pdf')
    )
    pyplot.close(fig)


def plotLines(ax, config, model_args) -> None:
    """
    Plot the lines of the spectra.

    Parameters
    ----------
    ax : Axes
        Axes to plot the lines
    config: list
        Configuration list
    model_args : tuple
        Arguments for the model

    Returns
    -------
    None

    """

    # Unpack model arguements
    spectra, _, _, _, _, _, _, _ = model_args
    oneplusz = 1 + spectra.redshift_initial

    # Get axis xlim and ylim
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # Iterate over configuration
    names, centers = [], []
    for group in config['Groups']:
        for species in group['Species']:
            for line in species['Lines']:
                # Get the line center
                line_center = (
                    (line['Wavelength'] * oneplusz * u.Unit(config['Unit']))
                    .to(spectra.λ_unit)
                    .value
                )

                # Check if line is in the axis limits
                if line_center < xlim[0] or line_center > xlim[1]:
                    continue

                # Append to names and centers
                names.append(species['Name'])
                centers.append(line_center)

    # Sort by wavelength
    names, centers = zip(*sorted(zip(names, centers), key=lambda x: x[1]))

    # Get name centers
    x0 = np.linspace(xlim[0], xlim[1], len(names) + 2)[1:-1]
    namecenters = minimize(
        logbarrier,
        x0,
        args=(xlim, centers, 1000),
        method='Nelder-Mead',
        options={'adaptive': True, 'maxiter': len(names) * 750},
    ).x

    # Plot the lines
    for name, ncenter, center in zip(names, namecenters, centers):
        ax.plot(
            [center, center, ncenter, ncenter],
            [ylim[0], ylim[1] * 0.92, ylim[1] * 0.93, ylim[1] * 0.95],
            color='k',
            linestyle='--',
            alpha=0.5,
        )

        ax.text(ncenter, ylim[1] * 0.96, name, va='center', ha='center', fontsize=15)

    # Set the axis limits
    ax.set(xlim=xlim, ylim=ylim)


# Log barrier constraints
def logbarrier(x, xlim, linelocs, norm):
    y = np.concatenate([[xlim[0]], x, [xlim[1]]])
    return np.square(x - linelocs).sum() - np.log(y[1:] - y[:-1]).sum() / norm
