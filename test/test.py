# Import packages
import json
from typing import Final

# JAX packages
from jax import random, numpy as jnp

# Astronomy packages
from astropy.table import Table
from astropy import units as u, constants as consts

# SimulSpecFit
from simul_specfit import utils, priors, optimized
from simul_specfit.spectra import RubiesSpectra
from simul_specfit.calibration import Calibration

# Import numpyro
import numpyro
from numpyro import sample, deterministic as determ, distributions as dist, infer

# Plotting
from matplotlib import pyplot

# Speed of light
C: Final[float] = consts.c.to(u.km / u.s).value

# Load config from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

# Load targets
targets = Table.read('RUBIES/Targets/targets.fits')

# Get testing targets
srcid = 154183  # 154183
rows = targets[targets['srcid'] == srcid]
spectra = RubiesSpectra(rows, 'RUBIES/Spectra')

# Restrict config to what we have coverage of
config = utils.restrictConfig(config, spectra)

# Get what we need for fitting
Z, Σ, F = utils.configToMatrices(config)
line_centers, line_guesses = utils.linesFluxesGuess(config, spectra)

# Get continuum regions
cont_regs, cont_guesses = utils.computeContinuumRegions(config, spectra)

# Rescale errorbars in the continuum regions
## Not implemented

# Restrict spectra to continuum regions
spectra.restrict(cont_regs)

# Define the model
def model(spectra, Z, Σ, F, line_centers, line_guesses, cont_regs, cont_guesses):
    # Redshift the continuum regions
    continuum_regions = cont_regs * (1 + spectra.redshift_initial)

    # Plate over the continua
    Nc = len(continuum_regions)  # Number of continuum regions
    with numpyro.plate(f'Nc = {Nc}', Nc):
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
    with numpyro.plate(f'Nz = {Nz}', Nz):
        # Sample redshifts
        redshift = sample('z', priors.redshift_prior(spectra.redshift_initial))
        centers = line_centers * (1 + redshift @ Z)

    # Plate for widths
    Nσ = Σ.shape[0]  # Number of independent widths
    with numpyro.plate(f'Nσ = {Nσ}', Nσ):
        # Sample widths
        widths = sample('σ', priors.sigma_prior()) / C
        widths = centers * (widths @ Σ)

    # Plate for fluxes
    Nf = F.shape[0]  # Number of independent fluxes
    with numpyro.plate(f'Nf = {Nf}', Nf):
        # Sample fluxes
        fluxes = sample('f', priors.flux_prior(F @ line_guesses))
        fluxes = fluxes @ F

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

        # Compute equivalent widths
        linecont = optimized.linearContinua(
            centers, cont_centers, angles, offsets, continuum_regions
        ).sum(1)
        determ(f'{spectrum.name}_ew', fluxes / linecont)

        # Compute model
        model = determ(
            f'{spectrum.name}_model', flux_scale * (lines.sum(1) + continuum)
        )

        # Compute likelihood
        sample(f'{spectrum.name}', dist.Normal(model, err), obs=flux)


# Render model
model_args = (spectra, Z, Σ, F, line_centers, line_guesses, cont_regs, cont_guesses)
# numpyro.render_model(
#     model,
#     filename='model.pdf',
#     model_args=model_args,
# )

# MCMC
rng = random.PRNGKey(0)
mcmc = infer.MCMC(infer.NUTS(model), num_samples=1000, num_warmup=1000)
mcmc.run(rng, *model_args)
logLs = infer.log_likelihood(model, samples := mcmc.get_samples(), *model_args)

# Number of regions and spectra
Nregs, Nspec = len(cont_regs), len(spectra.spectra)

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
        # Plot the spectrum
        ax.plot(wave, flux, color='k', ds='steps-mid')

        # Plot errorbars on the spectrum
        ax.errorbar(wave, flux, yerr=err, fmt='none', color='k')

        # Plot the models
        model = samples[f'{spectrum.name}_model']
        for k in range(model.shape[0]):
            ax.plot(wave, model[k], color='r', alpha=1 / 100, ds='steps-mid')

        # Label the axes
        if j == 0:
            ax.set(ylabel=f'{spectrum.name}')
        ax.set(xlim=cont_regs[j] * (1 + spectra.redshift_initial))

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
fig.supylabel(f'Flux [{spectrum.fλ_unit.to_string(format="latex", fraction=False)}]')
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
    f'{srcid} ({rows[0]['root']}): $z = {spectrum.redshift_initial}$',
    ha='center',
    va='center',
    fontsize='large',
)

# Show the plot
fig.savefig('test-result.pdf')
pyplot.close(fig)
