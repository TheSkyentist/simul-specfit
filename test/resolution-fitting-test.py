#! /usr/bin/env python

# Astropy packges
from astropy import units as u, constants as consts

# Plotting
from matplotlib import pyplot

# JAX packages
import jax
from jax import numpy as jnp, random, jit
from jax.scipy.special import erf

# Bayesian Inference
import numpyro
from numpyro import distributions as dist, sample, deterministic as determ, infer

# Set x64 precision
numpyro.enable_x64()

# Speed of light
C: float = consts.c.to(u.km / u.s).value

# Create random key(s)
rng = random.PRNGKey(0)
rng_centers, rng_fluxes, rng_mcmc = random.split(rng, 3)


# Compute the resolution curve
# Parametrized by central wavelength, slope, and resolution at central wavelength
@jit
def R(wave, λ0, θ, R0):
    return 1000 * ((wave - λ0) * jnp.tan(θ) + R0)


# Compute the LSF
@jit
def lsf(wave, *args):
    return wave / R(wave, *args)


# Comptute effectifve line width
@jit
def width_eff(center, line_width, inst_width):
    line_width = center * line_width / C

    return jnp.sqrt(line_width * line_width + inst_width * inst_width)


@jit
def erfcond(good: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Conditional vectorized erf for matrix
    Computes the erf of the input if the condition is met, otherwise returns 0
    Designed to minimize computation since most pixels will integrate to zero

    Parameters
    ----------
    good : jnp.ndarray
        Boolean array of whether the condition is met
    sigma : jnp.ndarray
        Sigma (1/2 variance) values

    Returns
    -------
    jnp.ndarray
        Conditional vectorized erf
    """

    return jax.vmap(jax.vmap(lambda b, λ: jax.lax.cond(b, erf, lambda x: 0.0, λ)))(
        good, sigma
    )


@jit
def integrate(
    low_edge: jnp.ndarray,
    high_edge: jnp.ndarray,
    centers: jnp.ndarray,
    fluxes: jnp.ndarray,
    widths: jnp.ndarray,
    threshold: float = 3.9,
) -> jnp.ndarray:
    """
    Integrate the flux of the emission lines over the pixels

    Parameters
    ----------
    low_edge : jnp.ndarray (λ,)
        Low edge of the bins
    high_edge : jnp.ndarray (λ,)
        High edge of the bins
    centers : jnp.ndarray (N,)
        Centers of the emission lines
    fluxes : jnp.ndarray (N,)
        Flux in the lines
    widths : jnp.ndarray (λ,)
        Instrumental widths at each line center

    Returns
    -------
    jnp.ndarray (λ,)
        Integrated flux over the pixels
    """

    # Adjust width to be for 1/2 variance for erf
    # Take the inverse of the width
    invwidths = 1 / (widths * jnp.sqrt(2))

    # Compute residual
    low_resid = (low_edge[:, jnp.newaxis] - centers) * invwidths
    high_resid = (high_edge[:, jnp.newaxis] - centers) * invwidths

    # Restrict to only those that won't compute to zero
    good = jnp.logical_and(-threshold < low_resid, high_resid < threshold)

    # Compute pixel integral
    pixel_ints = (erfcond(good, high_resid) - erfcond(good, low_resid)) / 2

    # Compute fluxes
    return (fluxes * pixel_ints) / (high_edge - low_edge)[:, jnp.newaxis]


######################################################################
# Generate synthetic data
######################################################################

# Wavelength in microns
low_edge: jnp.ndarray = jnp.linspace(2.9, 5.3, 1414)
δwave: jnp.ndarray = low_edge[1] - low_edge[0]
high_edge: jnp.ndarray = δwave + low_edge
wave: jnp.ndarray = (low_edge + high_edge) / 2

# Create N random emission lines
N: int = 50
centers: jnp.ndarray = random.uniform(rng_centers, (N,), minval=3.0, maxval=5.2)
fluxes: jnp.ndarray = random.uniform(rng_fluxes, (N,), minval=0.5, maxval=2)

# Compute the true flux (also warms up JIT)
pivot: float = 4.1
true_θ: float = 0.6
true_R0: float = 4.0
true_line_width: float = 5  # km/s
true_lsf: jnp.ndarray = lsf(centers, pivot, true_θ, true_R0)
true_flux: jnp.ndarray = integrate(
    low_edge, high_edge, centers, fluxes, width_eff(centers, true_line_width, true_lsf)
).sum(axis=1)

# Add some noise
err = jnp.sqrt((0.1 * true_flux) ** 2 + 10**2)
flux = true_flux + err * random.normal(rng, true_flux.shape)

######################################################################
# Build the model
######################################################################


# Build the model
def model(
    low_edge: jnp.ndarray,
    high_edge: jnp.ndarray,
    flux: jnp.ndarray,
    err: jnp.ndarray,
    centers: jnp.ndarray,
) -> None:
    """
    Model for the emission lines
    Parameters
    ----------
    high_edge : jnp.ndarray
        High edge of the bins
    low_edge : jnp.ndarray
        Low edge of the bins
    flux : jnp.ndarray
        Observed flux values
    err : jnp.ndarray
        Error in the flux values
    centers : jnp.ndarray
        Centers of the emission lines

    Returns
    -------
    None
    """
    # Parametrize the Resolution curve
    λ0_prior = determ('λ0', pivot)  # Center of the resolution curve
    θ_prior = sample(
        'θ', dist.Uniform(-jnp.pi / 2, jnp.pi / 2)
    )  # Slope of the resolution curve
    # θ_prior = determ('θ', true_θ)
    R0_prior = sample('R0', dist.Uniform(3, 5))  # Resolution at λ0
    # R0_prior = determ('R0', true_R0)

    # Prior on line width (shared)
    line_width_prior = sample('line_width', dist.HalfNormal(10))
    # line_width_prior = determ('line_width', 0)

    with numpyro.plate('lines', len(centers), dim=-1):
        # Priors on line flux and center (not shared)
        flux_priors = sample('fluxes', dist.Uniform(0, 3))
        # flux_priors = determ('fluxes', fluxes)

        # Sample the center of each line with a prior around the original values
        # center_priors = sample(
        #     'centers',
        #     dist.TruncatedNormal(
        #         centers, 0.001, low=centers - 0.005, high=centers + 0.005
        #     ),
        # )
        center_priors = determ('centers', centers)

    # Compute the total profile
    line_lsf = determ('line_lsf', lsf(center_priors, λ0_prior, θ_prior, R0_prior))
    effective_width = determ(
        'eff_width',
        width_eff(
            center_priors,
            line_width_prior,
            line_lsf,
        ),
    )
    flux_model = integrate(
        low_edge,
        high_edge,
        center_priors,
        flux_priors,
        effective_width,
    ).sum(axis=1)

    # Likelihood
    likelihood = dist.Normal(flux_model, err)
    sample('spectrum', likelihood, obs=flux)


# Plot the model
model_args = (low_edge, high_edge, flux, err, centers)
numpyro.render_model(
    model,
    model_args=model_args,
    render_distributions=True,
    # render_params=True,
    filename='model.pdf',
)

######################################################################
# Parameter Inference
######################################################################

# Infer Parameters
init_strategy = infer.init_to_value(
    values={
        'θ': true_θ,
        'R0': true_R0,
        'fluxes': fluxes,
        'centers': centers,
    }
)
kernel = infer.NUTS(model, init_strategy=init_strategy)

# Initialize MCMC sampler
mcmc = infer.MCMC(
    kernel,
    num_warmup=100,
    num_samples=500,
    num_chains=1,
)

# Run the sampler
mcmc.run(rng_mcmc, *model_args)

# Get the samples and log likelihood
samples = mcmc.get_samples()
logL = infer.util.log_likelihood(model, samples, *model_args)['spectrum'].sum(1)

######################################################################
# Plot the results
######################################################################


# Get flux of a model
def model_flux(i, samples=samples):
    # Compute the total profile
    return integrate(
        low_edge,
        high_edge,
        samples['centers'][i],
        samples['fluxes'][i],
        width_eff(
            samples['centers'][i],
            samples['line_width'][i],
            lsf(
                samples['centers'][i],
                samples['λ0'][i],
                samples['θ'][i],
                samples['R0'][i],
            ),
        ),
    ).sum(axis=1)


# Plot the models
fig = pyplot.figure(layout='constrained', figsize=(20, 15))
axes = fig.subplot_mosaic(
    [
        ['spec', 'spec'],
        ['curve', 'hist'],
    ],
)

# Plot the spectrum
axes['spec'].plot(wave, flux, color='gray', ds='steps-mid')
axes['spec'].errorbar(wave, flux, color='gray', yerr=err, fmt='none')
axes['spec'].set(xlabel='Wavelength', ylabel='Flux')
for i in range(N):
    axes['spec'].plot(wave, model_flux(i), alpha=0.05, ds='steps-mid', color='b')


# Plot the resolution curve
axes['curve'].plot(wave, R(wave, pivot, true_θ, true_R0), color='k')
for i in range(N):
    axes['curve'].plot(
        wave,
        R(wave, samples['λ0'][i], samples['θ'][i], samples['R0'][i]),
        alpha=0.05,
        color='b',
    )
axes['curve'].set(xlabel='Wavelength', ylabel='Resolution')

# 2D histogram of the resolution curve parameters
# axes[2].hist2d(samples['θ'], samples['R0'], bins=50, cmap='Blues')
axes['hist'].scatter(
    samples['θ'], samples['R0'], color='b', alpha=0.1, edgecolor='none'
)
# Plot best model
argbest = jnp.argmax(logL)
axes['hist'].scatter(
    samples['θ'][argbest], samples['R0'][argbest], color='b', marker='x'
)
axes['hist'].scatter(true_θ, true_R0, color='k', marker='x')
axes['hist'].set(xlabel='Theta', ylabel='Offset')

# Save the figure
fig.savefig('results.pdf')
pyplot.close(fig)
