"""
Module containing optimized routines
"""

# JAX packages
from jax.scipy.special import erf
from jax import jit, vmap, lax, numpy as jnp


# Could be replaced by one-liner, but this is more readable
# erfcond = jit(vmap(vmap(lambda b, λ: lax.cond(b, erf, lambda x: 0.0, λ))))
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
        Sigma (Variance of 1/2) values

    Returns
    -------
    jnp.ndarray
        Conditional vectorized erf
    """

    return vmap(vmap(lambda b, λ: lax.cond(b, erf, lambda x: 0.0, λ)))(good, sigma)


@jit
def integrate(
    low_edge: jnp.ndarray,
    high_edge: jnp.ndarray,
    centers: jnp.ndarray,
    widths: jnp.ndarray,
    fluxes: jnp.ndarray,
    threshold: float = 4.2,
) -> jnp.ndarray:
    """
    Integrate N emission lines over λ bins
    Return matrix of fluxes in each bin for each line

    Parameters
    ----------
    low_edge : jnp.ndarray (λ,)
        Low edge of the bins
    high_edge : jnp.ndarray (λ,)
        High edge of the bins
    centers : jnp.ndarray (N,)
        Centers of the emission lines
    widths : jnp.ndarray (N,)
        Effective widths at each line
    fluxes : jnp.ndarray (N,)
        Flux in the lines
    threshold : float, optional
        Threshold for the integral, defaults to 4.2
        erf(3.9) == 1 for 32 bit
        erf(4.2) == 1 for 64 bit

    Returns
    -------
    jnp.ndarray (λ, N)
        Fluxes in each bin for each line
    """

    # Adjust width to be for 1/2 variance for erf
    # Inverse width once for faster computation
    invwidths = 1 / (jnp.sqrt(2) * widths)

    # Compute residual
    low_resid = (low_edge[:, jnp.newaxis] - centers) * invwidths
    high_resid = (high_edge[:, jnp.newaxis] - centers) * invwidths

    # Restrict to only those that won't compute to zero
    good = jnp.logical_and(-threshold < low_resid, high_resid < threshold)

    # Compute pixel integral with error function (CDF)
    pixel_ints = (erfcond(good, high_resid) - erfcond(good, low_resid)) / 2

    # Compute fluxes
    # Divide by bin width to get flux density
    return (fluxes * pixel_ints) / (high_edge - low_edge)[:, jnp.newaxis]


@jit
def linearContinua(λ, cont_centers, angles, offsets, continuum_regions):
    """
    Compute the linear model

    Parameters
    ----------
    λ : jnp.ndarray
        Wavelength values
    cont_centers : jnp.ndarray
        Centers of the continua
    angles : jnp.ndarray
        Angles of the continua
    offsets : jnp.ndarray
        Offset of the continua
    continuum_regions : jnp.ndarray
        Bounds of the continuum region

    Returns
    -------
    jnp.ndarray
        Flux values
    """

    # Evaluate the linear model
    λ = λ[:, jnp.newaxis]
    continuum = jnp.tan(angles) * (λ - cont_centers) + offsets

    return jnp.where(
        jnp.logical_and(continuum_regions[:, 0] < λ, λ < continuum_regions[:, 1]),
        continuum,
        0.0,
    )
