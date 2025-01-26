"""
Module containing optimized routines
"""

# Typing
from typing import Final

# JAX packages
import jax
from jax.scipy.special import erf
from jax import jit, vmap, lax, numpy as jnp

# Set threshold where error function doesn't need to be computed
# erf(3.9) == 1 for 32 bit
# erf(4.2) == 1 for 64 bit
THRESHOLD: Final[float] = 4.2 if jax.config.jax_enable_x64 else 3.9

# Conversion factor from FWHM to sigma for variance = 1/2
# σ = fwhm / ( 2 * jnp.sqrt( 2 * jnp.log(2) ) )
# σ_halfvar = jnp.sqrt(2) * σ
FWHM_TO_SIGMA: Final[float] = 1 / (2 * jnp.sqrt(jnp.log(2)))

# Pseudo-Voigt Profile Magic Numbers
# From Thompson+ (1987) DOI:10.1107/S0021889887087090
MAGIC_FWHM: Final[jnp.ndarray] = jnp.array([1, 2.69268, 2.42843, 4.47163, 0.07842, 1])
MAGIC_ETA: Final[jnp.ndarray] = jnp.array([1.33603, -0.47719, 0.11116])


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
def integrateGaussian(
    low_edge: jnp.ndarray,
    high_edge: jnp.ndarray,
    centers: jnp.ndarray,
    fwhms: jnp.ndarray,
    threshold: float = 4.2,
) -> jnp.ndarray:
    """
    Integrate N emission lines over λ bins of Normal Distribution
    Return matrix of fluxes in each bin for each line

    Parameters
    ----------
    low_edge : jnp.ndarray (λ,)
        Low edge of the bins
    high_edge : jnp.ndarray (λ,)
        High edge of the bins
    centers : jnp.ndarray (N,)
        Centers of the emission lines
    fwhms : jnp.ndarray (N,)
        Fwhms at each line
    threshold : float, optional
        Threshold for the integral, defaults to 4.2
        erf(3.9) == 1 for 32 bit
        erf(4.2) == 1 for 64 bit

    Returns
    -------
    jnp.ndarray (λ, N)
        Fluxes in each bin for each line
    """

    # Transform to σ and adjust to be for variance = 1/2
    # Inverse width once for faster computation
    invfwhms = 1 / (fwhms * FWHM_TO_SIGMA)

    # Compute residual
    low_resid = (low_edge[:, jnp.newaxis] - centers) * invfwhms
    high_resid = (high_edge[:, jnp.newaxis] - centers) * invfwhms

    # Restrict to only those that won't compute to zero
    good = jnp.logical_and(-threshold < low_resid, high_resid < threshold)

    # Compute pixel integral with error function (CDF)
    pixel_ints = (erfcond(good, high_resid) - erfcond(good, low_resid)) / 2

    # Compute fluxes
    return pixel_ints


@jit
def integrateCauchy(
    low_edge: jnp.ndarray,
    high_edge: jnp.ndarray,
    centers: jnp.ndarray,
    fwhms: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integrate N emission lines over λ bins of Cauchy distribution
    Return matrix of integrals in each bin for each line

    Parameters
    ----------
    low_edge : jnp.ndarray (λ,)
        Low edge of the bins
    high_edge : jnp.ndarray (λ,)
        High edge of the bins
    centers : jnp.ndarray (N,)
        Centers of the emission lines
    fwhms : jnp.ndarray (N,)
        Fwhms at each line

    Returns
    -------
    jnp.ndarray (λ, N)
        Integral in each bin for each line
    """

    # Calculate inverse width
    invfwhms = 1 / fwhms

    # Compute residual
    low_resid = (low_edge[:, jnp.newaxis] - centers) * invfwhms
    high_resid = (high_edge[:, jnp.newaxis] - centers) * invfwhms

    # Compute Pixel integral with arctan
    pixel_ints = (jnp.arctan(high_resid) - jnp.arctan(low_resid)) / jnp.pi

    return pixel_ints


@jit
def integrateVoigt(
    low_edge: jnp.ndarray,
    high_edge: jnp.ndarray,
    centers: jnp.ndarray,
    fwhm_g: jnp.ndarray,
    fwhm_γ: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integrate N emission lines over λ bins of Voigt distribution
    Return matrix of integrals in each bin for each line
    Uses pseudo-Voigt profile from Thompson+ (1987) DOI:10.1107/S0021889887087090

    Parameters
    ----------
    low_edge : jnp.ndarray (λ,)
        Low edge of the bins
    high_edge : jnp.ndarray (λ,)
        High edge of the bins
    centers : jnp.ndarray (N,)
        Centers of the emission lines
    fwhm_g : jnp.ndarray (N,)
        Gaussian Component FWHM at each line
    fwhm_γ : jnp.ndarray (N,)
        Cauchy/Lorentzian Component FWHM at each line

    Returns
    -------
    jnp.ndarray (λ, N)
        Integral in each bin for each line
    """

    # Calculate FWHM for pseudo-Voigt components
    powers = jnp.arange(len(MAGIC_FWHM))
    fwhm = jnp.sum(MAGIC_FWHM * (fwhm_g**powers) * (fwhm_γ ** powers[::-1])) ** (1 / 5)

    # Compute contribution fraction for Pseudo-Voigt
    fwhm_ratio = fwhm_γ / fwhm
    η = jnp.sum(MAGIC_ETA * fwhm_ratio ** jnp.arange(1, len(MAGIC_ETA) + 1))

    # Compute components
    L = η * integrateCauchy(low_edge, high_edge, centers, fwhm)
    G = (1 - η) * integrateGaussian(low_edge, high_edge, centers, fwhm)

    # Compute total pixel integral
    return L + G


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
