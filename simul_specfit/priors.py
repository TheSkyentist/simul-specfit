"""
Module for defining the priors for the model parameters
"""

# JAX packages
from jax import numpy as jnp

# Bayesian Inference
from numpyro import distributions as dist


# Uniform prior for the redshift
def redshift_prior(
    z0: float, δz: float = 0.005, line_type: str = 'narrow'
) -> dist.Distribution:
    """
    Return a uniform prior for the redshift z0 with a width of δz

    Parameters
    ----------
    z0 : float
        Initial redshift value
    δz : float, optional
        (Half) Width of the prior
    line_type : str, optional
        Type of line to be used, narrow is assumed by default
        Currently does not affect the redshift prior

    Return
    ------
    dist.Distribution
        Prior distribution for the redshift
    """

    return dist.Uniform(low=z0 - δz, high=z0 + δz)


def sigma_prior(lineType: str = 'narrow') -> dist.Distribution:
    """
    Return a uniform prior for the velocity dispersion σ

    Parameters
    ----------
    linetype : str
        Type of line to be used, narrow is assumed by default

    Return
    ------
    dist.Distribution
        Prior distribution for the velocity dispersion σ
    """

    if lineType == 'narrow':
        return dist.Uniform(low=20.0, high=1000.0)
    elif lineType == 'broad':
        return dist.Uniform(low=700.0, high=5000.0)


def flux_prior(flux_guess: float) -> dist.Distribution:
    """
    Return a uniform prior for the flux of the line up to 2 times the initial guess

    Parameters
    ----------
    flux_guess : float
        Initial guess for the flux of the line

    Return
    ------
    dist.Distribution
        Prior distribution for the flux of the line
    """

    return dist.Uniform(low=-1.5 * flux_guess, high=1.5 * flux_guess)


def angle_prior() -> dist.Distribution:
    """
    Return a uniform prior for the angle of the angle of the continuum

    Parameters
    ----------
    None

    Return
    ------
    dist.Distribution
        Prior distribution for the angle of the continuum
    """

    return dist.Uniform(low=-jnp.pi / 2, high=jnp.pi / 2)


def height_prior(height_guess: float) -> dist.Distribution:
    """
    Return a uniform prior for the height of the continuum

    Parameters
    ----------
    intercept_guess : float
        Initial guess for the height of the continuum

    Return
    ------
    dist.Distribution
        Prior distribution for the height of the continuum
    """

    low = jnp.where(height_guess < 0, 2 * height_guess, -height_guess)
    high = jnp.where(height_guess < 0, -2 * height_guess, 2 * height_guess)
    return dist.Uniform(low=low, high=high)


def lsf_scale_prior(
    mean: float = 1.2, sig: float = 0.1, cutoff: float = 3.0
) -> dist.Distribution:
    """
    Return a truncated normal prior for the lsf scale
    Centered on 1.2 with a standard deviation of 0.1, but truncated at 3σ

    Parameters
    ----------
    mean : float, optional
        Mean of the prior
    sig : float, optional
        Standard deviation of the prior
    cutoff : float, optional
        Sigma cutoff for the prior


    Return
    ------
    dist.Distribution
        Prior distribution for the lsf scale
    """

    return dist.TruncatedNormal(
        loc=mean, scale=sig, low=mean - cutoff * sig, high=mean + cutoff * sig
    )


def pixel_offset_prior(mean: float = 0.2, half_width: float = 0.5) -> dist.Distribution:
    """
    Return a uniform prior for the pixel offset

    Parameters
    ----------
    mean : float, optional
        Mean of the prior
    half_width : float, optional
        Half width of the prior

    Return
    ------
    dist.Distribution
        Prior distribution for the pixel offset
    """

    return dist.Uniform(low=mean - half_width, high=mean + half_width)


def flux_scale_prior(mean=1.1, sig=0.2, cutoff=3.0) -> dist.Distribution:
    """
    Return a truncated normal prior for the flux scale

    Parameters
    ----------
    mean : float, optional
        Mean of the prior
    sig : float, optional
        Standard deviation of the prior
    cutoff : float, optional
        Sigma cutoff for the prior

    Return
    ------
    dist.Distribution
        Prior distribution for the flux scale
    """

    return dist.TruncatedNormal(
        loc=mean, scale=sig, low=mean - cutoff * sig, high=mean + cutoff * sig
    )
