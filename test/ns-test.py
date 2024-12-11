#! /usr/bin/env python

# JAX packages
from jax import random
from jax import numpy as jnp

# Bayesian Inference
import numpyro
from numpyro import distributions as dist, sample
from numpyro.contrib.nested_sampling import NestedSampler

# Random key
rng_key = random.PRNGKey(0)

# Define data
x, err = 0.0, 1 / jnp.sqrt(2 * jnp.pi)


def model(x, err, π):
    center = sample('center', dist.Uniform(-π, π))

    # Delta function likelihood
    sample('obs', dist.Normal(center, err), obs=x)


# Render the model
model_args = (x, err, 5)
numpyro.render_model(
    model,
    model_args=model_args,
    render_distributions=True,
    filename='model.pdf',
)

# # Nested Sampling
constructor_kwargs = {'num_live_points': 500, 'max_samples': 50000}
termination_kwargs = {'dlogZ': 0.01}
NS = NestedSampler(
    model=model,
    constructor_kwargs=constructor_kwargs,
    termination_kwargs=termination_kwargs,
)
NS.run(rng_key, *model_args)
NS.print_summary()

