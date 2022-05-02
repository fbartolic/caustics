# -*- coding: utf-8 -*-
__all__ = [
    "index_update",
    "first_nonzero",
    "last_nonzero",
    "first_zero",
    "min_zero_avoiding",
    "max_zero_avoiding",
    "mean_zero_avoiding",
    "sparse_argsort",
    "simpson_quadrature"
]

import jax.numpy as jnp
from jax import jit


@jit
def first_nonzero(x):
    return (x != 0.).argmax(axis=0)

@jit
def last_nonzero(x):
    return -(x[::-1] != 0.).argmax(axis=0) + len(x) - 1

@jit
def first_zero(x):
    return (x == 0.).argmax(axis=0)

@jit
def index_update(X, idx, x):
    X = X.at[idx].set(jnp.zeros_like(X[0]))
    Y = jnp.zeros_like(X)
    Y = Y.at[idx].set(x)
    return X + Y

@jit
def min_zero_avoiding(x):
    """
    Return the minimum  of a 1D array, avoiding 0.
    """
    x = jnp.sort(x)
    min_x = jnp.min(x)
    cond = min_x == 0.0
    return jnp.where(cond, x[(x != 0).argmax(axis=0)], min_x)


@jit
def max_zero_avoiding(x):
    """
    Return the maximum  of a 1D array, avoiding 0.
    """
    x = jnp.sort(x)
    max_x = jnp.max(x)
    cond = max_x == 0.0
    return jnp.where(cond, -min_zero_avoiding(jnp.abs(x)), max_x)

@jit
def mean_zero_avoiding(x):
    mask = x == 0.
    return jnp.where(
        jnp.all(mask),
        0.,
        jnp.nanmean(jnp.where(mask, jnp.nan, x))
    )

@jit
def sparse_argsort(a):
    return jnp.where(a != 0, a, jnp.nan).argsort()

@jit
def simpson_quadrature(x, y):
    """
    Compute an integral using Simpson's 1/3 rule assuming that x is uniformly
    distributed. `len(x)` has to be an odd number.
    """
    # len(x) must be odd
    h = (x[-1] - x[0]) / (len(x) - 1)
    return h/3.*jnp.sum(y[0:-1:2] + 4*y[1::2] + y[2::2], axis=0)