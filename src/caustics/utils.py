# -*- coding: utf-8 -*-
__all__ = [
    "first_nonzero",
    "last_nonzero",
    "first_zero",
    "min_zero_avoiding",
    "max_zero_avoiding",
    "mean_zero_avoiding",
    "sparse_argsort",
]
from functools import partial
import jax.numpy as jnp
from jax import jit, lax


@partial(jit, static_argnames=("axis"))
def first_nonzero(x, axis=0):
    return jnp.argmax(x != 0.0, axis=axis)


@partial(jit, static_argnames=("axis"))
def last_nonzero(x, axis=0):
    return lax.cond(
        jnp.any(x, axis=axis),  # if any non-zero
        lambda: (x.shape[axis] - 1)
        - jnp.argmax(jnp.flip(x, axis=axis) != 0, axis=axis),
        lambda: 0,
    )


@partial(jit, static_argnames=("axis"))
def first_zero(x, axis=0):
    return jnp.argmax(x == 0.0, axis=axis)


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
    mask = x == 0.0
    return jnp.where(jnp.all(mask), 0.0, jnp.nanmean(jnp.where(mask, jnp.nan, x)))


@jit
def sparse_argsort(a):
    return jnp.where(a != 0, a, jnp.nan).argsort()


@jit
def trapz_zero_avoiding(y, x, tail_idx):
    """
    Same as jnp.trapz(y[:tail_idx + 1] x=x[:tail_idx + 1], axis=0).
    """
    I = jnp.trapz(y, x=x)
    xt, yt = x[tail_idx], y[tail_idx]
    xtp1, ytp1 = x[tail_idx + 1], y[tail_idx + 1]
    return lax.cond(
        tail_idx == len(x) - 1,
        lambda: I,
        lambda: I - 0.5 * ((yt + ytp1) * (xtp1 - xt)),
    )
