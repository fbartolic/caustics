# -*- coding: utf-8 -*-
__all__ = [
    "match_points",
    "first_nonzero",
    "last_nonzero",
    "first_zero",
    "min_zero_avoiding",
    "max_zero_avoiding",
    "mean_zero_avoiding",
    "sparse_argsort",
]
import jax.numpy as jnp
from jax import lax

def match_points(a, b):
    """
    Iterate over elements of a in order and find the index of the closest 
    element in b in distance. Return indices which permute b.
    
    This algorithm is not guaranteed to find a permutation of b which minimizes 
    the sum of elementwise distances between every element of a and b 
    (linear sum assignment problem).

    Args:
        a (array_like): 1D array of complex numbers.
        b (array_like): 1D array of complex numbers.

    Returns:
        array_like: Indices which specify a permutation of b.
    """
    # First guess
    vals = jnp.argsort(jnp.abs(b - a[:, None]), axis=1)
    idcs = []
    for i, idx in enumerate(vals[:, 0]):
        # If index is duplicate choose the next best solution
        mask = ~jnp.isin(vals[i], jnp.array(idcs), assume_unique=True)
        idx = vals[i, first_nonzero(mask)]
        idcs.append(idx)

    return jnp.array(idcs)


def first_nonzero(x, axis=0):
    return jnp.argmax(x != 0.0, axis=axis)


def last_nonzero(x, axis=0):
    return lax.cond(
        jnp.any(x, axis=axis),  # if any non-zero
        lambda: (x.shape[axis] - 1)
        - jnp.argmax(jnp.flip(x, axis=axis) != 0, axis=axis),
        lambda: 0,
    )


def first_zero(x, axis=0):
    return jnp.argmax(x == 0.0, axis=axis)


def min_zero_avoiding(x):
    """
    Return the minimum  of a 1D array, avoiding 0.
    """
    x = jnp.sort(x)
    min_x = jnp.min(x)
    cond = min_x == 0.0
    return jnp.where(cond, x[(x != 0).argmax(axis=0)], min_x)


def max_zero_avoiding(x):
    """
    Return the maximum  of a 1D array, avoiding 0.
    """
    x = jnp.sort(x)
    max_x = jnp.max(x)
    cond = max_x == 0.0
    return jnp.where(cond, -min_zero_avoiding(jnp.abs(x)), max_x)


def mean_zero_avoiding(x):
    mask = x == 0.0
    return jnp.where(jnp.all(mask), 0.0, jnp.nanmean(jnp.where(mask, jnp.nan, x)))


def sparse_argsort(a):
    return jnp.where(a != 0, a, jnp.nan).argsort()


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

