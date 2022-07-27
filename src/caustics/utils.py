# -*- coding: utf-8 -*-
__all__ = [
    "first_nonzero",
    "last_nonzero",
    "first_zero",
    "min_zero_avoiding",
    "max_zero_avoiding",
    "mean_zero_avoiding",
    "sparse_argsort",
    "central_finite_difference",
]
from functools import partial
import jax.numpy as jnp
from jax import jit, lax


@partial(jit, static_argnames=("axis"))
def first_nonzero(x, axis=0):
    return jnp.argmax(x != 0.0, axis=axis)


@partial(jit, static_argnames=("axis"))
def last_nonzero(x, axis=0):
    return (x.shape[axis] - 1) - jnp.argmax(jnp.flip(x, axis=axis) != 0, axis=axis)


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
def central_finite_difference(x, y):
    """
    Evaluate the first derivative using central finite difference with
    a non-uniform grid. This function is equivalent to np.gradient(y, x).
    """
    x0, x1, x2 = x[:-2], x[1:-1], x[2:]
    y0, y1, y2 = y[:-2], y[1:-1], y[2:]
    f = (x2 - x1) / (x2 - x0)
    fp = (1 - f) * (y2 - y1) / (x2 - x1) + f * (y1 - y0) / (x1 - x0)
    fp0 = (y[1] - y[0]) / (x[1] - x[0])
    fpn = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return jnp.append(fp0, jnp.append(fp, fpn))


@partial(jit, static_argnames=("axis"))
def trapz_zero_avoiding(y, x, tail_idcs, axis=0):
    """Same as jnp.trapz but ignoring all the zeros after tidx."""
    I = jnp.trapz(y, x=x, axis=axis)

    xt = jnp.take_along_axis(x, tail_idcs[:, None], axis=axis)
    yt = jnp.take_along_axis(y, tail_idcs[:, None], axis=axis)
    xtp1 = jnp.take_along_axis(x, tail_idcs[:, None] + 1, axis=axis)
    ytp1 = jnp.take_along_axis(y, tail_idcs[:, None] + 1, axis=axis)

    return lax.cond(
        jnp.all(x.shape[axis] - 1 == tail_idcs),
        lambda: I,
        lambda: I - 0.5 * ((yt + ytp1) * (xtp1 - xt)).squeeze(),
    )
