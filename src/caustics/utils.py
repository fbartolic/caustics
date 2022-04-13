# -*- coding: utf-8 -*-
__all__ = [
    "index_update",
    "first_nonzero",
    "last_nonzero",
    "first_zero",
    "min_zero_avoiding",
    "max_zero_avoiding",
    "ang_dist",
    "ang_dist_diff",
    "add_angles",
    "sub_angles",
    "sparse_argsort",
]

import jax.numpy as jnp
from jax import jit, vmap


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
def ang_dist_diff(theta):
    """
    Angular distance between consecutive points for a 1D array. Last point of
    the output array is the distance between first and last point.
    """
    theta1 = theta
    theta2 = jnp.concatenate([theta[1:], jnp.atleast_1d(theta1[0])])
    return vmap(ang_dist)(theta1, theta2)


@jit
def add_angles(a, b):
    """a + b"""
    cos_apb = jnp.cos(a) * jnp.cos(b) - jnp.sin(a) * jnp.sin(b)
    sin_apb = jnp.sin(a) * jnp.cos(b) + jnp.cos(a) * jnp.sin(b)
    return jnp.arctan2(sin_apb, cos_apb)


@jit
def sub_angles(a, b):
    """a - b"""
    cos_amb = jnp.cos(a) * jnp.cos(b) + jnp.sin(a) * jnp.sin(b)
    sin_amb = jnp.sin(a) * jnp.cos(b) - jnp.cos(a) * jnp.sin(b)
    return jnp.arctan2(sin_amb, cos_amb)

@jit
def ang_dist(theta1, theta2):
    """
    Smallest separation between two angles.
    """
    return jnp.abs(sub_angles(theta1, theta2))

@jit
def sparse_argsort(a):
    return jnp.where(a != 0, a, jnp.nan).argsort()
