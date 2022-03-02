# -*- coding: utf-8 -*-

__all__ = [
    "min_zero_avoiding",
    "max_zero_avoiding",
    "ang_dist",
    "add_angles",
    "sub_angles",
]

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap


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
def ang_dist(theta1, theta2):
    """
    Smallest separation between two angles.
    """
    diff1 = (theta1 - theta2) % (2 * jnp.pi)
    diff2 = (theta2 - theta1) % (2 * jnp.pi)
    return jnp.min(jnp.array([diff1, diff2]), axis=0)


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
