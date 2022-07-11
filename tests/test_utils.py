# -*- coding: utf-8 -*-

import numpy as np

import jax.numpy as jnp
from jax.config import config

from caustics.utils import (
    min_zero_avoiding,
    max_zero_avoiding,
    trapz_zero_avoiding,
)

config.update("jax_enable_x64", True)


def test_min_zero_avoiding():
    x = jnp.array([0.0, 0.0, 3.4, 8.3])
    assert min_zero_avoiding(x) == 3.4
    x = jnp.array([-2.0, 0.0, 3.5])
    assert min_zero_avoiding(x) == -2.0


def test_max_zero_avoiding():
    x = jnp.array([0.0, 0.0, 3.4, 8.3])
    assert max_zero_avoiding(x) == 8.3
    x = jnp.array([-2.0, 0.0, -1.2])
    assert max_zero_avoiding(x) == -1.2


def test_trapz_zero_avoiding():
    x1 = jnp.linspace(0, 1, 13)
    x2 = jnp.linspace(0, 0.3, 10)

    y1 = jnp.cos(x1)
    y2 = jnp.cos(x2)

    I1 = jnp.trapz(y1, x1)
    I2 = jnp.trapz(y2, x2)

    x1 = jnp.append(x1, jnp.zeros(7))
    x2 = jnp.append(x2, jnp.zeros(10))
    y1 = jnp.append(y1, jnp.zeros(7))
    y2 = jnp.append(y2, jnp.zeros(10))
    x = jnp.stack([x1, x2])
    y = jnp.stack([y1, y2])
    tail_idcs = jnp.array([12, 9])
    res = trapz_zero_avoiding(y, x, tail_idcs, axis=1)

    np.testing.assert_array_equal(res, jnp.array([I1, I2]))
