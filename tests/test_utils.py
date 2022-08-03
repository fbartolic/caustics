# -*- coding: utf-8 -*-

import numpy as np

import jax.numpy as jnp
from jax.config import config

from caustics.utils import (
    min_zero_avoiding,
    max_zero_avoiding,
    trapz_zero_avoiding,
)

config.update("jax_platform_name", "cpu")
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
    x = jnp.linspace(0, 1, 13)
    y = jnp.cos(x)
    I = jnp.trapz(y, x)

    x = jnp.append(x, jnp.zeros(7))
    y = jnp.append(y, jnp.zeros(7))
    tidx = 12
    res = trapz_zero_avoiding(y, x, tidx)

    np.testing.assert_array_equal(res, I)
