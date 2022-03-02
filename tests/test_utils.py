# -*- coding: utf-8 -*-

import numpy as np
import pytest

import jax.numpy as jnp
from jax.config import config

from caustics import (
    min_zero_avoiding,
    max_zero_avoiding,
    ang_dist,
    ang_dist_diff,
    add_angles,
    sub_angles,
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


def test_ang_dist():
    t1 = jnp.deg2rad(-170.0)
    t2 = jnp.deg2rad(170.0)
    np.testing.assert_allclose(ang_dist(t1, t2), jnp.deg2rad(20.0))


def test_ang_dist_diff():
    theta = jnp.deg2rad(jnp.array([-5.0, -2.0, 2.0]))
    np.testing.assert_allclose(
        ang_dist_diff(theta), jnp.deg2rad(jnp.array([3.0, 4.0, 7.0]))
    )


def test_add_angles():
    t1 = jnp.deg2rad(170.0)
    t2 = jnp.deg2rad(20.0)
    np.testing.assert_allclose(add_angles(t1, t2), jnp.deg2rad(-170.0))


def test_sub_angles():
    t1 = jnp.deg2rad(-170.0)
    t2 = jnp.deg2rad(30.0)
    np.testing.assert_allclose(sub_angles(t1, t2), jnp.deg2rad(160.0))
