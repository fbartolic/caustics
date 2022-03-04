# -*- coding: utf-8 -*-

import numpy as np
import pytest

from jax import jit, vmap
import jax.numpy as jnp
from jax.config import config
from jax.test_util import check_grads

from caustics.ops import poly_roots
from caustics.point_source_magnification import poly_coeffs_binary


config.update("jax_enable_x64", True)


@pytest.fixture()
def get_coeffs():
    a = 0.5 * 0.9
    e1 = 0.8

    # Compute complex polynomial coefficients for each source position
    w_points = jnp.linspace(0.3, 0.35, 10).astype(jnp.complex128)
    coeffs = poly_coeffs_binary(w_points, a, e1).reshape(-1, 6)
    return coeffs.reshape((5, 2, 6))


@pytest.mark.parametrize("comp", [False, True])
def test_poly_roots(get_coeffs, comp):
    coeffs = get_coeffs
    roots = poly_roots(coeffs, compensated=comp)
    poly_eval = vmap(lambda z: jnp.polyval(coeffs[3, 1], z))(roots[3, 1])
    np.testing.assert_allclose(poly_eval, np.zeros_like(poly_eval), atol=1e-10)


def test_poly_roots_jit(get_coeffs):
    coeffs = get_coeffs
    roots = jit(poly_roots)(coeffs)
    poly_eval = vmap(lambda z: jnp.polyval(coeffs[3, 1], z))(roots[3, 1])
    np.testing.assert_allclose(poly_eval, np.zeros_like(poly_eval), atol=1e-10)


def test_poly_roots_vmap(get_coeffs):
    coeffs = get_coeffs
    roots = poly_roots(coeffs)

    # Batch over the first dimension
    res = vmap(poly_roots)(coeffs)
    poly_eval = vmap(lambda r: jnp.polyval(coeffs[4, 0], r))(res[4, 0])
    np.testing.assert_allclose(poly_eval, np.zeros_like(poly_eval), atol=1e-10)

    # Batch over the second dimension
    res = vmap(poly_roots, in_axes=1)(coeffs)
    poly_eval = vmap(lambda r: jnp.polyval(coeffs[4, 0], r))(res[0, 4])
    np.testing.assert_allclose(poly_eval, np.zeros_like(poly_eval), atol=1e-10)


def test_poly_roots_grad(get_coeffs):
    coeffs = get_coeffs
    roots = poly_roots(coeffs)

    fn = lambda c: poly_roots(c).sum()

    check_grads(fn, (coeffs,), 2, eps=1e-06, atol=1e-04, rtol=1e-04)
