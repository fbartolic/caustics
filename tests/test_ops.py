# -*- coding: utf-8 -*-

import numpy as np
import pytest

from jax import jit, vmap
import jax.numpy as jnp
from jax.config import config
from jax.test_util import check_grads
from functools import partial

from caustics import poly_roots

config.update("jax_enable_x64", True)


@partial(jit, static_argnames=("N",))
def compute_polynomial_coeffs(w, a, e1, N=2):
    wbar = jnp.conjugate(w)

    p_0 = -(a ** 2) + wbar ** 2
    p_1 = a ** 2 * w - 2 * a * e1 + a - w * wbar ** 2 + wbar
    p_2 = (
        2 * a ** 4
        - 2 * a ** 2 * wbar ** 2
        + 4 * a * wbar * e1
        - 2 * a * wbar
        - 2 * w * wbar
    )
    p_3 = (
        -2 * a ** 4 * w
        + 4 * a ** 3 * e1
        - 2 * a ** 3
        + 2 * a ** 2 * w * wbar ** 2
        - 4 * a * w * wbar * e1
        + 2 * a * w * wbar
        + 2 * a * e1
        - a
        - w
    )
    p_4 = (
        -(a ** 6)
        + a ** 4 * wbar ** 2
        - 4 * a ** 3 * wbar * e1
        + 2 * a ** 3 * wbar
        + 2 * a ** 2 * w * wbar
        + 4 * a ** 2 * e1 ** 2
        - 4 * a ** 2 * e1
        + 2 * a ** 2
        - 4 * a * w * e1
        + 2 * a * w
    )
    p_5 = (
        a ** 6 * w
        - 2 * a ** 5 * e1
        + a ** 5
        - a ** 4 * w * wbar ** 2
        - a ** 4 * wbar
        + 4 * a ** 3 * w * wbar * e1
        - 2 * a ** 3 * w * wbar
        + 2 * a ** 3 * e1
        - a ** 3
        - 4 * a ** 2 * w * e1 ** 2
        + 4 * a ** 2 * w * e1
        - a ** 2 * w
    )

    p = jnp.array([p_0, p_1, p_2, p_3, p_4, p_5])

    return p


@pytest.fixture()
def get_coeffs():
    a = 0.5 * 0.9
    e1 = 0.8
    ncoeffs = 6

    # Compute complex polynomial coefficients for each source position
    w_points = jnp.linspace(0.3, 0.35, 100).astype(jnp.complex128)
    coeffs = vmap(vmap(lambda w: compute_polynomial_coeffs(w, a, e1)))(
        w_points[:, None]
    ).reshape(-1, 6)
    return coeffs.reshape((20, 5, 6))


@pytest.mark.parametrize("comp", [False, True])
def test_poly_roots(get_coeffs, comp):
    coeffs = get_coeffs
    roots = poly_roots(coeffs, compensated=comp)
    poly_eval = vmap(lambda z: jnp.polyval(coeffs[13, 2], z))(roots[13, 2])
    np.testing.assert_allclose(poly_eval, np.zeros_like(poly_eval), atol=1e-10)


def test_poly_roots_jit(get_coeffs):
    coeffs = get_coeffs
    roots = jit(poly_roots)(coeffs)
    poly_eval = vmap(lambda z: jnp.polyval(coeffs[13, 2], z))(roots[13, 2])
    np.testing.assert_allclose(poly_eval, np.zeros_like(poly_eval), atol=1e-10)


def test_poly_roots_vmap(get_coeffs):
    coeffs = get_coeffs
    roots = poly_roots(coeffs)

    # Batch over the first dimension
    res = vmap(poly_roots)(coeffs)
    poly_eval = vmap(lambda r: jnp.polyval(coeffs[14, 3], r))(res[14, 3])
    np.testing.assert_allclose(poly_eval, np.zeros_like(poly_eval), atol=1e-10)

    # Batch over the second dimension
    res = vmap(poly_roots, in_axes=1)(coeffs)
    poly_eval = vmap(lambda r: jnp.polyval(coeffs[6, 2], r))(res[2, 6])
    np.testing.assert_allclose(poly_eval, np.zeros_like(poly_eval), atol=1e-10)


def test_poly_roots_grad(get_coeffs):
    coeffs = get_coeffs
    roots = poly_roots(coeffs)

    fn = lambda c: poly_roots(c).sum()

    check_grads(fn, (coeffs,), 2, eps=1e-06, atol=1e-04, rtol=1e-04)
