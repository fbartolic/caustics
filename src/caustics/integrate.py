# -*- coding: utf-8 -*-
"""
Functions for numerically integrating Green's integrals.
"""

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

from .utils import *

from .point_source_magnification import lens_eq
from .utils import trapz_zero_avoiding


@partial(jit, static_argnums=(0,), static_argnames=("n"))
def _integrate_gauss_legendre(f, a, b, n=100):
    pts, weights = np.polynomial.legendre.leggauss(n)
    pts_rescaled = 0.5 * (b - a) * pts[:, None] + 0.5 * (b + a)
    return jnp.sum(0.5 * (b - a) * f(pts_rescaled) * weights[:, None], axis=0)


@jit
def _integrate_unif(rho, z, parity, tail_idcs):
    # Trapezoidal rule
    I1 = trapz_zero_avoiding(0.5 * z.real, z.imag, tail_idcs, axis=1)
    I2 = trapz_zero_avoiding(-0.5 * z.imag, z.real, tail_idcs, axis=1)
    mag = (I1 + I2) / (np.pi * rho**2)

    # sum magnifications for each image, taking into account the parity
    # (per image)
    return jnp.abs(jnp.sum(mag * parity))


@partial(jit, static_argnames=("nlenses"))
def _brightness_profile(z, rho, w_center, u1=0.0, nlenses=2, **params):
    w = lens_eq(z, nlenses=nlenses, **params)
    r = jnp.abs(w - w_center) / rho

    def safe_for_grad_sqrt(x):
        return jnp.sqrt(jnp.where(x > 0.0, x, 0.0))

    # See Dominik 1998 for details
    B_r = jnp.where(
        r <= 1.0,
        1 + safe_for_grad_sqrt(1 - r**2),
        1 - safe_for_grad_sqrt(1 - 1.0 / r**2),
    )

    I = 3.0 / (3.0 - u1) * (u1 * B_r + 1.0 - 2.0 * u1)
    return I


@partial(jit, static_argnames=("nlenses", "npts"))
def _integrate_ld(
    w_center, rho, z, parity, tail_idcs, u1=0.0, nlenses=2, npts=100, **params
):
    def P(_, y0, xl, yl):
        # Construct grid in z2 and evaluate the brightness profile at each point
        a, b = y0 * jnp.ones_like(xl), yl  # lower and upper limits
        abs_delta = jnp.abs(b - a)

        # Split each integral into two intervals with the same number of points
        # and integrate each using Gauss Legendre quadrature
        mask = b > a
        split_points = jnp.where(
            mask,
            b - 2 * rho,
            b + 2 * rho,
        )
        split_points = jnp.where(
            0.5 * abs_delta <= 2 * rho, a + 0.5 * abs_delta, split_points
        )

        # First interval
        f = lambda y: _brightness_profile(
            xl + 1j * y, rho, w_center, u1=u1, nlenses=nlenses, **params
        )
        npts1 = int(npts / 2)
        npts2 = npts - npts1
        I1 = _integrate_gauss_legendre(f, a, split_points, n=npts1)

        # Second interval
        I2 = _integrate_gauss_legendre(f, split_points, b, n=npts2)
        I = I1 + I2

        return -0.5 * I

    def Q(x0, _, xl, yl):
        # Construct grid in z1 and evaluate the brightness profile at each point
        a, b = x0 * jnp.ones_like(yl), xl
        abs_delta = jnp.abs(b - a)

        # Split each integral into two intervals with the same number of points
        # and integrate each using Gauss Legendre quadrature
        mask = b > a
        split_points = jnp.where(
            mask,
            b - 2 * rho,
            b + 2 * rho,
        )
        split_points = jnp.where(
            0.5 * abs_delta <= 2 * rho, a + 0.5 * abs_delta, split_points
        )

        # First interval
        f = lambda x: _brightness_profile(
            x + 1j * yl, rho, w_center, u1=u1, nlenses=nlenses, **params
        )
        npts1 = int(npts / 2)
        npts2 = npts - npts1
        I1 = _integrate_gauss_legendre(f, a, split_points, n=npts1)

        # Second interval
        I2 = _integrate_gauss_legendre(f, split_points, b, n=npts2)
        I = I1 + I2

        return 0.5 * I

    # We choose the centroid of each contour to be lower limit for the P and Q
    # integrals
    z0 = vmap(lambda idx, z: z.sum() / (idx + 1))(tail_idcs, z)

    # Evaluate the P and Q integrals using Trapezoidal rule
    _P = vmap(P)(z0.real, z0.imag, z.real, z.imag)
    _Q = vmap(Q)(z0.real, z0.imag, z.real, z.imag)

    I1 = trapz_zero_avoiding(_P, z.real, tail_idcs, axis=1)
    I2 = trapz_zero_avoiding(_Q, z.imag, tail_idcs, axis=1)
    mag = (I1 + I2) / (np.pi * rho**2)

    # sum magnifications for each image, taking into account the parity of each
    # image
    return jnp.abs(jnp.sum(mag * parity))
