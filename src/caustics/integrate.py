# -*- coding: utf-8 -*-
"""
Functions for numerically integrating Green's integrals.
"""

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, checkpoint

from scipy.special import roots_legendre

from .utils import *

from .point_source_magnification import (
    lens_eq,
)


@checkpoint
@jit
def _integrate_unif(
    rho,
    contours,
    parity,
    tail_idcs,
):
    # Select k and (k + 1)th elements
    contours_k = contours
    contours_kp1 = jnp.pad(contours[:, 1:], ((0, 0), (0, 1)))
    contours_k = vmap(lambda idx, contour: contour.at[idx].set(0.0))(
        tail_idcs, contours_k
    )

    # Compute the integral using the trapezoidal rule
    z1_k, z2_k = jnp.real(contours_k), jnp.imag(contours_k)
    z1_kp1, z2_kp1 = jnp.real(contours_kp1), jnp.imag(contours_kp1)
    delta_z1, delta_z2 = z1_kp1 - z1_k, z2_kp1 - z2_k

    mag = jnp.sum(z1_k * delta_z2 - z2_k * delta_z1, axis=1) / (2 * np.pi * rho**2)

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
    w_center, rho, contours, parity, tail_idcs, u1=0.0, nlenses=2, npts=50, **params
):
    # Compute the Legendre roots and weights for use in Gaussian quadrature
    x_gl, w_gl = roots_legendre(npts)
    x_gl, w_gl = jnp.array(x_gl), jnp.array(w_gl)

    def P(_, y0, xl, yl):
        # Construct grid in z2 and evaluate the brightness profile at each point
        a, b = y0 * jnp.ones_like(xl), yl  # lower and upper limits

        # Rescale domain for Gauss-Legendre quadrature
        A = 0.5 * (b - a)
        y_eval = 0.5 * (b - a) * x_gl[:, None] + 0.5 * (b + a)

        # Integrate
        f_eval = _brightness_profile(
            xl + 1j * y_eval, rho, w_center, u1=u1, nlenses=nlenses, **params
        )
        I = jnp.sum(A * w_gl[:, None] * f_eval, axis=0)
        return -0.5 * I

    def Q(x0, _, xl, yl):
        # Construct grid in z1 and evaluate the brightness profile at each point
        a, b = x0 * jnp.ones_like(yl), xl

        # Rescale domain for Gauss-Legendre quadrature
        A = 0.5 * (b - a)
        x_eval = 0.5 * (b - a) * x_gl[:, None] + 0.5 * (b + a)

        # Integrate
        f_eval = _brightness_profile(
            x_eval + 1j * yl, rho, w_center, u1=u1, nlenses=nlenses, **params
        )
        I = jnp.sum(A * w_gl[:, None] * f_eval, axis=0)
        return 0.5 * I

    # We choose the centroid of each contour to be lower limit for the P and Q
    # integrals
    z0 = vmap(lambda idx, contour: contour.sum() / (idx + 1))(tail_idcs, contours)
    z10, z20 = jnp.real(z0), jnp.imag(z0)

    # Select k and (k + 1)th elements
    contours_k = contours
    contours_k = vmap(lambda idx, contour: contour.at[idx].set(0.0))(
        tail_idcs, contours_k
    )  # set last element to zero
    contours_kp1 = jnp.pad(contours[:, 1:], ((0, 0), (0, 1)))

    z1_k, z2_k = jnp.real(contours_k), jnp.imag(contours_k)
    z1_kp1, z2_kp1 = jnp.real(contours_kp1), jnp.imag(contours_kp1)
    delta_z1, delta_z2 = z1_kp1 - z1_k, z2_kp1 - z2_k
    z1_mid, z2_mid = 0.5 * (z1_k + z1_kp1), 0.5 * (z2_k + z2_kp1)

    # Evaluate the P and Q integrals using Midpoint rule
    P_mid = vmap(P)(z10, z20, z1_mid, z2_mid)
    Q_mid = vmap(Q)(z10, z20, z1_mid, z2_mid)

    # Compute the final integral using the trapezoidal rule
    mag = jnp.sum(P_mid * delta_z1 + Q_mid * delta_z2, axis=1) / (np.pi * rho**2)

    # sum magnifications for each image, taking into account the parity of each
    # image
    return jnp.abs(jnp.sum(mag * parity))
