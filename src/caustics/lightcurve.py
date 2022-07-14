# -*- coding: utf-8 -*-
"""
Function for computing the magnification of an extended source at an arbitrary
set of points in the source plane.
"""
__all__ = [
    "mag",
]

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax, checkpoint

from . import (
    images_point_source,
    mag_extended_source,
)

from caustics.multipole import (
    mag_hexadecapole,
)

from .utils import *


@partial(jit, static_argnames=("nlenses"))
def _extended_source_test(
    z,
    mask_z,
    w,
    rho,
    u1=0.0,
    c_hex=5e-03,
    c_ghost=0.9,
    c_cusp=1e-03,
    nlenses=2,
    **params
):
    """
    Test weather hexadecapole approximation is sufficient.
    """
    # For a single lens, use the full calculation if source center is
    # at least two source radii away from the central caustic
    mu_ps, mu_quad, mu_hex = mag_hexadecapole(
        z, mask_z, rho, u1=u1, nlenses=nlenses, **params
    )
    mu_multi = mu_ps + mu_quad + mu_hex

    if nlenses == 1:
        mask_valid = w.real**2 + w.imag**2 > 4.0 * rho**2
        return mask_valid, mu_multi

    elif nlenses == 2:
        a, e1 = params["a"], params["e1"]

        # Derivatives
        f = lambda z: -e1 / (z - a) - (1 - e1) / (z + a)
        f_p = lambda z: e1 / (z - a) ** 2 + (1 - e1) / (z + a) ** 2
        f_pp = lambda z: 2 * (e1 / (a - z) ** 3 - (1 - e1) / (a + z) ** 3)

    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]

        # Derivatives
        f = lambda z: -e1 / (z - a) - e2 / (z + a) - (1 - e1 - e2) / (z + r3)
        f_p = (
            lambda z: e1 / (z - a) ** 2
            + e2 / (z + a) ** 2
            + (1 - e1 - e2) / (z + r3) ** 2
        )
        f_pp = (
            lambda z: 2 * (e1 / (a - z) ** 3 - e2 / (a + z) ** 3)
            + (1 - e1 - e2) / (z + r3) ** 3
        )

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    zbar = jnp.conjugate(z)
    zhat = jnp.conjugate(w) - f(z)

    # Derivatives
    fp_z = f_p(z)
    fpp_z = f_pp(z)

    fp_zbar = f_p(zbar)
    fpp_zbar = f_pp(zbar)

    fp_zhat = f_p(zhat)

    fp_zhat_bar = jnp.conjugate(fp_zhat)

    J = 1.0 - jnp.abs(fp_z * fp_zbar)

    mask_inside = jnp.prod(mask_z, axis=0)  # True if there are no false images

    # Hexadecapole and cusp test
    hex_test = ((jnp.abs(mu_hex) + jnp.abs(mu_quad)) / mu_ps) < c_hex

    mu_cusp = 6 * jnp.imag(3 * fp_zbar**3.0 * fpp_z**2.0) / J**5 * rho**2
    mu_cusp = jnp.sum(jnp.abs(mu_cusp) * mask_z, axis=0)
    cusp_test = jnp.logical_or(mu_cusp / mu_ps < c_cusp, mask_inside)

    # Ghost images test
    pJ_pz = 1 - jnp.abs(fpp_z * fp_zbar)
    pJ_pzbar = 1 - jnp.abs(fp_z * fpp_zbar)

    # Compute theta for which J is maximized
    phi_max = np.pi - 0.5 * 1j * jnp.log(
        pJ_pzbar / (1 - fp_zhat_bar * fp_zbar) / (pJ_pz / (1 - fp_zhat * fp_z))
    )
    phi_max = jnp.real(phi_max)

    # Evaluate delta_J at phi_max
    delta_J_max = pJ_pz * rho * jnp.exp(1j * phi_max) / (
        1 - fp_zhat * fp_z
    ) + pJ_pzbar * rho * jnp.exp(-1j * phi_max) / (1 - fp_zhat_bar * fp_zbar)

    ghost_test = jnp.abs((~mask_z * (J + delta_J_max)).sum(axis=0)) > c_ghost
    ghost_test = jnp.logical_or(ghost_test, mask_inside)

    mask_valid = hex_test & cusp_test & ghost_test

    return mask_valid, mu_multi


@partial(
    jit,
    static_argnames=(
        "nlenses",
        "npts_limb",
        "limb_darkening",
        "npts_ld",
        "roots_itmax",
        "roots_compensated",
    ),
)
def mag(
    w_points,
    rho,
    nlenses=2,
    npts_limb=200,
    limb_darkening=False,
    u1=0.0,
    npts_ld=100,
    roots_itmax=2500,
    roots_compensated=False,
    **params
):
    """
    Compute the magnification for a system with `nlenses` and an extended
    limb-darkned source at a set of complex points `w_points` in the source plane.
    This function calls either `mag_hexadecapole` or `mag_extended_source` for
    each point in `w_points` depending on whether or not the hexadecapole
    approximation is good enough.

    If `nlenses` is 2 (binary lens) or 3 (triple lens), the coordinate
    system is set such that the first two lenses with mass fractions
    `$e1=m_1/m_\mathrm{total}$` and `$e2=m_2/m_\mathrm{total}$` are positioned
    on the x-axis at locations $r_1=a$ and $r_2=-a$ respectively. The third
    lens is at an arbitrary position in the complex plane $r_3$. For a single lens
    lens the magnification is computed analytically. For binary and triple
    lenses computing the magnification involves solving for the roots of a
    complex polynomial with degree (`nlenses`**2 + 1) using the Elrich-Aberth
    algorithm. Optional keywords `itmax` and `compensated` can be passed to the
    root solver as a dictionary. `itmax` is the number of root solver iterations,
    it defaults to `2500`, and `compensated` specifies whether the root solver
    should use the compensated version of or the Elrich-Aberth algorithm or the
    regular version, it defaults to `False`.

    Args:
        w_points (array_like): Source positions in the complex plane.
        rho (float): Source radius in Einstein radii.
        npts_limb (int, optional): Initial number of points uniformly distributed
            on the source limb when computing the point source magnification.
            The final number of points is greater than this value because
            the number of points is decreased geometrically by a factor of
            1/2 until it reaches 2.
        limb_darkening (bool, optional): If True, compute the magnification of
            a limb-darkened source. If limb_darkening is enabled the u1 linear
            limb-darkening coefficient needs to be specified. Defaults to False.
        u1 (float, optional): Linear limb darkening coefficient. Defaults to 0..
        npts_ld (int, optional): Number of points at which the stellar brightness
            function is evaluated when computing the integrals P and Q from
            Dominik 1998. Defaults to 50.
        **a (float): Half the separation between the first two lenses located on
            the real line with $r_1 = a$ and $r_2 = -a$.
        **r3 (float): The position of the third lens at arbitrary location in
            the complex plane.
        **e1 (array_like): Mass fraction of the first lens located at $r_1=a$.
        **e2 (array_like): Mass fraction of the second lens located at $r_2=-a$.
        **roots_itmax (int, optional): Number of iterations for the root solver.
        **roots_compensated (bool, optional): Whether to use the compensated
            arithmetic version of the Ehrlich-Aberth root solver.

    Returns:
        array_like: Magnification array.
    """
    # Compute point images for a point source
    z, mask_z = images_point_source(
        w_points,
        nlenses=nlenses,
        roots_itmax=roots_itmax,
        roots_compensated=roots_compensated,
        **params
    )

    # Compute hexadecapole approximation at every point and a test where it is
    # sufficient
    mask_test, mu_approx = _extended_source_test(
        z, mask_z, w_points, rho, nlenses=nlenses, u1=u1, **params
    )

    mag_full = lambda w: mag_extended_source(
        w,
        rho,
        nlenses=nlenses,
        npts_limb=npts_limb,
        limb_darkening=limb_darkening,
        u1=u1,
        npts_ld=npts_ld,
        roots_itmax=roots_itmax,
        roots_compensated=roots_compensated,
    )

    # Iterate over w_points and execute either the hexadecapole  approximation
    # or the full extended source calculation. `vmap` cannot be used here because
    # `lax.cond` executes both branches within vmap.
    return lax.map(
        lambda xs: lax.cond(
            xs[0],
            lambda _: xs[1],
            mag_full,
            xs[2],
        ),
        [mask_test, mu_approx, w_points],
        #        jnp.stack([mask_test, mu_approx,  w_points]).T,
    )
