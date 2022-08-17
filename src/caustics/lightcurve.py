# -*- coding: utf-8 -*-
"""
Function for computing the magnification of an extended source at an arbitrary
set of points in the source plane.
"""
__all__ = [
    "mag",
]

from functools import partial

import jax.numpy as jnp
from jax import jit, lax 

from . import (
    images_point_source,
    mag_extended_source,
)

from caustics.multipole import (
    mag_hexadecapole,
)

from .utils import *


def _multipole_and_false_image_test(
    w, z, z_mask, rho, delta_mu_multi, nlenses=2, c_m=5e-03, c_g=7, rho_min=1e-03, delta=1e-04, **params
):
    if nlenses == 2:
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
    zbar = jnp.conjugate(z)
    zhat = jnp.conjugate(w) - f(z)

    # Derivatives
    fp_z = f_p(z)
    fpp_z = f_pp(z)
    fp_zbar = f_p(zbar)
    fp_zhat = f_p(zhat)
    fpp_zbar = f_pp(zbar)
    J = 1.0 - jnp.abs(fp_z * fp_zbar)

    # Multipole test 
    multipole_test = c_m*(delta_mu_multi) < delta 

    # False images test
    mask_inside = jnp.prod(z_mask, axis=0)  # True if there are no false images
    Jhat = 1 - fp_z*fp_zhat
    factor = jnp.abs(
        J*Jhat**2/(Jhat*fpp_zbar*fp_z - jnp.conjugate(Jhat)*fpp_z*fp_zbar*fp_zhat)
    )
    test_ghost = 0.5*(~z_mask*factor).sum(axis=0) > c_g*(rho + rho_min)
    test_ghost = jnp.logical_or(test_ghost, mask_inside)

    return test_ghost & multipole_test


def _planetary_caustic_test(w, rho, c_p=2., **params):
    e1, a = params["e1"], params["a"]
    s = 2*a
    q = e1/(1-e1)
    x_cm = (2*e1 - 1)*a
    w_pc = -1/s - x_cm
    delta_pc = 3*jnp.sqrt(q)/s
    return (w_pc - w).real**2 + (w_pc - w).imag**2 > c_p*(rho**2 + delta_pc**2)


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
    z, z_mask = images_point_source(
        w_points,
        nlenses=nlenses,
        roots_itmax=roots_itmax,
        roots_compensated=roots_compensated,
        **params
    )

    # Compute hexadecapole approximation at every point and a test where it is
    # sufficient
    mu_multi, delta_mu_multi = mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **params)
    test = _multipole_and_false_image_test(
        w_points, z, z_mask, rho, delta_mu_multi, 
        c_c=4e-03, c_g=7, rho_min=1e-03, nlenses=nlenses,  **params
    )

    if nlenses == 2:
        e1 = params['e1']
        q = e1/(1-e1)
        test = lax.cond(
            q < 0.01, 
            lambda:test & _planetary_caustic_test(w_points, rho, **params),
            lambda:test,
        )

    # Trigger the full calculation everywhere because I haven't figured out 
    # how to implement the ghost image test for nlenses > 2 yet
    elif nlenses == 3:
        test = jnp.zeros_like(w_points).astype(jnp.bool_)
    else:
        raise ValueError("nlenses must be <= 3")
    
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
        **params,
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
        [test, mu_multi, w_points],
        #        jnp.stack([mask_test, mu_approx,  w_points]).T,
    )
