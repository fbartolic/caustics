# -*- coding: utf-8 -*-
"""
Function for computing the magnification of an extended source at an arbitrary
set of points in the source plane.
"""
__all__ = [
    "mag_binary",
    "mag_triple",
]

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax

from . import (
    images_point_source_binary,
    images_point_source_triple,
    mag_extended_source_binary,
    mag_extended_source_triple,
)

from caustics.multipole import (
    mag_hexadecapole_binary,
    mag_hexadecapole_triple,
)

from .utils import *


@partial(jit, static_argnames=("nlenses"))
def _extended_source_test(
    z,
    mask_z,
    w,
    rho,
    u=0.0,
    c_hex=5e-03,
    c_ghost=0.9,
    c_cusp=1e-03,
    nlenses=2,
    **kwargs
):
    # Evaluate hexadecapole approximation
    if nlenses == 2:
        a, e1 = kwargs["a"], kwargs["e1"]
        mu_ps, mu_quad, mu_hex = mag_hexadecapole_binary(z, mask_z, a, e1, rho, u=u)

        # Derivatives
        f = lambda z: -e1 / (z - a) - (1 - e1) / (z + a)
        f_p = lambda z: e1 / (z - a) ** 2 + (1 - e1) / (z + a) ** 2
        f_pp = lambda z: 2 * (e1 / (a - z) ** 3 - (1 - e1) / (a + z) ** 3)

    else:
        a, r3, e1, e2 = kwargs["a"], kwargs["r3"], kwargs["e1"], kwargs["e2"]
        mu_ps, mu_quad, mu_hex = mag_hexadecapole_triple(
            z, mask_z, a, r3, e1, e2, rho, u=u
        )

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
    fpp_zbar = f_pp(zbar)

    fp_zhat = f_p(zhat)

    fp_zhat_bar = jnp.conjugate(fp_zhat)

    J = 1.0 - jnp.abs(fp_z * fp_zbar)

    mask_inside = jnp.prod(mask_z, axis=0)  # True if there are no false images

    # Hexadecapole and cusp test
    mu_multi = mu_ps + mu_quad + mu_hex
    hex_test = ((jnp.abs(mu_hex) + jnp.abs(mu_quad)) / mu_ps) < c_hex

    mu_cusp = 6 * jnp.imag(3 * fp_zbar ** 3.0 * fpp_z ** 2.0) / J ** 5 * rho ** 2
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
    jit, static_argnames=("npts_limb", "niter_limb", "npts_ld",),
)
def mag_binary(
    w_points, a, e1, rho, u=0.0, npts_limb=300, niter_limb=8, npts_ld=601,
):
    """
    Compute the binary lens magnification for an extended limb-darkned source 
    at a set of points `w_points` in the source plane. Depending on the proximity
    of each point to a caustic, this function will call either 
    `mag_extended_source_binary` or `mag_hexadecapole_binary`.

    Args:
        w_points (array_like): Center of the source in the lens plane.
        a (float): Half the separation between the two lenses. We use the
            convention where both lenses are located on the real line with
            r1 = a and r2 = -a.
        e1 (array_like): Mass fraction of the first lens e1 = m1/(m1+m2). It
            follows that e2 = 1 - e1.
        rho (float): Source radius in Einstein radii.
        u (float, optional): Linear limb darkening coefficient. Defaults to 0..
        npts_limb (int, optional): Initial number of points uniformly distributed
            on the source limb when computing the point source magnification.
            The final number of points depends on this value and `niter_limb`
            because additional points are added iteratively in a geometric
            fashion. This parameters determines the precision of the magnification
            calculation (absent limb-darkening, in which case `npts_ld` is also
            important). The default value should keep the relative error well
            below 10^{-3} in all cases. Defaults to 300.
        niter_limb (int, optional): Number of iterations to use for the point
            source magnification evaluation on the source limb. At each
            iteration we geometrically decrease the number of points starting
            with `npts_limb` and ending with 2 for the final iteration. The new
            points are placed where the gradient of the magnification is
            largest. Deaults to 8.
        npts_ld (int, optional): Number of points at which the stellar
            brightness function is evaluated when computing the integrals P and
            Q defined in Dominik 1998. Defaults to 601.

    Returns:
        array_like: Magnification array.
    """
    z, mask_z = images_point_source_binary(w_points, a, e1)

    mask_test, mu_approx = _extended_source_test(
        z, mask_z, w_points, rho, a, e1, **{"a": a, "e1": e1}
    )

    # Iterate over w_points and execute either the hexadecapole  approximation
    # or the full extended source calculation. `vmap` cannot be used here because
    # `lax.cond` executes both branches within vmap.
    def body_fn(_, x):
        w, c, _mu_approx = x
        mag = lax.cond(
            c,
            lambda _: _mu_approx,
            lambda w: mag_extended_source_binary(
                w,
                a,
                e1,
                rho,
                u=u,
                npts_limb=npts_limb,
                niter_limb=niter_limb,
                npts_ld=npts_ld,
            ),
            w,
        )
        return 0, mag

    return lax.scan(body_fn, 0, [w_points, mask_test, mu_approx])[1]


@partial(
    jit, static_argnames=("npts_limb", "niter_limb", "npts_ld",),
)
def mag_triple(
    w_points, a, r3, e1, e2, rho, u=0.0, npts_limb=300, niter_limb=8, npts_ld=601,
):
    """
    Compute the triple lens magnification for an extended limb-darkned source 
    at a set of points `w_points` in the source plane. Depending on the 
    proximity of each point to a caustic, this function will call either 
    `mag_extended_source_triple` or `mag_hexadecapole_triple`.

    Args:
        w_points (array_like): Center of the source in the lens plane.
        a (float): Half the separation between the two lenses. We use the
            convention where both lenses are located on the real line with
            r1 = a and r2 = -a.
        e1 (array_like): Mass fraction of the first lens e1 = m1/(m1+m2). It
            follows that e2 = 1 - e1.
        rho (float): Source radius in Einstein radii.
        u (float, optional): Linear limb darkening coefficient. Defaults to 0..
        npts_limb (int, optional): Initial number of points uniformly distributed
            on the source limb when computing the point source magnification.
            The final number of points depends on this value and `niter_limb`
            because additional points are added iteratively in a geometric
            fashion. This parameters determines the precision of the magnification
            calculation (absent limb-darkening, in which case `npts_ld` is also
            important). The default value should keep the relative error well
            below 10^{-3} in all cases. Defaults to 300.
        niter_limb (int, optional): Number of iterations to use for the point
            source magnification evaluation on the source limb. At each
            iteration we geometrically decrease the number of points starting
            with `npts_limb` and ending with 2 for the final iteration. The new
            points are placed where the gradient of the magnification is
            largest. Deaults to 8.
        npts_ld (int, optional): Number of points at which the stellar
            brightness function is evaluated when computing the integrals P and
            Q defined in Dominik 1998. Defaults to 601.

    Returns:
        array_like: Magnification array.
    """
    z, mask_z = images_point_source_triple(w_points, a, e1, e2, r3)

    mask_test, mu_approx = _extended_source_test(
        z, mask_z, w_points, rho, a, e1, **{"a": a, "e1": e1, "r3": r3, "e2": e2}
    )

    # Iterate over w_points and execute either the hexadecapole  approximation
    # or the full extended source calculation. `vmap` cannot be used here because
    # `lax.cond` executes both branches within vmap.
    def body_fn(_, x):
        w, c, _mu_approx = x
        mag = lax.cond(
            c,
            lambda _: _mu_approx,
            lambda w: mag_extended_source_triple(
                w,
                a,
                r3,
                e1,
                e2,
                rho,
                u=u,
                npts_limb=npts_limb,
                niter_limb=niter_limb,
                npts_ld=npts_ld,
            ),
            w,
        )
        return 0, mag

    return lax.scan(body_fn, 0, [w_points, mask_test, mu_approx])[1]
