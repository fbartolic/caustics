# -*- coding: utf-8 -*-
"""
Computing the magnification of an extended source at an arbitrary
set of points in the source plane.
"""
__all__ = [
    "mag",
]

from functools import partial

import jax.numpy as jnp
from jax import jit, lax 

from . import mag_extended_source
from .point_source import _images_point_source

from caustics.multipole import _mag_hexadecapole

from .utils import *


@partial(jit, static_argnames=("nlenses"))
def _caustics_proximity_test(
    w, z, z_mask, rho, delta_mu_multi, nlenses=2, c_m=1e-02, gamma=0.02, c_f=4., rho_min=1e-03, **params
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

    # Multipole test and cusp test
    mu_cusp = 6 * jnp.imag(3 * fp_zbar**3.0 * fpp_z**2.0) / J**5 * (rho + rho_min)**2
    mu_cusp = jnp.sum(jnp.abs(mu_cusp) * z_mask, axis=0)
    test_multipole_and_cusp = gamma*mu_cusp + delta_mu_multi < c_m

    # False images test
    Jhat = 1 - jnp.abs(fp_z*fp_zhat)
    factor = jnp.abs(
        J*Jhat**2/(Jhat*fpp_zbar*fp_z - jnp.conjugate(Jhat)*fpp_z*fp_zbar*fp_zhat)
    )
    test_false_images = 0.5*(~z_mask*factor).sum(axis=0) > c_f*(rho + rho_min)
    test_false_images = jnp.where(
        (~z_mask).sum(axis=0)==0, 
        jnp.ones_like(test_false_images, dtype=jnp.bool_), 
        test_false_images
    )

    return test_false_images & test_multipole_and_cusp


def _planetary_caustic_test(w, rho, c_p=2., **params):
    e1, a = params["e1"], params["a"]
    s = 2*a
    q = e1/(1-e1)
    x_cm = (2*e1 - 1)*a
    w_pc = -1/s 
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
    Compute the extended source magnification for a system with `nlenses` lenses 
    and a source star radius `rho` at a set of complex points `w_points` in the 
    source plane. This function calls either [`caustics.mag_hexadecapole`][] or 
    [`caustics.mag_extended_source`][] at each point in `w_points` depending on
    whether or not the hexadecapole approximation is accurate enough at that point. 

    If `nlenses` is 2 (binary lens) or 3 (triple lens), the coordinate system is
    set up such that the the origin is at the center of mass of the first two 
    lenses which are both located on the real line. The location of the first 
    lens is $-sq/(1 + q)$ and the second lens is at $s/(1 + q)$. The optional 
    third lens is located at an arbitrary position in the complex plane 
    $r_3e^{-i\psi}$. The magnification is computed using contour integration in
    the image plane. Boolean flag `limb_darkening` indicates whether linear 
    limb-darkening needs to taken into account. If `limb_darkening` is set to 
    `True` the linear limb-darkening coefficient `u1` needs to be specified as 
    well. Note that turning on this flag slows down the computation by up to an 
    order of magnitude.

    If `nlenses` is 2 only the parameters `s` and `q` should be specified. If 
    `nlenses` is 3, the parameters `s`, `q`, `q3`, `r3` and `psi` should be 
    specified.

    !!! note

        Turning on limb-darkening (`limb_darkening=True`) slows down the 
        computation by up to an order of magnitude.
    
    !!! warning

        At the moment the test determining whether or not to use the hexadecapole
        approximation does not work for triple lenses so the function will use
        full contour integration at every point. This substantially slows down
        the computation. See https://github.com/fbartolic/caustics/issues/19.

    Args:
        w_points (array_like): Source positions in the complex plane.
        rho (float): Source radius in Einstein radii.
        nlenses (int): Number of lenses in the system.
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
            function is evaluated when computing contour integrals 
            $\int P(z_1^\prime, z_2) dz_1^\prime$ and 
            $\int Q(z_1, z_2^\prime) dz_2^\prime$ (see Dominik 1998). Defaults 
            to 100.
         s (float): Separation between the two lenses. The first lens is located 
            at $-sq/(1 + q)$ and the second lens is at $s/(1 + q)$ on the real line.
        q (float): Mass ratio defined as $m_2/m_1$.
        q3 (float): Mass ratio defined as $m_3/m_1$.
        r3 (float): Magnitude of the complex position of the third lens.
        psi (float): Phase angle of the complex position of the third lens.
        roots_itmax (int, optional): Number of iterations for the root solver.
        roots_compensated (bool, optional): Whether to use the compensated
            arithmetic version of the Ehrlich-Aberth root solver.

    Returns:
        array_like: Magnification array.
    """
    if nlenses == 1:
        _params = {}
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5*s
        e1 = 1/(1 + q)
        _params = {"a": a, "e1": e1}
        x_cm = a*(1 - q)/(1 + q)

    # Trigger the full calculation everywhere because I haven't figured out 
    # how to implement the ghost image test for nlenses > 2 yet
    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a = 0.5*s
        e1 = q/(1 + q + q3)
        e2 = q*e1
        r3 = r3*jnp.exp(1j*psi)
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2}
        x_cm = a*(1 - q)/(1 + q)

    else:
        raise ValueError("nlenses must be <= 3")


    # Compute point images for a point source
    z, z_mask = _images_point_source(
        w_points + x_cm,
        nlenses=nlenses,
        roots_itmax=roots_itmax,
        roots_compensated=roots_compensated,
        **_params
    )

    if nlenses==1:
        test = w_points > 2*rho
    elif nlenses==2:
        # Compute hexadecapole approximation at every point and a test where it is
        # sufficient
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
        test1 = _caustics_proximity_test(
            w_points + x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params
        )
        test2 = _planetary_caustic_test(w_points + x_cm, rho, **_params)

        test = lax.cond(
            q < 0.01, 
            lambda:test1 & test2,
            lambda:test1,
        )
    elif nlenses == 3:
        test = jnp.zeros_like(w_points).astype(jnp.bool_)
    
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
