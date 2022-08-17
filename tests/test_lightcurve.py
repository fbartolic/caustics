# -*- coding: utf-8 -*-
import numpy as np
import pytest
from functools import partial

import jax.numpy as jnp
from jax import jit, random, vmap
from jax.config import config
from jax import random

from caustics import (
    critical_and_caustic_curves,
    images_point_source,
)
from caustics.multipole import mag_hexadecapole
from caustics.lightcurve import _multipole_cusp_and_ghost_tests, _planetary_caustic_test

config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)

import MulensModel as mm

def mag_vbb_binary_vec(w_pts, rho, a, e1, u1=0., accuracy=5e-05):
    e2 = 1 - e1
    x_cm = (e1 - e2)*a
    bl = mm.BinaryLens(e2, e1, 2*a)

    res = []
    for w0 in w_pts:
        res.append(bl.vbbl_magnification(float(w0.real - x_cm), float(w0.imag), rho, accuracy=accuracy))
    return np.array(res)

def test_finite_source_switch(niter=15, rtol=5e-04):
    key = random.PRNGKey(0)
    npts = 250 # 1000 pts around caustics

    # Draw random values of (q, s, rho) niter times
    for i in range(niter):
        key, subkey = random.split(key)
        q = float(10**random.uniform(subkey, minval=-6, maxval=0))
        s = float(10**random.uniform(subkey, minval=-1, maxval=np.log10(4)))
        rho = float(10**random.uniform(subkey, minval=-3, maxval=-1))

        a = 0.5*s
        e1 = q/(1+q)
        _, caustic_curves = critical_and_caustic_curves(
            npts=npts, nlenses=2, a=a, e1=e1
        )
        caustic_curves = caustic_curves.reshape(-1)

        # Generate random test points near the caustics
        key, subkey1, subkey2 = random.split(key, num=3)
        phi = random.uniform(subkey1, caustic_curves.shape, minval=-np.pi, maxval=np.pi)
        r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=15*rho)
        w_test = caustic_curves + r*np.exp(1j*phi)
        mags = mag_vbb_binary_vec(w_test, rho, a, e1)

        z, z_mask = images_point_source(w_test, nlenses=2, a=a, e1=e1)
        mu_multi, delta_multi = mag_hexadecapole(z, z_mask, rho, nlenses=2, a=a,e1=e1)
        err_hex = jnp.abs(mu_multi - mags)/mags

        test = _multipole_cusp_and_ghost_tests(
            w_test, z, z_mask, rho, delta_multi, 
            nlenses=2,  a=a, e1=e1
        )

        if q < 0.01:
            test_planetary = _planetary_caustic_test(w_test, rho, c_p=2., a=a,e1=e1)
            test = test & test_planetary

        mask_fail = test & (err_hex > rtol)
        assert sum(mask_fail) == 0

