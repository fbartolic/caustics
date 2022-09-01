# -*- coding: utf-8 -*-
import numpy as np
import pytest

import jax.numpy as jnp
from jax.config import config
from jax.test_util import check_grads

from caustics import (
    mag_point_source,
)

import MulensModel as mm

def mag_vbb_binary_ps(w0, s, q):
    e1 = 1/(1 + q)
    e2 = q/(1 + q)
    bl = mm.BinaryLens(e2, e1, s)
    return bl.point_source_magnification(w0.real, w0.imag)

config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)


@pytest.fixture()
def get_data():
    s = 0.9
    q = 0.2
    return s, q


def test_mag_point_source_binary(get_data):
    s, q = get_data

    width = 1.0
    npts = 50
    x = jnp.linspace(-0.5 * width, 0.5 * width, npts)
    y = jnp.linspace(-0.5 * width, 0.5 * width, npts)
    xgrid, ygrid = jnp.meshgrid(x, y)
    wgrid = xgrid + 1j * ygrid

    mag = mag_point_source(wgrid, nlenses=2, s=s, q=q)

    # Compare to VBBinaryLensing
    mag_vbb = np.zeros(wgrid.shape)
    for i in range(wgrid.shape[0]):
        for j in range(wgrid.shape[1]):
            mag_vbb[i, j] = mag_vbb_binary_ps(wgrid[i, j], s, q)
    np.testing.assert_allclose(mag, mag_vbb, atol=1e-10)


def test_mag_point_source_grad(get_data):
    s, q = get_data

    # Check gradient of mag. with respect to lens separation
    fn = lambda s: mag_point_source(jnp.array([0.0 + 0.1j])[0], nlenses=2, s=s, q=q)
    check_grads(fn, (s,), 2)
