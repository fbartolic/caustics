# -*- coding: utf-8 -*-

import numpy as np
import pytest

import jax.numpy as jnp
from jax.config import config
from jax.test_util import check_grads

from caustics import (
    mag_point_source,
)

import VBBinaryLensing

VBBL = VBBinaryLensing.VBBinaryLensing()
VBBL.RelTol = 1e-10

config.update("jax_enable_x64", True)


@pytest.fixture()
def get_data():
    a = 0.5 * 0.9
    e1 = 0.8
    e2 = 1.0 - e1

    q = e1 / e2
    x_cm = (e1 - e2) * a

    return a, e1, q, x_cm


def test_mag_point_source_binary(get_data):
    a, e1, q, x_cm = get_data

    width = 1.0
    npts = 50
    x = jnp.linspace(-0.5 * width, 0.5 * width, npts)
    y = jnp.linspace(-0.5 * width, 0.5 * width, npts)
    xgrid, ygrid = jnp.meshgrid(x, y)
    wgrid = xgrid + 1j * ygrid

    mag = mag_point_source(wgrid, nlenses=2, a=a, e1=e1)

    # Compare to VBBinaryLensing
    mag_vbb = np.zeros(wgrid.shape)
    for i in range(wgrid.shape[0]):
        for j in range(wgrid.shape[1]):
            mag_vbb[i, j] = VBBL.BinaryMag0(
                2 * a,
                q,
                jnp.real(wgrid[i, j]) - x_cm,
                jnp.imag(wgrid[i, j]),
            )

    np.testing.assert_allclose(mag, mag_vbb, atol=1e-10)


def test_mag_point_source_grad(get_data):
    a, e1, _, _ = get_data

    # Check gradient of mag. with respect to lens separation
    fn = lambda a: mag_point_source(jnp.array([0.0 + 0.1j])[0], nlenses=2, a=a, e1=e1)
    check_grads(fn, (a,), 2)
