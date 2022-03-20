# -*- coding: utf-8 -*-

import numpy as np
import pytest

import jax.numpy as jnp
from jax.config import config
from jax.test_util import check_grads

from caustics.extended_source_magnification import (
    get_bbox_polar,
    merge_polar_intervals,
)
from caustics.utils import min_zero_avoiding, ang_dist
from caustics.integrate_image_primitive import integrate_image_binary

import VBBinaryLensing

VBBL = VBBinaryLensing.VBBinaryLensing()
VBBL.RelTol = 1e-10

config.update("jax_enable_x64", True)


def test_get_bbox_polar():
    # Case 1 - points aren't scattered across both quadrant II and III
    theta = np.deg2rad(np.linspace(20.0, 160.0, 100))
    r = np.random.uniform(0.8, 1.0, 100)

    # Add zero points
    theta = jnp.array(np.concatenate([theta, np.zeros(50)]))
    r = jnp.array(np.concatenate([theta, np.zeros(50)]))

    bbox = get_bbox_polar(r, theta)

    assert bbox[0] == min_zero_avoiding(r)
    assert bbox[1] == jnp.max(r)

    assert bbox[2] == jnp.deg2rad(20.0)
    assert bbox[3] == jnp.deg2rad(160.0)

    # Case 2a: the elongated arc of points covers quadrants II, III and IV fully
    # and there's a gap in quadrant I
    theta1 = np.deg2rad(np.linspace(-180.0, 20.0, 50))
    theta2 = np.deg2rad(np.linspace(60.0, 180.0, 50))
    theta = np.concatenate([theta1, theta2])

    theta = jnp.array(np.concatenate([theta, np.zeros(50)]))

    bbox = get_bbox_polar(r, theta)

    assert bbox[2] == jnp.deg2rad(60.0)
    assert bbox[3] == jnp.deg2rad(20.0)

    # Case 2b: the elongated arc of points covers quadrants II and II fully but
    # the gap also contains the +x axis which requires special treatment
    theta1 = np.deg2rad(np.linspace(-180.0, -20.0, 50))
    theta2 = np.deg2rad(np.linspace(20.0, 180.0, 50))
    theta = np.concatenate([theta1, theta2])

    theta = jnp.array(np.concatenate([theta, np.zeros(50)]))

    bbox = get_bbox_polar(r, theta)

    assert bbox[2] == jnp.deg2rad(20.0)
    assert bbox[3] == jnp.deg2rad(-20.0)


def test_merge_polar_intervals():
    a = jnp.array([jnp.deg2rad(160.0), jnp.deg2rad(-173.0)])
    b = jnp.array([jnp.deg2rad(160.05), jnp.deg2rad(-170.0)])
    c_sol = jnp.array([jnp.deg2rad(160.0), jnp.deg2rad(-170.0)])

    np.testing.assert_allclose(merge_polar_intervals(a, b), c_sol)
    np.testing.assert_allclose(merge_polar_intervals(b, a), c_sol)

    a = jnp.array([jnp.deg2rad(45.0), jnp.deg2rad(-70.0)])
    b = jnp.array([jnp.deg2rad(-75.0), jnp.deg2rad(15.0)])
    c_sol = jnp.array([jnp.deg2rad(45.0), jnp.deg2rad(15.0)])

    np.testing.assert_allclose(merge_polar_intervals(a, b), c_sol)
    np.testing.assert_allclose(merge_polar_intervals(b, a), c_sol)

    a = jnp.array([jnp.deg2rad(130.0), jnp.deg2rad(150.0)])
    b = jnp.array([jnp.deg2rad(-178), jnp.deg2rad(-150.0)])
    c_sol = jnp.array([jnp.deg2rad(130.0), jnp.deg2rad(-150.0)])

    np.testing.assert_allclose(merge_polar_intervals(a, b), c_sol)
    np.testing.assert_allclose(merge_polar_intervals(b, a), c_sol)

def test_integrate_image_binary_grad():
    rmin = 1.2460687963709596
    rmax = 1.3807432495443732
    theta_min = -0.6736152464838706
    theta_max = 0.6736152464838707
    dr = 0.0005
    dtheta = 0.002
    rho = 0.01
    a1 = 0.2
    a = 0.45
    e1 = 0.8
    w_cent_real = 0.3955
    w_cent_imag = 0.

    fn = lambda a: integrate_image_binary(
        rmin, rmax, theta_min, theta_max, dr, dtheta, rho, a1, a, e1, w_cent_real, w_cent_imag
    )
    check_grads(fn, (e1,), 1, eps=1e-08)