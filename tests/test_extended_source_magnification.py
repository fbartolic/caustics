# -*- coding: utf-8 -*-
import numpy as np
import pytest

import jax.numpy as jnp
from jax import jit, random, vmap
from jax.config import config
from jax import lax
from jax.test_util import check_grads

from caustics.extended_source_magnification import (
    _images_of_source_limb,
    _match_two_sets_of_images,
    _permute_images,
    _split_single_segment,
    _get_segments,
    _get_contours,
    _connection_condition,
)
from caustics import (
    images_point_source,
    critical_and_caustic_curves,
    mag_extended_source,
)
from caustics.point_source_magnification import lens_eq_det_jac

config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)

import MulensModel as mm

params = mm.ModelParameters({"t_0": 200.0, "u_0": 0.1, "t_E": 3.0, "rho": 0.1})
point_lens = mm.PointLens(params)


def mag_espl_lee_unif(w_points, rho):
    return point_lens.get_point_lens_uniform_integrated_magnification(
        np.abs(w_points), rho
    )


def mag_espl_lee_ld(w_points, rho, u1=0.0):
    Gamma = 2 * u1 / (3.0 - u1)
    return point_lens.get_point_lens_LD_integrated_magnification(
        np.abs(w_points), rho, Gamma
    )


# from MulensModel.binarylensimports import _adaptive_contouring_linear


# def mag_adaptive_cont(w_center, rho, a, e1, u=0.1, eps=1e-04, eps_ld=1e-04):
#    x_cm = (e1 - (1.0 - e1)) * a
#    return _adaptive_contouring_linear(
#        2 * a,
#        (1.0 - e1) / e1,
#        jnp.real(w_center) - x_cm,
#        jnp.imag(w_center),
#        rho,
#        (2.0 * u) / (3.0 - u),
#        eps,
#        eps_ld,
#    )


@pytest.mark.parametrize("rho", [1e-01, 1e-02, 1e-03])
def test_images_of_source_limb(rho):
    a, e1, e2, r3 = 0.698, 0.02809, 0.9687, -0.0197 - 0.95087j

    critical_curves, caustic_curves = critical_and_caustic_curves(
        nlenses=3, npts=10, a=a, e1=e1, e2=e2, r3=r3, rho=rho
    )

    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, num=3)

    # Generate random test points near the caustics
    w_test = (
        caustic_curves
        + random.uniform(subkey1, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
        + 1j
        * random.uniform(subkey2, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
    )
    for w in w_test:
        z, z_mask, z_parity = _images_of_source_limb(
            w, rho, nlenses=3, a=a, e1=e1, e2=e2, r3=r3, npts=200
        )

        # Check that there are no identical points
        u, c = jnp.unique(z, return_counts=True)
        assert (c > 1).sum() == 0


@pytest.mark.parametrize("rho", [1.0, 1e-01, 1e-02, 1e-03])
def test_mag_extended_source_single_uniform(rho, rtol=1e-03):
    npts_limb = 100
    w_points = jnp.linspace(0.0, 3.0 * rho, 11)

    mags = vmap(
        lambda w: mag_extended_source(
            w,
            rho,
            nlenses=1,
            npts_limb=npts_limb,
        )
    )(w_points)

    mags_lee = mag_espl_lee_unif(w_points, rho)
    np.testing.assert_allclose(mags, mags_lee, rtol=rtol)

    # Check that u1 = 0. reduces to uniform case
    mags2 = vmap(
        lambda w: mag_extended_source(
            w,
            rho,
            limb_darkening=True,
            u1=0.0,
            nlenses=1,
            npts_limb=npts_limb,
            npts_ld=50,
        )
    )(w_points)

    np.testing.assert_allclose(mags, mags2, rtol=rtol)


@pytest.mark.parametrize("rho", [1.0, 1e-01, 1e-02])
def test_mag_extended_source_single_ld(rho, rtol=1e-03):
    npts_limb = 300
    npts_ld = 150
    u1 = 0.7
    w_points = jnp.linspace(0.0, 3.0 * rho, 11)

    mags = vmap(
        lambda w: mag_extended_source(
            w,
            rho,
            limb_darkening=True,
            u1=u1,
            nlenses=1,
            npts_limb=npts_limb,
            npts_ld=npts_ld,
        )
    )(w_points)

    mags_lee = mag_espl_lee_ld(w_points, rho, u1=u1)
    np.testing.assert_allclose(mags, mags_lee, rtol=rtol)


def test_match_two_sets_of_images():
    x = jnp.array([2.3, 3.2, 6.3, 143.0, 0.3, 753.0])
    y = jnp.array([0.4, 653.0, 6.1, 125.0, 3.1, 2.45])
    assert jnp.all(_match_two_sets_of_images(x, y) == jnp.array([5, 4, 2, 3, 0, 1]))


def test_permute_images():
    a, e1, e2, r3 = 0.698, 0.02809, 0.9687, -0.0197 - 0.95087j
    w_center = -0.65
    rho = 1e-02

    npts_init = 100
    theta = jnp.linspace(-np.pi, np.pi, npts_init - 1, endpoint=False)
    z, z_mask = images_point_source(
        rho * jnp.exp(1j * theta) + w_center,
        nlenses=3,
        a=a,
        r3=r3,
        e1=e1,
        e2=e2,
    )
    det = lens_eq_det_jac(z, nlenses=3, a=a, r3=r3, e1=e1, e2=e2)
    z_parity = jnp.sign(det)

    # Sort
    idcs_sorted = jnp.argsort(z_mask, axis=0)[::-1, :]
    z = jnp.take_along_axis(z, idcs_sorted, axis=0)
    z_mask = jnp.take_along_axis(z_mask, idcs_sorted, axis=0)
    z_parity = jnp.take_along_axis(z_parity, idcs_sorted, axis=0)

    z, z_mask, z_parity = _permute_images(z, z_mask, z_parity)

    # Make sure that the distance between consecutive points is small
    assert jnp.all(jnp.max(jnp.diff(jnp.abs(z), axis=1), axis=0) < 0.1)


def test_split_single_segment():
    array = jnp.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1]).astype(float)
    seg = jnp.zeros((2, array.shape[0]))
    seg = seg.at[0].set(array)
    seg_split = _split_single_segment(seg, n_parts=4)

    assert jnp.all(
        seg_split[0, 0, :] == jnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
    )
    assert jnp.all(
        seg_split[1, 0, :] == jnp.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]).astype(float)
    )
    assert jnp.all(
        seg_split[2, 0, :] == jnp.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).astype(float)
    )
    assert jnp.all(seg_split[3, 0, :] == 0.0)


@pytest.mark.parametrize("rho", [1e-01, 1e-02, 1e-03])
def test_get_segments(rho):
    a, e1, e2, r3 = 0.698, 0.02809, 0.9687, -0.0197 - 0.95087j

    _, caustic_curves = critical_and_caustic_curves(
        npts=10, nlenses=3, a=a, e1=e1, e2=e2, r3=r3
    )

    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, num=3)

    # Generate random test points near the caustics
    w_test = (
        caustic_curves
        + random.uniform(subkey1, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
        + 1j
        * random.uniform(subkey2, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
    )

    def f(w):
        z, z_mask, z_parity = _images_of_source_limb(
            w,
            rho,
            nlenses=3,
            a=a,
            r3=r3,
            e1=e1,
            e2=e2,
            npts_init=200,
        )
        segments, cond_closed = _get_segments(z, z_mask, z_parity, nlenses=3)
        return segments

    segments_list = jnp.array([f(w) for w in w_test])

    def check_single_segment(segment):
        z, p = segment

        # Check that the parity of each nonzero point is the same
        cond_parity = jnp.logical_xor((p == -1.0).sum() > 0, (p == 1).sum() > 0)

        # Check that the segment is continous
        mask = jnp.abs(z) > 0.0
        cond_continuous = (jnp.abs(jnp.diff(mask)) > 0.0).sum() <= 1

        return lax.cond(
            jnp.logical_and(cond_parity, cond_continuous), lambda: 1.0, lambda: jnp.nan
        )

    def check_segments(segments):
        f = lambda seg: lax.cond(
            jnp.all(seg == 0 + 0j), lambda _: 1.0, check_single_segment, seg
        )
        return jnp.any(jnp.isnan(vmap(f)(segments)))

    assert jnp.all(
        jnp.isnan(jnp.array([check_segments(seg) for seg in segments_list])) == False
    )


@pytest.mark.parametrize("rho", [1e-01, 1e-02, 1e-03])
def test_get_contours(rho):
    a, e1, e2, r3 = 0.698, 0.02809, 0.9687, -0.0197 - 0.95087j

    critical_curves, caustic_curves = critical_and_caustic_curves(
        nlenses=3, npts=10, a=a, e1=e1, e2=e2, r3=r3, rho=rho
    )

    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, num=3)

    # Generate random test points near the caustics
    w_test = (
        caustic_curves
        + random.uniform(subkey1, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
        + 1j
        * random.uniform(subkey1, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
    )

    def f(w):
        z, z_mask, z_parity = _images_of_source_limb(
            w,
            rho,
            nlenses=3,
            a=a,
            r3=r3,
            e1=e1,
            e2=e2,
            npts_init=200,
        )
        segments, cond_closed = _get_segments(z, z_mask, z_parity, nlenses=3)
        contours, parity, tail_idcs = _get_contours(
            segments,
            cond_closed,
            n_contours=10,
        )
        return contours, parity, tail_idcs

    contours_list, parity_list, tail_idcs_list = [], [], []
    for w in w_test:
        contours, parity, tail_idcs = f(w)
        contours_list.append(contours)
        parity_list.append(parity)
        tail_idcs_list.append(tail_idcs)

    contours_list, parity_list, tail_idcs_list = (
        jnp.stack(contours_list),
        jnp.stack(parity_list),
        jnp.stack(tail_idcs_list),
    )

    tail_idcs_list = tail_idcs_list - 1

    def test_connection_cond(cont, tidx):
        """ "Test that contour is closed"""
        return jnp.where(
            jnp.all(cont == 0 + 0j),
            True,
            _connection_condition(
                lax.dynamic_slice_in_dim(cont, tidx - 1, 2),
                lax.dynamic_slice_in_dim(cont, 0, 2)[::-1],
                max_dist=1e-01,
            ),
        )

    connection_conds = np.zeros((contours_list.shape[:2]))
    for i in range(contours_list.shape[0]):
        for j in range(contours_list.shape[1]):
            connection_conds[i, j] = test_connection_cond(
                contours_list[i, j], tail_idcs_list[i, j]
            )

    assert jnp.all(jnp.prod(connection_conds, axis=1))


# @pytest.mark.parametrize("rho", [1e-01, 1e-02, 1e-03, 1e-04])
# def test_mag_extended_source_binary_unif(rho, max_err=1e-03):
#    a, e1, = (
#        0.45,
#        0.8,
#    )
#    critical_curves, caustic_curves = critical_and_caustic_curves(
#        npts=10, nlenses=2, a=a, e1=e1
#    )
#
#    key = random.PRNGKey(42)
#    key, subkey1, subkey2 = random.split(key, num=3)
#
#    # Generate random test points near the caustics
#    w_test = (
#        caustic_curves
#        + random.uniform(subkey1, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
#        + 1j
#        * random.uniform(subkey1, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
#    )
#
#    # Compute the magnification with `caustics` and adaptive contouring
#    mags = vmap(
#        lambda w: mag_extended_source(
#            w,
#            rho,
#            u=0.0,
#            nlenses=2,
#            npts_limb=250,
#            niter_limb=8,
#            a=a,
#            e1=e1,
#        )
#    )(w_test)
#
#    mags_ac = np.array(
#        [
#            mag_adaptive_cont(w, rho, a, e1, u=0.0, eps=1e-02, eps_ld=1e-02)
#            for w in w_test
#        ]
#    )
#
#    assert np.all(np.abs((mags - mags_ac) / mags_ac) < max_err)
#

# @pytest.mark.parametrize("rho", [1e-01, 1e-02])
# def test_mag_extended_source_binary_limb_darkening(rho, max_err=1e-03):
#    a, e1, = (
#        0.45,
#        0.8,
#    )
#    critical_curves, caustic_curves = critical_and_caustic_curves(
#        npts=10, nlenses=2, a=a, e1=e1
#    )
#
#    key = random.PRNGKey(42)
#    key, subkey1, subkey2 = random.split(key, num=3)
#
#    # Generate random test points near the caustics
#    w_test = (
#        caustic_curves
#        + random.uniform(subkey1, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
#        + 1j
#        * random.uniform(subkey1, caustic_curves.shape, minval=-2 * rho, maxval=2 * rho)
#    )
#
#    # Compute the magnification with `caustics` and adaptive contouring
#    mags = vmap(
#        lambda w: mag_extended_source(
#            w,
#            rho,
#            u=0.25,
#            nlenses=2,
#            npts_limb=250,
#            niter_limb=8,
#            a=a,
#            e1=e1,
#        )
#    )(w_test)
#
#    mags_ac = np.array(
#        [
#            mag_adaptive_cont(w, rho, a, e1, u=0.25, eps=1e-02, eps_ld=1e-04)
#            for w in w_test
#        ]
#    )
#
#    assert np.all(np.abs((mags - mags_ac) / mags_ac) < max_err)


def test_grad_mag_extended_source_binary(rho=1e-02):
    a, e1, = (
        0.45,
        0.8,
    )
    critical_curves, caustic_curves = critical_and_caustic_curves(
        npts=1, nlenses=2, a=a, e1=e1
    )
    w = caustic_curves[0]

    f = lambda a: mag_extended_source(w, rho, u=0.2, nlenses=2, a=a, e1=e1)
    check_grads(f, (a,), 1, rtol=5e-03)

    f = lambda rho: mag_extended_source(w, rho, u=0.2, a=a, e1=e1)
    check_grads(f, (rho,), 1, rtol=5e-03)
