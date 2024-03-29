# -*- coding: utf-8 -*-
import numpy as np
import pytest
from functools import partial

import jax.numpy as jnp
from jax import jit, random, vmap
from jax.config import config
from jax import lax, jacrev, jacfwd

from caustics.extended_source import (
    _images_of_source_limb,
    _split_segment,
    _get_segments,
    _contours_from_open_segments,
    _connection_condition,
)
from caustics import (
    critical_and_caustic_curves,
    mag_extended_source,
)
from caustics.utils import last_nonzero

config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)

from scipy.optimize import approx_fprime

import MulensModel as mm

params = mm.ModelParameters({"t_0": 200.0, "u_0": 0.1, "t_E": 3.0, "rho": 0.1})
point_lens = mm.PointLens(params)

# Helper functions
def count_duplicates(a):
    _, ix = jnp.unique(a, return_index=True, size=len(a))
    mask = jnp.full(a.shape, True)
    mask = mask.at[ix].set(False)
    return mask.sum()

def mag_espl_lee_unif(w_points, rho):
    return point_lens.get_point_lens_uniform_integrated_magnification(
        np.abs(w_points), rho
    )


def mag_espl_lee_ld(w_points, rho, u1=0.0):
    Gamma = 2 * u1 / (3.0 - u1)
    return point_lens.get_point_lens_LD_integrated_magnification(
        np.abs(w_points), rho, Gamma
    )


def mag_vbb_binary(w0, rho, s, q, u1=0.0, accuracy=1e-05):
    e1 = 1/(1 + q)
    e2 = 1 - e1
    bl = mm.BinaryLens(e2, e1, s)
    return bl.vbbl_magnification(
        w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1
    )

@partial(jit, static_argnames=("npts_limb", "npts_ld", "limb_darkening"))
def mag_binary(w_points, rho, s, q, u1=0., npts_limb=300, npts_ld=100, limb_darkening=False):
    def body_fn(_, w):
        mag = mag_extended_source(
            w,
            rho,
            nlenses=2,
            npts_limb=npts_limb,
            limb_darkening=limb_darkening,
            npts_ld=npts_ld,
            u1=u1,
            s=s,
            q=q,
        )
        return 0, mag

    _, mags = lax.scan(body_fn, 0, w_points)
    return mags


def get_points_and_params_binary(rho, npts=50):
    s, q = 0.9, 0.2
    a, e1  = 0.5*s, 1/(1 + q)

    _, caustic_curves = critical_and_caustic_curves(
        npts=npts, nlenses=2, s=s, q=q
    )
    caustic_curves = caustic_curves.reshape(-1)

    # Generate random test points near the caustics
    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, num=3)
    phi = random.uniform(subkey1, caustic_curves.shape, minval=-np.pi, maxval=np.pi)
    r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=2*rho)
    w_test = caustic_curves + r*np.exp(1j*phi)
    return w_test, {'a': a, 'e1': e1, 's': s, 'q': q}

def get_points_and_params_triple(rho, npts=50):
    a, e1, e2, r3 = 0.698, 0.02809, 0.9687, -0.0197 - 0.95087j
    psi = jnp.arctan2(r3.imag, r3.real)
    s = 0.5*a
    q = e2/e1
    q3 = (1 - e1-e2)/e1

    _, caustic_curves = critical_and_caustic_curves(
        npts=npts, nlenses=3, s=s, q=q, q3=q3, r3=jnp.abs(r3), psi=psi
    )
    caustic_curves = caustic_curves.reshape(-1)

    # Generate random test points near the caustics
    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, num=3)
    phi = random.uniform(subkey1, caustic_curves.shape, minval=-np.pi, maxval=np.pi)
    r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=2*rho)
    w_test = caustic_curves + r*np.exp(1j*phi)
    return w_test, {'a': a, 'e1': e1, 'e2': e2, 'r3': r3}


# Tests
@pytest.mark.parametrize("rho", [1e-01, 1e-02, 1e-03])
def test_images_of_source_limb(rho, npts_limb=500):
    w_test, params = get_points_and_params_triple(rho)
    a, e1, e2, r3 = params['a'], params['e1'], params['e2'], params['r3']

    for w in w_test:
        z, z_mask, z_parity = _images_of_source_limb(
            w, rho, nlenses=3, npts=npts_limb,  a=a, e1=e1, e2=e2, r3=r3
        )

        # Check that there are no identical points
        assert count_duplicates(z.reshape(-1)) == 0

@pytest.mark.parametrize("rho", [1.0, 1e-01, 1e-02, 1e-03])
def test_mag_extended_source_single_unif(rho, npts_limb=150, rtol=1e-03):
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
def test_mag_extended_source_single_ld(rho, u1=0.7, npts_limb=300, npts_ld=100, rtol=1e-03):
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


def test_split_segment():
    z = jnp.array([0.0, 1.0, 1.01, 1.02, 1.03, 0.0, 4.0, 4.01, 0.0, 0.0, 6.0]).astype(
        jnp.complex128
    )
    z_parity = jnp.array([-1., -1., -1., 1., 1., 1., 1., 1., 1., 1., 1.])
    z_mask = jnp.array([0., 1., 1., 1., 1., 0., 1., 1., 0., 0., 1.])
    res = _split_segment(jnp.stack([z, z_parity, z_mask]), n_parts=5)[:, 0, :].real

    np.testing.assert_equal(
        res[0], np.array([0.0, 1.0, 1.01, 0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_equal(
        res[1], np.array([0.0, 0., 0., 1.02, 1.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_equal(
        res[2], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0., 4.0, 4.01, 0.0, 0.0, 0.0])
    )
    np.testing.assert_equal(
        res[3], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0., 0.0, 0.0, 0.0, 0.0, 6.0])
    )
    np.testing.assert_equal(res[4], np.zeros_like(res.shape[0]))


@pytest.mark.parametrize("rho", [1e-0, 1e-01, 1e-02, 1e-03, 1e-04])
def test_get_contours(rho, npts_limb=200):
    w_test, params = get_points_and_params_triple(rho)

    @jit
    def fn(w):
        z, z_mask, z_parity = _images_of_source_limb(
            w,
            rho,
            nlenses=3,
            npts=npts_limb,
            **params,
        )
        segments_closed, segments_open, all_closed = _get_segments(
            z, z_mask, z_parity, nlenses=3
        )
        contours, contours_p = _contours_from_open_segments(segments_open)
        tail_idcs = vmap(last_nonzero)(contours.real)
        return contours, contours_p, tail_idcs



    contours_list, contours_p_list, tail_idcs_list = [], [], []
    for w in w_test:
        contours, contours_p, tail_idcs = fn(w)
        contours_list.append(contours)
        contours_p_list.append(contours_p)
        tail_idcs_list.append(tail_idcs)

    contours_list, contours_p_list, tail_idcs_list = (
        jnp.stack(contours_list),
        jnp.stack(contours_p_list),
        jnp.stack(tail_idcs_list),
    )

    tail_idcs_list = tail_idcs_list - 1

    def test_connection_cond(cont):
        """Test that contour is closed"""
        tidx = last_nonzero(jnp.abs(cont))
        return jnp.where(
            jnp.all(cont == 0 + 0j),
            True,
            _connection_condition(cont[None, :], cont[None, :], tidx, tidx, 0),
        )

    connection_conds = vmap(vmap(test_connection_cond))(contours_list)
    assert jnp.all(jnp.prod(connection_conds, axis=1))


@pytest.mark.parametrize("rho", [1., 1e-01, 1e-02, 1e-03, 1e-04])
def test_mag_extended_source_binary_unif(rho, npts_limb=400, rtol=1e-03):
    # Compute the magnification with caustics and VBBinaryLensing
    w_test, params = get_points_and_params_binary(rho)

    mags = mag_binary(
        w_test, rho, params['s'], params['q'], npts_limb=npts_limb, 
    )

    mags_vbb = np.array(
        [
            mag_vbb_binary(w, rho, params['s'], params['q'], u1=0.0, accuracy=1e-05) for w in w_test
        ]
    )

    assert np.all(np.abs((mags - mags_vbb) / mags_vbb) < rtol)


@pytest.mark.parametrize("rho", [1., 1e-01, 1e-02, 1e-03])
def test_mag_extended_source_binary_ld(rho, u1=0.7, npts_limb=500, npts_ld=100, rtol=1e-03):
    w_test, params = get_points_and_params_binary(rho, npts=5)

    # Compute the magnification with `caustics` and VBB
    mags = mag_binary(
        w_test, rho, params['s'], params['q'], npts_limb=npts_limb, 
        limb_darkening=True, npts_ld=npts_ld, u1=u1
    )

    mags_vbb = np.array(
        [mag_vbb_binary(w, rho, params['s'], params['q'], u1=u1, accuracy=1e-05) for w in w_test]
    )

    assert np.all(np.abs((mags - mags_vbb) / mags_vbb) < rtol)


@pytest.mark.parametrize("rho", [1e-02])
def test_grad_mag_extended_source_binary(
    rho, u1=0.7, npts_limb=300, npts_ld=100, rtol=1e-03
):
    s, q = 0.9, 0.2
    _, caustic_curves = critical_and_caustic_curves(
        npts=50, nlenses=2, s=s, q=q
    )
    caustic_curves = caustic_curves.reshape(-1)

    w0 = caustic_curves[0]

    def fn(params):
        s, q, rho, u1 = params
        return mag_extended_source(
            w0,
            rho,
            u1=u1,
            nlenses=2,
            npts_limb=npts_limb,
            limb_darkening=True,
            npts_ld=npts_ld,
            s=s,
            q=q,
        )

    grad_fn_fwd = jit(jacfwd(fn))
    grad_fn_rev = jit(jacrev(fn))

    jac_fwd = grad_fn_fwd(jnp.array([s, q, rho, u1]))
    jac_rev = grad_fn_rev(jnp.array([s, q, rho, u1]))
    jac_finite_diff = approx_fprime(np.array([s, q, rho, u1]), fn, epsilon=1e-09)

    err_fwd = jnp.abs((jac_fwd - jac_finite_diff)/jac_finite_diff)
    err_rev = jnp.abs((jac_rev - jac_finite_diff)/jac_finite_diff)

    assert jnp.all(err_fwd < rtol)
    assert jnp.all(err_rev < rtol)

