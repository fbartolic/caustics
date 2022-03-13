__all__ = [
    "mag_extended_source_direct_integration_binary",
    "mag_extended_source_binary",
]

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax

from . import (
    lens_eq_binary,
    lens_eq_triple,
    images_point_source_binary,
    images_point_source_triple,
    mag_point_source_binary,
    mag_point_source_triple,
    integrate_image,
)
from .utils import *


@partial(jit, static_argnames=("npts",))
def mag_extended_source_direct_integration_binary(w_point, a, e1, rho_source, npts=100):
    r = jnp.linspace(0.0, rho_source, npts)
    theta = jnp.linspace(-jnp.pi, jnp.pi, npts)
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]

    rgrid, thetagrid = jnp.meshgrid(r, theta)

    xgrid = rgrid * jnp.cos(thetagrid) + jnp.real(w_point)
    ygrid = rgrid * jnp.sin(thetagrid) + jnp.imag(w_point)
    wgrid = xgrid + 1j * ygrid

    mag = mag_point_source_binary(wgrid, a, e1)

    integrate_2d = lambda r, f: jnp.trapz(
        jnp.trapz(f * r, dx=dr, axis=0), dx=dtheta, axis=0
    )

    return integrate_2d(r, mag) / (np.pi * rho_source ** 2)


@jit
def get_bbox_polar(r, theta):
    """
    Given a set of points in polar coordinates, return the bounding box around
    those points. All of the complexity here is for dealing with the singularity
    at [-pi, pi).

    Args:
        r (array): Array of radial distances.
        theta (array): Array of angular distances in range[-pi, pi).
    Returns:
        tuple: (rmin, rmax, thetamin, thetamax)
    """
    # Sort points by theta [-pi, pi)
    sorted_idcs = jnp.argsort(theta)
    theta = theta[sorted_idcs]
    r = r[sorted_idcs]

    # Get bbox in r
    rmin = min_zero_avoiding(r)
    rmax = jnp.max(r)
    bbox_r = jnp.array([rmin, rmax])

    # Check in which quadrants are the points located
    cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
    quad_i = (cos_theta > 0.0) * (sin_theta > 0.0)
    quad_ii = (cos_theta < 0.0) * (sin_theta > 0.0)
    quad_iii = (cos_theta < 0.0) * (sin_theta < 0.0)
    quad_iv = (cos_theta > 0.0) * (sin_theta < 0.0)

    # Identify large separations between consecutive points to find bbox edges
    # in theta
    theta_diff = ang_dist_diff(theta)
    sorted_idcs = jnp.argsort(theta_diff)
    idx_best = sorted_idcs[-1]
    idx_secbest = sorted_idcs[-2]
    vals = jnp.array(
        [
            theta[idx_best],
            theta[idx_best + 1],
            theta[idx_secbest],
            theta[idx_secbest + 1],
        ]
    )

    def case1(theta, vals):
        cond = jnp.any(quad_i) * jnp.any(quad_iv)
        bbox_theta = jnp.where(
            cond,
            jnp.array([jnp.min(theta), jnp.max(theta)]),
            jnp.array([min_zero_avoiding(theta), max_zero_avoiding(theta)]),
        )
        return bbox_theta

    def case2(theta, vals):
        # The elongeated arc of points covers quadrants II, III and IV fully
        # and there's a gap in quadrant I
        a = (vals[0] > 0.0) * (vals[1] > 0.0)

        # The elongeated arc of points covers quadrants II and II fully but
        # the gap also contains the +x axis which requires special treatment
        b = (vals[0] < 0.0) * (vals[1] < 0.0)

        cond = jnp.logical_or(a, b)

        bbox_theta = jnp.where(
            cond,
            jnp.array([jnp.max(vals[:2]), jnp.min(vals[:2])]),
            jnp.array([jnp.max(vals), jnp.min(vals)]),
        )
        return bbox_theta

    bbox_theta = jnp.where(
        jnp.any(quad_ii) * jnp.any(quad_iii),
        case2(theta, vals),
        case1(theta, vals),
    )

    return jnp.concatenate([bbox_r, bbox_theta])


@jit
def merge_polar_intervals(a, b):
    """
    Given two angular intervals (a[0], a[1]) and (b[0], b[1]) in range[-pi, pi),
    where each interval is defined in the CCW direction, merge them into one.

    Args:
        a (array_like): Beginning and end of the first interval in range[-pi, pi).
        b (array_like): Beginning and end of the second interval in range[-pi, pi).

    Returns:
        array_like: The merged interval.
    """
    # 4 possible cases: a containss b, b contains a, a and b overlap such that
    # a is the first interval (CCW direction from -pi) or a and b overlap such
    # that b is the first interval
    wa = ang_dist(a[0], a[1])
    wb = ang_dist(b[0], b[1])

    # A inside B
    a_inside_b = jnp.logical_and(
        jnp.logical_and(sub_angles(b[1], a[1]) > 0.0, sub_angles(a[0], b[0]) > 0.0),
        wb > wa,
    )
    b_inside_a = jnp.logical_and(
        jnp.logical_and(sub_angles(a[1], b[1]) > 0.0, sub_angles(b[0], a[0]) > 0.0),
        wa > wb,
    )
    neither = jnp.logical_not(jnp.logical_or(a_inside_b, b_inside_a))

    a_comes_first = jnp.logical_and(
        neither, ang_dist(a[1], b[0]) < ang_dist(b[1], a[0])
    )
    b_comes_first = jnp.logical_and(neither, jnp.logical_not(a_comes_first))

    def fn_a_comes_first():
        return jnp.array([a[0], b[1]])

    def fn_b_comes_first():
        return jnp.array([b[0], a[1]])

    index = jnp.nonzero(
        jnp.array([a_inside_b, b_inside_a, a_comes_first, b_comes_first]), size=1
    )[0][0]

    return lax.switch(index, [lambda: b, lambda: a, fn_a_comes_first, fn_b_comes_first])


@jit
def merge_two_bboxes(bbox_a, bbox_b):
    """
    Given two overlapping bounding boxes, merge them into a single bounding box.
    """
    rmin = min_zero_avoiding(jnp.array([bbox_a[0], bbox_b[0]]))
    rmax = max_zero_avoiding(jnp.array([bbox_a[1], bbox_b[1]]))
    return jnp.concatenate(
        [jnp.array([rmin, rmax]), merge_polar_intervals(bbox_a[2:], bbox_b[2:])]
    )


@jit
def extend_bbox(bbox, f_r, f_theta):
    """
    Extend the bounding box by a factor of f_r*(rmax - rmin) in the +/- r
    directions and a factor of f_theta*ang_dist(thetamin, thetamax) in the +/-
    theta directions.
    """
    theta_width = ang_dist(bbox[2], bbox[3])
    rwidth = bbox[1] - bbox[0]
    return jnp.array(
        [
            bbox[0] - f_r * rwidth,
            bbox[1] + f_r * rwidth,
            sub_angles(bbox[2], f_theta * theta_width),
            add_angles(bbox[3], f_theta * theta_width),
        ]
    )


@jit
def bboxes_near_overlap(bbox_a, bbox_b):
    """True if `bbox_a` and `bbox_b` overlap or nearly overlap."""
    delta_r_a = bbox_a[1] - bbox_a[0]
    delta_r_b = bbox_b[1] - bbox_b[0]

    delta_theta_a = ang_dist(bbox_a[2], bbox_a[3])
    delta_theta_b = ang_dist(bbox_b[2], bbox_b[3])

    # Condition for nearly overlapping in r
    cond_r = jnp.abs(bbox_b[0] - bbox_a[0]) < 1.2 * jnp.max(
        jnp.array([delta_r_a, delta_r_b])
    )

    # Condition for nearly overlapping in theta
    bbox_merged = merge_two_bboxes(bbox_a, bbox_b)
    cond_theta = ang_dist(bbox_merged[2], bbox_merged[3]) < 1.2 * (
        delta_theta_a + delta_theta_b
    )

    return jnp.where(jnp.all(bbox_a == bbox_b), False, cond_r * cond_theta)


@partial(jit, static_argnames=("f_r", "f_theta"))
def process_bboxes_polar(bboxes, f_r=0.1, f_theta=0.1):
    """
    Given a set of bounding boxes, slightly extend each bounding box in r and
    theta, then determine if it is overlapping any of the neighbours and merge
    it with the overlapping neighbour. Repeat for each bounding box until
    we are left with only isolated nonoveralping bounding boxes.

    Args:
        bboxes (array_like): Bounding boxes, shape (n_bboxes, 4).
        f_r (float): Factor to extend the radial bounding box by.
        f_theta (float): Factor to extend the angular bounding box by.
    Returns:
        array_like: Extended and merged bounding boxes. Those bboxes that
            end up getting merged are set to all zeros.
    """
    # Extend bboxes
    bboxes = vmap(lambda bbox: extend_bbox(bbox, f_r, f_theta))(bboxes)

    def merge_and_remove(bboxes, i, j):
        bbox_merged = merge_two_bboxes(bboxes[i], bboxes[j])
        bboxes = bboxes.at[i].set(bbox_merged)
        return bboxes.at[j].set(jnp.zeros(4))

    for i in range(len(bboxes)):
        bbox = bboxes[i]

        # Check if the bbox overlaps with any of the neighbours
        mask_overlap = vmap(lambda bbox_other: bboxes_near_overlap(bbox, bbox_other))(
            bboxes
        )

        # Return new bboxes which area either unchanged or one pair of bboxes
        # gets merged
        bboxes = jnp.where(
            jnp.logical_or(mask_overlap.sum() == 0, jnp.all(bbox == 0.0)),
            bboxes,  # nothing happens if there are no overlaps or the bbox is all zeros
            merge_and_remove(bboxes, i, jnp.argsort(mask_overlap)[-1]),
        )
    return bboxes


@jit
def linear_limbdark(r, I0, c=0.1):
    return I0 * (1.0 - c * (1.0 - jnp.sqrt(1.0 - r ** 2)))


@partial(jit, static_argnames=("npts",))
def images_of_source_limb_binary(w_center, rho_source, a, e1, npts=1000):
    """
    Solve for the images of uniformly distributed points on the limb of a
    circular disc with radius `rho_source` and center `w_center`.

    WARNING: Using too few points may lead to cases where a part of the source
    disc which crossed the caustic is not accounted for. Also, since the images
    of the limb determine the extent of the integration grids in the image plane
    using too few points may lead to situations where the grids end up being
    too small and don't cover the entire image.

    Args:
        w_center (complex128): Center of the disc in the source plane.
        rho_source (float): Radius of the source disc in angular Einstein radii.
        a (float): Half the lens separation.
        e1 (float): m1/(m1 + m2).
        npts (int, optional): Number of points on the limb. Defaults to 1000.

    Returns:
        array_like: The images for each point on the limb.
    """
    theta = jnp.linspace(-np.pi, np.pi, npts)
    x = rho_source * jnp.cos(theta) + jnp.real(w_center)
    y = rho_source * jnp.sin(theta) + jnp.imag(w_center)
    wgrid = x + 1j * y
    images, mask = images_point_source_binary(wgrid, a, e1)
    images = images * mask
    return images.T


@partial(
    jit,
    static_argnames=(
        "grid_size",
        "grid_size_ratio",
        "npts_limb",
        "f_r",
        "f_theta",
        "eps",
    ),
)
def mag_extended_source_binary(
    a,
    e1,
    w_center,
    rho_source,
    grid_size=0.05,
    grid_size_ratio=4.0,
    eps=1e-04,
    npts_limb=1000,
    a1=0.1,
):
    # Set the resolution of the grid in r and theta
    dr = grid_size * rho_source
    dtheta = grid_size_ratio * dr

    # Get images of source limb
    z_limb = images_of_source_limb_binary(w_center, rho_source, a, e1, npts=npts_limb)

    # Get seed locations within all images
    get_centroids_init = lambda z: z[(z != 0).prod(axis=1).argsort()[-1]]
    centroids_init = get_centroids_init(z_limb)
    centroids_init = jnp.stack([jnp.real(centroids_init), jnp.imag(centroids_init)]).T

    # Add (0,0) as the last cluster, all points in this cluster are ignored in subsequent
    # simulations
    centroids_init = jnp.vstack([centroids_init, jnp.array([0.0, 0.0])])

    X = jnp.stack([jnp.real(z_limb.reshape(-1)), jnp.imag(z_limb.reshape(-1))]).T

    closest_centroid = lambda p, centroids: jnp.sqrt(
        ((p - centroids) ** 2).sum(axis=1)
    ).argmin()
    labels = vmap(lambda p: closest_centroid(p, centroids_init))(X)

    r_limb = jnp.abs(z_limb.reshape(-1))
    theta_limb = jnp.arctan2(jnp.imag(z_limb.reshape(-1)), jnp.real(z_limb.reshape(-1)))

    # Compute bboxes
    n_clusters = 5
    label_idcs = jnp.arange(n_clusters)

    r_limb = jnp.abs(z_limb.reshape(-1))
    theta_limb = jnp.arctan2(jnp.imag(z_limb.reshape(-1)), jnp.real(z_limb.reshape(-1)))

    bboxes = vmap(
        lambda c: get_bbox_polar(
            r_limb * (labels == c).astype(float),
            theta_limb * (labels == c).astype(float),
        ),
    )(label_idcs)

    bboxes = process_bboxes_polar(bboxes, f_r=0.1, f_theta=0.1)

    # Compute the integrals for each of the bboxes
    integrals = []
    for bbox in bboxes:
        rmin, rmax, tmin, tmax = bbox

        I = integrate_image(
            rmin,
            rmax,
            tmin,
            tmax,
            dr,
            dtheta,
            rho_source,
            a1,
            a,
            e1,
            jnp.complex128(w_center),
            eps=eps,
            grid_size_ratio=grid_size_ratio,
        )
        integrals.append(I)

    I = jnp.sum(jnp.array(integrals))
    mag = I / (np.pi * rho_source ** 2)

    return mag
