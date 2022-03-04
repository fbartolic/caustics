__all__ = [
    "mag_extended_source_direct_integration_binary",
    "mag_extended_source_binary",
]

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

from . import (
    lens_eq_binary,
    lens_eq_triple,
    images_point_source_binary,
    images_point_source_triple,
    mag_point_source_binary,
    mag_point_source_triple,
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
def bboxes_near_overlap_pos_dir(bbox_c, bbox_r):
    """
    True if positive (CCW direction) theta edge of `bbox_c` is nearly adjacent to
    the negative edge of `bbox_r`.
    """
    wc = bbox_c[1] - bbox_c[0]
    wr = bbox_r[1] - bbox_r[0]
    cond_r = jnp.abs(bbox_c[0] - bbox_r[0]) < 1.2 * jnp.max(jnp.array([wc, wr]))
    cond_theta = ang_dist(bbox_c[3], bbox_r[2]) < jnp.deg2rad(5.0)
    return jnp.where(cond_r * cond_theta, True, False)


@jit
def bboxes_near_overlap_neg_dir(bbox_c, bbox_l):
    """
    True if negative (CW direction) theta edge of `bbox_c` is nearly adjacent to
    the positive edge of `bbox_l`.
    """
    wc = bbox_c[1] - bbox_c[0]
    wl = bbox_l[1] - bbox_l[0]
    cond_r = jnp.abs(bbox_c[0] - bbox_l[0]) < 1.2 * jnp.max(jnp.array([wc, wl]))
    cond_theta = ang_dist(bbox_l[3], bbox_c[2]) < jnp.deg2rad(5.0)
    return jnp.where(cond_r * cond_theta, True, False)


@jit
def get_closest_bbox_pos_dir(bbox, bboxes):
    rmin, rmax, tmin, tmax = bbox
    rmean = rmin + 0.5 * (rmax - rmin)

    x_edge, y_edge = rmean * jnp.cos(bbox[3]), rmean * jnp.sin(bbox[3])

    def get_dist(bbox_other):
        rmean_other = bbox_other[0] + 0.5 * (bbox_other[1] - bbox_other[0])
        x_edge_other, y_edge_other = rmean_other * jnp.cos(
            bbox_other[2]
        ), rmean * jnp.sin(bbox_other[2])
        return jnp.sqrt((x_edge_other - x_edge) ** 2 + (y_edge_other - y_edge) ** 2)

    distances = vmap(get_dist)(bboxes)
    return bboxes[jnp.argsort(distances)][0]


@jit
def get_closest_bbox_neg_dir(bbox, bboxes):
    rmin, rmax, tmin, tmax = bbox
    rmean = rmin + 0.5 * (rmax - rmin)

    x_edge, y_edge = rmean * jnp.cos(bbox[2]), rmean * jnp.sin(bbox[2])

    def get_dist(bbox_other):
        rmean_other = bbox_other[0] + 0.5 * (bbox_other[1] - bbox_other[0])
        x_edge_other, y_edge_other = rmean_other * jnp.cos(
            bbox_other[3]
        ), rmean * jnp.sin(bbox_other[3])
        return jnp.sqrt((x_edge_other - x_edge) ** 2 + (y_edge_other - y_edge) ** 2)

    distances = vmap(get_dist)(bboxes)
    return bboxes[jnp.argsort(distances)][0]


@partial(jit, static_argnames=("f_r", "f_theta"))
def extend_bboxes_polar(bboxes, f_r=0.2, f_theta=0.05):
    """
    Given a set of bounding boxes in polar coordinates, extend them by a factor
    of `f_r` in the radial direction and `f_theta` in the angular direction
    while making sure that if two edges of are nearly adjecent, the two edges
    are extended to match exactly.

    The purpose of this is to make sure that that all parts of the images are
    covered by the bounding boxes.

    Args:
        bboxes (array_like): Bounding boxes, shape (n_bboxes, 4).
        f_r (float): Factor to extend the radial bounding box by.
        f_theta (float): Factor to extend the angular bounding box by.
    Returns:
        array_like: Extended bounding boxes.
    """
    # Sort bboxes based on the inner theta edge
    bboxes = bboxes[jnp.argsort(bboxes[:, 2])]
    size = len(bboxes)

    for c, bbox in enumerate(bboxes):
        # Get shape of the central bbox
        theta_width = ang_dist(bboxes[c, 2], bboxes[c, 3])
        rwidth = bboxes[c, 1] - bboxes[c, 0]

        # Extend each bbox in theta and r but if two bboxes are close to each
        # other in theta merge their edges

        # Adjacent bboxes in the positive and negative theta directions
        bbox_adjacent_pos = get_closest_bbox_pos_dir(
            bbox, jnp.delete(bboxes, c, axis=0)
        )
        bbox_adjacent_neg = get_closest_bbox_neg_dir(
            bbox, jnp.delete(bboxes, c, axis=0)
        )

        cond_pos = bboxes_near_overlap_pos_dir(bbox, bbox_adjacent_pos)
        cond_neg = bboxes_near_overlap_neg_dir(bbox, bbox_adjacent_neg)

        def case2(bboxes):
            """bbox close to another bbox in the positive or negative bbox direction."""
            return jnp.where(
                cond_pos,
                jnp.array(
                    [
                        bboxes[c, 0] - f_r * rwidth,
                        bboxes[c, 1] + f_r * rwidth,
                        sub_angles(bbox[2], f_theta * theta_width),
                        bbox_adjacent_pos[2],
                    ]
                ),
                jnp.array(
                    [
                        bbox[0] - f_r * rwidth,
                        bbox[1] + f_r * rwidth,
                        bbox_adjacent_neg[3],
                        add_angles(bbox[3], f_theta * theta_width),
                    ]
                ),
            )

        def case1(bboxes):
            return jnp.where(
                cond_pos + cond_neg,  # close to another bbox on one of the sides
                case2(bboxes),
                jnp.array(
                    [
                        bbox[0] - f_r * rwidth,
                        bbox[1] + f_r * rwidth,
                        sub_angles(bbox[2], f_theta * theta_width),
                        add_angles(bbox[3], f_theta * theta_width),
                    ]
                ),
            )

        bbox_new = jnp.where(
            cond_pos * cond_neg,  # close to another bbox at both sides
            jnp.array(
                [
                    bbox[0] - f_r * rwidth,
                    bbox[1] + f_r * rwidth,
                    bbox_adjacent_neg[3],
                    bbox_adjacent_pos[2],
                ]
            ),
            case1(bboxes),
        )

        bboxes = bboxes.at[c].set(bbox_new)

    return bboxes


@partial(
    jit,
    static_argnames=(
        "npts_r",
        "npts_theta",
        "a1",
    ),
)
def integrate_within_bbox_binary(
    bbox, a, e1, w_center, rho_source, npts_r=100, npts_theta=1000, a1=0.1
):
    rmin, rmax, tmin, tmax = bbox

    # equivalent of np.linspace but ignoring the singularity at [-pi, pi]
    linspace_ang = lambda t1, t2, npts: add_angles(
        t1, jnp.linspace(0, ang_dist(t1, t2), npts)
    )

    # Image plane
    r = jnp.linspace(rmin, rmax, npts_r)
    theta = linspace_ang(tmin, tmax, npts_theta)
    dr = r[1] - r[0]
    dtheta = ang_dist(theta[1], theta[0])
    rgrid, thetagrid = jnp.meshgrid(r, theta)

    xgrid = rgrid * jnp.cos(thetagrid)
    ygrid = rgrid * jnp.sin(thetagrid)
    zgrid = xgrid + 1j * ygrid

    # Lens plane
    w = lens_eq_binary(zgrid, a, e1)
    x_w, y_w = jnp.real(w), jnp.imag(w)
    fn = (
        (x_w - jnp.real(w_center)) ** 2 + (y_w - jnp.imag(w_center)) ** 2
        <= rho_source ** 2
    ).astype(float)

    integrate_2d = lambda r, f: jnp.trapz(
        jnp.trapz(f * r, dx=dr, axis=0), dx=dtheta, axis=0
    )
    _rgrid = (x_w - jnp.real(w_center)) ** 2 + (y_w - jnp.imag(w_center)) ** 2
    integrand = fn * linear_limbdark(_rgrid, 1.0, c=a1)

    return integrate_2d(r, integrand)


@jit
def linear_limbdark(r, I0, c=0.1):
    return I0 * (1.0 - c * (1.0 - jnp.sqrt(1.0 - r ** 2)))


@partial(jit, static_argnames=("npts",))
def images_of_source_limb(w_center, rho_source, a, e1, npts=800):
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
        "npts_r",
        "npts_theta",
        "npts_limb",
        "a1",
    ),
)
def mag_extended_source_binary(
    a, e1, w_center, rho_source, npts_r=100, npts_theta=1000, npts_limb=800, a1=0.1
):
    # Get images of source limb and center
    z_limb = images_of_source_limb(w_center, rho_source, a, e1, npts=npts_limb)

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

    bboxes_final = extend_bboxes_polar(bboxes, f_r=0.2, f_theta=0.1)

    # Compute integrals
    integrals = []
    for bbox in bboxes_final:
        I = integrate_within_bbox_binary(
            bbox,
            a,
            e1,
            w_center,
            rho_source,
            npts_r=npts_r,
            npts_theta=npts_theta,
            a1=a1,
        )
        integrals.append(I)
    I = jnp.sum(jnp.array(integrals))

    mag = I / (np.pi * rho_source ** 2)

    return mag
