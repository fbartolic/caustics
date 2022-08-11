# -*- coding: utf-8 -*-
"""
Compute the magnification of an extended source using contour integration.
"""
__all__ = [
    "mag_extended_source",
]

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax, random

from . import (
    images_point_source,
)
from .integrate import (
    _integrate_unif,
    _integrate_ld,
)
from .utils import (
    first_nonzero,
    first_zero,
    last_nonzero,
    sparse_argsort,
)

from .point_source_magnification import (
    lens_eq_det_jac,
    _images_point_source_sequential,
)


@jit
def _match_two_sets_of_images(a, b):
    """
    For each image in a and find the index of the closest image in b. This
    procedure is not guaranteed to find a permutation of b which minimizes
    the sum of elementwise distances between every element of a and b but
    it is good enough.

    Args:
        a (array_like): 1D array.
        b (array_like): 1D array.

    Returns:
        array_like: Indices which specify a permutation of b.
    """
    # First guess
    vals = jnp.argsort(jnp.abs(b - a[:, None]), axis=1)
    idcs = []
    for i, idx in enumerate(vals[:, 0]):
        # If index is duplicate choose the next best solution
        mask = ~jnp.isin(vals[i], jnp.array(idcs), assume_unique=True)
        idx = vals[i, first_nonzero(mask)]
        idcs.append(idx)

    return jnp.array(idcs)


@jit
def _permute_images(z, z_mask, z_parity):
    """
    Sequantially permute the images corresponding to points on the source limb
    starting with the first point such that each point source image is assigned
    to the correct curve. This procedure does not differentiate between real
    and false images, false images are set to zero after the permutation
    operation.
    """
    xs = jnp.stack([z, z_mask, z_parity])

    def apply_match_two_sets_of_images(carry, xs):
        z, z_mask, z_parity = xs
        idcs = _match_two_sets_of_images(carry, z)
        return z[idcs], jnp.stack([z[idcs], z_mask[idcs], z_parity[idcs]])

    init = xs[0, :, 0]
    _, xs = lax.scan(apply_match_two_sets_of_images, init, jnp.moveaxis(xs, -1, 0))
    z, z_mask, z_parity = jnp.moveaxis(xs, 1, 0)

    return z.T, z_mask.real.astype(bool).T, z_parity.T


@partial(
    jit,
    static_argnames=(
        "nlenses",
        "npts",
        "niter",
        "roots_itmax",
        "roots_compensated",
    ),
)
def _images_of_source_limb(
    w0,
    rho,
    nlenses=2,
    npts=300,
    niter=10,
    roots_itmax=2500,
    roots_compensated=False,
    **params,
):
    key = random.PRNGKey(0)
    key1, key2 = random.split(key)

    def fn(theta, z_init):
        # Add a small perturbation to z_init to avoid situations where
        # there is convergence to the exact same roots when difference in theta
        # is very small and z_init is very close to the exact roots
        u1 = random.uniform(key1, shape=z_init.shape, minval=-1e-6, maxval=1e-6)
        u2 = random.uniform(key2, shape=z_init.shape, minval=-1e-6, maxval=1e-6)
        z_init = z_init + u1 + u2 * 1j

        z, z_mask = images_point_source(
            rho * jnp.exp(1j * theta) + w0,
            nlenses=nlenses,
            roots_itmax=roots_itmax,
            roots_compensated=roots_compensated,
            z_init=z_init.T,
            custom_init=True,
            **params,
        )
        det = lens_eq_det_jac(z, nlenses=nlenses, **params)
        z_parity = jnp.sign(det)
        return z, z_mask, z_parity

    # Initial sampling on the source limb
    npts_init = int(0.5 * npts)
    theta = jnp.linspace(-np.pi, np.pi, npts_init - 1, endpoint=False)
    theta = jnp.pad(theta, (0, 1), constant_values=np.pi - 1e-8)
    z, z_mask = _images_point_source_sequential(
        rho * jnp.exp(1j * theta) + w0, nlenses=nlenses, roots_itmax=roots_itmax, **params
    )
    z_parity = jnp.sign(lens_eq_det_jac(z, nlenses=nlenses, **params))

    # Refine sampling by adding npts_init additional points a fraction
    # 1 / niter at a time
    npts_additional = int(0.5 * npts)
    n = int(npts_additional / niter)

    for i in range(niter):
        # Find the indices (along the contour axis) with the biggest distance
        # gap for consecutive points under the condition that at least
        # one of the two points is a real image
        delta_z = jnp.abs(z[:, 1:] - z[:, :-1])
        delta_z = jnp.where(
            jnp.logical_or(z_mask[:, 1:], z_mask[:, :-1]),
            delta_z,
            jnp.zeros_like(delta_z.real),
        )
        delta_z_max = jnp.max(delta_z, axis=0) # max over images at each phi_n
        idcs_theta = jnp.argsort(delta_z_max)[::-1][:n]

        # Add new points at the midpoints of the top-ranking intervals
        theta_new = 0.5 * (theta[idcs_theta] + theta[idcs_theta + 1])
        z_new, z_mask_new, z_parity_new = fn(theta_new, z[:, idcs_theta])

        # Insert new points
        theta = jnp.insert(theta, idcs_theta + 1, theta_new, axis=0)
        z = jnp.insert(z, idcs_theta + 1, z_new, axis=1)
        z_mask = jnp.insert(z_mask, idcs_theta + 1, z_mask_new, axis=1)
        z_parity = jnp.insert(z_parity, idcs_theta + 1, z_parity_new, axis=1)

    # Get rid of duplicate values of images which may occur in rare cases
    # by adding a very small random perturbation to the images
    _, c = jnp.unique(z.reshape(-1), return_counts=True, size=len(z.reshape(-1)))
    c = c.reshape(z.shape)
    mask_dup = c > 1
    z = jnp.where(
        mask_dup,
        z + random.uniform(key, shape=z.shape, minval=1e-9, maxval=1e-9),
        z,
    )

    # Permute images
    z, z_mask, z_parity = _permute_images(z, z_mask, z_parity)

    return z, z_mask, z_parity


@partial(jit, static_argnames=("n_parts"))
def _split_single_segment(segment, n_parts=5):
    """
    Split a single contour segment with shape `(2, npts)` which has at most
    `n_parts` parts seperated by zeros, split it into `n_parts` segments
    such that each segment is contiguous.
    """
    npts = segment.shape[1]

    # adapted from https://stackoverflow.com/questions/43385877/efficient-numpy-subarrays-extraction-from-a-mask
    def separate_regions(m):
        m0 = jnp.concatenate([jnp.array([False]), m, jnp.array([False])])
        idcs = jnp.flatnonzero(m0[1:] != m0[:-1], size=2 * n_parts)
        return idcs[::2], idcs[1::2]

    mask_nonzero = jnp.abs(segment[0].real) > 0.0

    # Split into n_parts parts
    idcs_start, idcs_end = separate_regions(mask_nonzero)

    n = jnp.arange(npts)

    def mask_region(carry, xs):
        l, r = xs
        mask = jnp.where(
            (l == 0.0) & (r == 0.0),  # check if region is empty
            jnp.zeros(npts),
            (n >= l) & (n < r),
        )
        return 0, mask

    _, masks = lax.scan(mask_region, 0, (idcs_start, idcs_end))
    segments_split = vmap(lambda mask: segment * mask)(masks)
    return segments_split


@partial(jit, static_argnames=("nr_of_segments"))
def _process_segments(segments, nr_of_segments=20):
    """
    Process raw contour segments such that each segment is contigous (meaning
    that there are no gaps between the segments head and tail) and that the head
    of the segment is at the 0th index in the array. The resultant number of
    segments is in general greater than the number of images and is at most
    equal to `nr_of_segments`. Segments with fewer than 3 points are removed.
    """
    # Split segments
    segments = jnp.concatenate(vmap(_split_single_segment)(segments))

    # If a segment consists of less than 3 points, set it to zero
    mask = jnp.sum((segments[:, 0] != 0 + 0j).astype(int), axis=1) < 3
    segments = segments * (~mask[:, None, None])

    # Sort segments such that the nonempty segments appear first and shrink array
    sorted_idcs = jnp.argsort(jnp.any(jnp.abs(segments[:, 0, :]) > 0.0, axis=1))[::-1]
    segments = segments[sorted_idcs]
    segments = segments[:nr_of_segments, :, :]

    # Find head of each segment
    head_idcs = vmap(first_nonzero)(jnp.abs(segments[:, 0, :]))

    # Roll each segment such that head is at idx 0
    segments = vmap(lambda segment, head_idx: jnp.roll(segment, -head_idx, axis=-1))(
        segments, head_idcs
    )

    return segments


@partial(jit, static_argnames=("nlenses"))
def _get_segments(z, z_mask, z_parity, nlenses=2):
    """
    Given the raw images corresponding to a sequence of points on the source
    limb, return two arrays with `open` and `closed` contour segments.
    Closed segments are those images which do not cross the critical curve
    and do not require any extra processing. Open segments need to be stiched
    together to form closed contours.

    WARNING: The number of these open segments is variable and I assume that
        there are at most 2*n_images open segments. This is a heuristic which
        may potentially fail.

    Args:
        z (array_like): Images corresponding to points on the source limb.
        z_mask (array_like): Mask which indicates which images are real.
        z_parity (array_like): Parity of each image.
        nlenses (int, optional): _description_. Defaults to 2.

    Returns:
        tuple (array_like, array_like, bool): A tuple with the following
        elements:
           array_like: Array with shape (n_images, 2, npts) containing  the
            closed segments.
           array_like: Array with shape (2*n_images, 2, npts) containing  the
            open segments.
           bool: A boolean indicating if the last array is empty (source does
            not cross the critical curve).
    """
    n_images = nlenses**2 + 1
    nr_of_segments = 2 * n_images

    # Apply mask and bundle the images and parity into a single array
    z = z * z_mask
    z_parity = z_parity * z_mask
    segments = jnp.stack([z, z_parity])
    segments = jnp.moveaxis(segments, 0, 1)

    # Split segments into segments which are already closed contours (they don't
    # cross the critical curve) and open segments
    mask_closed = (jnp.abs((z[:, 0] - z[:, -1])) < 1e-5) & jnp.all(z_mask, axis=1)

    segments_closed = segments * mask_closed[:, None, None]
    segments_open = segments * (~mask_closed[:, None, None])

    # If there are open segments split them such that each row is a single
    # contigous segment. This expands the size of total number of segments_open
    # to `nr_of_segments`.
    all_closed = jnp.all(mask_closed)
    segments_open = lax.cond(
        all_closed,
        lambda s: jnp.pad(s, ((0, nr_of_segments - s.shape[0]), (0, 0), (0, 0))),
        lambda s: _process_segments(s, nr_of_segments=nr_of_segments),
        segments_open,
    )

    # Set open segments to NaN if the pointwise parity for each segment is not
    # consistent TODO: think about how to avoid this issue entirely
#    cond_parity = jnp.all(
#        jnp.all(segments_open[:, 1, :].real >= 0.0, axis=1)
#        | jnp.all(segments_open[:, 1, :].real <= 0, axis=1),
#    )
#    segments_open = lax.cond(
#        cond_parity,
#        lambda: segments_open,
#        lambda: segments_open * jnp.nan,
#    )

    return segments_closed, segments_open, all_closed


@jit
def _concatenate_segments(segment_first, segment_second):
    segment_first_length = first_zero(jnp.abs(segment_first[0]))
    return segment_first + jnp.roll(segment_second, segment_first_length, axis=-1)


@jit
def _get_segment_length(segment, tail_idx):
    """Get the physical length of a segment."""
    diff = jnp.diff(segment[0])
    diff = diff.at[tail_idx].set(0.0)
    return jnp.abs(diff).sum()


@partial(jit, static_argnames=("min_dist", "max_dist"))
def _connection_condition(
    seg1, seg2, tidx1, tidx2, ctype, min_dist=1e-05, max_dist=1e-01, max_ang=60.0
):
    """
    Dermine wether two segments should be connected or not for a specific
    type of connection. We differentiate between four types of connections:
       - `ctype` == 0: Tail-Head connection
       - `ctype` == 1: Head-Tail connection
       - `ctype` == 2: Head-Head connection
       - `ctype` == 3: Tail-Tail connection

    We use four criterions to determine if the segments should be connected:
        1. For T-H and H-T the two segments need to have the same parity and
            for H-H and T-T connections they need to have opposite parity.
        2. If we form a line consisting of two points at the end of each segment,
            such that the second point is the connection point, the distance
            between two potential connection points of the segments must be
            less than the distance between the other two points.
        3. The smaller of the two angles formed by the intersection of `line1`
            and `line2` must be less than `max_ang` degrees.
        4. The distance between two potential connection points must be less
            than `max_dist`.

    If the distance between the two connection points is less than `min_dist`,
    and the parity condition is satisfied the function returns `True`
    irrespective of the other conditions.

    All of this is to ensure that we avoid connecting two segments which
    shouldn't be connected.

    Args:
        seg1 (array_like): First segment.
        seg2 (array_like): Second segment.
        tidx1 (int): Index specifying the tail of the first segment.
        tidx2 (int): Index specifying the tail of the second segment.
        ctype (int): Type of connection. 0 = T-H, 1 = H-T, 2 = H-H, 3 = T-T.
        min_dist (float, optional): If the distance between the connection points
            of the two segments is less than this value and the parity
            condition evaluates to `True`, the segments are connected
            irrespective of other conditions.
        max_dist (float, optional): Maximum distance between the ends of the
            two segments. If the distance is greater than this value function
            return `False`. Defaults to 1e-01.
        max_ang (float, optional): Angle in degrees. See criterion 2 above.

    Returns:
        bool: True if the two segments should be connected with a `ctype`
            connection.
    """

    def get_segment_head(seg):
        # Sometimes two points at end or beginning are nearly identical, need to
        # avoid those situations by picking the next point
        x = seg[0]
        cond = jnp.abs(x[1] - x[0]) > 1e-5
        line = lax.cond(
            cond,
            lambda: x[:2],
            lambda: x[1:3],
        )
        return line[::-1]

    def get_segment_tail(seg, t):
        x = seg[0]
        cond = jnp.abs(x[t] - x[t - 1]) > 1e-5
        line = lax.cond(
            cond,
            lambda: lax.dynamic_slice(x, (t - 1,), (2,)),
            lambda: lax.dynamic_slice(x, (t - 2,), (2,)),
        )
        return line

    # Evaluate parity condition
    same_parity = seg1[1, 0].real * seg2[1, 0].real > 0.0
    conds_parity = jnp.stack(
        [
            same_parity,
            same_parity,
            ~same_parity,
            ~same_parity,
        ]
    )
    cond_parity = conds_parity[ctype]

    # Evaluate conditions 2-4
    line1, line2 = lax.switch(
        ctype,
        [
            lambda s1, s2, t1, t2: (get_segment_tail(s1, t1), get_segment_head(s2)),
            lambda s1, s2, t1, t2: (get_segment_head(s1), get_segment_tail(s2, t2)),
            lambda s1, s2, t1, t2: (get_segment_head(s1), get_segment_head(s2)),
            lambda s1, s2, t1, t2: (
                get_segment_tail(s1, tidx1),
                get_segment_tail(s2, t2),
            ),
        ],
        seg1,
        seg2,
        tidx1,
        tidx2,
    )

    dist = jnp.abs(line1[1] - line2[1])
    cond1 = dist < max_dist

    # Angle between two vectors
    vec1 = (line1[1] - line1[0]) / jnp.abs(line1[1] - line1[0])
    vec2 = (line2[1] - line2[0]) / jnp.abs(line2[1] - line2[0])
    alpha = jnp.arccos(
        jnp.real(vec1) * jnp.real(vec2) + jnp.imag(vec1) * jnp.imag(vec2)
    )
    cond2 = (180.0 - jnp.rad2deg(alpha)) < max_ang

    # Distance between endpoints of line1 and line2 has to be smaller than the
    # distance between points where the line begins
    cond3 = jnp.abs(line1[1] - line2[1]) < jnp.abs(line1[0] - line2[0])

    cond_geom = jnp.logical_or(cond1 & cond2 & cond3, dist < min_dist)

    return cond_parity & cond_geom


@jit
def _merge_two_segments(seg1, seg2, tidx1, tidx2, ctype):
    """
    Merge two segments into one assuming that the length of the merged
    segments is equal to at most the shape of `seg1.shape[1]`.

    Args:
        seg1 (array_like): First segment.
        seg2 (array_like): Second segment.
        tidx1 (int): Index specifying the tail of the first segment.
        tidx2 (int): Index specifying the tail of the second segment.
        ctype (int): Type of connection. 0 = T-H, 1 = H-T, 2 = H-H, 3 = T-T.

    Returns
        (array_like, int): The merged segment and the index of the tail of
            the merged segment.
    """

    def hh_connection(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1]  # flip
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg2 = seg2.at[1].set(-1 * seg2[1])
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2 + 1

    def tt_connection(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1]  # flip
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg2 = seg2.at[1].set(-1 * seg2[1])
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2 + 1

    def th_connection(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2 + 1

    def ht_connection(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2 + 1

    seg_merged, tidx_merged = lax.switch(
        ctype,
        [th_connection, ht_connection, hh_connection, tt_connection],
        seg1,
        seg2,
        tidx1,
        tidx2,
    )
    return seg_merged, tidx_merged


@partial(
    jit,
    static_argnames=("max_nr_of_contours", "max_nr_of_segments_in_contour"),
)
def _merge_open_segments(
    segments,
    max_nr_of_contours=2,
    max_nr_of_segments_in_contour=10,
):
    """
    Sequentially merge open segments until they form a closed contour. The
    merging algorithm is as follows:

        1. Select shortest (in length) open segment, set it as the active segment.
        2. Find segments closest to current segment in (H-H, T-T, T-H and H-T),
            select 4 of the closest segments.
        3. Evaluate the `_connection_condition` for the active segment and each
            of the 4 closest segments. Pick the segment for which the condition
            is True and the distance is minimized.
        4. Merge the active segment with the selected segment.
        5. Repeat steps 2-4 until the active segment becomes a closed contour.
            This will happen when there are no more open segments which satisfy
            the connection condition.

    The whole process is repeated `max_nr_of_contours` times.

    Args:
        segments (array_like): Array containing the segments, shape
            (`n_segments`, 2, `n_points`).
        segments_mask (array_like): Mask indicating which segments are closed to
            begin with. Shape (`n_segments`).
        max_nr_of_contours (int): Maximum number of contours comprised of the
            input segments.
        max_nr_of_segments_in_contour (int): Maximum number of segments for one
            contour. Should be at least 12 in case of triple lensing. Default is
            10.
        max_dist (float): Maximum distance between two segment end points that
            are allowed to be connected.

    Returns:
        array_like: Array with shape (`n_segments`, 2, `n_points`) containing
        the merged segments contours.
    """

    def merge_with_another_segment(seg_active, tidx_active, segments, tidcs):
        # Compute all T-H, H-T, H-H, T-T distances between the current segment
        # and all other open segments
        dist_th = jnp.abs(seg_active[0, tidx_active] - segments[:, 0, 0])
        dist_ht = vmap(lambda seg, tidx: jnp.abs(seg_active[0, 0] - seg[0, tidx]))(
            segments, tidcs
        )
        dist_hh = jnp.abs(seg_active[0, 0] - segments[:, 0, 0])
        dist_tt = vmap(
            lambda seg, tidx: jnp.abs(seg_active[0, tidx_active] - seg[0, tidx])
        )(segments, tidcs)

        distances = jnp.stack([dist_th, dist_ht, dist_hh, dist_tt])

        # Get the connection type and the index of the "closest" segment, the
        # second closest segment and so on
        ctype1, idx1 = jnp.unravel_index(
            jnp.argsort(distances.reshape(-1))[0], distances.shape
        )
        ctype2, idx2 = jnp.unravel_index(
            jnp.argsort(distances.reshape(-1))[1], distances.shape
        )
        ctype3, idx3 = jnp.unravel_index(
            jnp.argsort(distances.reshape(-1))[2], distances.shape
        )
        ctype4, idx4 = jnp.unravel_index(
            jnp.argsort(distances.reshape(-1))[3], distances.shape
        )

        # Evaluate the connection conditions for each of the four closest segments
        success1 = _connection_condition(
            seg_active, segments[idx1], tidx_active, tidcs[idx1], ctype1
        )
        success2 = _connection_condition(
            seg_active, segments[idx2], tidx_active, tidcs[idx2], ctype2
        )
        success3 = _connection_condition(
            seg_active, segments[idx3], tidx_active, tidcs[idx3], ctype3
        )
        success4 = _connection_condition(
            seg_active, segments[idx4], tidx_active, tidcs[idx4], ctype4
        )

        def branch1(segments, tidcs):
            # Select the closest segment that satisfies the connection condition
            idx_best = first_nonzero(
                jnp.array([success1, success2, success3, success4]).astype(float)
            )
            seg_best = jnp.stack(
                [segments[idx1], segments[idx2], segments[idx3], segments[idx4]]
            )[idx_best]
            tidx_best = jnp.stack([tidcs[idx1], tidcs[idx2], tidcs[idx3], tidcs[idx4]])[
                idx_best
            ]
            ctype = jnp.stack([ctype1, ctype2, ctype3, ctype4])[idx_best]

            # Merge that segment with the active segment
            seg_active_new, tidx_active_new = _merge_two_segments(
                seg_active,
                seg_best,
                tidx_active,
                tidx_best,
                ctype,
            )

            # Zero-out the segment that was merged
            idx_seg = jnp.array([idx1, idx2, idx3, idx4])[idx_best]
            segments = segments.at[idx_seg].set(
                jnp.zeros_like(segments[0]), segments[idx1]
            )
            return seg_active_new, tidx_active_new, segments, tidcs

        def branch2(segments, tidcs):
            return seg_active, tidx_active, segments, tidcs

        return lax.cond(
            jnp.any(jnp.array([success1, success2, success3, success4])),
            branch1,
            branch2,
            segments,tidcs
        )


    def body_fn(carry, _):
        # Get the active segment, the index of its tail and all the other
        # open segments and their tail indices from previous iteration
        seg_active, tidx_active, segments, tidcs = carry

        # If all segments are empty don't do anything
        stopping_criterion = ~jnp.any(segments[:, 0, 0])
        seg_active, tidx_active, segments, tidcs = lax.cond(
            stopping_criterion,
            lambda: (seg_active, tidx_active, segments, tidcs),
            lambda: merge_with_another_segment(
                seg_active, tidx_active, segments, tidcs
            ),
        )

        return (
            seg_active,
            tidx_active,
            segments,
            tidcs,
        ), 0.0

    # Compute tail index for each segment
    tail_idcs = vmap(last_nonzero)(segments[:, 0, :].real)

    # Pad segments with zeros to make room for merged segments as this array
    # be modified in place
    # WARNING: this will fail silently if a contour is longer than 3*segments.shape[-1]
    segments = jnp.pad(
        segments, ((0, 0), (0, 0), (0, 3 * segments.shape[-1])), constant_values=0.0
    )

    # Merge all open segments: start by selecting 0th open segment and then
    # sequentially merging it with other ones until a closed segment is formed
    segments_merged_list = []

    for i in range(max_nr_of_contours):
        # Sort open segments such that the shortest segments appear first
        segment_lengths = vmap(_get_segment_length)(segments, tail_idcs)
        _idcs = sparse_argsort(segment_lengths)
        segments, tail_idcs = segments[_idcs], tail_idcs[_idcs]

        idcs = jnp.arange(0, max_nr_of_segments_in_contour)
        init = (segments[0], tail_idcs[0], segments[1:], tail_idcs[1:])
        carry, _ = lax.scan(body_fn, init, idcs)
        seg_merged, tidx_merged, segments, tail_idcs = carry
        segments_merged_list.append(seg_merged)
        max_nr_of_segments_in_contour -= 2

    return jnp.stack(segments_merged_list)


@jit
def _contours_from_closed_segments(segments):
    """
    Process closed segments by extracting the parity information and adding a
    single point at the end so that the last point is the same as the first.
    """
    # Extract per contour parity values
    contours_p = segments[:, 1, 0].real

    # Remove parity dimension
    contours = segments[:, 0]

    # Set the last point in each contour to be equal to the first point
    contours = jnp.hstack([contours, contours[:, 0][:, None]])

    return contours, contours_p


@partial(
    jit,
    static_argnames=("max_nr_of_contours", "max_nr_of_segments_in_contour"),
)
def _contours_from_open_segments(
    segments,
    max_nr_of_contours=2,
    max_nr_of_segments_in_contour=10,
):
    """
    Given a set of open image segments, some of which may be open, return
    closed contours which are obtained by merging the open segments together.

    Args:
        segments (array_like): Array of shape `(n_segments, 2, n_points)`
            containing the segment points and the parity for each point.
        n_contours (int, optional): Final number of contours. Defaults to 5.
        max_nr_of_segments_in_contour (int): Maximum number of segments for a
            closed contour. Should be at least 12 in case of triple lensing.
            Default is 10.

    Returns:
        tuple: A tuple of (contours, parity) where `contours` is an array with
        shape `(n_contours, n_points)` and parity is an array of shape `n_contours`
        indicating the parity of each contour.
    """

    # Run the merging algorithm
    segments_merged = _merge_open_segments(
        segments,
        max_nr_of_contours=max_nr_of_contours,
        max_nr_of_segments_in_contour=max_nr_of_segments_in_contour,
    )

    # Extract per contour parity values
    contours_p = segments_merged[:, 1, 0].real

    # Remove the parity dimension
    contours = segments_merged[:, 0]

    # Set the last point in each contour to be equal to the first point
    tail_idcs = vmap(last_nonzero)(contours.real)
    contours = vmap(lambda idx, c: c.at[idx + 1].set(c[0]))(tail_idcs, contours)

    return contours, contours_p


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
def mag_extended_source(
    w0,
    rho,
    nlenses=2,
    npts_limb=150,
    limb_darkening=False,
    u1=0.0,
    npts_ld=100,
    roots_itmax=2500,
    roots_compensated=False,
    **params,
):
    """
    Compute the magnification of an extended source with radius `rho` for a
    system with `nlenses` lenses.

    Args:
        w0 (complex): Source position in the complex plane.
        rho (float): Source radius in Einstein radii.
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
            function is evaluated when computing the integrals P and Q from
            Dominik 1998. Defaults to 100.
        **a (float): Half the separation between the first two lenses located on
            the real line with $r_1 = a$ and $r_2 = -a$.
        **r3 (float): The position of the third lens at arbitrary location in
            the complex plane.
        **e1 (array_like): Mass fraction of the first lens located at $r_1=a$.
        **e2 (array_like): Mass fraction of the second lens located at $r_2=-a$.
        **roots_itmax (int, optional): Number of iterations for the root solver.
        **roots_compensated (bool, optional): Whether to use the compensated
            arithmetic version of the Ehrlich-Aberth root solver.

    Returns:
        float: Total magnification.
    """
    # Get ordered point source images at the source limb
    z, z_mask, z_parity = _images_of_source_limb(
        w0,
        rho,
        nlenses=nlenses,
        npts=npts_limb,
        roots_itmax=roots_itmax,
        roots_compensated=roots_compensated,
        **params,
    )

    # Integration function depending on whether the source is limb-darkened
    if limb_darkening:
        integrate = lambda contour, tidx: _integrate_ld(
            contour,
            tidx,
            w0,
            rho,
            u1=u1,
            nlenses=nlenses,
            npts=npts_ld,
            **params,
        )

    else:
        integrate = lambda contour, tidx: _integrate_unif(contour, tidx)

    # For N = 1 the contours are trivially to obtain
    if nlenses == 1:
        contours, contours_p = _contours_from_closed_segments(
            jnp.moveaxis(jnp.stack([z, z_parity]), 0, 1)
        )
        tail_idcs = jnp.array([z.shape[1] - 1, z.shape[1] - 1])
        I = vmap(_integrate_unif)(contours, tail_idcs)
        return jnp.abs(jnp.sum(I * contours_p)) / (np.pi * rho**2)

    # For N = 2 we first have to obtain segments and then convert those to
    # closed contours
    elif (nlenses == 2) or (nlenses == 3):
        max_nr_of_contours = 2
        # Get segments. If `all_closed` is True there are no caustic crossings
        # and everything is easy
        segments_closed, segments_open, all_closed = _get_segments(
            z, z_mask, z_parity, nlenses=nlenses
        )

        # Get contours from closed segments
        contours1, contours_p1 = _contours_from_closed_segments(segments_closed)

        # Integrate over contours obtained from closed segments
        tail_idcs = jnp.repeat(contours1.shape[1] - 1, contours1.shape[0])
        I1 = vmap(integrate)(contours1, tail_idcs)
        mags1 = I1 * contours_p1

        # If there are caustic crossings things are a lot more complicated and
        # we have to stitch together the open segments to form closed contours.
        branch1 = lambda _: jnp.zeros(max_nr_of_contours)

        def branch2(segments):
            contours, contours_p = _contours_from_open_segments(
                segments, max_nr_of_contours=max_nr_of_contours
            )
            tail_idcs = vmap(last_nonzero)(contours.real)
            I = vmap(integrate)(contours, tail_idcs)
            return I * contours_p

        mags2 = lax.cond(all_closed, branch1, branch2, segments_open)
        mag = jnp.abs(mags1.sum() + mags2.sum()) / (np.pi * rho**2)

        return mag

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")
