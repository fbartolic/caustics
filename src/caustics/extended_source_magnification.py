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
from .utils import *

from .point_source_magnification import (
    lens_eq,
    lens_eq_det_jac,
)


@partial(
    jit,
    static_argnames=(
        "nlenses",
        "npts_init",
        "niter",
        "roots_itmax",
        "roots_compensated",
    ),
)
def _images_of_source_limb(
    w_center,
    rho,
    nlenses=2,
    npts_init=500,
    niter=2,
    roots_itmax=2500,
    roots_compensated=False,
    **params,
):
    # Initial sampling on the source limb
    theta = jnp.linspace(-np.pi, np.pi, npts_init - 1, endpoint=False)
    theta = jnp.pad(theta, (0, 1), constant_values=np.pi - 1e-05)
    images, mask_images = images_point_source(
        rho * jnp.exp(1j * theta) + w_center,
        nlenses=nlenses,
        roots_itmax=roots_itmax,
        roots_compensated=roots_compensated,
        **params,
    )
    det = lens_eq_det_jac(images, nlenses=nlenses, **params)
    parity = jnp.sign(det)
    mag = jnp.sum((1.0 / jnp.abs(det)) * mask_images, axis=0)

    # Refine sampling by placing geometrically fewer points each iteration
    # in the regions where the magnification gradient is largest
    npts_list = np.geomspace(2, npts_init, niter, endpoint=False, dtype=int)[::-1]
    key = random.PRNGKey(42)

    for _npts in npts_list:
        # Resample theta
        delta_mag = jnp.gradient(mag)
        idcs_maxdelta = jnp.argsort(jnp.abs(delta_mag))[::-1][:_npts]

        theta_patch = 0.5 * (theta[idcs_maxdelta] + theta[idcs_maxdelta + 1])

        # Make sure that there are no exact duplicate values in `theta_patch`
        # and no common values with `theta`
        mask_duplicate = jnp.ones(len(theta_patch), dtype=bool)
        mask_duplicate = mask_duplicate.at[
            jnp.unique(theta_patch, return_index=True, size=len(theta_patch))[1]
        ].set(False)
        mask_common = jnp.isin(theta_patch, theta, assume_unique=True)
        mask = jnp.logical_or(mask_duplicate, mask_common)
        theta_patch += mask * random.uniform(
            key, theta_patch.shape, maxval=1e-05
        )  # small perturbation

        images_patch, mask_images_patch = images_point_source(
            rho * jnp.exp(1j * theta_patch) + w_center,
            nlenses=nlenses,
            roots_itmax=roots_itmax,
            roots_compensated=roots_compensated,
            **params,
        )
        det_patch = lens_eq_det_jac(images_patch, nlenses=nlenses, **params)
        mag_patch = jnp.sum((1.0 / jnp.abs(det_patch)) * mask_images_patch, axis=0)

        theta = jnp.concatenate([theta, theta_patch])
        sorted_idcs = jnp.argsort(theta)
        theta = theta[sorted_idcs]

        mag = jnp.concatenate([mag, mag_patch])[sorted_idcs]
        images = jnp.hstack([images, images_patch])[:, sorted_idcs]
        mask_images = jnp.hstack([mask_images, mask_images_patch])[:, sorted_idcs]
        det = jnp.hstack([det, det_patch])[:, sorted_idcs]
        parity = jnp.sign(det)

    return images, mask_images, parity


@partial(jit, backend="cpu")
def _linear_sum_assignment(a, b):
    """
    Given 1D arrays a and b, return the indices which specify the permutation of
    b for which the element-wise distance between the two arrays is minimized.

    For an alternative solution to this problem see the Hungarian algorithm
    and similar problems in optimal transport.

    Args:
        a (array_like): 1D array.
        b (array_like): 1D array.

    Returns:
        array_like: Indices which specify the desired permutation of b.
    """
    # This is the first guess for a solution but sometimes we get duplicate
    # indices so for those values we need to choose the next best best
    # solution.
    idcs_initial = jnp.argsort(jnp.abs(b - a[:, None]), axis=1)
    idcs_final = jnp.repeat(999, len(a))

    # Make sure that each index is assigned to at most one value
    def body_fn(carry, idcs_initial_row):
        i, idcs_final = carry

        conds = ~jnp.isin(idcs_initial_row, idcs_final)
        branches = [lambda j=j: idcs_initial_row[j] for j in range(len(a))]

        idx_closest = lax.switch(
            first_nonzero(conds),
            branches,
        )

        idcs_final = idcs_final.at[i].set(idx_closest)
        return (i + 1, idcs_final), idx_closest

    _, res = lax.scan(body_fn, (0, idcs_final), idcs_initial)

    return res


@partial(jit, backend="cpu")  # this function has to be executed on the CPU
def _permute_images(images, mask_solutions, parity):
    """
    Sequantially permute the images corresponding to points on the source limb
    starting with the first point such that each point source image is assigned
    to the correct curve. This procedure does not differentiate between real
    and false images, false images are set to zero after the permutation
    operation.
    """
    segments = jnp.stack([images, mask_solutions, parity])

    def apply_linear_sum_assignment(carry, array_slice):
        z, mask_sols, parity = array_slice
        idcs = _linear_sum_assignment(carry, z)
        return z[idcs], jnp.stack([z[idcs], mask_sols[idcs], parity[idcs]])

    init = segments[0, :, 0]
    _, segments = lax.scan(
        apply_linear_sum_assignment, init, jnp.moveaxis(segments, -1, 0)
    )
    z, mask_sols, parity = jnp.moveaxis(segments, 1, 0)

    return z, mask_sols, parity


@partial(jit, static_argnames=("n_parts"))
def _split_single_segment(segment, n_parts=5):
    """
    Split a single contour segment with shape `(2, npts)` which has at most
    `n_parts` parts seperated by zeros, split it into `n_parts` segments
    such that each segment is contiguous.
    """
    npts = segment.shape[1]
    z = segment[0]
    z_diff = jnp.diff((jnp.abs(z) > 0.0).astype(float), prepend=0, append=0)
    z_diff = z_diff.at[-2].set(z_diff[-1])
    z_diff = z_diff[:-1]

    left_edges = jnp.nonzero(z_diff > 0.0, size=n_parts)[0]
    right_edges = jnp.nonzero(z_diff < 0.0, size=n_parts)[0]

    # Split into n_parts parts
    n = jnp.arange(npts)

    def body_fn(carry, xs):
        l, r = xs
        mask = jnp.where((l == 0.0) & (r == 0.0), jnp.zeros(npts), (n >= l) & (n <= r))
        return 0, mask

    _, masks = lax.scan(body_fn, 0, (left_edges, right_edges))

    segments_split = vmap(lambda mask: segment * mask)(masks)

    return segments_split


@partial(jit, static_argnames=("nr_of_segments"))
def _process_segments(segments, nr_of_segments=20):
    """
    Process raw contour segments such that each segment is contigous (meaning
    that there are no gaps between the segments head and tail) and that the head
    of the segment is at the 0th index in the array. The resultant number of
    segments is in general greater than the number of images and is at most
    equal to `nr_of_segments`.
    """
    # Split segments
    segments = jnp.concatenate(vmap(_split_single_segment)(segments))

    # If a segment consists of a single point, set it to zero
    mask_onepoint_segments = (
        jnp.sum((segments[:, 0] != 0 + 0j).astype(int), axis=1) == 1
    )
    segments = segments * (~mask_onepoint_segments[:, None, None])

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
def _get_segments(images, images_mask, images_parity, nlenses=2):
    """
    Given the raw images corresponding to a sequence of points on the source
    limb, return a collection of contour segments some open and some closed.

    Args:
        images (array_like): Images corresponding to points on the source limb.
        images_mask (array_like): Mask which indicated which images are real.
        images_parity (array_like): Parity of each image.
        nlenses (int, optional): _description_. Defaults to 2.

    Returns:
        tuple: (array_like, bool) Array containing the segments and
        a boolean variable indicating whether all segments are closed.
        The "head" of each segment starts at index 0 and the "tail" index is
        variable depending on the segment.
    """
    n_images = nlenses**2 + 1
    nr_of_segments = 2 * n_images

    # Untangle the images
    images, images_mask, parity = _permute_images(images, images_mask, images_parity)

    # Apply mask and bundle the images and parity into a single array
    images = (images * images_mask).T
    parity = (parity * images_mask).T
    segments = jnp.stack([images, parity])
    segments = jnp.moveaxis(segments, 0, 1)

    # Expand size of the array to make room for splitted segments in case there
    # are critical curve crossings
    segments = jnp.pad(
        segments, ((0, nr_of_segments - segments.shape[0]), (0, 0), (0, 0))
    )

    # Check if all segments are already closed, if not, do
    # extra processing to obtain splitted segments
    cond1 = jnp.all(segments[:, 0, :] != 0 + 0j, axis=1)
    cond2 = jnp.abs(segments[:, 0, 0] - segments[:, 0, -1]) < 1e-05
    all_closed = jnp.all(jnp.logical_and(cond1, cond2))

    segments = lax.cond(
        all_closed,
        lambda _: segments,
        lambda s: _process_segments(s, nr_of_segments=nr_of_segments),
        segments,
    )

    return segments, all_closed


@jit
def _concatenate_segments(segment_first, segment_second):
    segment_first_length = first_zero(jnp.abs(segment_first[0]))
    return segment_first + jnp.roll(segment_second, segment_first_length, axis=-1)


@partial(jit, static_argnames=("max_dist"))
def _connection_condition(line1, line2, max_dist=1e-01):
    """
    Dermine weather two segments should be connected or not.

    The input to the function are two arrays `line1` and `line2` consisting of
    the last (or first) two points of points of contour segments. `line1` and
    `line2` each consist of a starting point and an endpoint (in that order).
    We use four criterions to determine if the segments should be connected:
        1. Distance between the endpoints of `line1` and `line2` is less than
        `max_dist`.
        2. The smaller of the two angles formed by the intersection of `line1`
            and `line2` is less than 60 degrees.
        3. The distance between the point of intersection of `line1` and `line2`
            and each of the endpoints of `line1` and `line2` is less than
            `max_dist`.
        4. Distance between ending points of `line1` and `line2` is less than
            the distance between start points.

    If the distance between the two endpoints is less than 1e-05, the output
    is `True` irrespective of the other conditions.

    Args:
        line1(array_like): Size 2 array containing two points in the complex
            plane where the second point is the end-point.
        line1(array_like): Size 2 array containing two points in the complex
            plane where the second point is the end-point.
        max_dist (float, optional): Maximum distance between the ends of the
            two segments. If the distance is greater than this value function
            return `False`. Defaults to 1e-01.

    Returns:
        bool: True if the two segments should be connected.
    """
    x1, y1 = jnp.real(line1[0]), jnp.imag(line1[0])
    x2, y2 = jnp.real(line1[1]), jnp.imag(line1[1])
    x3, y3 = jnp.real(line2[0]), jnp.imag(line2[0])
    x4, y4 = jnp.real(line2[1]), jnp.imag(line2[1])

    dist = jnp.abs(line1[1] - line2[1])
    cond1 = dist < max_dist

    # Angle between two vectors
    vec1 = (line1[1] - line1[0]) / jnp.abs(line1[1] - line1[0])
    vec2 = (line2[1] - line2[0]) / jnp.abs(line2[1] - line2[0])
    alpha = jnp.arccos(
        jnp.real(vec1) * jnp.real(vec2) + jnp.imag(vec1) * jnp.imag(vec2)
    )
    cond2 = (180 - jnp.rad2deg(alpha)) < 60.0

    # Point of intersection point of the two lines
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / D
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / D
    p = px + 1j * py

    # Require that the intersection point is at most `max_dist` away from the
    # two endpoints
    cond3 = (jnp.abs(line1[1] - p) < max_dist) & (jnp.abs(line2[1] - p) < max_dist)

    # Distance between endpoints of line1 and line2 has to be smaller than the
    # distance between points where the line begins
    cond4 = jnp.abs(line1[1] - line2[1]) < jnp.abs(line1[0] - line2[0])

    return jnp.logical_or(cond1 & cond2 & cond3 & cond4, dist < 1e-05)


@partial(jit, static_argnames=("max_dist"))
def _merge_two_segments(seg1, seg2, tidx1, tidx2, ctype, max_dist=1e-01):
    """
    Given two segments seg1 and seg2, merge them if the condition for merging
    is satisfied, while keeping track of the parity of each segment.

    Args:
        seg1 (array_like): First segment.
        seg2 (array_like): Second segment.
        tidx1 (int): Index specifying the tail of the first segment.
        tidx2 (int): Index specifying the tail of the second segment.
        ctype (int): Type of connection. 0 = T-H, 1 = H-T, 2 = H-H, 3 = T-T.
        max_dist (float, optional): Maximum allowed distance between the ends
            of the the two segments.

    Returns:
        (array_like, int, bool): The merged segment, the index of the tail of
            the merged segment, and a boolean indicating if the merging was
            sucessful. If the merging condition is not satisfied the function
            returns (`seg1, tidx1`, `False`).
    """

    def hh_connection(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1]  # flip
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg1_parity = jnp.sign(seg1[1].sum())
        # seg2 = seg2.at[1].set(-1*seg2[1])
        seg2 = index_update(seg2, 1, -1 * seg2[1])
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2 + 1, True

    def tt_connection(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1]  # flip
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg1_parity = jnp.sign(seg1[1].sum())
        # seg2 = seg2.at[1].set(-1*seg2[1])
        seg2 = index_update(seg2, 1, -1 * seg2[1])
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2 + 1, True

    def th_connection(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2 + 1, True

    def ht_connection(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2 + 1, True

    def case1(seg1, seg2, tidx1, tidx2):
        cond = _connection_condition(
            lax.dynamic_slice_in_dim(seg1[0], tidx1 - 1, 2),
            lax.dynamic_slice_in_dim(seg2[0], 0, 2)[::-1],
            max_dist=max_dist,
        )
        return lax.cond(
            cond,
            th_connection,
            lambda *args: (seg1, tidx1, False),
            seg1,
            seg2,
            tidx1,
            tidx2,
        )

    def case2(seg1, seg2, tidx1, tidx2):
        cond = _connection_condition(
            lax.dynamic_slice_in_dim(seg2[0], tidx2 - 1, 2),
            lax.dynamic_slice_in_dim(seg1[0], 0, 2)[::-1],
            max_dist=max_dist,
        )
        return lax.cond(
            cond,
            ht_connection,
            lambda *args: (seg1, tidx1, False),
            seg1,
            seg2,
            tidx1,
            tidx2,
        )

    def case3(seg1, seg2, tidx1, tidx2):
        cond = _connection_condition(
            lax.dynamic_slice_in_dim(seg1[0], 0, 2)[::-1],
            lax.dynamic_slice_in_dim(seg2[0], 0, 2)[::-1],
            max_dist=max_dist,
        )
        return lax.cond(
            cond,
            hh_connection,
            lambda *args: (seg1, tidx1, False),
            seg1,
            seg2,
            tidx1,
            tidx2,
        )

    def case4(seg1, seg2, tidx1, tidx2):
        cond = _connection_condition(
            lax.dynamic_slice_in_dim(seg1[0], tidx1 - 1, 2),
            lax.dynamic_slice_in_dim(seg2[0], tidx2 - 1, 2),
            max_dist=max_dist,
        )
        return lax.cond(
            cond,
            tt_connection,
            lambda *args: (seg1, tidx1, False),
            seg1,
            seg2,
            tidx1,
            tidx2,
        )

    seg_merged, tidx_merged, success = lax.switch(
        ctype,
        [case1, case2, case3, case4],
        seg1,
        seg2,
        tidx1,
        tidx2,
    )

    return success, seg_merged, tidx_merged


@jit
def _get_segment_length(segment, tail_idx):
    """Get the physical length of a segment."""
    diff = jnp.diff(segment[0])
    diff = diff.at[tail_idx].set(0.0)
    return jnp.abs(diff).sum()


@partial(jit, static_argnames=("max_dist", "max_nr_of_segments_in_contour"))
def _merge_open_segments(
    segments, mask_closed, max_nr_of_segments_in_contour=10, max_dist=5e-02
):
    """
    Sequentially merge open segments until they form a closed contour. The
    merging algorithm is as follows:

        1. Pick random open segment, this is the current segment.
        2. Find segments closest to current segment in terms of distance (H-H,
        T-T, T-H and H-T), select 4 of the closest segments.
        3. Evaluate the `_connection_condition` function for the current segment
            and each of the 4 closest segments. Pick the segment for which the
            condition is True and the distance is minimized.
        4. Merge the current segment with the selected segment.
        5. Repeat steps 2-4 until the current segment becomes a closed contour.
            This will happen when there are no more open segments which satisfy
            the connection condition.

    Steps 1-5 are then repeated once more to account for the possibility of there
    being a second contour composed of disjoint segments.

    WARNING: This function is the most failure prone part of the whole method.
    Here are some examples of possible failure modes:
       - If too few points are used on the source star limb, there could be a
       situation where the distance between the endpoints of two segments which
       should be connected over a critical curve is greater than `max_dist`
       in which case `_connection_condition` evaluates to False. In this case
       the two segments won't be connected and we'll have an incomplete contour.
       The solution is to increase the number of points used to solve the lens
       equation on the source star limb.
       - If none of the 4 closest segment ends satisfy the connection condition
       and the contour is not closed. In this case the contour will be incomplete.
       - In extreme cases, there could be a situation where a connection is made
       between a segment belonging to one contour and a segment belonging to a
       different contour because the segment belonging to the other contour is
       closer in the distance than any of the segments from the first contour.
       There is no simple way to handle this situation besides modifying the
       `_connection_condition` function such that it somehow recognizes that
       these two segments shouldn't be connected.
       - The algorithm if there are more than two closed contours composed of
       disjoint segments. I haven't seen an example of this situation, even for
       triple lenses. It is trivial to fix by adding another iteration of the
       merging algorithm.

    Args:
        segments (array_like): Array containing the segments, shape
            (`n_segments`, 2, `n_points`).
        mask_closed (array_like): Mask indicating which segments are closed to
            begin with. Shape (`n_segments`).
        max_nr_of_segments_in_contour (int): Maximum number of segments for a
            closed contour. Should be at least 12 in case of triple lensing.
            Default is 10.
        max_dist (float): Maximum distance between two segment end points that
            are allowed to be connected.

    Returns:
        array_like: Array of shape (`n_segments`, 2, `n_points`) containing
        the closed contours.
    """
    # Compute tail index for each segment
    tail_idcs = vmap(last_nonzero)(jnp.abs(segments[:, 0, :]))

    # Split segments into closed and open ones
    closed_segments = mask_closed[:, None, None] * segments
    sorted_idcs = lambda segments: jnp.argsort(jnp.abs(segments[:, 0, 0]))[::-1]
    closed_segments = closed_segments[sorted_idcs(closed_segments)]  # sort
    open_segments = jnp.logical_not(mask_closed)[:, None, None] * segments

    # Sort open segments such that the shortest segments appear first
    segment_lengths = vmap(_get_segment_length)(open_segments, tail_idcs)
    _idcs = sparse_argsort(segment_lengths)
    open_segments, tail_idcs = open_segments[_idcs], tail_idcs[_idcs]

    # Merge all open segments: start by selecting 0th open segment and then
    # sequentially merge it with other ones until a closed segment is formed
    idcs = jnp.arange(0, max_nr_of_segments_in_contour)

    def body_fn(carry, idx_dummy):
        # Get the current segment, the index of its tail and all the other
        # open segments and their tail indices from previous iteration
        seg_current, tidx_current, open_segments, tidcs = carry

        # Compute all T-H, H-T, H-H, T-T distances between the current segment
        # and all other open segments
        dist_th = jnp.abs(seg_current[0, tidx_current] - open_segments[:, 0, 0])
        dist_ht = vmap(lambda seg, tidx: jnp.abs(seg_current[0, 0] - seg[0, tidx]))(
            open_segments, tidcs
        )
        dist_hh = jnp.abs(seg_current[0, 0] - open_segments[:, 0, 0])
        dist_tt = vmap(
            lambda seg, tidx: jnp.abs(seg_current[0, tidx_current] - seg[0, tidx])
        )(open_segments, tidcs)

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

        seg1, tidx1 = open_segments[idx1], tidcs[idx1]
        seg2, tidx2 = open_segments[idx2], tidcs[idx2]
        seg3, tidx3 = open_segments[idx3], tidcs[idx3]
        seg4, tidx4 = open_segments[idx4], tidcs[idx4]

        # First merge attempt
        success1, seg_current_new, tidx_current_new = lax.cond(
            jnp.all(seg1[0] == 0.0 + 0j),
            lambda: (False, seg_current, tidx_current),
            lambda: _merge_two_segments(
                seg_current, seg1, tidx_current, tidx1, ctype1, max_dist=max_dist
            ),
        )

        # Zero out the other segment if merge was successfull, otherwise do nothing
        open_segments = open_segments.at[idx1].set(
            jnp.where(success1, jnp.zeros_like(open_segments[0]), open_segments[idx1])
        )

        # Second merge attempt
        success2, seg_current_new, tidx_current_new = lax.cond(
            ~success1,
            lambda: _merge_two_segments(
                seg_current, seg2, tidx_current, tidx2, ctype2, max_dist=max_dist
            ),
            lambda: (False, seg_current_new, tidx_current_new),
        )

        open_segments = open_segments.at[idx2].set(
            jnp.where(
                ~success1 & success2,
                jnp.zeros_like(open_segments[0]),
                open_segments[idx2],
            )
        )

        # Third merge attempt
        success3, seg_current_new, tidx_current_new = lax.cond(
            ~success1 & ~success2,
            lambda: _merge_two_segments(
                seg_current, seg3, tidx_current, tidx3, ctype3, max_dist=max_dist
            ),
            lambda: (False, seg_current_new, tidx_current_new),
        )

        open_segments = open_segments.at[idx3].set(
            jnp.where(
                ~success1 & ~success2 & success3,
                jnp.zeros_like(open_segments[0]),
                open_segments[idx3],
            )
        )

        # Fourth merge attempt
        success4, seg_current_new, tidx_current_new = lax.cond(
            ~success1 & ~success2 & ~success3,
            lambda: _merge_two_segments(
                seg_current, seg4, tidx_current, tidx4, ctype4, max_dist=max_dist
            ),
            lambda: (False, seg_current_new, tidx_current_new),
        )

        open_segments = open_segments.at[idx4].set(
            jnp.where(
                ~success1 & ~success2 & ~success3 & success4,
                jnp.zeros_like(open_segments[0]),
                open_segments[idx4],
            )
        )

        return (
            seg_current_new,
            tidx_current_new,
            open_segments,
            tidcs,
        ), 0.0

    init = (open_segments[0], tail_idcs[0], open_segments[1:], tail_idcs[1:])
    carry, _ = lax.scan(body_fn, init, idcs)
    seg_merged, tidx_merged, open_segments, tail_idcs = carry

    # Store the merged closed segment
    closed_segments = closed_segments.at[-1].set(seg_merged)

    # Repeat the whole procedure once again
    idcs = jnp.arange(0, max_nr_of_segments_in_contour - 2)

    init = (open_segments[0], tail_idcs[0], open_segments[1:], tail_idcs[1:])
    carry, _ = lax.scan(body_fn, init, idcs)
    seg_merged, tidx_merged, open_segments, tail_idcs = carry

    # Store the second merged contour
    closed_segments = closed_segments.at[-2].set(seg_merged)

    return closed_segments


@partial(
    jit, static_argnames=("n_contours", "max_nr_of_segments_in_contour", "max_dist")
)
def _get_contours(
    segments,
    segments_are_closed,
    n_contours=5,
    max_nr_of_segments_in_contour=10,
    max_dist=1e-01,
):
    """
    Given a set of image segments, some of which may be open, return closed contours
    which are obtained by merging the open segments together.

    Args:
        segments (array_like): Array of shape `(n_segments, 2, n_points)`
            containing the segment points and the parity for each point.
        segments_are_closed (bool): False if not all segments are closed.
        n_contours (int, optional): Final number of contours. Defaults to 5.
        max_nr_of_segments_in_contour (int): Maximum number of segments for a
            closed contour. Should be at least 12 in case of triple lensing.
            Default is 10.


    Returns:
        tuple: A tuple of (contours, parity) where `contours` is an array with
        shape `(n_contours, n_points)` and parity is an array of shape `n_contours`
        indicating the parity of each contour.
    """
    # Mask for closed segments
    cond1 = jnp.all(segments[:, 0, :] != 0 + 0j, axis=1)
    cond2 = jnp.abs(segments[:, 0, 0] - segments[:, 0, -1]) < 1e-05
    mask_closed = jnp.logical_and(cond1, cond2)

    # Pad segments with zeros to make room for merged segments
    segments = jnp.pad(
        segments, ((0, 0), (0, 0), (0, 3 * segments.shape[-1])), constant_values=0.0
    )

    # If the segments are closed do nothing, otherwise run the merging algorithm
    contours = lax.cond(
        segments_are_closed,
        lambda: segments,
        lambda: _merge_open_segments(
            segments,
            mask_closed,
            max_nr_of_segments_in_contour=max_nr_of_segments_in_contour,
            max_dist=max_dist,
        ),
    )

    # Sort such that nonempty contours appear first
    sorted_idcs = lambda segments: jnp.argsort(jnp.abs(segments[:, 0, 0]))[::-1]
    contours = contours[sorted_idcs(contours)][:n_contours]

    # Extract per-contour parity values
    parity = jnp.sign(contours[:, 1, 0].astype(float))
    contours = contours[:, 0, :]

    # Set the last point in each contour to be equal to the first point
    tail_idcs = vmap(last_nonzero)(jnp.abs(contours))
    contours = vmap(lambda idx, contour: contour.at[idx + 1].set(contour[0]))(
        tail_idcs, contours
    )
    tail_idcs += 1

    return contours, parity, tail_idcs


@jit
def _integrate_unif(
    rho,
    contours,
    parity,
    tail_idcs,
):
    # Select k and (k + 1)th elements
    contours_k = contours
    contours_kp1 = jnp.pad(contours[:, 1:], ((0, 0), (0, 1)))
    contours_k = vmap(lambda idx, contour: contour.at[idx].set(0.0))(
        tail_idcs, contours_k
    )

    # Compute the integral using the midpoint rule
    x_k = jnp.real(contours_k)
    y_k = jnp.imag(contours_k)

    x_kp1 = jnp.real(contours_kp1)
    y_kp1 = jnp.imag(contours_kp1)

    delta_x = x_kp1 - x_k
    delta_y = y_kp1 - y_k

    x_mid = 0.5 * (x_k + x_kp1)
    y_mid = 0.5 * (y_k + y_kp1)

    I1 = 0.5 * x_mid * delta_y
    I2 = -0.5 * y_mid * delta_x

    mag = jnp.sum(I1 + I2, axis=1) / (np.pi * rho**2)

    # sum magnifications for each image, taking into account the parity
    # (per image)
    return jnp.abs(jnp.sum(mag * parity))


@partial(jit, static_argnames=("nlenses"))
def _brightness_profile(z, rho, w_center, u1=0.0, nlenses=2, **params):
    w = lens_eq(z, nlenses=nlenses, **params)
    r = jnp.abs(w - w_center) / rho

    def safe_for_grad_sqrt(x):
        return jnp.sqrt(jnp.where(x > 0.0, x, 0.0))

    # See Dominik 1998 for details
    B_r = jnp.where(
        r <= 1.0,
        1 + safe_for_grad_sqrt(1 - r**2),
        1 - safe_for_grad_sqrt(1 - 1.0 / r**2),
    )

    I = 3.0 / (3.0 - u1) * (u1 * B_r + 1.0 - 2.0 * u1)
    return I


@partial(jit, static_argnames=("nlenses", "npts"))
def _integrate_ld(
    w_center, rho, contours, parity, tail_idcs, u1=0.0, nlenses=2, npts=201, **params
):
    # Make sure that npts is odd
    #    npts = jnp.where(npts % 2 == 0, npts + 1, npts)

    def P(x0, y0, xl, yl):
        """Integrate from x0 to xl for each yl."""
        # Construct grid in z2 and evaluate the brightness profile at each point
        y = jnp.linspace(y0 * jnp.ones_like(xl), yl, npts)

        integrands = _brightness_profile(
            xl + 1j * y, rho, w_center, u1=u1, nlenses=nlenses, **params
        )
        I = simpson_quadrature(y, integrands)
        return -0.5 * I

    def Q(x0, y0, xl, yl):
        """Integrate from y0 to yl for each xl."""
        # Construct grid in z1 and evaluate the brightness profile at each point
        x = jnp.linspace(x0 * jnp.ones_like(xl), xl, npts)

        integrands = _brightness_profile(
            x + 1j * yl, rho, w_center, u1=u1, nlenses=nlenses, **params
        )
        I = simpson_quadrature(x, integrands)
        return 0.5 * I

    # We choose the centroid of each contour to be lower limit for the P and Q
    # integrals
    z0 = vmap(lambda idx, contour: contour.sum() / (idx + 1))(tail_idcs, contours)
    x0, y0 = jnp.real(z0), jnp.imag(z0)

    # Select k and (k + 1)th elements
    contours_k = contours
    contours_kp1 = jnp.pad(contours[:, 1:], ((0, 0), (0, 1)))
    contours_k = vmap(lambda idx, contour: contour.at[idx].set(0.0))(
        tail_idcs, contours_k
    )

    # Compute the integral using the midpoint rule
    x_k = jnp.real(contours_k)
    y_k = jnp.imag(contours_k)

    x_kp1 = jnp.real(contours_kp1)
    y_kp1 = jnp.imag(contours_kp1)

    delta_x = x_kp1 - x_k
    delta_y = y_kp1 - y_k

    x_mid = 0.5 * (x_k + x_kp1)
    y_mid = 0.5 * (y_k + y_kp1)

    Pmid = vmap(P)(x0, y0, x_mid, y_mid)
    Qmid = vmap(Q)(x0, y0, x_mid, y_mid)

    I1 = Pmid * delta_x
    I2 = Qmid * delta_y

    mag = jnp.sum(I1 + I2, axis=1) / (np.pi * rho**2)

    # sum magnifications for each image, taking into account the parity of each
    # image
    return jnp.abs(jnp.sum(mag * parity))


@partial(
    jit,
    static_argnames=(
        "nlenses",
        "npts_limb",
        "niter_limb",
        "npts_ld",
        "roots_itmax",
        "roots_compensated",
    ),
)
def mag_extended_source(
    w,
    rho,
    u1=0.0,
    nlenses=2,
    npts_limb=300,
    niter_limb=8,
    npts_ld=601,
    roots_itmax=2500,
    roots_compensated=False,
    **params,
):
    """
    Compute the magnification of an extended source with radius `rho` for a
    system with `nlenses` lenses.

    Args:
        w (array_like): Source position in the complex plane.
        rho (float): Source radius in Einstein radii.
        u1 (float, optional): Linear limb darkening coefficient. Defaults to 0..
        npts_limb (int, optional): Initial number of points uniformly distributed
            on the source limb when computing the point source magnification.
            The final number of points depends on this value and `niter_limb`
            because additional points are added iteratively in a geometric
            fashion. This parameters determines the precision of the magnification
            calculation (absent limb-darkening, in which case `npts_ld` is also
            important). The default value should keep the relative error well
            below 10^{-3} in all cases. Defaults to 300.
        niter_limb (int, optional): Number of iterations to use for the point
            source magnification evaluation on the source limb. At each
            iteration we geometrically decrease the number of points starting
            with `npts_limb` and ending with 2 for the final iteration. The new
            points are placed where the gradient of the magnification is
            largest. Deaults to 8.
        npts_ld (int, optional): Number of points at which the stellar
            brightness function is evaluated when computing the integrals P and
            Q defined in Dominik 1998. Defaults to 601.
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
    images, images_mask, images_parity = _images_of_source_limb(
        w,
        rho,
        nlenses=nlenses,
        npts_init=npts_limb,
        niter=niter_limb,
        roots_itmax=roots_itmax,
        roots_compensated=roots_compensated,
        **params,
    )

    if nlenses == 1:
        # Per image parity
        parity = images_parity[:, 0]

        # Last point is equal to first point
        contours = jnp.hstack([images, images[:, 0][:, None]])
        tail_idcs = jnp.array([images.shape[1] - 1, images.shape[1] - 1])

    elif nlenses == 2:
        segments, cond_closed = _get_segments(
            images, images_mask, images_parity, nlenses=2
        )
        contours, parity, tail_idcs = _get_contours(
            segments,
            cond_closed,
            n_contours=5,
            max_nr_of_segments_in_contour=10,
        )

    elif nlenses == 3:
        segments, cond_closed = _get_segments(
            images, images_mask, images_parity, nlenses=3
        )
        contours, parity, tail_idcs = _get_contours(
            segments,
            cond_closed,
            n_contours=5,
            max_nr_of_segments_in_contour=15,
        )
    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    mag = lax.cond(
        u1 == 0.0,
        lambda: _integrate_unif(
            rho,
            contours,
            parity,
            tail_idcs,
        ),
        lambda: _integrate_ld(
            w,
            rho,
            contours,
            parity,
            tail_idcs,
            u1=u1,
            nlenses=nlenses,
            npts=npts_ld,
            **params,
        ),
    )

    return mag
