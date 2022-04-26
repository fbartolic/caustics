# -*- coding: utf-8 -*-
"""
Compute the magnification of an extended source using contour integration.
"""
__all__ = [
    "mag_extended_source_binary",
    "mag_extended_source_triple",
]

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax, random
import jax.scipy as jsp

from . import (
    images_point_source_binary,
    images_point_source_triple,
)
from .utils import *

from .point_source_magnification import (
    lens_eq_binary,
    lens_eq_triple,
    lens_eq_jac_det_binary,
    lens_eq_jac_det_triple,
)

from jaxopt import Bisection

@partial(jit, static_argnames=('n_samples'))
def _inverse_transform_sampling(key, xp, fp, n_samples=1000):
    """
    Given function values fp at points xp, return `n_samples` samples from
    a pdf which approximates this function using inverse transform sampling.

    Args:
        key (jax rng key): JAX PRNG key.
        xp (array_like): points at which fp is evaluated.
        fp (array_like): function values.
        n_samples (int, optional): Number of samples from the pdf. Defaults to 
            1000.

    Returns:
        array_like: Samples from the target pdf.
    """
    cum_values = jnp.cumsum(fp)
    cum_values /= jnp.max(cum_values)
    r = random.uniform(key, (n_samples,))
    inv_cdf = jnp.interp(r, cum_values, xp)
    return inv_cdf

@partial(jit, static_argnames=("nlenses", "npts", "npts_init_fraction", "compensated"))
def _images_of_source_limb(
    key, w_center, rho, nlenses=2, npts=1500, npts_init_fraction=0.2, compensated=False, **kwargs
):
    """
    Compute the images of a sequence of points on the limb of the source star.
    We use inverse transform sampling to get a higher density of points where
    the gradient in the magnifcation is large (at caustic crossings).
    """
    npts_init, npts_dense = int(npts_init_fraction*npts), int((1. - npts_init_fraction)*npts)

    theta_init = jnp.linspace(-np.pi, np.pi, npts_init)
    x = rho * jnp.cos(theta_init) + jnp.real(w_center)
    y = rho * jnp.sin(theta_init) + jnp.imag(w_center)
    w_init = x + 1j*y
    
    if nlenses == 2:
        get_images = lambda w: images_point_source_binary(w, kwargs['a'], kwargs['e1'], compensated=compensated)
        get_det = lambda im: lens_eq_jac_det_binary(im, kwargs['a'], kwargs['e1'])
    else:
        get_images = lambda w: images_point_source_triple(
            w, kwargs['a'], kwargs['r3'], kwargs['e1'], kwargs['e2'], compensated=compensated
        )
        get_det = lambda im: lens_eq_jac_det_triple(im, kwargs['a'], kwargs['r3'], kwargs['e1'], kwargs['e2'])
    
    images_init, mask_init = get_images(w_init)
    det_init = get_det(images_init)
    mag_init = jnp.sum((1.0 / jnp.abs(det_init)) * mask_init, axis=0)

    # Resample around high magnification regions
    window_size = 20
    window = jsp.stats.norm.pdf(jnp.linspace(-3, 3, window_size))
    conv = lambda x: jnp.convolve(x, window, mode='same')
    pdf = conv(jnp.abs(jnp.gradient(mag_init)))
    theta_dense = _inverse_transform_sampling(
        key, theta_init, pdf, n_samples=npts_dense)
    x = rho * jnp.cos(theta_dense) + jnp.real(w_center)
    y = rho * jnp.sin(theta_dense) + jnp.imag(w_center)
    w_dense = x + 1j*y

    # Compute images at points biased towards higher magnification
    images_dense, mask_dense = get_images(w_dense)
    det_dense = get_det(images_dense)
    
    # Combine pts
    theta = jnp.hstack([theta_init, theta_dense])
    sorted_idcs = jnp.argsort(theta)
    images = jnp.hstack([images_init, images_dense])[:, sorted_idcs]
    mask = jnp.hstack([mask_init, mask_dense])[:, sorted_idcs]
    parity = jnp.sign(jnp.hstack([det_init, det_dense]))[:, sorted_idcs]

    return images, mask, parity


@partial(jit, backend='cpu')
def _linear_sum_assignment(a, b):
    """
    Given 1D arrays a and b, return the indices which specify the permutation of
    b for which the element-wise distance between the two arrays is minimized.

    Args:
        a (array_like): 1D array.
        b (array_like): 1D array. 

    Returns:
        array_like: Indices which specify the desired permutation of b.
    """
    # This is the first guess for a solution but sometimes we get duplicate 
    # indices so for those values we need to choose the 2nd or 3rd best 
    # solution. This approach can fail if there are too many elements in b which 
    # map tothe same element of a but it's good enough for our purposes. For a 
    # general solution see the Hungarian algorithm/optimal transport algorithms.
    idcs_initial = jnp.argsort(jnp.abs(b - a[:, None]), axis=1)    
    idcs_final = jnp.repeat(999, len(a))
    
    def f(carry, idcs_initial_row):
        i, idcs_final = carry
        cond1 = jnp.isin(idcs_initial_row[0], jnp.array(idcs_final))
        cond2 = jnp.isin(idcs_initial_row[1], jnp.array(idcs_final))
        
        idx_closest = jnp.where(
            cond1, jnp.where(cond2, idcs_initial_row[2], idcs_initial_row[1]), idcs_initial_row[0]
        )
        idcs_final = idcs_final.at[i].set(idx_closest)
        return (i + 1, idcs_final), idx_closest
    
    _, res = lax.scan(f, (0, idcs_final), idcs_initial)
    
    return res


@partial(jit, backend='cpu') # this function has to be executed on the CPU
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

    # Apply mask
    z, mask_sols, parity = jnp.moveaxis(segments, 1, 0)
    z = (z*mask_sols).T
    parity = (parity*mask_sols).T

    segments = jnp.stack([z, parity])
    return jnp.moveaxis(segments, 0, 1)

@partial(jit, static_argnames=("n_parts"))
def _split_single_segment(segment, n_parts=5):
    """
    Split a single contour segment with shape `(2, npts)` which has at most
    `n_parts` parts seperated by zeros, split it into `n_parts` segments 
    such that each segment is contiguous.
    """
    npts = segment.shape[1]
    z = segment[0]
    z_diff = jnp.diff((jnp.abs(z) > 0.).astype(float), prepend=0, append=0)
    z_diff = z_diff.at[-2].set(z_diff[-1])
    z_diff = z_diff[:-1]
   
    left_edges = jnp.nonzero(z_diff > 0., size=n_parts)[0]
    right_edges = jnp.nonzero(z_diff < 0., size=n_parts)[0]
        
    # Split into n_parts parts 
    n = jnp.arange(npts)
    masks = []
    for i in range(n_parts):
        l, r = left_edges[i], right_edges[i]
        masks.append(
            jnp.where(
                (l == 0.) & (r ==0.),
                jnp.zeros(npts),
                (n >= l) & (n <= r)
            )
        )
            
    segments_split = jnp.stack(jnp.array([segment*mask for mask in masks]))
    
    return segments_split

@partial(jit, static_argnames=('nr_of_segments'))
def _process_segments(segments, nr_of_segments=20):
    """
    Process raw contour segments such that each segment is contigous (meaning 
    that there are no gaps between the segments head and tail) and that the head
    of the segment is at the 0th index in the array. The resultant number of 
    segments is in general greater than the number of images and is at most 
    equal to `nr_of_segments`.
    """
    # Split segments
    segments = jnp.concatenate(vmap(_split_single_segment)(segments))
    
    # Sort segments such that the nonempty segments appear first and shrink array
    sorted_idcs = jnp.argsort(jnp.any(jnp.abs(segments[:, 0, :]) > 0., axis=1))[::-1]
    segments = segments[sorted_idcs]
    segments = segments[:nr_of_segments, :, :]
    
    # Find head of each segment
    head_idcs = vmap(first_nonzero)(jnp.abs(segments[:, 0, :]))
    
    # Roll each segment such that head is at idx 0
    segments = vmap(lambda segment, head_idx: jnp.roll(segment, -head_idx, axis=-1))(segments, head_idcs)
    
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
    nr_of_segments = 2*n_images

    # Untangle the images
    segments = _permute_images(images, images_mask, images_parity)

    # Expand size of the array to make room for splitted segments in case there
    # are critical curve crossings
    segments = jnp.pad(
        segments, ((0, nr_of_segments - segments.shape[0]), (0,0), (0,0))
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
        segments
    )

    return segments, all_closed
    

@jit
def _concatenate_segments(segment_first, segment_second):
    segment_first_length = first_zero(jnp.abs(segment_first[0]))
    return segment_first + jnp.roll(segment_second, segment_first_length, axis=-1)


@partial(jit, static_argnames=("max_dist"))
def _connection_condition(line1, line2, max_dist=1e-01):
    """
    Dermine weather two segments should be connected or not. The input to the
    function are two arrays `line1` and `line2` consisting of the last (or first)
    two points of points of contour segments. `line1` and `line2` each consist
    of a starting point and an endpoint (in that order).We use five criterions 
    to determine if the segments should be connected:

        1. The endpoints of `line1` and `line2` are at most `max_dist` apart.
        2. The point of interesection of lines `line1` and `line2` is at most
        `max_dist` away from the endpoints of `line1` and `line2`.
        3. The angle between lines `line1` and `line2` is greater than 90 degrees.
        4. If we divide the image plane into two half-planes by forming a line
        passing throught the starting point of `line1` and the starting point 
        of `line2`, the endpoints of `line1` and `line2` must belong to the same
        half-plane.
        5. Distance between ending points of `line1` and `line2` is less than
            the distance between start points.

    Args:
        line1(array_like): Size 2 array containing two points in the complex
            plane where the second point is the end-point.
        line1(array_like): Size 2 array containing two points in the complex
            plane where the second point is the end-point.
        max_dist (float, optional): Maximum distance between the ends of the 
            two segments. If the distance is greater than this value function 
            return `False`. Defaults to 5e-02.

    Returns:
        bool: True if the two segments should be connected.
    """
    x1, y1 = jnp.real(line1[0]), jnp.imag(line1[0])
    x2, y2 = jnp.real(line1[1]), jnp.imag(line1[1])
    x3, y3 = jnp.real(line2[0]), jnp.imag(line2[0])
    x4, y4 = jnp.real(line2[1]), jnp.imag(line2[1])
    
    # Intersection of two lines
    D = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))/D
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))/D
    p = px + 1j*py
    
    # Require that z and w are close and that the intersection point is roughly equidistant
    # from z and w
    cond1 = jnp.abs(line1[1] - line2[1]) < max_dist
    cond2 = (jnp.abs(line1[1] - p) < max_dist) & (jnp.abs(line2[1] - p) < max_dist) 

    # Angle between two vectors
    vec1 = (line1[1] - line1[0])/jnp.abs(line1[1] - line1[0])
    vec2 = (line2[1] - line2[0])/jnp.abs(line2[1] - line2[0])
    alpha = jnp.arccos(jnp.real(vec1)*jnp.real(vec2) + jnp.imag(vec1)*jnp.imag(vec2))
    cond3 = jnp.rad2deg(alpha) > 90.0

    # Eq. of a line through points (x1, y1), (x3, y3)
    y = lambda x: y1 + (y3 - y1)/(x3 - x1)*(x - x1)

    # Check if points (x2, y2) and (x4, y4) are in the same half-plane
    cond4 = jnp.logical_xor(y2 > y(x2), y4 > y(x4)) == False

    # Distance between endpoints of line1 and line2 has to be smaller than the
    # distance between points where the line begins
    cond5 = jnp.abs(line1[1] - line2[1]) < jnp.abs(line1[0] - line2[0])
    
    return cond1 & cond2 & cond3 & cond4 & cond5

@jit
def _merge_two_segments(seg1, seg2, tidx1, tidx2):
    """
    Given two segments seg1 and seg2, merge them if the condition for merging
    is satisfied, while keeping track of the parity of each segment.

    Args:
        seg1 (array_like): First segment.
        seg2 (array_like): Second segment. 
        tidx1 (int): Index specifying the tail of the first segment.
        tidx2 (int): Index specifying the tail of the second segment.

    Returns:
        (array_like, tidx): The merged segment and the tail index of the merged
            segment. If the merging condition is not satisfied the function 
            returns (`seg1, tidx`). 
    """
    # Evaluate TH, HT, HH, and TT distances
    dist_th = jnp.abs(seg1[0, tidx1] - seg2[0, 0]) 
    dist_ht = jnp.abs(seg1[0, 0] - seg2[0, tidx2]) 
    dist_hh = jnp.abs(seg1[0, 0] - seg2[0, 0]) 
    dist_tt = jnp.abs(seg1[0, tidx1] - seg2[0, tidx2]) 

    # Evaluate connection conditions for each of the 4 possible connections
    cc_th = _connection_condition(
        lax.dynamic_slice_in_dim(seg1[0], tidx1 - 1, 2),
        lax.dynamic_slice_in_dim(seg2[0], 0, 2)[::-1],
    )
    cc_ht = _connection_condition(
        lax.dynamic_slice_in_dim(seg2[0], tidx2 - 1, 2),
        lax.dynamic_slice_in_dim(seg1[0], 0, 2)[::-1],
    )
    cc_tt = _connection_condition(
        lax.dynamic_slice_in_dim(seg1[0], tidx1 - 1, 2),
        lax.dynamic_slice_in_dim(seg2[0], tidx2 - 1, 2),
    )
    cc_hh = _connection_condition(
        lax.dynamic_slice_in_dim(seg1[0], 0, 2)[::-1],
        lax.dynamic_slice_in_dim(seg2[0], 0, 2)[::-1],
    )

    # Evaluate conditions for all possible cases

    # Case 1: Either of the two segments is all zeros
    case1 = jnp.all(seg1[0] == 0. + 0j) | jnp.all(seg2[0] == 0. + 0j)

    # Case 2: Short range TH connection
    case2 = (dist_th < 1e-05) & ~case1

    # Case 3: Short range HT connection
    case3 = (dist_ht < 1e-05) & ~case1 & ~case2

    # Case 4: Short range HH connection
    case4 = (dist_hh < 1e-05) & ~case1 & ~case2 & ~case3

    # Case 5: Short range TT connection
    case5 = (dist_tt < 1e-05) & ~case1 & ~case2 & ~case3 & ~case4

    # Case 6: Long range HH connection 
    case6 = cc_hh & ~case1 & ~case2 & ~case3 & ~case4 & ~case5

    # Case 7: Long range TT connection 
    case7 = cc_tt & ~case1 & ~case2 & ~case3 & ~case4 & ~case5 & ~case6

    # Case 8: Long range TH connection
    case8 = cc_th & ~case1 & ~case2 & ~case3 & ~case4 & ~case5 & ~case6 & ~case7

    # Case 9: Long range HT connection
    case9 = cc_ht & ~case1 & ~case2 & ~case3 & ~case4 & ~case5 & ~case6 & ~case7 & ~case8 

    # Case 10: None of the above
    case10 = ~case1 & ~case2 & ~case3 & ~case4 & ~case5 & ~case6 & ~case7 & ~case8 & ~case9

    # Possible output functions, some shared by multiple cases
    def return_seg1(seg1, seg2, tidx1, tidx2):
        return seg1, tidx1
    
    def hh_connection(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1] # flip
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg1_parity = jnp.sign(seg1[1].sum())   
        # seg2 = seg2.at[1].set(-1*seg2[1])
        seg2 = index_update(seg2, 1, -1*seg2[1])
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2 + 1
    
    def tt_connection(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1] # flip
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg1_parity = jnp.sign(seg1[1].sum())   
        # seg2 = seg2.at[1].set(-1*seg2[1])
        seg2 = index_update(seg2, 1, -1*seg2[1])
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2 + 1

    def th_connection(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2 + 1
        
    def ht_connection(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2 + 1

    case_idx = jnp.argmax(
        jnp.array(
            [case1, case2, case3, case4, case5, case6, case7, case8, case9, case10]
        )
    )
    branches = [
        return_seg1, th_connection, ht_connection, hh_connection, tt_connection,
        hh_connection, tt_connection, th_connection, ht_connection, return_seg1
    ]

    seg_merged, tidx_merged = lax.switch(
        case_idx,
        branches,
        seg1, seg2, tidx1, tidx2
    )
    
    return seg_merged, tidx_merged 


@jit
def _get_segment_length(segment, tail_idx):
    """Get the physical length of a segment."""
    diff = jnp.diff(segment[0])
    diff = diff.at[tail_idx].set(0.)
    return jnp.abs(diff).sum()

@jit
def _merge_open_segments(segments, mask_closed):
    # Compute tail index for each segment
    tail_idcs = vmap(last_nonzero)(jnp.abs(segments[:, 0, :]))

    # Split segments into closed and open ones
    closed_segments = mask_closed[:, None, None]*segments
    sorted_idcs = lambda segments: jnp.argsort(jnp.abs(segments[:, 0, 0]))[::-1]
    closed_segments = closed_segments[sorted_idcs(closed_segments)] # sort 
    open_segments = jnp.logical_not(mask_closed)[:, None, None]*segments

    # Sort open segments such that the shortest segments appear first
    segment_lengths = vmap(_get_segment_length)(open_segments, tail_idcs)
    _idcs = sparse_argsort(segment_lengths)
    open_segments, tail_idcs = open_segments[_idcs], tail_idcs[_idcs] 
    
    # Merge all open segments: start by selecting 0th open segment and then
    # sequantially merge it with other ones until a closed segment is formed
    n_remaining = segments.shape[0] - 1
    idcs = jnp.arange(0, n_remaining)
    
    # Repeat loop three times, each time the nr. of open segments within a contour
    # decreases by 2 so this will fail if there are > 7 open segments within a contour
    idcs = jnp.concatenate([idcs, idcs, idcs])
    
    def body_fn(carry, idx):
        seg_carry, tidx_carry, open_segments, tidcs = carry 
        seg_other, tidx_other = open_segments[idx], tidcs[idx]
                
        # Merge segment carried over from previous iteration with the other segment. 
        # If the merging condition
        # isn't satisfied, the function returns the first argument. 
        seg_carry_new, tidx_carry_new = _merge_two_segments(
            seg_carry, seg_other, tidx_carry, tidx_other, 
        )
        
        # Zero out the other segment if merge was successfull, otherwise do nothing
        val = jnp.where(
            tidx_carry_new > tidx_carry, 
            jnp.zeros_like(open_segments[0]),
            open_segments[idx]
        )
        open_segments = open_segments.at[idx].set(val)
        
        return (seg_carry_new, tidx_carry_new, open_segments, tidcs), 0.
    
    init = (open_segments[0], tail_idcs[0], open_segments[1:], tail_idcs[1:])
    carry, _ = lax.scan(body_fn, init, idcs)
    seg_merged, tail_idx_merged, open_segments, tail_idcs = carry 
    
    # Store the merged closed segment 
    closed_segments = closed_segments.at[-1].set(seg_merged)
    
    # Repeat the procedure once again
    seg_lengths = vmap(_get_segment_length)(open_segments, tail_idcs)
    _idcs = sparse_argsort(seg_lengths)
    open_segments, tail_idcs = open_segments[_idcs], tail_idcs[_idcs]

    # The remaining number of open segments 
    n_remaining -= 2 # previous run used up at least 2 segments
    idcs = jnp.arange(0, n_remaining)
    idcs = jnp.concatenate([idcs, idcs, idcs])
    
    init = (open_segments[0], tail_idcs[0], open_segments[1:], tail_idcs[1:])
    carry, _ = lax.scan(body_fn, init,  idcs)
    seg_merged, tail_idx_merged, open_segments, tail_idcs = carry

    # Store the second merged contour 
    closed_segments = closed_segments.at[-2].set(seg_merged)
    
    return closed_segments

@partial(jit, static_argnames=("n_contours"))
def _get_contours(segments, segments_are_closed, n_contours=5):
    """
    Given disjoint segments as an input, merge them into closed contours - the 
    boundaries of the final images.

    Args:
        segments (array_like): Array of shape `(n_segments, 2, n_points)` 
            containing the segment points and the parity for each point.
        segments_are_closed (bool): False if not all segments are closed.
        n_contours (int, optional): Final number of contours. Defaults to 5.
    """
    # Mask for closed segments
    cond1 = jnp.all(segments[:, 0, :] != 0 + 0j, axis=1)
    cond2 = jnp.abs(segments[:, 0, 0] - segments[:, 0, -1]) < 1e-05
    mask_closed = jnp.logical_and(cond1, cond2)

    # Pad segments with zeros to make room for merged segments 
    segments = jnp.pad(segments, ((0,0), (0,0),(0, 3*segments.shape[-1])), constant_values=0.)
    
    contours = lax.cond(
        segments_are_closed,
        lambda a, b: segments,
        _merge_open_segments,
        segments,
        mask_closed
    )

    sorted_idcs = lambda segments: jnp.argsort(jnp.abs(segments[:, 0, 0]))[::-1]
    contours = contours[sorted_idcs(contours)] # sort 

    return contours[:n_contours]
     
@partial(jit, static_argnames=("nlenses"))
def _brightness_profile(z, rho, w_center, u=0.1, nlenses=2, **kwargs):
    if nlenses==2:
        w = lens_eq_binary(z, kwargs['a'], kwargs['e1'])
    elif nlenses==3:
        w = lens_eq_triple(z, kwargs['a'], kwargs['r3'], kwargs['e1'], kwargs['e2'])
    else:
        raise ValueError("`nlenses` has to be set to either 2 or 3")
        
    r = jnp.abs(w - w_center)/rho
    # See Dominik 1998 for details 
    def safe_for_grad_sqrt(x):
        return jnp.sqrt(jnp.where(x > 0., x, 0.))
    
    # See Dominik 1998 for details 
    B_r = jnp.where(
        r <= 1.,
        1 + safe_for_grad_sqrt(1 - r**2),
        1 - safe_for_grad_sqrt(1 - 1./r**2),
    )

    I = 3./(3. - u)*(u*B_r + 1. - 2.*u)
    return I

@partial(jit, static_argnames=("nlenses"))
def _fn_indicator(x, y, w_cent, rho, nlenses=2, **kwargs):
    """
    Evaluates to 0.5 if point (x, y) is inside the image,
    otherwise it is -0.5.
    """
    if nlenses == 2:
        w = lens_eq_binary(x + 1j*y, kwargs['a'], kwargs['e1'])
    elif nlenses == 3:
        w = lens_eq_triple(x + 1j*y, kwargs['a'], kwargs['r3'], kwargs['e1'], kwargs['e2'])
    else:
        raise ValueError("`nlenses` has to be set to either 2 or 3")
    xs = jnp.real(w) - jnp.real(w_cent)
    ys = jnp.imag(w) - jnp.imag(w_cent)
    cond = xs**2 + ys**2 <= rho**2
    return cond.astype(float) - 0.5


@partial(jit, static_argnames=("nlenses"))
def _refine_xmid_bisect(x_mid, y_mid, delta_x, w_center, rho, nlenses=2, **kwargs):
    delta = mean_zero_avoiding(jnp.abs(delta_x))
    upper = x_mid + 0.5*delta
    lower = x_mid - 0.5*delta

    def bisect_x(y, lower, upper, tol=1e-08):
        val, state = Bisection(
            lambda x: _fn_indicator(x, y, w_center, rho, nlenses=nlenses, **kwargs),
            lower=lower, upper=upper, maxiter=50, tol=tol, check_bracket=False
        ).run()
        return val
    
    return vmap(vmap(bisect_x))(y_mid, lower, upper)


@partial(jit, static_argnames=("nlenses"))
def _refine_ymid_bisect(x_mid, y_mid, delta_y, w_center, rho, nlenses=2, **kwargs):
    delta = mean_zero_avoiding(jnp.abs(delta_y))
    upper = y_mid + 0.5*delta
    lower = y_mid - 0.5*delta

    def bisect_y(x, lower, upper, tol=1e-08):
        val, state = Bisection(
            lambda y: _fn_indicator(x, y, w_center, rho, nlenses=nlenses, **kwargs),
            lower=lower, upper=upper, maxiter=50, tol=tol, check_bracket=False
        ).run()
        return val

    return vmap(vmap(bisect_y))(x_mid, lower, upper)


    
@partial(jit, static_argnames=("nlenses", "npts", "inner_integration_rule", "outer_integration_rule"))
def _integrate(
    w_center, rho, contours, u=0., nlenses=2, npts=301, 
    outer_integration_rule="midpoint", inner_integration_rule="trapezoidal", **kwargs
):  
    # Make sure that npts is odd
#    npts = jnp.where(npts % 2 == 0, npts + 1, npts)

    def P(x0, y0, xl, yl):
        """Integrate from x0 to xl for each yl."""    
        # Construct grid in z2 and evaluate the brightness profile at each point
        y = jnp.linspace(y0*jnp.ones_like(xl), yl, npts)
        integrands = _brightness_profile(xl + 1j*y, rho, w_center, u=u, nlenses=nlenses, **kwargs)
        if inner_integration_rule == "simpson":
            I = simpson_quadrature(y, integrands)
        elif inner_integration_rule == "trapezoidal":
            I = jnp.trapz(integrands, x=y, axis=0)
        else:
            raise ValueError(
                "`inner_integration_rule` has to be either `simpson` or `trapezoidal`"
            )
        return -0.5*I

    def Q(x0, y0, xl, yl):
        """Integrate from y0 to yl for each xl."""
        # Construct grid in z1 and evaluate the brightness profile at each point
        x = jnp.linspace(x0*jnp.ones_like(xl), xl, npts)
        integrands = _brightness_profile(x + 1j*yl, rho, w_center, u=u, nlenses=nlenses, **kwargs)

        if inner_integration_rule == "simpson":
            I = simpson_quadrature(x, integrands)
        elif inner_integration_rule == "trapezoidal":
            I = jnp.trapz(integrands, x=x, axis=0)
        else:
            raise ValueError(
                "`inner_integration_rule` has to be either `simpson` or `trapezoidal`"
            )
        return 0.5*I

    # Compute the tail indices for each contour 
    end_points = vmap(last_nonzero)(jnp.abs(contours[:, 0, :]))

    # Set the last point in each contour to be equal to the first point
    end_points = vmap(last_nonzero)(jnp.abs(contours[:, 0, :]))
    contours = vmap(
        lambda idx, contour: contour.at[:, idx + 1].set(contour[:, 0])
    )(end_points, contours)
    end_points += 1

    # We choose the centroid of each contour to be lower limit for the P and Q
    # integrals
    z0 = vmap(lambda idx, contour: contour[0].sum()/(idx + 1))(end_points, contours)
    x0, y0 = jnp.real(z0), jnp.imag(z0)

    # Select k and (k + 1)th elements
    contours_k = contours
    contours_kp1 = jnp.pad(contours[:, :, 1:], ((0,0),(0,0),(0,1)))
    contours_k = vmap(lambda idx, contour: contour.at[0, idx].set(0.))(end_points, contours_k)

    # Compute the integral using the midpoint rule
    x_k = jnp.real(contours_k[:, 0, :])
    y_k = jnp.imag(contours_k[:, 0, :])

    x_kp1 = jnp.real(contours_kp1[:, 0, :])
    y_kp1 = jnp.imag(contours_kp1[:, 0, :])

    delta_x = x_kp1 - x_k
    delta_y = y_kp1 - y_k

    x_mid = 0.5*(x_k + x_kp1)
    y_mid = 0.5*(y_k + y_kp1)

    if outer_integration_rule == "midpoint":
        Pmid = vmap(P)(x0, y0, x_mid, y_mid)
        Qmid = vmap(Q)(x0, y0, x_mid, y_mid)

        I1 = Pmid*delta_x 
        I2 = Qmid*delta_y 

    elif outer_integration_rule == "simpson":
        x_mid_refined = _refine_xmid_bisect(
            x_mid, y_mid, delta_x, w_center, rho, nlenses=nlenses, **kwargs
        )
        y_mid_refined = _refine_ymid_bisect(
            x_mid, y_mid, delta_y, w_center, rho, nlenses=nlenses, **kwargs
        )

        Pmid = vmap(P)(x0, y0, x_mid, y_mid_refined)
        Qmid = vmap(Q)(x0, y0, x_mid_refined, y_mid)
        
        Pk = vmap(P)(x0, y0, x_k, y_k)
        Qk = vmap(Q)(x0, y0, x_k, y_k)
        
        Pkp1 = vmap(P)(x0, y0, x_k, y_k)
        Qkp1 = vmap(Q)(x0, y0, x_k, y_k)
        
        I1 = delta_x/6*(Pk + 4*Pmid + Pkp1)
        I2 = delta_y/6*(Qk + 4*Qmid + Qkp1)

    else:
        raise ValueError(
            "`integration_rule` has to be set to either 'midpoint' or 'simpson'"
        )
   
    mag = jnp.sum(I1 + I2, axis=1)/(np.pi*rho**2)
    parity = jnp.sign(jnp.real(contours[:, 1, 0]))

    # sum magnifications for each image, taking into account the parity of each 
    # image
    return jnp.abs(mag*parity).sum()    


@partial(
    jit, 
    static_argnames=(
        "npts_limb", "npts_ld", "npts_init_fraction", "inner_integration_rule",
        "outer_integration_rule"
        ))
def mag_extended_source_binary(
    rng_key, w, a, e1, rho, u=0., 
    npts_limb=1000, npts_ld=301, 
    npts_init_fraction=0.2, inner_integration_rule="trapezoidal", 
    outer_integration_rule="midpoint"
):
    """
    Compute the magnification for an extended source lensed by a binary lens
    using contour integration.

    Args:
        rng_key (jax PRNG key): PRNG key which is used in the method for 
            determining optimal locations on the limb of the source for 
            computing the images.
        w (jnp.complex128): Center of the source in the lens plane.
        a (float): Half the separation between the two lenses. We use the
            convention where both lenses are located on the real line with
            r1 = a and r2 = -a.
        e1 (array_like): Mass fraction of the first lens e1 = m1/(m1+m2). It
            follows that e2 = 1 - e1.
        rho (float): Source radius in Einstein radii. 
        u (float, optional): Linear limb darkening coefficient. Defaults to 0..
        npts_limb (int, optional): Total number of points on the source limb for
            which we evaluate the images This is a key input to the method which
            determines the accuracy of the magnification calculation. Defaults 
            to 2000.
        npts_ld (int, optional): Number of points at which the stellar 
            brightness function is evaluated when computing the integrals P and
            Q defined in Dominik 1998. Defaults to 100.

    Returns:
        float: Total magnification factor.
    """
    images, images_mask, images_parity = _images_of_source_limb(
        rng_key, w, rho, nlenses=2, a=a, e1=e1, npts=npts_limb, npts_init_fraction=npts_init_fraction
    )
    segments, cond_closed = _get_segments(images, images_mask, images_parity, nlenses=2)
    contours = _get_contours(segments, cond_closed, n_contours=5)
    mag = _integrate(w, rho, contours, u=u, nlenses=2, npts=npts_ld, 
                     inner_integration_rule=inner_integration_rule, 
                     outer_integration_rule=outer_integration_rule,
                     a=a, e1=e1,
                     )

    return mag


@partial(
    jit, 
    static_argnames=(
        "npts_limb", "npts_ld", "npts_init_fraction", "inner_integration_rule",
        "outer_integration_rule"
        ))
def mag_extended_source_triple(
    rng_key, w, a, r3, e1, e2, rho, u=0., npts_limb=1000, npts_ld=301, npts_init_fraction=0.1,
     inner_integration_rule="trapezoidal", 
    outer_integration_rule="midpoint"

):
    """
    Compute the magnification for an extended source lensed by a triple lens
    using contour integration.

    Args:
        rng_key (jax PRNG key): PRNG key which is used in the method for 
            determining optimal locations on the limb of the source for 
            computing the images.
        w (jnp.complex128): Center of the source in the lens plane.
        a (float): Half the separation between the first two lenses located on
            the real line with r1 = a and r2 = -a.
        r3 (float): The position of the third lens.
        e1 (array_like): Mass fraction of the first lens e1 = m1/(m1 + m2 + m3).
        e2 (array_like): Mass fraction of the second lens e2 = m2/(m1 + m2 + m3).
        rho (float): Source radius in Einstein radii. 
        u (float, optional): Linear limb darkening coefficient. Defaults to 0..
        npts_limb (int, optional): Total number of points on the source limb for
            which we evaluate the images This is a key input to the method which
            determines the accuracy of the magnification calculation. Defaults 
            to 2000.
        npts_ld (int, optional): Number of points at which the stellar 
            brightness function is evaluated when computing the integrals P and
            Q defined in Dominik 1998. Defaults to 100.

    Returns:
        float: Total magnification factor.
    """
    images, images_mask, images_parity = _images_of_source_limb(
        rng_key, w, rho, nlenses=3, a=a, r3=r3, e1=e1, e2=e2, npts=npts_limb, 
        npts_init_fraction=npts_init_fraction
    )
    segments = _get_segments(images, images_mask, images_parity, nlenses=3)
    contours = _get_contours(segments, n_contours=10)

    mag = _integrate(w, rho, contours, u=u, nlenses=3, npts=npts_ld, 
                     inner_integration_rule=inner_integration_rule, 
                     outer_integration_rule=outer_integration_rule,
                     a=a, r3=r3, e1=e1, e2=e2
                     )


    return mag
