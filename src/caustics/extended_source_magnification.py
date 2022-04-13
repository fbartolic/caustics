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
    key, w_center, rho, nlenses=2, npts=2000, npts_init_fraction=0.1, compensated=False, **kwargs
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
    theta_dense = _inverse_transform_sampling(
        key, theta_init, jnp.abs(jnp.gradient(mag_init)), n_samples=npts_dense)
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
    # solution. This can fail if there are too many elements in b which map to
    # the same element of a but it's good enough for our purposes. For a general
    # solution see the Hungarian algorithm/optimal transport algorithms.
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
                jnp.logical_and(l == 0., r == 0.), 
                jnp.zeros(npts),
                jnp.logical_and(n >= l, n <= r)
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
        array_like: Array containing the segments. The "head" of each segment
        starts at index 0 and the "tail" index is variable depending on the 
        segment.
    """
    if nlenses == 2:
        nr_of_segments = 12
    elif nlenses == 3:
        nr_of_segments = 20
    else:
        raise ValueError("`nlenses` has to be set to either 2 or 3")

    # Untangle the images
    segments = _permute_images(images, images_mask, images_parity)

    # Construct disjoint segments of images
    segments = _process_segments(segments, nr_of_segments=nr_of_segments)

    return segments

@jit
def _concatenate_segments(segment_first, segment_second):
    segment_first_length = first_zero(jnp.abs(segment_first[0]))
    return segment_first + jnp.roll(segment_second, segment_first_length, axis=-1)


@partial(jit, static_argnames=("max_dist"))
def _connection_condition(z, w, z2, w2, max_dist=1e-01):
    """
    Given two points which define the end of one segment (z, z2) and two points
    (w, w2) which define the end of the other segment where the "end" can be 
    either the head of the tail of the segment. Evaluates to True if the two
    segments should be connected. To determine whether the two segments should
    be connected we form a line defined by the two points of the first segment 
    and do the same for the second segment, we then find the intersection of 
    the two lines and check that the intersection if "in between" the ends of 
    the two segments. We also check that the angle between the two intersecting 
    lines is greater than 90 degrees.

    Args:
        z (jnp.complex128): Point defining the end of the first segment.
        z2 (_type_): Point adjacent to z. 
        w (_type_): Point defining the end of the second segment.
        w2 (_type_): Point adjacent to w.
        max_dist (float, optional): Maximum distance between the ends of the 
            two segments. If the distance is greater than this value function 
            return `False`. Defaults to 1e-01.

    Returns:
        bool: True if the two segments should be connected.
    """
    x1, y1 = jnp.real(z), jnp.imag(z)
    x2, y2 = jnp.real(z2), jnp.imag(z2)
    x3, y3 = jnp.real(w), jnp.imag(w)
    x4, y4 = jnp.real(w2), jnp.imag(w2)
    
    # Intersection of two lines
    D = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))/D
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))/D
    p = px + 1j*py
    
    # Angle between two vectors
    vec1 = (z - z2)/jnp.abs(z - z2)
    vec2 = (w - w2)/jnp.abs(w - w2)
    alpha = jnp.arccos(jnp.real(vec1)*jnp.real(vec2) + jnp.imag(vec1)*jnp.imag(vec2))
    
    # Require that z and w are close and that the intersection point is close to both
    cond1 = jnp.logical_and(jnp.abs(z - w) < max_dist, jnp.rad2deg(alpha) > 90.0)
    cond2 = jnp.logical_and(jnp.abs(z - p) < max_dist, jnp.abs(w - p) < max_dist)
    
    return jnp.logical_and(cond1, cond2)

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
    # Case 1 - Either of the two segments is all zeros
    case1 = jnp.logical_or(
            jnp.all(seg1[0] == 0. + 0j),
            jnp.all(seg2[0] == 0. + 0j),
    )

    # Case 2 - Tail of segment i connects to head of segment j
    connection = _connection_condition(
        seg1[0, tidx1], seg2[0, 0],
        seg1[0, tidx1 - 1], seg2[0, 1]
    )
    case2 = jnp.logical_or(connection, jnp.abs(seg1[0, tidx1] - seg2[0, 0]) < 1e-05)
    
    # Case 3 - Tail of segment j connects to head of segment i
    connection = _connection_condition(
        seg2[0, tidx2], seg1[0, 0],
        seg2[0, tidx2 - 1], seg1[0, 1]
    )
    case3 = jnp.logical_or(connection, jnp.abs(seg1[0, 0] - seg2[0, tidx2]) < 1e-05)

    # Case 4 - Heads of segments i and j are connected 
    case4 = jnp.logical_or(
        jnp.abs(seg1[0, 0] - seg2[0, 0]) < 1e-05,
        _connection_condition(seg1[0, 0], seg2[0, 0], seg1[0, 1], seg2[0, 1])
    )
    
    # Case 5 - Tails of segments i and j are connected
    case5 = jnp.logical_or(
        jnp.abs(seg1[0, tidx1] - seg2[0, tidx2]) < 1e-05,
        _connection_condition(seg1[0, tidx1], seg2[0, tidx2],seg1[0, tidx1 - 1], seg2[0, tidx2 - 1])
    )
    
    # Case 6 - None of the above
    case6 = (case1 + case2 + case3 + case4 + case5) == False
    
    # Different outputs depending on the case
    def branch1(seg1, seg2, tidx1, tidx2):
        return seg1, tidx1
    
    def branch2(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2
        
    def branch3(seg1, seg2, tidx1, tidx2):
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2

    def branch4(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1] # flip
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg1_parity = jnp.sign(seg1[1].sum())   
        # seg2 = seg2.at[1].set(-1*seg2[1])
        seg2 = index_update(seg2, 1, -1*seg2[1])
        seg_merged = _concatenate_segments(seg2, seg1)
        return seg_merged, tidx1 + tidx2
    
    def branch5(seg1, seg2, tidx1, tidx2):
        seg2 = seg2[:, ::-1] # flip
        seg2 = jnp.roll(seg2, -(seg2.shape[-1] - tidx2 - 1), axis=-1)
        seg1_parity = jnp.sign(seg1[1].sum())   
        # seg2 = seg2.at[1].set(-1*seg2[1])
        seg2 = index_update(seg2, 1, -1*seg2[1])
        seg_merged = _concatenate_segments(seg1, seg2)
        return seg_merged, tidx1 + tidx2

    seg_merged, tidx_merged = lax.switch(
        jnp.argmax(jnp.array([case1, case2, case3, case4, case5, case6])), 
        [branch1, branch2, branch3, branch4, branch5, branch1],
        seg1, seg2, tidx1, tidx2
    )
    
    return seg_merged, tidx_merged 

@jit
def _get_segment_length(segment, tail_idx):
    """Get the physical length of a segment."""
    diff = jnp.diff(segment[0])
    diff = diff.at[tail_idx].set(0.)
    return jnp.abs(diff).sum()

@partial(jit, static_argnames=("n_contours"))
def _get_contours(segments, n_contours=5):
    """
    Given disjoint segments as an input, merge them into closed contours - the 
    boundaries of the final images.

    Args:
        segments (array_like): Array of shape `(n_segments, 2, n_points)` 
            containing the segment points and the parity for each point.
        n_contours (int, optional): Final number of contours. Defaults to 5.
    """
    def sorted_idcs(segments):
        return jnp.argsort(jnp.any(jnp.abs(segments[:, 0, :]) > 0., axis=1))[::-1]
    
    # Pad segments with zeros to make room for merged segments 
    segments = jnp.pad(segments, ((0,0), (0,0),(0, 3*segments.shape[-1])), constant_values=0.)
        
    # Compute tail index for each segment
    tail_idcs = vmap(last_nonzero)(jnp.abs(segments[:, 0, :]))

    # Select segments which are already closed
    segment_is_closed = lambda seg, tidx: jnp.abs(seg[0, 0] - seg[0, tidx]) < 1e-05
    mask_closed = vmap(segment_is_closed)(segments, tail_idcs)
    closed_segments = mask_closed[:, None, None]*segments
    closed_segments = closed_segments[sorted_idcs(closed_segments)] # sort 

    # The rest of the segments are open and need to be merged     
    open_segments = jnp.logical_not(mask_closed)[:, None, None]*segments

    # Sort such that the nonempty segments come first in order of increasing length
    seg_lengths = vmap(_get_segment_length)(open_segments, tail_idcs)
    _idcs = sparse_argsort(seg_lengths)
    open_segments, tail_idcs = open_segments[_idcs], tail_idcs[_idcs] 

    # Store already closed segments to final output
    contours = closed_segments[:n_contours, :, :]

    # Merge open segments by selecting one segment (with index 0) belonging to 
    # a composite image, then progressively merge it with other open segments 
    # until the resultant merged segment is closed. 
    n_remaining = segments.shape[0] - 1
    idcs = jnp.arange(0, n_remaining)
    idcs = jnp.concatenate([idcs, idcs, idcs, idcs])
    
    def body_fn(carry, idx):
        seg_merged, tail_idx_merged, open_segments, tail_idcs = carry
        seg_other, tail_idx_other = open_segments[idx], tail_idcs[idx]
                
        # Merge the main segment with the other segment. If the merging condition
        # isn't satisfied, the function returns the first argument. 
        seg_merged_new, tail_idx_merged_new = _merge_two_segments(
            seg_merged, seg_other, tail_idx_merged, tail_idx_other
        )
        
        val = jnp.where(
            tail_idx_merged_new > tail_idx_merged, 
            jnp.zeros_like(open_segments[0]),
            open_segments[idx]
        )
        open_segments = open_segments.at[idx].set(val)
        
        return (seg_merged_new, tail_idx_merged_new, open_segments, tail_idcs), 0.

    init = (open_segments[0], tail_idcs[0], open_segments[1:], tail_idcs[1:])
    carry, _ = lax.scan(body_fn, init,  idcs)
    seg_merged, tail_idx_merged, open_segments, tail_idcs = carry 
    
    # Store the merged contour 
    contours = contours.at[-1].set(seg_merged)
    
    # Repeat the procedure once again
    seg_lengths = vmap(_get_segment_length)(open_segments, tail_idcs)
    _idcs = sparse_argsort(seg_lengths)
    open_segments, tail_idcs = open_segments[_idcs], tail_idcs[_idcs]

    # The remaining number of open segments 
    n_remaining = n_remaining - 2 # previous run used up at least 2 segments
    idcs = jnp.arange(0, n_remaining)
    idcs = jnp.concatenate([idcs, idcs, idcs, idcs])
    
    init = (open_segments[0], tail_idcs[0], open_segments[1:], tail_idcs[1:])
    carry, _ = lax.scan(body_fn, init,  idcs)
    seg_merged, tail_idx_merged, open_segments, tail_idcs = carry

    # Store the second merged contour 
    contours = contours.at[-2].set(seg_merged)
    return contours 

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
    B_r = jnp.where(r <= 1., 1. + jnp.sqrt(1. - r**2), 1. - jnp.sqrt(1. - 1./r**2)) 
    I = 3./(3. - u)*(u*B_r + 1. - 2.*u)
    return I
    
@partial(jit, static_argnames=("nlenses", "npts"))
def _integrate(w_center, rho, contours, u=0., nlenses=2, npts=500, **kwargs):  
    """
    Integrate over each closed contour using Green's theorem, taking into account
    limb-darkening. See Dominik 1998 for details.

    Args:
        w_center (jnp.complex128): Source star center.
        rho (float): Source star radius.
        contours (array_like): Closed contours representing the images.
        u (float, optional): Linear limb darkening coefficient. Defaults to 0..
        nlenses (int, optional): Number of lenses. Defaults to 2.
        npts (int, optional): Number of points used when evaluating the surface
        brightness for computing the P and Q integrals (See Dominik 1998). 
        Defaults to 500.
    """
    # 1D integral in the y direction in the image plane where the limits of 
    # integration are z0 and (z1, z2) and the integration variable is z2'
    def P(z0, z1, z2):
        # Construct grid in z2 and evaluate the brightness profile at each point
        z2p = jnp.linspace(jnp.imag(z0)*jnp.ones_like(z1), z2, npts)
        integrand = _brightness_profile(z2p*1j + z1, rho, w_center, u=u, nlenses=nlenses, **kwargs)

        # Integrate over z2' using the trapezoidal rule
        delta_z2 = z2p[1, :] - z2p[0, :]
        I = jnp.trapz(integrand, x=z2p, axis=0)
        return -0.5*I

    # Similar to P except the integration variable is z1' and z2 is fixed
    def Q(z0, z1, z2):
        # Construct grid in z1 and evaluate the brightness profile at each point
        z1p = jnp.linspace(jnp.real(z0)*jnp.ones_like(z1), z1, npts)
        integrand = _brightness_profile(z2*1j + z1p, rho, w_center, u=u, nlenses=nlenses, **kwargs)

        # Integrate over z1' using the trapezoidal rule
        delta_z1 = z1p[1, :] - z1p[0, :]
        I = jnp.trapz(integrand, x=z1p, axis=0)
        return 0.5*I

    # Compute the tail indices for each contour 
    end_points = vmap(last_nonzero)(jnp.abs(contours[:, 0, :]))

    # Set the last point in each contour to be equal to the first point
    end_points = vmap(last_nonzero)(jnp.abs(contours[:, 0, :]))
    contours = vmap(lambda idx, contour: contour.at[:, idx + 1].set(contour[:, 0]))(end_points, contours)
    end_points += 1

    # Select k and (k + 1)th elements
    contours_k = contours
    contours_kp1 = contours[:, :, 1:]
    contours_k = vmap(lambda idx, contour: contour.at[0, idx].set(0.))(end_points, contours_k)

    z2_k = jnp.imag(contours_k[:, 0, :])
    z1_k = jnp.real(contours_k[:, 0, :])

    z2_kp1 = jnp.imag(contours_kp1[:, 0, :])
    z1_kp1 = jnp.real(contours_kp1[:, 0, :])

    z1_kp1 = jnp.pad(z1_kp1, ((0,0), (0, 1)))
    z2_kp1 = jnp.pad(z2_kp1, ((0,0), (0, 1))) 

    # We choose the centroid of each contour to be lower limit for the P and Q
    # integrals
    z0 = vmap(lambda idx, contour: contour[0].sum()/(idx + 1))(end_points, contours_k)

    # Compute magnification for each of the images
    mag = (vmap(Q)(z0, z1_k, 0.5*(z2_k + z2_kp1)) + vmap(Q)(z0, z1_kp1, 0.5*(z2_k + z2_kp1)))*(z2_kp1 - z2_k) +\
        (vmap(P)(z0, 0.5*(z1_k + z1_kp1), z2_k) + vmap(P)(z0, 0.5*(z1_k + z1_kp1), z2_kp1))*(z1_kp1 - z1_k)
    mag = mag.sum(axis=1)/(2*jnp.pi*rho**2)
    parity = jnp.sign(jnp.real(contours[:, 1, 0]))

    # sum magnifications for each image, taking into account the parity of each 
    # image
    return jnp.abs(mag*parity).sum() 

@partial(jit, static_argnames=("npts_limb", "npts_ld"))
def mag_extended_source_binary(rng_key, w, a, e1, rho, u=0., npts_limb=2000, npts_ld=100):
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
        rng_key, w, rho, nlenses=2, a=a, e1=e1, npts=npts_limb, npts_init_fraction=0.3
    )
    segments = _get_segments(images, images_mask, images_parity, nlenses=2)
    contours = _get_contours(segments, n_contours=5)
    mag = _integrate(w, rho, contours, u=u, nlenses=2, npts=npts_ld, a=a, e1=e1)
    return mag

@partial(jit, static_argnames=("npts_limb", "npts_ld"))
def mag_extended_source_triple(rng_key, w, a, r3, e1, e2, rho, u=0., npts_limb=2000, npts_ld=100):
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
        rng_key, w, rho, nlenses=3, a=a, r3=r3, e1=e1, e2=e2, npts=npts_limb, npts_init_fraction=0.3
    )
    segments = _get_segments(images, images_mask, images_parity, nlenses=3)
    contours = _get_contours(segments, n_contours=10)
    mag = _integrate(w, rho, contours, u=u, nlenses=3, npts=npts_ld, a=a, r3=r3, e1=e1, e2=e2)
    return mag
