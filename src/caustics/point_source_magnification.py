# -*- coding: utf-8 -*-
__all__ = [
    "images_point_source",
    "mag_point_source",
    "critical_and_caustic_curves",
    "lens_eq",
]

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, lax

from .ehrlich_aberth_primitive import poly_roots
from .utils import match_points


@jit
def _poly_coeffs_binary(w, a, e1):
    """
    Compute the coefficients of the complex polynomial equation corresponding
    to the binary lens equation. The function returns a vector of coefficients
    starting with the highest order term.

    Args:
        w (array_like): Source plane positions in the complex plane.
        a (float): Half the separation between the two lenses. We use the
            convention where both lenses are located on the real line with
            r1 = a and r2 = -a.
        e1 (array_like): Mass fraction of the first lens e1 = m1/(m1+m2). It
            follows that e2 = 1 - e1.

    Returns:
        array_like: Polynomial coefficients, same shape as w with an added
            dimension for the polynomial coefficients.
    """
    wbar = jnp.conjugate(w)

    p_0 = -(a**2) + wbar**2
    p_1 = a**2 * w - 2 * a * e1 + a - w * wbar**2 + wbar
    p_2 = (
        2 * a**4
        - 2 * a**2 * wbar**2
        + 4 * a * wbar * e1
        - 2 * a * wbar
        - 2 * w * wbar
    )
    p_3 = (
        -2 * a**4 * w
        + 4 * a**3 * e1
        - 2 * a**3
        + 2 * a**2 * w * wbar**2
        - 4 * a * w * wbar * e1
        + 2 * a * w * wbar
        + 2 * a * e1
        - a
        - w
    )
    p_4 = (
        -(a**6)
        + a**4 * wbar**2
        - 4 * a**3 * wbar * e1
        + 2 * a**3 * wbar
        + 2 * a**2 * w * wbar
        + 4 * a**2 * e1**2
        - 4 * a**2 * e1
        + 2 * a**2
        - 4 * a * w * e1
        + 2 * a * w
    )
    p_5 = (
        a**6 * w
        - 2 * a**5 * e1
        + a**5
        - a**4 * w * wbar**2
        - a**4 * wbar
        + 4 * a**3 * w * wbar * e1
        - 2 * a**3 * w * wbar
        + 2 * a**3 * e1
        - a**3
        - 4 * a**2 * w * e1**2
        + 4 * a**2 * w * e1
        - a**2 * w
    )

    p = jnp.stack([p_0, p_1, p_2, p_3, p_4, p_5])

    return jnp.moveaxis(p, 0, -1)


@jit
def _poly_coeffs_triple(w, a, r3, e1, e2):
    """
    Compute the coefficients of the complex polynomial equation corresponding
    to the triple lens equation. The function returns a vector of coefficients
    starting with the highest order term.

    Args:
        w (array_like): Source plane positions in the complex plane.
        a (float): Half the separation between the first two lenses located on
            the real line with r1 = a and r2 = -a.
        r3 (float): The position of the third lens.
        e1 (array_like): Mass fraction of the first lens e1 = m1/(m1 + m2 + m3).
        e2 (array_like): Mass fraction of the second lens e2 = m2/(m1 + m2 + m3).

    Returns:
        array_like: Polynomial coefficients, same shape as w with an added
            dimension for the polynomial coefficients.
    """
    wbar = jnp.conjugate(w)
    r3bar = jnp.conjugate(r3)

    p_0 = -(a**2) * wbar + a**2 * r3bar + wbar**3 - wbar**2 * r3bar

    p_1 = (
        a**2 * w * wbar
        - a**2 * w * r3bar
        + 3 * a**2 * wbar * r3
        - 3 * a**2 * r3bar * r3
        - a**2 * e1
        - a**2 * e2
        - a * wbar * e1
        + a * wbar * e2
        + a * r3bar * e1
        - a * r3bar * e2
        - w * wbar**3
        + w * wbar**2 * r3bar
        - 3 * wbar**3 * r3
        + 3 * wbar**2 * r3bar * r3
        + 2 * wbar**2
        + wbar * r3bar * e1
        + wbar * r3bar * e2
        - 2 * wbar * r3bar
    )

    p_2 = (
        3 * a**4 * wbar
        - 3 * a**4 * r3bar
        - a**3 * e1
        + a**3 * e2
        - 3 * a**2 * w * wbar * r3
        + 3 * a**2 * w * r3bar * r3
        + a**2 * w
        - 3 * a**2 * wbar**3
        + 3 * a**2 * wbar**2 * r3bar
        - 3 * a**2 * wbar * r3**2
        + 3 * a**2 * r3bar * r3**2
        + 4 * a**2 * e1 * r3
        + 4 * a**2 * e2 * r3
        - a**2 * r3
        + 3 * a * wbar**2 * e1
        - 3 * a * wbar**2 * e2
        - 2 * a * wbar * r3bar * e1
        + 2 * a * wbar * r3bar * e2
        + 3 * a * wbar * e1 * r3
        - 3 * a * wbar * e2 * r3
        - 3 * a * r3bar * e1 * r3
        + 3 * a * r3bar * e2 * r3
        - a * e1
        + a * e2
        + 3 * w * wbar**3 * r3
        - 3 * w * wbar**2 * r3bar * r3
        - 3 * w * wbar**2
        + 2 * w * wbar * r3bar
        + 3 * wbar**3 * r3**2
        - 3 * wbar**2 * r3bar * r3**2
        - 3 * wbar**2 * e1 * r3
        - 3 * wbar**2 * e2 * r3
        - 3 * wbar**2 * r3
        - wbar * r3bar * e1 * r3
        - wbar * r3bar * e2 * r3
        + 4 * wbar * r3bar * r3
        + wbar
        + r3bar * e1
        + r3bar * e2
        - r3bar
    )

    p_3 = (
        -3 * a**4 * w * wbar
        + 3 * a**4 * w * r3bar
        - 9 * a**4 * wbar * r3
        + 9 * a**4 * r3bar * r3
        + 2 * a**4 * e1
        + 2 * a**4 * e2
        + a**3 * w * e1
        - a**3 * w * e2
        + 3 * a**3 * wbar * e1
        - 3 * a**3 * wbar * e2
        - 3 * a**3 * r3bar * e1
        + 3 * a**3 * r3bar * e2
        + 3 * a**3 * e1 * r3
        - 3 * a**3 * e2 * r3
        + 3 * a**2 * w * wbar**3
        - 3 * a**2 * w * wbar**2 * r3bar
        + 3 * a**2 * w * wbar * r3**2
        - 3 * a**2 * w * r3bar * r3**2
        - a**2 * w * e1 * r3
        - a**2 * w * e2 * r3
        - 2 * a**2 * w * r3
        + 9 * a**2 * wbar**3 * r3
        - 9 * a**2 * wbar**2 * r3bar * r3
        + 3 * a**2 * wbar**2 * e1
        + 3 * a**2 * wbar**2 * e2
        - 6 * a**2 * wbar**2
        - 5 * a**2 * wbar * r3bar * e1
        - 5 * a**2 * wbar * r3bar * e2
        + 6 * a**2 * wbar * r3bar
        + a**2 * wbar * r3**3
        - a**2 * r3bar * r3**3
        - a**2 * e1**2
        + 2 * a**2 * e1 * e2
        - 5 * a**2 * e1 * r3**2
        - a**2 * e2**2
        - 5 * a**2 * e2 * r3**2
        + 2 * a**2 * r3**2
        - 3 * a * w * wbar**2 * e1
        + 3 * a * w * wbar**2 * e2
        + 2 * a * w * wbar * r3bar * e1
        - 2 * a * w * wbar * r3bar * e2
        - 9 * a * wbar**2 * e1 * r3
        + 9 * a * wbar**2 * e2 * r3
        + 6 * a * wbar * r3bar * e1 * r3
        - 6 * a * wbar * r3bar * e2 * r3
        - 3 * a * wbar * e1 * r3**2
        + 4 * a * wbar * e1
        + 3 * a * wbar * e2 * r3**2
        - 4 * a * wbar * e2
        + a * r3bar * e1**2
        + 3 * a * r3bar * e1 * r3**2
        - 2 * a * r3bar * e1
        - a * r3bar * e2**2
        - 3 * a * r3bar * e2 * r3**2
        + 2 * a * r3bar * e2
        + a * e1**2 * r3
        + 2 * a * e1 * r3
        - a * e2**2 * r3
        - 2 * a * e2 * r3
        - 3 * w * wbar**3 * r3**2
        + 3 * w * wbar**2 * r3bar * r3**2
        + 3 * w * wbar**2 * e1 * r3
        + 3 * w * wbar**2 * e2 * r3
        + 6 * w * wbar**2 * r3
        - 2 * w * wbar * r3bar * e1 * r3
        - 2 * w * wbar * r3bar * e2 * r3
        - 4 * w * wbar * r3bar * r3
        - 3 * w * wbar
        + w * r3bar
        - wbar**3 * r3**3
        + wbar**2 * r3bar * r3**3
        + 6 * wbar**2 * e1 * r3**2
        + 6 * wbar**2 * e2 * r3**2
        - wbar * r3bar * e1 * r3**2
        - wbar * r3bar * e2 * r3**2
        - 2 * wbar * r3bar * r3**2
        - 4 * wbar * e1 * r3
        - 4 * wbar * e2 * r3
        + wbar * r3
        - r3bar * e1**2 * r3
        - 2 * r3bar * e1 * e2 * r3
        - r3bar * e2**2 * r3
        + r3bar * r3
    )

    p_4 = (
        -3 * a**6 * wbar
        + 3 * a**6 * r3bar
        + 2 * a**5 * e1
        - 2 * a**5 * e2
        + 9 * a**4 * w * wbar * r3
        - 9 * a**4 * w * r3bar * r3
        + a**4 * w * e1
        + a**4 * w * e2
        - 3 * a**4 * w
        + 3 * a**4 * wbar**3
        - 3 * a**4 * wbar**2 * r3bar
        + 9 * a**4 * wbar * r3**2
        - 9 * a**4 * r3bar * r3**2
        - 9 * a**4 * e1 * r3
        - 9 * a**4 * e2 * r3
        + 3 * a**4 * r3
        - 3 * a**3 * w * e1 * r3
        + 3 * a**3 * w * e2 * r3
        - 6 * a**3 * wbar**2 * e1
        + 6 * a**3 * wbar**2 * e2
        + 4 * a**3 * wbar * r3bar * e1
        - 4 * a**3 * wbar * r3bar * e2
        - 9 * a**3 * wbar * e1 * r3
        + 9 * a**3 * wbar * e2 * r3
        + 9 * a**3 * r3bar * e1 * r3
        - 9 * a**3 * r3bar * e2 * r3
        - a**3 * e1**2
        - 3 * a**3 * e1 * r3**2
        + 3 * a**3 * e1
        + a**3 * e2**2
        + 3 * a**3 * e2 * r3**2
        - 3 * a**3 * e2
        - 9 * a**2 * w * wbar**3 * r3
        + 9 * a**2 * w * wbar**2 * r3bar * r3
        - 3 * a**2 * w * wbar**2 * e1
        - 3 * a**2 * w * wbar**2 * e2
        + 9 * a**2 * w * wbar**2
        + 2 * a**2 * w * wbar * r3bar * e1
        + 2 * a**2 * w * wbar * r3bar * e2
        - 6 * a**2 * w * wbar * r3bar
        - a**2 * w * wbar * r3**3
        + a**2 * w * r3bar * r3**3
        + 2 * a**2 * w * e1 * r3**2
        + 2 * a**2 * w * e2 * r3**2
        + a**2 * w * r3**2
        - 9 * a**2 * wbar**3 * r3**2
        + 9 * a**2 * wbar**2 * r3bar * r3**2
        + 9 * a**2 * wbar**2 * r3
        + 9 * a**2 * wbar * r3bar * e1 * r3
        + 9 * a**2 * wbar * r3bar * e2 * r3
        - 12 * a**2 * wbar * r3bar * r3
        + 3 * a**2 * wbar * e1**2
        - 6 * a**2 * wbar * e1 * e2
        + 4 * a**2 * wbar * e1
        + 3 * a**2 * wbar * e2**2
        + 4 * a**2 * wbar * e2
        - 3 * a**2 * wbar
        + 4 * a**2 * r3bar * e1 * e2
        - 5 * a**2 * r3bar * e1
        - 5 * a**2 * r3bar * e2
        + 3 * a**2 * r3bar
        + 3 * a**2 * e1**2 * r3
        - 6 * a**2 * e1 * e2 * r3
        + 2 * a**2 * e1 * r3**3
        + 3 * a**2 * e2**2 * r3
        + 2 * a**2 * e2 * r3**3
        - a**2 * r3**3
        + 9 * a * w * wbar**2 * e1 * r3
        - 9 * a * w * wbar**2 * e2 * r3
        - 6 * a * w * wbar * r3bar * e1 * r3
        + 6 * a * w * wbar * r3bar * e2 * r3
        - 6 * a * w * wbar * e1
        + 6 * a * w * wbar * e2
        + 2 * a * w * r3bar * e1
        - 2 * a * w * r3bar * e2
        + 9 * a * wbar**2 * e1 * r3**2
        - 9 * a * wbar**2 * e2 * r3**2
        - 6 * a * wbar * r3bar * e1 * r3**2
        + 6 * a * wbar * r3bar * e2 * r3**2
        - 6 * a * wbar * e1**2 * r3
        + a * wbar * e1 * r3**3
        - 6 * a * wbar * e1 * r3
        + 6 * a * wbar * e2**2 * r3
        - a * wbar * e2 * r3**3
        + 6 * a * wbar * e2 * r3
        - a * r3bar * e1**2 * r3
        - a * r3bar * e1 * r3**3
        + 4 * a * r3bar * e1 * r3
        + a * r3bar * e2**2 * r3
        + a * r3bar * e2 * r3**3
        - 4 * a * r3bar * e2 * r3
        - 2 * a * e1**2 * r3**2
        - a * e1 * r3**2
        + a * e1
        + 2 * a * e2**2 * r3**2
        + a * e2 * r3**2
        - a * e2
        + w * wbar**3 * r3**3
        - w * wbar**2 * r3bar * r3**3
        - 6 * w * wbar**2 * e1 * r3**2
        - 6 * w * wbar**2 * e2 * r3**2
        - 3 * w * wbar**2 * r3**2
        + 4 * w * wbar * r3bar * e1 * r3**2
        + 4 * w * wbar * r3bar * e2 * r3**2
        + 2 * w * wbar * r3bar * r3**2
        + 6 * w * wbar * e1 * r3
        + 6 * w * wbar * e2 * r3
        + 3 * w * wbar * r3
        - 2 * w * r3bar * e1 * r3
        - 2 * w * r3bar * e2 * r3
        - w * r3bar * r3
        - w
        - 3 * wbar**2 * e1 * r3**3
        - 3 * wbar**2 * e2 * r3**3
        + wbar**2 * r3**3
        + wbar * r3bar * e1 * r3**3
        + wbar * r3bar * e2 * r3**3
        + 3 * wbar * e1**2 * r3**2
        + 6 * wbar * e1 * e2 * r3**2
        + 2 * wbar * e1 * r3**2
        + 3 * wbar * e2**2 * r3**2
        + 2 * wbar * e2 * r3**2
        - 2 * wbar * r3**2
        + r3bar * e1**2 * r3**2
        + 2 * r3bar * e1 * e2 * r3**2
        - r3bar * e1 * r3**2
        + r3bar * e2**2 * r3**2
        - r3bar * e2 * r3**2
        - e1 * r3
        - e2 * r3
        + r3
    )

    p_5 = (
        3 * a**6 * w * wbar
        - 3 * a**6 * w * r3bar
        + 9 * a**6 * wbar * r3
        - 9 * a**6 * r3bar * r3
        - a**6 * e1
        - a**6 * e2
        - 2 * a**5 * w * e1
        + 2 * a**5 * w * e2
        - 3 * a**5 * wbar * e1
        + 3 * a**5 * wbar * e2
        + 3 * a**5 * r3bar * e1
        - 3 * a**5 * r3bar * e2
        - 6 * a**5 * e1 * r3
        + 6 * a**5 * e2 * r3
        - 3 * a**4 * w * wbar**3
        + 3 * a**4 * w * wbar**2 * r3bar
        - 9 * a**4 * w * wbar * r3**2
        + 9 * a**4 * w * r3bar * r3**2
        + 6 * a**4 * w * r3
        - 9 * a**4 * wbar**3 * r3
        + 9 * a**4 * wbar**2 * r3bar * r3
        - 6 * a**4 * wbar**2 * e1
        - 6 * a**4 * wbar**2 * e2
        + 6 * a**4 * wbar**2
        + 7 * a**4 * wbar * r3bar * e1
        + 7 * a**4 * wbar * r3bar * e2
        - 6 * a**4 * wbar * r3bar
        - 3 * a**4 * wbar * r3**3
        + 3 * a**4 * r3bar * r3**3
        + 2 * a**4 * e1**2
        - 4 * a**4 * e1 * e2
        + 12 * a**4 * e1 * r3**2
        + 2 * a**4 * e2**2
        + 12 * a**4 * e2 * r3**2
        - 6 * a**4 * r3**2
        + 6 * a**3 * w * wbar**2 * e1
        - 6 * a**3 * w * wbar**2 * e2
        - 4 * a**3 * w * wbar * r3bar * e1
        + 4 * a**3 * w * wbar * r3bar * e2
        + 3 * a**3 * w * e1 * r3**2
        - 3 * a**3 * w * e2 * r3**2
        + 18 * a**3 * wbar**2 * e1 * r3
        - 18 * a**3 * wbar**2 * e2 * r3
        - 12 * a**3 * wbar * r3bar * e1 * r3
        + 12 * a**3 * wbar * r3bar * e2 * r3
        + 6 * a**3 * wbar * e1**2
        + 9 * a**3 * wbar * e1 * r3**2
        - 8 * a**3 * wbar * e1
        - 6 * a**3 * wbar * e2**2
        - 9 * a**3 * wbar * e2 * r3**2
        + 8 * a**3 * wbar * e2
        - 4 * a**3 * r3bar * e1**2
        - 9 * a**3 * r3bar * e1 * r3**2
        + 4 * a**3 * r3bar * e1
        + 4 * a**3 * r3bar * e2**2
        + 9 * a**3 * r3bar * e2 * r3**2
        - 4 * a**3 * r3bar * e2
        + a**3 * e1 * r3**3
        - 6 * a**3 * e1 * r3
        - a**3 * e2 * r3**3
        + 6 * a**3 * e2 * r3
        + 9 * a**2 * w * wbar**3 * r3**2
        - 9 * a**2 * w * wbar**2 * r3bar * r3**2
        - 18 * a**2 * w * wbar**2 * r3
        + 12 * a**2 * w * wbar * r3bar * r3
        - 3 * a**2 * w * wbar * e1**2
        + 6 * a**2 * w * wbar * e1 * e2
        - 6 * a**2 * w * wbar * e1
        - 3 * a**2 * w * wbar * e2**2
        - 6 * a**2 * w * wbar * e2
        + 9 * a**2 * w * wbar
        + a**2 * w * r3bar * e1**2
        - 2 * a**2 * w * r3bar * e1 * e2
        + 2 * a**2 * w * r3bar * e1
        + a**2 * w * r3bar * e2**2
        + 2 * a**2 * w * r3bar * e2
        - 3 * a**2 * w * r3bar
        - a**2 * w * e1 * r3**3
        - a**2 * w * e2 * r3**3
        + 3 * a**2 * wbar**3 * r3**3
        - 3 * a**2 * wbar**2 * r3bar * r3**3
        - 9 * a**2 * wbar**2 * e1 * r3**2
        - 9 * a**2 * wbar**2 * e2 * r3**2
        - 3 * a**2 * wbar * r3bar * e1 * r3**2
        - 3 * a**2 * wbar * r3bar * e2 * r3**2
        + 6 * a**2 * wbar * r3bar * r3**2
        - 15 * a**2 * wbar * e1**2 * r3
        + 6 * a**2 * wbar * e1 * e2 * r3
        + 6 * a**2 * wbar * e1 * r3
        - 15 * a**2 * wbar * e2**2 * r3
        + 6 * a**2 * wbar * e2 * r3
        - 3 * a**2 * wbar * r3
        + 5 * a**2 * r3bar * e1**2 * r3
        - 2 * a**2 * r3bar * e1 * e2 * r3
        + 4 * a**2 * r3bar * e1 * r3
        + 5 * a**2 * r3bar * e2**2 * r3
        + 4 * a**2 * r3bar * e2 * r3
        - 3 * a**2 * r3bar * r3
        - 3 * a**2 * e1**2 * r3**2
        + 2 * a**2 * e1**2
        + 6 * a**2 * e1 * e2 * r3**2
        - 4 * a**2 * e1 * e2
        + a**2 * e1
        - 3 * a**2 * e2**2 * r3**2
        + 2 * a**2 * e2**2
        + a**2 * e2
        - 9 * a * w * wbar**2 * e1 * r3**2
        + 9 * a * w * wbar**2 * e2 * r3**2
        + 6 * a * w * wbar * r3bar * e1 * r3**2
        - 6 * a * w * wbar * r3bar * e2 * r3**2
        + 6 * a * w * wbar * e1**2 * r3
        + 12 * a * w * wbar * e1 * r3
        - 6 * a * w * wbar * e2**2 * r3
        - 12 * a * w * wbar * e2 * r3
        - 2 * a * w * r3bar * e1**2 * r3
        - 4 * a * w * r3bar * e1 * r3
        + 2 * a * w * r3bar * e2**2 * r3
        + 4 * a * w * r3bar * e2 * r3
        - 3 * a * w * e1
        + 3 * a * w * e2
        - 3 * a * wbar**2 * e1 * r3**3
        + 3 * a * wbar**2 * e2 * r3**3
        + 2 * a * wbar * r3bar * e1 * r3**3
        - 2 * a * wbar * r3bar * e2 * r3**3
        + 12 * a * wbar * e1**2 * r3**2
        - 12 * a * wbar * e2**2 * r3**2
        - a * r3bar * e1**2 * r3**2
        - 2 * a * r3bar * e1 * r3**2
        + a * r3bar * e2**2 * r3**2
        + 2 * a * r3bar * e2 * r3**2
        + a * e1**2 * r3**3
        - 4 * a * e1**2 * r3
        + a * e1 * r3
        - a * e2**2 * r3**3
        + 4 * a * e2**2 * r3
        - a * e2 * r3
        + 3 * w * wbar**2 * e1 * r3**3
        + 3 * w * wbar**2 * e2 * r3**3
        - 2 * w * wbar * r3bar * e1 * r3**3
        - 2 * w * wbar * r3bar * e2 * r3**3
        - 3 * w * wbar * e1**2 * r3**2
        - 6 * w * wbar * e1 * e2 * r3**2
        - 6 * w * wbar * e1 * r3**2
        - 3 * w * wbar * e2**2 * r3**2
        - 6 * w * wbar * e2 * r3**2
        + w * r3bar * e1**2 * r3**2
        + 2 * w * r3bar * e1 * e2 * r3**2
        + 2 * w * r3bar * e1 * r3**2
        + w * r3bar * e2**2 * r3**2
        + 2 * w * r3bar * e2 * r3**2
        + 3 * w * e1 * r3
        + 3 * w * e2 * r3
        - 3 * wbar * e1**2 * r3**3
        - 6 * wbar * e1 * e2 * r3**3
        + 2 * wbar * e1 * r3**3
        - 3 * wbar * e2**2 * r3**3
        + 2 * wbar * e2 * r3**3
        + 2 * e1**2 * r3**2
        + 4 * e1 * e2 * r3**2
        - 2 * e1 * r3**2
        + 2 * e2**2 * r3**2
        - 2 * e2 * r3**2
    )

    p_6 = (
        a**8 * wbar
        - a**8 * r3bar
        - a**7 * e1
        + a**7 * e2
        - 9 * a**6 * w * wbar * r3
        + 9 * a**6 * w * r3bar * r3
        - 2 * a**6 * w * e1
        - 2 * a**6 * w * e2
        + 3 * a**6 * w
        - a**6 * wbar**3
        + a**6 * wbar**2 * r3bar
        - 9 * a**6 * wbar * r3**2
        + 9 * a**6 * r3bar * r3**2
        + 6 * a**6 * e1 * r3
        + 6 * a**6 * e2 * r3
        - 3 * a**6 * r3
        + 6 * a**5 * w * e1 * r3
        - 6 * a**5 * w * e2 * r3
        + 3 * a**5 * wbar**2 * e1
        - 3 * a**5 * wbar**2 * e2
        - 2 * a**5 * wbar * r3bar * e1
        + 2 * a**5 * wbar * r3bar * e2
        + 9 * a**5 * wbar * e1 * r3
        - 9 * a**5 * wbar * e2 * r3
        - 9 * a**5 * r3bar * e1 * r3
        + 9 * a**5 * r3bar * e2 * r3
        + 2 * a**5 * e1**2
        + 6 * a**5 * e1 * r3**2
        - 3 * a**5 * e1
        - 2 * a**5 * e2**2
        - 6 * a**5 * e2 * r3**2
        + 3 * a**5 * e2
        + 9 * a**4 * w * wbar**3 * r3
        - 9 * a**4 * w * wbar**2 * r3bar * r3
        + 6 * a**4 * w * wbar**2 * e1
        + 6 * a**4 * w * wbar**2 * e2
        - 9 * a**4 * w * wbar**2
        - 4 * a**4 * w * wbar * r3bar * e1
        - 4 * a**4 * w * wbar * r3bar * e2
        + 6 * a**4 * w * wbar * r3bar
        + 3 * a**4 * w * wbar * r3**3
        - 3 * a**4 * w * r3bar * r3**3
        - 3 * a**4 * w * e1 * r3**2
        - 3 * a**4 * w * e2 * r3**2
        - 3 * a**4 * w * r3**2
        + 9 * a**4 * wbar**3 * r3**2
        - 9 * a**4 * wbar**2 * r3bar * r3**2
        + 9 * a**4 * wbar**2 * e1 * r3
        + 9 * a**4 * wbar**2 * e2 * r3
        - 9 * a**4 * wbar**2 * r3
        - 15 * a**4 * wbar * r3bar * e1 * r3
        - 15 * a**4 * wbar * r3bar * e2 * r3
        + 12 * a**4 * wbar * r3bar * r3
        + 12 * a**4 * wbar * e1 * e2
        - 8 * a**4 * wbar * e1
        - 8 * a**4 * wbar * e2
        + 3 * a**4 * wbar
        - 2 * a**4 * r3bar * e1**2
        - 8 * a**4 * r3bar * e1 * e2
        + 7 * a**4 * r3bar * e1
        - 2 * a**4 * r3bar * e2**2
        + 7 * a**4 * r3bar * e2
        - 3 * a**4 * r3bar
        - 6 * a**4 * e1**2 * r3
        + 12 * a**4 * e1 * e2 * r3
        - 5 * a**4 * e1 * r3**3
        - 6 * a**4 * e2**2 * r3
        - 5 * a**4 * e2 * r3**3
        + 3 * a**4 * r3**3
        - 18 * a**3 * w * wbar**2 * e1 * r3
        + 18 * a**3 * w * wbar**2 * e2 * r3
        + 12 * a**3 * w * wbar * r3bar * e1 * r3
        - 12 * a**3 * w * wbar * r3bar * e2 * r3
        - 6 * a**3 * w * wbar * e1**2
        + 12 * a**3 * w * wbar * e1
        + 6 * a**3 * w * wbar * e2**2
        - 12 * a**3 * w * wbar * e2
        + 2 * a**3 * w * r3bar * e1**2
        - 4 * a**3 * w * r3bar * e1
        - 2 * a**3 * w * r3bar * e2**2
        + 4 * a**3 * w * r3bar * e2
        - a**3 * w * e1 * r3**3
        + a**3 * w * e2 * r3**3
        - 18 * a**3 * wbar**2 * e1 * r3**2
        + 18 * a**3 * wbar**2 * e2 * r3**2
        + 12 * a**3 * wbar * r3bar * e1 * r3**2
        - 12 * a**3 * wbar * r3bar * e2 * r3**2
        - 6 * a**3 * wbar * e1**2 * r3
        - 3 * a**3 * wbar * e1 * r3**3
        + 12 * a**3 * wbar * e1 * r3
        + 6 * a**3 * wbar * e2**2 * r3
        + 3 * a**3 * wbar * e2 * r3**3
        - 12 * a**3 * wbar * e2 * r3
        + 8 * a**3 * r3bar * e1**2 * r3
        + 3 * a**3 * r3bar * e1 * r3**3
        - 8 * a**3 * r3bar * e1 * r3
        - 8 * a**3 * r3bar * e2**2 * r3
        - 3 * a**3 * r3bar * e2 * r3**3
        + 8 * a**3 * r3bar * e2 * r3
        + a**3 * e1**3
        - 3 * a**3 * e1**2 * e2
        + 3 * a**3 * e1**2 * r3**2
        + 4 * a**3 * e1**2
        + 3 * a**3 * e1 * e2**2
        + 3 * a**3 * e1 * r3**2
        - 2 * a**3 * e1
        - a**3 * e2**3
        - 3 * a**3 * e2**2 * r3**2
        - 4 * a**3 * e2**2
        - 3 * a**3 * e2 * r3**2
        + 2 * a**3 * e2
        - 3 * a**2 * w * wbar**3 * r3**3
        + 3 * a**2 * w * wbar**2 * r3bar * r3**3
        + 9 * a**2 * w * wbar**2 * e1 * r3**2
        + 9 * a**2 * w * wbar**2 * e2 * r3**2
        + 9 * a**2 * w * wbar**2 * r3**2
        - 6 * a**2 * w * wbar * r3bar * e1 * r3**2
        - 6 * a**2 * w * wbar * r3bar * e2 * r3**2
        - 6 * a**2 * w * wbar * r3bar * r3**2
        + 15 * a**2 * w * wbar * e1**2 * r3
        - 6 * a**2 * w * wbar * e1 * e2 * r3
        - 6 * a**2 * w * wbar * e1 * r3
        + 15 * a**2 * w * wbar * e2**2 * r3
        - 6 * a**2 * w * wbar * e2 * r3
        - 9 * a**2 * w * wbar * r3
        - 5 * a**2 * w * r3bar * e1**2 * r3
        + 2 * a**2 * w * r3bar * e1 * e2 * r3
        + 2 * a**2 * w * r3bar * e1 * r3
        - 5 * a**2 * w * r3bar * e2**2 * r3
        + 2 * a**2 * w * r3bar * e2 * r3
        + 3 * a**2 * w * r3bar * r3
        - 3 * a**2 * w * e1**2
        + 6 * a**2 * w * e1 * e2
        - 3 * a**2 * w * e1
        - 3 * a**2 * w * e2**2
        - 3 * a**2 * w * e2
        + 3 * a**2 * w
        + 6 * a**2 * wbar**2 * e1 * r3**3
        + 6 * a**2 * wbar**2 * e2 * r3**3
        - 3 * a**2 * wbar**2 * r3**3
        - a**2 * wbar * r3bar * e1 * r3**3
        - a**2 * wbar * r3bar * e2 * r3**3
        + 12 * a**2 * wbar * e1**2 * r3**2
        - 12 * a**2 * wbar * e1 * e2 * r3**2
        - 6 * a**2 * wbar * e1 * r3**2
        + 12 * a**2 * wbar * e2**2 * r3**2
        - 6 * a**2 * wbar * e2 * r3**2
        + 6 * a**2 * wbar * r3**2
        - 7 * a**2 * r3bar * e1**2 * r3**2
        - 2 * a**2 * r3bar * e1 * e2 * r3**2
        + a**2 * r3bar * e1 * r3**2
        - 7 * a**2 * r3bar * e2**2 * r3**2
        + a**2 * r3bar * e2 * r3**2
        - 3 * a**2 * e1**3 * r3
        + 3 * a**2 * e1**2 * e2 * r3
        + a**2 * e1**2 * r3**3
        - 7 * a**2 * e1**2 * r3
        + 3 * a**2 * e1 * e2**2 * r3
        - 2 * a**2 * e1 * e2 * r3**3
        - 2 * a**2 * e1 * e2 * r3
        + 4 * a**2 * e1 * r3
        - 3 * a**2 * e2**3 * r3
        + a**2 * e2**2 * r3**3
        - 7 * a**2 * e2**2 * r3
        + 4 * a**2 * e2 * r3
        - 3 * a**2 * r3
        + 3 * a * w * wbar**2 * e1 * r3**3
        - 3 * a * w * wbar**2 * e2 * r3**3
        - 2 * a * w * wbar * r3bar * e1 * r3**3
        + 2 * a * w * wbar * r3bar * e2 * r3**3
        - 12 * a * w * wbar * e1**2 * r3**2
        - 6 * a * w * wbar * e1 * r3**2
        + 12 * a * w * wbar * e2**2 * r3**2
        + 6 * a * w * wbar * e2 * r3**2
        + 4 * a * w * r3bar * e1**2 * r3**2
        + 2 * a * w * r3bar * e1 * r3**2
        - 4 * a * w * r3bar * e2**2 * r3**2
        - 2 * a * w * r3bar * e2 * r3**2
        + 6 * a * w * e1**2 * r3
        + 3 * a * w * e1 * r3
        - 6 * a * w * e2**2 * r3
        - 3 * a * w * e2 * r3
        - 6 * a * wbar * e1**2 * r3**3
        + 2 * a * wbar * e1 * r3**3
        + 6 * a * wbar * e2**2 * r3**3
        - 2 * a * wbar * e2 * r3**3
        + a * r3bar * e1**2 * r3**3
        - a * r3bar * e2**2 * r3**3
        + 3 * a * e1**3 * r3**2
        + 3 * a * e1**2 * e2 * r3**2
        + 2 * a * e1**2 * r3**2
        - 3 * a * e1 * e2**2 * r3**2
        - 2 * a * e1 * r3**2
        - 3 * a * e2**3 * r3**2
        - 2 * a * e2**2 * r3**2
        + 2 * a * e2 * r3**2
        + 3 * w * wbar * e1**2 * r3**3
        + 6 * w * wbar * e1 * e2 * r3**3
        + 3 * w * wbar * e2**2 * r3**3
        - w * r3bar * e1**2 * r3**3
        - 2 * w * r3bar * e1 * e2 * r3**3
        - w * r3bar * e2**2 * r3**3
        - 3 * w * e1**2 * r3**2
        - 6 * w * e1 * e2 * r3**2
        - 3 * w * e2**2 * r3**2
        - e1**3 * r3**3
        - 3 * e1**2 * e2 * r3**3
        + e1**2 * r3**3
        - 3 * e1 * e2**2 * r3**3
        + 2 * e1 * e2 * r3**3
        - e2**3 * r3**3
        + e2**2 * r3**3
    )

    p_7 = (
        -(a**8) * w * wbar
        + a**8 * w * r3bar
        - 3 * a**8 * wbar * r3
        + 3 * a**8 * r3bar * r3
        + a**7 * w * e1
        - a**7 * w * e2
        + a**7 * wbar * e1
        - a**7 * wbar * e2
        - a**7 * r3bar * e1
        + a**7 * r3bar * e2
        + 3 * a**7 * e1 * r3
        - 3 * a**7 * e2 * r3
        + a**6 * w * wbar**3
        - a**6 * w * wbar**2 * r3bar
        + 9 * a**6 * w * wbar * r3**2
        - 9 * a**6 * w * r3bar * r3**2
        + 3 * a**6 * w * e1 * r3
        + 3 * a**6 * w * e2 * r3
        - 6 * a**6 * w * r3
        + 3 * a**6 * wbar**3 * r3
        - 3 * a**6 * wbar**2 * r3bar * r3
        + 3 * a**6 * wbar**2 * e1
        + 3 * a**6 * wbar**2 * e2
        - 2 * a**6 * wbar**2
        - 3 * a**6 * wbar * r3bar * e1
        - 3 * a**6 * wbar * r3bar * e2
        + 2 * a**6 * wbar * r3bar
        + 3 * a**6 * wbar * r3**3
        - 3 * a**6 * r3bar * r3**3
        - a**6 * e1**2
        + 2 * a**6 * e1 * e2
        - 9 * a**6 * e1 * r3**2
        - a**6 * e2**2
        - 9 * a**6 * e2 * r3**2
        + 6 * a**6 * r3**2
        - 3 * a**5 * w * wbar**2 * e1
        + 3 * a**5 * w * wbar**2 * e2
        + 2 * a**5 * w * wbar * r3bar * e1
        - 2 * a**5 * w * wbar * r3bar * e2
        - 6 * a**5 * w * e1 * r3**2
        + 6 * a**5 * w * e2 * r3**2
        - 9 * a**5 * wbar**2 * e1 * r3
        + 9 * a**5 * wbar**2 * e2 * r3
        + 6 * a**5 * wbar * r3bar * e1 * r3
        - 6 * a**5 * wbar * r3bar * e2 * r3
        - 6 * a**5 * wbar * e1**2
        - 9 * a**5 * wbar * e1 * r3**2
        + 4 * a**5 * wbar * e1
        + 6 * a**5 * wbar * e2**2
        + 9 * a**5 * wbar * e2 * r3**2
        - 4 * a**5 * wbar * e2
        + 3 * a**5 * r3bar * e1**2
        + 9 * a**5 * r3bar * e1 * r3**2
        - 2 * a**5 * r3bar * e1
        - 3 * a**5 * r3bar * e2**2
        - 9 * a**5 * r3bar * e2 * r3**2
        + 2 * a**5 * r3bar * e2
        - 3 * a**5 * e1**2 * r3
        - 2 * a**5 * e1 * r3**3
        + 6 * a**5 * e1 * r3
        + 3 * a**5 * e2**2 * r3
        + 2 * a**5 * e2 * r3**3
        - 6 * a**5 * e2 * r3
        - 9 * a**4 * w * wbar**3 * r3**2
        + 9 * a**4 * w * wbar**2 * r3bar * r3**2
        - 9 * a**4 * w * wbar**2 * e1 * r3
        - 9 * a**4 * w * wbar**2 * e2 * r3
        + 18 * a**4 * w * wbar**2 * r3
        + 6 * a**4 * w * wbar * r3bar * e1 * r3
        + 6 * a**4 * w * wbar * r3bar * e2 * r3
        - 12 * a**4 * w * wbar * r3bar * r3
        - 12 * a**4 * w * wbar * e1 * e2
        + 12 * a**4 * w * wbar * e1
        + 12 * a**4 * w * wbar * e2
        - 9 * a**4 * w * wbar
        + 4 * a**4 * w * r3bar * e1 * e2
        - 4 * a**4 * w * r3bar * e1
        - 4 * a**4 * w * r3bar * e2
        + 3 * a**4 * w * r3bar
        + 2 * a**4 * w * e1 * r3**3
        + 2 * a**4 * w * e2 * r3**3
        - 3 * a**4 * wbar**3 * r3**3
        + 3 * a**4 * wbar**2 * r3bar * r3**3
        + 9 * a**4 * wbar * r3bar * e1 * r3**2
        + 9 * a**4 * wbar * r3bar * e2 * r3**2
        - 6 * a**4 * wbar * r3bar * r3**2
        + 12 * a**4 * wbar * e1**2 * r3
        - 12 * a**4 * wbar * e1 * e2 * r3
        + 12 * a**4 * wbar * e2**2 * r3
        + 3 * a**4 * wbar * r3
        - a**4 * r3bar * e1**2 * r3
        + 10 * a**4 * r3bar * e1 * e2 * r3
        - 8 * a**4 * r3bar * e1 * r3
        - a**4 * r3bar * e2**2 * r3
        - 8 * a**4 * r3bar * e2 * r3
        + 3 * a**4 * r3bar * r3
        + 3 * a**4 * e1**3
        - 3 * a**4 * e1**2 * e2
        + 6 * a**4 * e1**2 * r3**2
        - 3 * a**4 * e1 * e2**2
        - 12 * a**4 * e1 * e2 * r3**2
        + 8 * a**4 * e1 * e2
        - 2 * a**4 * e1
        + 3 * a**4 * e2**3
        + 6 * a**4 * e2**2 * r3**2
        - 2 * a**4 * e2
        + 18 * a**3 * w * wbar**2 * e1 * r3**2
        - 18 * a**3 * w * wbar**2 * e2 * r3**2
        - 12 * a**3 * w * wbar * r3bar * e1 * r3**2
        + 12 * a**3 * w * wbar * r3bar * e2 * r3**2
        + 6 * a**3 * w * wbar * e1**2 * r3
        - 24 * a**3 * w * wbar * e1 * r3
        - 6 * a**3 * w * wbar * e2**2 * r3
        + 24 * a**3 * w * wbar * e2 * r3
        - 2 * a**3 * w * r3bar * e1**2 * r3
        + 8 * a**3 * w * r3bar * e1 * r3
        + 2 * a**3 * w * r3bar * e2**2 * r3
        - 8 * a**3 * w * r3bar * e2 * r3
        - a**3 * w * e1**3
        + 3 * a**3 * w * e1**2 * e2
        - 6 * a**3 * w * e1**2
        - 3 * a**3 * w * e1 * e2**2
        + 6 * a**3 * w * e1
        + a**3 * w * e2**3
        + 6 * a**3 * w * e2**2
        - 6 * a**3 * w * e2
        + 6 * a**3 * wbar**2 * e1 * r3**3
        - 6 * a**3 * wbar**2 * e2 * r3**3
        - 4 * a**3 * wbar * r3bar * e1 * r3**3
        + 4 * a**3 * wbar * r3bar * e2 * r3**3
        - 6 * a**3 * wbar * e1**2 * r3**2
        + 6 * a**3 * wbar * e2**2 * r3**2
        - 4 * a**3 * r3bar * e1**2 * r3**2
        + 4 * a**3 * r3bar * e1 * r3**2
        + 4 * a**3 * r3bar * e2**2 * r3**2
        - 4 * a**3 * r3bar * e2 * r3**2
        - 9 * a**3 * e1**3 * r3
        + 3 * a**3 * e1**2 * e2 * r3
        - 2 * a**3 * e1**2 * r3**3
        + 2 * a**3 * e1**2 * r3
        - 3 * a**3 * e1 * e2**2 * r3
        - 2 * a**3 * e1 * r3
        + 9 * a**3 * e2**3 * r3
        + 2 * a**3 * e2**2 * r3**3
        - 2 * a**3 * e2**2 * r3
        + 2 * a**3 * e2 * r3
        - 6 * a**2 * w * wbar**2 * e1 * r3**3
        - 6 * a**2 * w * wbar**2 * e2 * r3**3
        + 4 * a**2 * w * wbar * r3bar * e1 * r3**3
        + 4 * a**2 * w * wbar * r3bar * e2 * r3**3
        - 12 * a**2 * w * wbar * e1**2 * r3**2
        + 12 * a**2 * w * wbar * e1 * e2 * r3**2
        + 12 * a**2 * w * wbar * e1 * r3**2
        - 12 * a**2 * w * wbar * e2**2 * r3**2
        + 12 * a**2 * w * wbar * e2 * r3**2
        + 4 * a**2 * w * r3bar * e1**2 * r3**2
        - 4 * a**2 * w * r3bar * e1 * e2 * r3**2
        - 4 * a**2 * w * r3bar * e1 * r3**2
        + 4 * a**2 * w * r3bar * e2**2 * r3**2
        - 4 * a**2 * w * r3bar * e2 * r3**2
        + 3 * a**2 * w * e1**3 * r3
        - 3 * a**2 * w * e1**2 * e2 * r3
        + 12 * a**2 * w * e1**2 * r3
        - 3 * a**2 * w * e1 * e2**2 * r3
        - 6 * a**2 * w * e1 * r3
        + 3 * a**2 * w * e2**3 * r3
        + 12 * a**2 * w * e2**2 * r3
        - 6 * a**2 * w * e2 * r3
        + 12 * a**2 * wbar * e1 * e2 * r3**3
        - 4 * a**2 * wbar * e1 * r3**3
        - 4 * a**2 * wbar * e2 * r3**3
        + 2 * a**2 * r3bar * e1**2 * r3**3
        + 2 * a**2 * r3bar * e2**2 * r3**3
        + 9 * a**2 * e1**3 * r3**2
        + 3 * a**2 * e1**2 * e2 * r3**2
        - 4 * a**2 * e1**2 * r3**2
        + 3 * a**2 * e1 * e2**2 * r3**2
        - 8 * a**2 * e1 * e2 * r3**2
        + 4 * a**2 * e1 * r3**2
        + 9 * a**2 * e2**3 * r3**2
        - 4 * a**2 * e2**2 * r3**2
        + 4 * a**2 * e2 * r3**2
        + 6 * a * w * wbar * e1**2 * r3**3
        - 6 * a * w * wbar * e2**2 * r3**3
        - 2 * a * w * r3bar * e1**2 * r3**3
        + 2 * a * w * r3bar * e2**2 * r3**3
        - 3 * a * w * e1**3 * r3**2
        - 3 * a * w * e1**2 * e2 * r3**2
        - 6 * a * w * e1**2 * r3**2
        + 3 * a * w * e1 * e2**2 * r3**2
        + 3 * a * w * e2**3 * r3**2
        + 6 * a * w * e2**2 * r3**2
        - 3 * a * e1**3 * r3**3
        - 3 * a * e1**2 * e2 * r3**3
        + 2 * a * e1**2 * r3**3
        + 3 * a * e1 * e2**2 * r3**3
        + 3 * a * e2**3 * r3**3
        - 2 * a * e2**2 * r3**3
        + w * e1**3 * r3**3
        + 3 * w * e1**2 * e2 * r3**3
        + 3 * w * e1 * e2**2 * r3**3
        + w * e2**3 * r3**3
    )

    p_8 = (
        3 * a**8 * w * wbar * r3
        - 3 * a**8 * w * r3bar * r3
        + a**8 * w * e1
        + a**8 * w * e2
        - a**8 * w
        + 3 * a**8 * wbar * r3**2
        - 3 * a**8 * r3bar * r3**2
        - a**8 * e1 * r3
        - a**8 * e2 * r3
        + a**8 * r3
        - 3 * a**7 * w * e1 * r3
        + 3 * a**7 * w * e2 * r3
        - 3 * a**7 * wbar * e1 * r3
        + 3 * a**7 * wbar * e2 * r3
        + 3 * a**7 * r3bar * e1 * r3
        - 3 * a**7 * r3bar * e2 * r3
        - a**7 * e1**2
        - 3 * a**7 * e1 * r3**2
        + a**7 * e1
        + a**7 * e2**2
        + 3 * a**7 * e2 * r3**2
        - a**7 * e2
        - 3 * a**6 * w * wbar**3 * r3
        + 3 * a**6 * w * wbar**2 * r3bar * r3
        - 3 * a**6 * w * wbar**2 * e1
        - 3 * a**6 * w * wbar**2 * e2
        + 3 * a**6 * w * wbar**2
        + 2 * a**6 * w * wbar * r3bar * e1
        + 2 * a**6 * w * wbar * r3bar * e2
        - 2 * a**6 * w * wbar * r3bar
        - 3 * a**6 * w * wbar * r3**3
        + 3 * a**6 * w * r3bar * r3**3
        + 3 * a**6 * w * r3**2
        - 3 * a**6 * wbar**3 * r3**2
        + 3 * a**6 * wbar**2 * r3bar * r3**2
        - 6 * a**6 * wbar**2 * e1 * r3
        - 6 * a**6 * wbar**2 * e2 * r3
        + 3 * a**6 * wbar**2 * r3
        + 7 * a**6 * wbar * r3bar * e1 * r3
        + 7 * a**6 * wbar * r3bar * e2 * r3
        - 4 * a**6 * wbar * r3bar * r3
        - 3 * a**6 * wbar * e1**2
        - 6 * a**6 * wbar * e1 * e2
        + 4 * a**6 * wbar * e1
        - 3 * a**6 * wbar * e2**2
        + 4 * a**6 * wbar * e2
        - a**6 * wbar
        + 2 * a**6 * r3bar * e1**2
        + 4 * a**6 * r3bar * e1 * e2
        - 3 * a**6 * r3bar * e1
        + 2 * a**6 * r3bar * e2**2
        - 3 * a**6 * r3bar * e2
        + a**6 * r3bar
        + 3 * a**6 * e1**2 * r3
        - 6 * a**6 * e1 * e2 * r3
        + 4 * a**6 * e1 * r3**3
        + 3 * a**6 * e2**2 * r3
        + 4 * a**6 * e2 * r3**3
        - 3 * a**6 * r3**3
        + 9 * a**5 * w * wbar**2 * e1 * r3
        - 9 * a**5 * w * wbar**2 * e2 * r3
        - 6 * a**5 * w * wbar * r3bar * e1 * r3
        + 6 * a**5 * w * wbar * r3bar * e2 * r3
        + 6 * a**5 * w * wbar * e1**2
        - 6 * a**5 * w * wbar * e1
        - 6 * a**5 * w * wbar * e2**2
        + 6 * a**5 * w * wbar * e2
        - 2 * a**5 * w * r3bar * e1**2
        + 2 * a**5 * w * r3bar * e1
        + 2 * a**5 * w * r3bar * e2**2
        - 2 * a**5 * w * r3bar * e2
        + 2 * a**5 * w * e1 * r3**3
        - 2 * a**5 * w * e2 * r3**3
        + 9 * a**5 * wbar**2 * e1 * r3**2
        - 9 * a**5 * wbar**2 * e2 * r3**2
        - 6 * a**5 * wbar * r3bar * e1 * r3**2
        + 6 * a**5 * wbar * r3bar * e2 * r3**2
        + 12 * a**5 * wbar * e1**2 * r3
        + 3 * a**5 * wbar * e1 * r3**3
        - 6 * a**5 * wbar * e1 * r3
        - 12 * a**5 * wbar * e2**2 * r3
        - 3 * a**5 * wbar * e2 * r3**3
        + 6 * a**5 * wbar * e2 * r3
        - 7 * a**5 * r3bar * e1**2 * r3
        - 3 * a**5 * r3bar * e1 * r3**3
        + 4 * a**5 * r3bar * e1 * r3
        + 7 * a**5 * r3bar * e2**2 * r3
        + 3 * a**5 * r3bar * e2 * r3**3
        - 4 * a**5 * r3bar * e2 * r3
        + 3 * a**5 * e1**3
        + 3 * a**5 * e1**2 * e2
        - 4 * a**5 * e1**2
        - 3 * a**5 * e1 * e2**2
        - 3 * a**5 * e1 * r3**2
        + a**5 * e1
        - 3 * a**5 * e2**3
        + 4 * a**5 * e2**2
        + 3 * a**5 * e2 * r3**2
        - a**5 * e2
        + 3 * a**4 * w * wbar**3 * r3**3
        - 3 * a**4 * w * wbar**2 * r3bar * r3**3
        - 9 * a**4 * w * wbar**2 * r3**2
        + 6 * a**4 * w * wbar * r3bar * r3**2
        - 12 * a**4 * w * wbar * e1**2 * r3
        + 12 * a**4 * w * wbar * e1 * e2 * r3
        - 6 * a**4 * w * wbar * e1 * r3
        - 12 * a**4 * w * wbar * e2**2 * r3
        - 6 * a**4 * w * wbar * e2 * r3
        + 9 * a**4 * w * wbar * r3
        + 4 * a**4 * w * r3bar * e1**2 * r3
        - 4 * a**4 * w * r3bar * e1 * e2 * r3
        + 2 * a**4 * w * r3bar * e1 * r3
        + 4 * a**4 * w * r3bar * e2**2 * r3
        + 2 * a**4 * w * r3bar * e2 * r3
        - 3 * a**4 * w * r3bar * r3
        - 3 * a**4 * w * e1**3
        + 3 * a**4 * w * e1**2 * e2
        + 3 * a**4 * w * e1 * e2**2
        - 12 * a**4 * w * e1 * e2
        + 6 * a**4 * w * e1
        - 3 * a**4 * w * e2**3
        + 6 * a**4 * w * e2
        - 3 * a**4 * w
        - 3 * a**4 * wbar**2 * e1 * r3**3
        - 3 * a**4 * wbar**2 * e2 * r3**3
        + 3 * a**4 * wbar**2 * r3**3
        - a**4 * wbar * r3bar * e1 * r3**3
        - a**4 * wbar * r3bar * e2 * r3**3
        - 15 * a**4 * wbar * e1**2 * r3**2
        + 6 * a**4 * wbar * e1 * e2 * r3**2
        + 6 * a**4 * wbar * e1 * r3**2
        - 15 * a**4 * wbar * e2**2 * r3**2
        + 6 * a**4 * wbar * e2 * r3**2
        - 6 * a**4 * wbar * r3**2
        + 5 * a**4 * r3bar * e1**2 * r3**2
        - 2 * a**4 * r3bar * e1 * e2 * r3**2
        + a**4 * r3bar * e1 * r3**2
        + 5 * a**4 * r3bar * e2**2 * r3**2
        + a**4 * r3bar * e2 * r3**2
        - 9 * a**4 * e1**3 * r3
        - 3 * a**4 * e1**2 * e2 * r3
        - 2 * a**4 * e1**2 * r3**3
        + 8 * a**4 * e1**2 * r3
        - 3 * a**4 * e1 * e2**2 * r3
        + 4 * a**4 * e1 * e2 * r3**3
        + 4 * a**4 * e1 * e2 * r3
        - 5 * a**4 * e1 * r3
        - 9 * a**4 * e2**3 * r3
        - 2 * a**4 * e2**2 * r3**3
        + 8 * a**4 * e2**2 * r3
        - 5 * a**4 * e2 * r3
        + 3 * a**4 * r3
        - 6 * a**3 * w * wbar**2 * e1 * r3**3
        + 6 * a**3 * w * wbar**2 * e2 * r3**3
        + 4 * a**3 * w * wbar * r3bar * e1 * r3**3
        - 4 * a**3 * w * wbar * r3bar * e2 * r3**3
        + 6 * a**3 * w * wbar * e1**2 * r3**2
        + 12 * a**3 * w * wbar * e1 * r3**2
        - 6 * a**3 * w * wbar * e2**2 * r3**2
        - 12 * a**3 * w * wbar * e2 * r3**2
        - 2 * a**3 * w * r3bar * e1**2 * r3**2
        - 4 * a**3 * w * r3bar * e1 * r3**2
        + 2 * a**3 * w * r3bar * e2**2 * r3**2
        + 4 * a**3 * w * r3bar * e2 * r3**2
        + 9 * a**3 * w * e1**3 * r3
        - 3 * a**3 * w * e1**2 * e2 * r3
        + 3 * a**3 * w * e1 * e2**2 * r3
        - 6 * a**3 * w * e1 * r3
        - 9 * a**3 * w * e2**3 * r3
        + 6 * a**3 * w * e2 * r3
        + 6 * a**3 * wbar * e1**2 * r3**3
        - 4 * a**3 * wbar * e1 * r3**3
        - 6 * a**3 * wbar * e2**2 * r3**3
        + 4 * a**3 * wbar * e2 * r3**3
        + 9 * a**3 * e1**3 * r3**2
        - 3 * a**3 * e1**2 * e2 * r3**2
        - 4 * a**3 * e1**2 * r3**2
        + 3 * a**3 * e1 * e2**2 * r3**2
        + 4 * a**3 * e1 * r3**2
        - 9 * a**3 * e2**3 * r3**2
        + 4 * a**3 * e2**2 * r3**2
        - 4 * a**3 * e2 * r3**2
        - 12 * a**2 * w * wbar * e1 * e2 * r3**3
        + 4 * a**2 * w * r3bar * e1 * e2 * r3**3
        - 9 * a**2 * w * e1**3 * r3**2
        - 3 * a**2 * w * e1**2 * e2 * r3**2
        - 3 * a**2 * w * e1 * e2**2 * r3**2
        + 12 * a**2 * w * e1 * e2 * r3**2
        - 9 * a**2 * w * e2**3 * r3**2
        - 3 * a**2 * e1**3 * r3**3
        + 3 * a**2 * e1**2 * e2 * r3**3
        + 3 * a**2 * e1 * e2**2 * r3**3
        - 4 * a**2 * e1 * e2 * r3**3
        - 3 * a**2 * e2**3 * r3**3
        + 3 * a * w * e1**3 * r3**3
        + 3 * a * w * e1**2 * e2 * r3**3
        - 3 * a * w * e1 * e2**2 * r3**3
        - 3 * a * w * e2**3 * r3**3
    )

    p_9 = (
        -3 * a**8 * w * wbar * r3**2
        + 3 * a**8 * w * r3bar * r3**2
        - 2 * a**8 * w * e1 * r3
        - 2 * a**8 * w * e2 * r3
        + 2 * a**8 * w * r3
        - a**8 * wbar * r3**3
        + a**8 * r3bar * r3**3
        + 2 * a**8 * e1 * r3**2
        + 2 * a**8 * e2 * r3**2
        - 2 * a**8 * r3**2
        + 3 * a**7 * w * e1 * r3**2
        - 3 * a**7 * w * e2 * r3**2
        + 3 * a**7 * wbar * e1 * r3**2
        - 3 * a**7 * wbar * e2 * r3**2
        - 3 * a**7 * r3bar * e1 * r3**2
        + 3 * a**7 * r3bar * e2 * r3**2
        + 2 * a**7 * e1**2 * r3
        + a**7 * e1 * r3**3
        - 2 * a**7 * e1 * r3
        - 2 * a**7 * e2**2 * r3
        - a**7 * e2 * r3**3
        + 2 * a**7 * e2 * r3
        + 3 * a**6 * w * wbar**3 * r3**2
        - 3 * a**6 * w * wbar**2 * r3bar * r3**2
        + 6 * a**6 * w * wbar**2 * e1 * r3
        + 6 * a**6 * w * wbar**2 * e2 * r3
        - 6 * a**6 * w * wbar**2 * r3
        - 4 * a**6 * w * wbar * r3bar * e1 * r3
        - 4 * a**6 * w * wbar * r3bar * e2 * r3
        + 4 * a**6 * w * wbar * r3bar * r3
        + 3 * a**6 * w * wbar * e1**2
        + 6 * a**6 * w * wbar * e1 * e2
        - 6 * a**6 * w * wbar * e1
        + 3 * a**6 * w * wbar * e2**2
        - 6 * a**6 * w * wbar * e2
        + 3 * a**6 * w * wbar
        - a**6 * w * r3bar * e1**2
        - 2 * a**6 * w * r3bar * e1 * e2
        + 2 * a**6 * w * r3bar * e1
        - a**6 * w * r3bar * e2**2
        + 2 * a**6 * w * r3bar * e2
        - a**6 * w * r3bar
        - a**6 * w * e1 * r3**3
        - a**6 * w * e2 * r3**3
        + a**6 * wbar**3 * r3**3
        - a**6 * wbar**2 * r3bar * r3**3
        + 3 * a**6 * wbar**2 * e1 * r3**2
        + 3 * a**6 * wbar**2 * e2 * r3**2
        - 5 * a**6 * wbar * r3bar * e1 * r3**2
        - 5 * a**6 * wbar * r3bar * e2 * r3**2
        + 2 * a**6 * wbar * r3bar * r3**2
        + 3 * a**6 * wbar * e1**2 * r3
        + 6 * a**6 * wbar * e1 * e2 * r3
        - 2 * a**6 * wbar * e1 * r3
        + 3 * a**6 * wbar * e2**2 * r3
        - 2 * a**6 * wbar * e2 * r3
        - a**6 * wbar * r3
        - 3 * a**6 * r3bar * e1**2 * r3
        - 6 * a**6 * r3bar * e1 * e2 * r3
        + 4 * a**6 * r3bar * e1 * r3
        - 3 * a**6 * r3bar * e2**2 * r3
        + 4 * a**6 * r3bar * e2 * r3
        - a**6 * r3bar * r3
        + a**6 * e1**3
        + 3 * a**6 * e1**2 * e2
        - 3 * a**6 * e1**2 * r3**2
        - 2 * a**6 * e1**2
        + 3 * a**6 * e1 * e2**2
        + 6 * a**6 * e1 * e2 * r3**2
        - 4 * a**6 * e1 * e2
        + a**6 * e1
        + a**6 * e2**3
        - 3 * a**6 * e2**2 * r3**2
        - 2 * a**6 * e2**2
        + a**6 * e2
        - 9 * a**5 * w * wbar**2 * e1 * r3**2
        + 9 * a**5 * w * wbar**2 * e2 * r3**2
        + 6 * a**5 * w * wbar * r3bar * e1 * r3**2
        - 6 * a**5 * w * wbar * r3bar * e2 * r3**2
        - 12 * a**5 * w * wbar * e1**2 * r3
        + 12 * a**5 * w * wbar * e1 * r3
        + 12 * a**5 * w * wbar * e2**2 * r3
        - 12 * a**5 * w * wbar * e2 * r3
        + 4 * a**5 * w * r3bar * e1**2 * r3
        - 4 * a**5 * w * r3bar * e1 * r3
        - 4 * a**5 * w * r3bar * e2**2 * r3
        + 4 * a**5 * w * r3bar * e2 * r3
        - 3 * a**5 * w * e1**3
        - 3 * a**5 * w * e1**2 * e2
        + 6 * a**5 * w * e1**2
        + 3 * a**5 * w * e1 * e2**2
        - 3 * a**5 * w * e1
        + 3 * a**5 * w * e2**3
        - 6 * a**5 * w * e2**2
        + 3 * a**5 * w * e2
        - 3 * a**5 * wbar**2 * e1 * r3**3
        + 3 * a**5 * wbar**2 * e2 * r3**3
        + 2 * a**5 * wbar * r3bar * e1 * r3**3
        - 2 * a**5 * wbar * r3bar * e2 * r3**3
        - 6 * a**5 * wbar * e1**2 * r3**2
        + 6 * a**5 * wbar * e2**2 * r3**2
        + 5 * a**5 * r3bar * e1**2 * r3**2
        - 2 * a**5 * r3bar * e1 * r3**2
        - 5 * a**5 * r3bar * e2**2 * r3**2
        + 2 * a**5 * r3bar * e2 * r3**2
        - 3 * a**5 * e1**3 * r3
        - 3 * a**5 * e1**2 * e2 * r3
        + a**5 * e1**2 * r3**3
        + 2 * a**5 * e1**2 * r3
        + 3 * a**5 * e1 * e2**2 * r3
        + a**5 * e1 * r3
        + 3 * a**5 * e2**3 * r3
        - a**5 * e2**2 * r3**3
        - 2 * a**5 * e2**2 * r3
        - a**5 * e2 * r3
        + 3 * a**4 * w * wbar**2 * e1 * r3**3
        + 3 * a**4 * w * wbar**2 * e2 * r3**3
        - 2 * a**4 * w * wbar * r3bar * e1 * r3**3
        - 2 * a**4 * w * wbar * r3bar * e2 * r3**3
        + 15 * a**4 * w * wbar * e1**2 * r3**2
        - 6 * a**4 * w * wbar * e1 * e2 * r3**2
        - 6 * a**4 * w * wbar * e1 * r3**2
        + 15 * a**4 * w * wbar * e2**2 * r3**2
        - 6 * a**4 * w * wbar * e2 * r3**2
        - 5 * a**4 * w * r3bar * e1**2 * r3**2
        + 2 * a**4 * w * r3bar * e1 * e2 * r3**2
        + 2 * a**4 * w * r3bar * e1 * r3**2
        - 5 * a**4 * w * r3bar * e2**2 * r3**2
        + 2 * a**4 * w * r3bar * e2 * r3**2
        + 9 * a**4 * w * e1**3 * r3
        + 3 * a**4 * w * e1**2 * e2 * r3
        - 12 * a**4 * w * e1**2 * r3
        + 3 * a**4 * w * e1 * e2**2 * r3
        + 3 * a**4 * w * e1 * r3
        + 9 * a**4 * w * e2**3 * r3
        - 12 * a**4 * w * e2**2 * r3
        + 3 * a**4 * w * e2 * r3
        + 3 * a**4 * wbar * e1**2 * r3**3
        - 6 * a**4 * wbar * e1 * e2 * r3**3
        + 2 * a**4 * wbar * e1 * r3**3
        + 3 * a**4 * wbar * e2**2 * r3**3
        + 2 * a**4 * wbar * e2 * r3**3
        - 2 * a**4 * r3bar * e1**2 * r3**3
        - 2 * a**4 * r3bar * e2**2 * r3**3
        + 3 * a**4 * e1**3 * r3**2
        - 3 * a**4 * e1**2 * e2 * r3**2
        + 2 * a**4 * e1**2 * r3**2
        - 3 * a**4 * e1 * e2**2 * r3**2
        + 4 * a**4 * e1 * e2 * r3**2
        - 2 * a**4 * e1 * r3**2
        + 3 * a**4 * e2**3 * r3**2
        + 2 * a**4 * e2**2 * r3**2
        - 2 * a**4 * e2 * r3**2
        - 6 * a**3 * w * wbar * e1**2 * r3**3
        + 6 * a**3 * w * wbar * e2**2 * r3**3
        + 2 * a**3 * w * r3bar * e1**2 * r3**3
        - 2 * a**3 * w * r3bar * e2**2 * r3**3
        - 9 * a**3 * w * e1**3 * r3**2
        + 3 * a**3 * w * e1**2 * e2 * r3**2
        + 6 * a**3 * w * e1**2 * r3**2
        - 3 * a**3 * w * e1 * e2**2 * r3**2
        + 9 * a**3 * w * e2**3 * r3**2
        - 6 * a**3 * w * e2**2 * r3**2
        - a**3 * e1**3 * r3**3
        + 3 * a**3 * e1**2 * e2 * r3**3
        - 2 * a**3 * e1**2 * r3**3
        - 3 * a**3 * e1 * e2**2 * r3**3
        + a**3 * e2**3 * r3**3
        + 2 * a**3 * e2**2 * r3**3
        + 3 * a**2 * w * e1**3 * r3**3
        - 3 * a**2 * w * e1**2 * e2 * r3**3
        - 3 * a**2 * w * e1 * e2**2 * r3**3
        + 3 * a**2 * w * e2**3 * r3**3
    )

    p_10 = (
        a**8 * w * wbar * r3**3
        - a**8 * w * r3bar * r3**3
        + a**8 * w * e1 * r3**2
        + a**8 * w * e2 * r3**2
        - a**8 * w * r3**2
        - a**8 * e1 * r3**3
        - a**8 * e2 * r3**3
        + a**8 * r3**3
        - a**7 * w * e1 * r3**3
        + a**7 * w * e2 * r3**3
        - a**7 * wbar * e1 * r3**3
        + a**7 * wbar * e2 * r3**3
        + a**7 * r3bar * e1 * r3**3
        - a**7 * r3bar * e2 * r3**3
        - a**7 * e1**2 * r3**2
        + a**7 * e1 * r3**2
        + a**7 * e2**2 * r3**2
        - a**7 * e2 * r3**2
        - a**6 * w * wbar**3 * r3**3
        + a**6 * w * wbar**2 * r3bar * r3**3
        - 3 * a**6 * w * wbar**2 * e1 * r3**2
        - 3 * a**6 * w * wbar**2 * e2 * r3**2
        + 3 * a**6 * w * wbar**2 * r3**2
        + 2 * a**6 * w * wbar * r3bar * e1 * r3**2
        + 2 * a**6 * w * wbar * r3bar * e2 * r3**2
        - 2 * a**6 * w * wbar * r3bar * r3**2
        - 3 * a**6 * w * wbar * e1**2 * r3
        - 6 * a**6 * w * wbar * e1 * e2 * r3
        + 6 * a**6 * w * wbar * e1 * r3
        - 3 * a**6 * w * wbar * e2**2 * r3
        + 6 * a**6 * w * wbar * e2 * r3
        - 3 * a**6 * w * wbar * r3
        + a**6 * w * r3bar * e1**2 * r3
        + 2 * a**6 * w * r3bar * e1 * e2 * r3
        - 2 * a**6 * w * r3bar * e1 * r3
        + a**6 * w * r3bar * e2**2 * r3
        - 2 * a**6 * w * r3bar * e2 * r3
        + a**6 * w * r3bar * r3
        - a**6 * w * e1**3
        - 3 * a**6 * w * e1**2 * e2
        + 3 * a**6 * w * e1**2
        - 3 * a**6 * w * e1 * e2**2
        + 6 * a**6 * w * e1 * e2
        - 3 * a**6 * w * e1
        - a**6 * w * e2**3
        + 3 * a**6 * w * e2**2
        - 3 * a**6 * w * e2
        + a**6 * w
        - a**6 * wbar**2 * r3**3
        + a**6 * wbar * r3bar * e1 * r3**3
        + a**6 * wbar * r3bar * e2 * r3**3
        - 2 * a**6 * wbar * e1 * r3**2
        - 2 * a**6 * wbar * e2 * r3**2
        + 2 * a**6 * wbar * r3**2
        + a**6 * r3bar * e1**2 * r3**2
        + 2 * a**6 * r3bar * e1 * e2 * r3**2
        - a**6 * r3bar * e1 * r3**2
        + a**6 * r3bar * e2**2 * r3**2
        - a**6 * r3bar * e2 * r3**2
        + a**6 * e1**2 * r3**3
        - a**6 * e1**2 * r3
        - 2 * a**6 * e1 * e2 * r3**3
        - 2 * a**6 * e1 * e2 * r3
        + 2 * a**6 * e1 * r3
        + a**6 * e2**2 * r3**3
        - a**6 * e2**2 * r3
        + 2 * a**6 * e2 * r3
        - a**6 * r3
        + 3 * a**5 * w * wbar**2 * e1 * r3**3
        - 3 * a**5 * w * wbar**2 * e2 * r3**3
        - 2 * a**5 * w * wbar * r3bar * e1 * r3**3
        + 2 * a**5 * w * wbar * r3bar * e2 * r3**3
        + 6 * a**5 * w * wbar * e1**2 * r3**2
        - 6 * a**5 * w * wbar * e1 * r3**2
        - 6 * a**5 * w * wbar * e2**2 * r3**2
        + 6 * a**5 * w * wbar * e2 * r3**2
        - 2 * a**5 * w * r3bar * e1**2 * r3**2
        + 2 * a**5 * w * r3bar * e1 * r3**2
        + 2 * a**5 * w * r3bar * e2**2 * r3**2
        - 2 * a**5 * w * r3bar * e2 * r3**2
        + 3 * a**5 * w * e1**3 * r3
        + 3 * a**5 * w * e1**2 * e2 * r3
        - 6 * a**5 * w * e1**2 * r3
        - 3 * a**5 * w * e1 * e2**2 * r3
        + 3 * a**5 * w * e1 * r3
        - 3 * a**5 * w * e2**3 * r3
        + 6 * a**5 * w * e2**2 * r3
        - 3 * a**5 * w * e2 * r3
        + 2 * a**5 * wbar * e1 * r3**3
        - 2 * a**5 * wbar * e2 * r3**3
        - a**5 * r3bar * e1**2 * r3**3
        + a**5 * r3bar * e2**2 * r3**3
        + 2 * a**5 * e1**2 * r3**2
        - 2 * a**5 * e1 * r3**2
        - 2 * a**5 * e2**2 * r3**2
        + 2 * a**5 * e2 * r3**2
        - 3 * a**4 * w * wbar * e1**2 * r3**3
        + 6 * a**4 * w * wbar * e1 * e2 * r3**3
        - 3 * a**4 * w * wbar * e2**2 * r3**3
        + a**4 * w * r3bar * e1**2 * r3**3
        - 2 * a**4 * w * r3bar * e1 * e2 * r3**3
        + a**4 * w * r3bar * e2**2 * r3**3
        - 3 * a**4 * w * e1**3 * r3**2
        + 3 * a**4 * w * e1**2 * e2 * r3**2
        + 3 * a**4 * w * e1**2 * r3**2
        + 3 * a**4 * w * e1 * e2**2 * r3**2
        - 6 * a**4 * w * e1 * e2 * r3**2
        - 3 * a**4 * w * e2**3 * r3**2
        + 3 * a**4 * w * e2**2 * r3**2
        - a**4 * e1**2 * r3**3
        + 2 * a**4 * e1 * e2 * r3**3
        - a**4 * e2**2 * r3**3
        + a**3 * w * e1**3 * r3**3
        - 3 * a**3 * w * e1**2 * e2 * r3**3
        + 3 * a**3 * w * e1 * e2**2 * r3**3
        - a**3 * w * e2**3 * r3**3
    )

    p = jnp.stack([p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10])

    return jnp.moveaxis(p, 0, -1)


@jit
def _poly_coeffs_critical_binary(phi, a, e1):
    """
    Compute the coefficients of 2*Nth order polynomial which defines the critical
    curves for the binary lens case (N = 2).
    """
    p_0 = jnp.exp(-1j * phi)
    p_1 = jnp.zeros_like(phi)
    p_2 = -2 * a**2 * jnp.exp(-1j * phi) - 1.0
    p_3 = (-4 * a * e1 + 2 * a) * jnp.ones_like(phi)
    p_4 = a**4 * jnp.exp(-1j * phi) - a**2

    p = jnp.stack([p_0, p_1, p_2, p_3, p_4])

    return p


@jit
def _poly_coeffs_critical_triple(phi, a, r3, e1, e2):
    x = jnp.exp(-1j * phi)

    p_0 = x
    p_1 = -2 * x * r3
    p_2 = -2 * a**2 * x - 1 + x * r3**2
    p_3 = 4 * a**2 * x * r3 - 2 * a * e1 + 2 * a * e2 + 2 * e1 * r3 + 2 * e2 * r3
    p_4 = (
        a**4 * x
        - 3 * a**2 * e1
        - 3 * a**2 * e2
        + 2 * a**2
        - 2 * a**2 * x * r3**2
        + 4 * a * e1 * r3
        - 4 * a * e2 * r3
        - e1 * r3**2
        - e2 * r3**2
    )
    p_5 = (
        -2 * a**4 * x * r3
        + 2 * a**2 * e1 * r3
        + 2 * a**2 * e2 * r3
        - 2 * a * e1 * r3**2
        + 2 * a * e2 * r3**2
    )
    p_6 = (
        a**4 * e1
        + a**4 * e2
        - a**4
        + a**4 * x * r3**2
        - a**2 * e1 * r3**2
        - a**2 * e2 * r3**2
    )

    p = jnp.stack([p_0, p_1, p_2, p_3, p_4, p_5, p_6])

    return p


@partial(jit, static_argnames=("nlenses"))
def lens_eq(z, nlenses=2, **params):
    zbar = jnp.conjugate(z)

    if nlenses == 1:
        return z - 1 / zbar

    elif nlenses == 2:
        a, e1 = params["a"], params["e1"]
        return z - e1 / (zbar - a) - (1.0 - e1) / (zbar + a)

    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        return (
            z
            - e1 / (zbar - a)
            - e2 / (zbar + a)
            - (1.0 - e1 - e2) / (zbar - jnp.conjugate(r3))
        )

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")


@partial(jit, static_argnames=("nlenses"))
def lens_eq_det_jac(z, nlenses=2, **params):
    zbar = jnp.conjugate(z)

    if nlenses == 1:
        return 1.0 - 1.0 / jnp.abs(zbar**2)

    elif nlenses == 2:
        a, e1 = params["a"], params["e1"]
        return 1.0 - jnp.abs(e1 / (zbar - a) ** 2 + (1.0 - e1) / (zbar + a) ** 2) ** 2

    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        return (
            1.0
            - jnp.abs(
                e1 / (zbar - a) ** 2
                + e2 / (zbar + a) ** 2
                + (1.0 - e1 - e2) / (zbar - jnp.conjugate(r3)) ** 2
            )
            ** 2
        )
    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")


@partial(jit, static_argnames=("npts", "nlenses"))
def critical_and_caustic_curves(npts=200, nlenses=2, **params):
    phi = jnp.linspace(-np.pi, np.pi, npts)

    def apply_match_points(carry, z):
        idcs = match_points(carry, z)
        return z[idcs], z[idcs]

    if nlenses == 1:  # trivial
        return jnp.exp(-1j * phi), jnp.zeros(npts).astype(jnp.complex128)

    if nlenses == 2:
        a, e1 = params["a"], params["e1"]
        coeffs = jnp.moveaxis(_poly_coeffs_critical_binary(phi, a, e1), 0, -1)

    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        coeffs = jnp.moveaxis(_poly_coeffs_critical_triple(phi, a, r3, e1, e2), 0, -1)

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    # Compute roots
    z_cr = poly_roots(coeffs)

    # Permute roots so that they form contiguous curves
    init = z_cr[0, :]
    _, z_cr = lax.scan(apply_match_points, init, z_cr)
    z_cr = z_cr.T

    # Caustics are critical curves mapped by the lens equation
    z_ca = lens_eq(z_cr, nlenses=nlenses, **params)

    return z_cr, z_ca



@partial(
    jit, static_argnames=("nlenses", "roots_itmax", "roots_compensated", "custom_init")
)
def images_point_source(
    w,
    nlenses=2,
    roots_itmax=2500,
    roots_compensated=False,
    custom_init=False,
    z_init=None,
    **params
):
    if nlenses == 1:
        w_abs_sq = w.real**2 + w.imag**2
        w_bar = jnp.conjugate(w)
        #  Compute the image locations using the quadratic formula
        z1 = 0.5 * w * (1.0 + jnp.sqrt(1 + 4 / w_abs_sq))
        z2 = 0.5 * w * (1.0 - jnp.sqrt(1 + 4 / w_abs_sq))
        z = jnp.stack(jnp.array([z1, z2]))

        return z, jnp.ones(z.shape).astype(bool)

    elif nlenses == 2:
        a, e1 = params["a"], params["e1"]
        # Compute complex polynomial coefficients for each element of w
        coeffs = _poly_coeffs_binary(w, a, e1)

    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        # Compute complex polynomial coefficients for each element of w
        coeffs = _poly_coeffs_triple(w, a, r3, e1, e2)

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    # Compute roots
    if custom_init:
        z = poly_roots(
            coeffs,
            itmax=roots_itmax,
            compensated=roots_compensated,
            custom_init=True,
            roots_init=z_init,
        )
    else:
        z = poly_roots(
            coeffs,
            itmax=roots_itmax,
            compensated=roots_compensated,
        )
    z = jnp.moveaxis(z, -1, 0)

    # Evaluate the lens equation at the roots
    lens_eq_eval = lens_eq(z, nlenses=nlenses, **params) - w

    # Mask out roots which don't satisfy the lens equation
    z_mask = jnp.abs(lens_eq_eval) < 1e-6

    return z, z_mask 

@partial(
    jit, static_argnames=("nlenses", "roots_itmax", "roots_compensated")
)
def _images_point_source_sequential(
    w,  
    nlenses=2,
    roots_itmax=2500,
    roots_compensated=False,
    **params,
):
    """
    Same as `images_point_source` except w is a 1D arrray and the images 
    are computed sequentially using `lax.scan` such that the first set 
    of images is initialized using the default initialization and the 
    subsequent images are initialized using the previous images as a starting
    point.
    """

    def fn(w, z_init=None, custom_init=False):
        if custom_init:
            z, z_mask = images_point_source(
                w,
                nlenses=nlenses,
                roots_itmax=roots_itmax,
                roots_compensated=roots_compensated,
                z_init=z_init,
                custom_init=True,
                **params,
            )
        else:
            z, z_mask = images_point_source(
                w,
                nlenses=nlenses,
                roots_itmax=roots_itmax,
                roots_compensated=roots_compensated,
                **params,
            )
        return z, z_mask

    z_first, z_mask_first = fn(w[0])

    def body_fn(z_prev, w):
        z, z_mask = fn(w, z_init=z_prev, custom_init=True)
        return z, (z, z_mask)

    _, xs = lax.scan(body_fn, z_first, w[1:])
    z, z_mask = xs

    # Append to the initial point
    z = jnp.concatenate([z_first[None, :], z])
    z_mask = jnp.concatenate([z_mask_first[None, :], z_mask])

    return z.T, z_mask.T  


@partial(jit, static_argnames=("nlenses", "roots_itmax", "roots_compensated"))
def mag_point_source(w, nlenses=2, roots_itmax=2500, roots_compensated=False, **params):
    """
    Compute the magnification of a point source for a system with `nlenses`
    lenses. If `nlenses` is 2 (binary lens) or 3 (triple lens), the coordinate
    system is set such that the first two lenses with mass fractions
    `$e1=m_1/m_\mathrm{total}$` and `$e2=m_2/m_\mathrm{total}$` are positioned
    on the x-axis at locations $r_1=a$ and $r_2=-a$ respectively. The third
    lens is at an arbitrary position in the complex plane $r_3$. For a single lens
    lens the magnification is computed analytically. For binary and triple
    lenses computing the magnification involves solving for the roots of a
    complex polynomial with degree (`nlenses`**2 + 1) using the Elrich-Aberth
    algorithm.

    Args:
        w (array_like): Source position in the complex plane.
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
        array_like: The point source magnification evaluated at w.
    """
    z, z_mask = images_point_source(
        w,
        nlenses=nlenses,
        roots_itmax=roots_itmax,
        roots_compensated=roots_compensated,
        **params
    )
    det = lens_eq_det_jac(z, nlenses=nlenses, **params)
    mag = (1.0 / jnp.abs(det)) * z_mask 
    return mag.sum(axis=0).reshape(w.shape)
