# -*- coding: utf-8 -*-

__all__ = ["ehrlich_aberth"]

from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax, jit, vmap
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray
import jax.numpy as jnp

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")
print("gpu_ops:", gpu_ops)

xops = xla_client.ops


# This function exposes the primitive to user code and this is the only
# public-facing function in this module
@partial(jit, static_argnames=("itmax"))
def ehrlich_aberth(coeffs, itmax=2000, compensated=False):
    """Compute complex polynomial roots."""
    # Reshape to shape (size * (deg + 1),)
    coeffs = coeffs.reshape(-1, coeffs.shape[-1])

    # The C++ function expects the coefficients to be ordered as p[0] + p[1] * x + ...
    coeffs = coeffs[:, ::-1]

    return _ehrlich_aberth_prim.bind(coeffs, itmax=itmax)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _ehrlich_aberth_abstract(coeffs, itmax=2000, compensated=False):
    shape = coeffs.shape
    size = shape[0]  # number of polynomials
    deg = shape[1] - 1  # degree of polynomials
    dtype = dtypes.canonicalize_dtype(coeffs.dtype)
    return ShapedArray((size * deg,), dtype)


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _ehrlich_aberth_translation(
    c, coeffs, itmax=2000, compensated=False, platform="cpu"
):
    # The inputs have "shapes" that provide both the shape and the dtype
    coeffs_shape = c.get_shape(coeffs)

    # Input shapes
    dims_input = coeffs_shape.dimensions()

    size = dims_input[0]  # number of polynomials
    deg = dims_input[1] - 1  # degree of polynomials

    # Output shapes
    dims_output = (size * deg,)

    # Extract the dtype
    dtype = coeffs_shape.element_type()
    assert coeffs_shape.element_type() == dtype
    assert coeffs_shape.dimensions() == dims_input

    shape_input = xla_client.Shape.array_shape(
        np.dtype(dtype), dims_input, tuple(range(len(dims_input) - 1, -1, -1))
    )
    shape_output = xla_client.Shape.array_shape(
        np.dtype(dtype), dims_output, tuple(range(len(dims_output) - 1, -1, -1))
    )

    if dtype == np.complex128:
        op_name = platform.encode() + b"_ehrlich_aberth"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return xops.CustomCallWithLayout(
            c,
            op_name,
            # The inputs:
            operands=(
                xops.ConstantLiteral(c, size),
                xops.ConstantLiteral(c, deg),
                xops.ConstantLiteral(c, itmax),
                xops.ConstantLiteral(c, compensated),
                coeffs,
            ),
            # The input shapes:
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.bool), (), ()),
                shape_input,
            ),
            # The output shapes:
            shape_with_layout=shape_output,
        )

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'ehrlich_aberth_jax' module was not compiled with CUDA support"
            )

        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_ehrlich_aberth_descriptor(size, deg, itmax, compensated)

        return xops.CustomCallWithLayout(
            c,
            op_name,
            operands=(coeffs,),
            operand_shapes_with_layout=(shape_input,),
            shape_with_layout=shape_output,
            opaque=opaque,
        )

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************
# For each primitive numerical operation that the original function would have
# applied, the jvp-transformed function executes a “JVP rule” for that primitive
# that both evaluates the primitive on the primals and applies the primitive’s
# JVP at those primal values.
def _ehrlich_aberth_jvp(args, tangents, **kwargs):
    """
    Compute the Jacobian-vector product of the Ehrlich-Aberth complex polynomial
    root solver.

    Let f(z, p) be a polynomial function where p is a vector of coefficients.
    For each root z0 we have f(z0, p0) = 0 where p0 is a specific set of
    coefficients. The implicit function theorem says that there exists an
    implicit function such tat h(p) = z0 (in our case this is the `ehrlich_aberth`
    function). The Jacobian of this function is given by:

    \partial_z0 h(p0) = -[\partial_z f(z0, p0)]^(-1) \partial_p f(z0, p0)

    THe first part of the above equation is the first derivative of f(z, p)
    evaluated at the root z0 and the second part is a vector of partial derivatives
    of f with respect to each of the coefficients which gets dotted into the
    tangent vector dp.

    Args:
        args (tuple): The arguments to the function `ehrlich_aberth`.
        tangents (tuple): Small perturbation to the arguments.

    Returns:
        tuple: (z, dz) where z are the roots and dz is JVP.
    """
    p = args[0]
    dp = tangents[0]

    size = p.shape[0]  # number of polynomials
    deg = p.shape[1] - 1  # degree of polynomials

    # We use "bind" here because we don't want to mod the roots again
    z = _ehrlich_aberth_prim.bind(p)  # shape (size * deg,)
    z = z.reshape((size, deg))

    # Evaluate the derivative of the polynomials at the roots
    p_deriv = vmap(jnp.polyder)(p)
    df_dz = vmap(lambda coeffs, root: jnp.polyval(coeffs, root))(p_deriv, z)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    # The Jacobian of f with respect to coefficient p evaluated at each of the
    # roots. Shape (size, deg, deg + 1).
    df_dp = vmap(vmap(lambda z: jnp.power(z, jnp.arange(deg + 1)[::-1])))(z)

    # Jacobian of the roots multiplied by the tangents, shape (size, deg)
    dz = (
        vmap(
            lambda df_dp_i: jnp.sum(df_dp_i * zero_tangent(dp, p), axis=1),
            in_axes=1,  # vmap over all roots
        )(df_dp).T
        / (-df_dz)
    )

    return (
        z.reshape(-1),
        dz.reshape(-1),
    )


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
# def _kepler_batch(args, axes):
#    assert axes[0] == axes[1]
#    return ehrlich_aberth(*args), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_ehrlich_aberth_prim = core.Primitive("ehrlich_aberth")
_ehrlich_aberth_prim.def_impl(partial(xla.apply_primitive, _ehrlich_aberth_prim))
_ehrlich_aberth_prim.def_abstract_eval(_ehrlich_aberth_abstract)

# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][_ehrlich_aberth_prim] = partial(
    _ehrlich_aberth_translation, platform="cpu"
)
xla.backend_specific_translations["gpu"][_ehrlich_aberth_prim] = partial(
    _ehrlich_aberth_translation, platform="gpu"
)

# Connect the JVP and batching rules
ad.primitive_jvps[_ehrlich_aberth_prim] = _ehrlich_aberth_jvp
# batching.primitive_batchers[_ehrlich_aberth_prim] = _kepler_batch
