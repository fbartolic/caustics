# -*- coding: utf-8 -*-

__all__ = ["poly_roots"]

from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax, jit, vmap
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray
import jax.numpy as jnp

# Register the CPU XLA custom calls
from . import ehrlich_aberth_cpu_op

for _name, _value in ehrlich_aberth_cpu_op.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

# If the GPU version exists, also register those
try:
    from . import ehrlich_aberth_gpu_op
except ImportError:
    ehrlich_aberth_gpu_op = None
else:
    for _name, _value in ehrlich_aberth_gpu_op.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

xops = xla_client.ops

# This function is just a wrapper around ehrlich_aberth to make it easier to
# handle arrays with different shapes
@partial(jit, static_argnames=("itmax", "compensated"))
def poly_roots(coeffs, itmax=2000, compensated=False):
    """
    Computes the roots of a complex polynomial using the Ehrlich-Aberth method.
    The function wraps a C++ CUDA version of the C code that is available here:
    https://github.com/trcameron/CompEA . One of two algorithms is used depending
    on the value of `compensated`. The first is the standard Ehrlich-Aberth
    method and the second is the Ehrlich-Aberth method with compensated
    arithmetic (see https://hal.archives-ouvertes.fr/hal-03335604) which computes
    the roots using double precision but with accuracy as if they were computed
    in quadruple precision. The compensated EA algorithm is about 2X slower than
    the regular one and is only necessary for polynomials with very large
    condition numbers.

    Args:
        coeffs (array_like): A JAX array of complex polynomial coefficients
            where the last dimension is stores the coefficients starting from
            the coefficient of the highest order term.
        itmax (int, optional): Maximum number of iteration of the root solver.
            Defaults to 2000.
        compensated (bool, optional): Use the compensated arithmetic version
            of the Ehrlich-Aberth algorithm. Defaults to False.

    Returns:
        array_like: The complex roots of the polynomial. Same shape as `coeffs`
        except the last dimension is shrunk by one.
    """
    ncoeffs = coeffs.shape[-1]
    output_shape = coeffs.shape[:-1] + (ncoeffs - 1,)

    # The `ehrlich_aberth` function expects the coefficients to be ordered as
    # p[0] + p[1] * x + ...
    coeffs = coeffs.reshape((-1, ncoeffs))[:, ::-1]
    roots = ehrlich_aberth(coeffs, itmax=itmax, compensated=compensated)
    return roots.reshape(output_shape)


# This function exposes the primitive to user code
@partial(jit, static_argnames=("itmax", "compensated"))
def ehrlich_aberth(coeffs, itmax=None, compensated=None):
    roots = _ehrlich_aberth_prim.bind(
        coeffs,
        itmax=itmax,
        compensated=compensated,
    )
    return roots


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _ehrlich_aberth_abstract(coeffs, **kwargs):
    """
    Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.
    """
    ncoeffs = coeffs.shape[-1]
    shape = (coeffs.shape[0] * (ncoeffs - 1),)
    dtype = dtypes.canonicalize_dtype(coeffs.dtype)
    return ShapedArray(shape, dtype)


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _ehrlich_aberth_translation(
    c, coeffs, itmax=None, compensated=None, platform="cpu"
):
    """
    The compilation to XLA of the primitive.

    JAX compilation works by compiling each primitive into a graph of XLA
    operations.

    This is the biggest hurdle to adding new functionality to JAX, because the
    set of XLA operations is limited, and JAX already has pre-defined primitives
    for most of them. However, XLA includes a CustomCall operation that can be
    used to encapsulate arbitrary functionality defined using C++.

    Here we specify the interaction between XLA and the the C++ code implementing
    the CPU and CUDA versions of the Ehrlich-Aberth algorithm.

    For more details see the tutorial https://github.com/dfm/extending-jax.
    """
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
        if ehrlich_aberth_gpu_op is None:
            raise ValueError(
                "The 'ehrlich_aberth_jax' module was not compiled with CUDA support"
            )

        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = ehrlich_aberth_gpu_op.build_ehrlich_aberth_descriptor(
            size, deg, itmax, compensated
        )

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
@partial(jit, static_argnames=("itmax", "compensated"))
def _ehrlich_aberth_jvp(args, tangents, itmax=None, compensated=None):
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

    We do not need a corresponding VJP rule "because JAX will produce the reverse
    differentiation computation by processing the JVP computation backwards.
    For each operation in the tangent computation, it accumulates the cotangents
    of the variables used by the operation, using the cotangent of the result
    of the operation."

    For more details on implicit differentiation and JAX, see this excellent
    tutorial: http://implicit-layers-tutorial.org/implicit_functions/.

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

    # Evaluate the roots (shape (size*deg,))
    z = _ehrlich_aberth_prim.bind(p, itmax=itmax, compensated=compensated)
    z = z.reshape((size, deg))

    # Evaluate the derivative of the polynomials at the roots
    p_deriv = vmap(lambda x: jnp.polyder(x[::-1]))(p)
    df_dz = vmap(lambda coeffs, root: jnp.polyval(coeffs, root))(p_deriv, z)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    # The Jacobian of f with respect to coefficient p evaluated at each of the
    # roots. Shape (size, deg, deg + 1).
    df_dp = vmap(vmap(lambda z: jnp.power(z, jnp.arange(deg + 1))))(z)

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
def _ehrlich_aberth_batch(args, axes, **kwargs):
    """
    Computes the batched version of the primitive. This must be a JAX-traceable
    function.

    Args:
        args: a tuple of two arguments, each being a tensor of matching
            shape.
        axes: the axes that are being batched. See vmap documentation.
    Returns:
        a tuple of the result, and the result axis that was batched.
    """
    coeffs = args[0]
    axis = axes[0]
    ncoeffs = coeffs.shape[-1]
    output_shape = coeffs.shape[:-1] + (ncoeffs - 1,)
    coeffs = coeffs.reshape(-1, ncoeffs)
    res = ehrlich_aberth(coeffs, **kwargs)

    return res.reshape(output_shape), axis


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
batching.primitive_batchers[_ehrlich_aberth_prim] = _ehrlich_aberth_batch
