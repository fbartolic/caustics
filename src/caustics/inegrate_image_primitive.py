# -*- coding: utf-8 -*-

__all__ = ["integrate_image"]

from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax, jit, vmap
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray
import jax.numpy as jnp

# Register the CPU XLA custom calls
from . import integrate_image_cpu_op

for _name, _value in integrate_image_cpu_op.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

# If the GPU version exists, also register those
try:
    from . import integrate_image_gpu_op
except ImportError:
    integrate_image_gpu_op = None
else:
    for _name, _value in integrate_image_gpu_op.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

xops = xla_client.ops

# This function exposes the primitive to user code
@jit
def integrate_image(
    rmin, theta_min, dr, dtheta, nr, ntheta, rho, a1, a, e1, source_center
):
    return _integrate_image_prim.bind(
        rmin, theta_min, dr, dtheta, nr, ntheta, rho, a1, a, e1, source_center
    )


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _integrate_image_abstract(
    rmin, theta_min, dr, dtheta, nr, ntheta, rho, a1, a, e1, source_center
):
    """
    Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.
    """
    dtype = dtypes.canonicalize_dtype(np.float64)
    return ShapedArray((), dtype)


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _integrate_image_translation(
    c,
    rmin,
    theta_min,
    dr,
    dtheta,
    nr,
    ntheta,
    rho,
    a1,
    a,
    e1,
    source_center,
    platform="cpu",
):
    """
    The compilation to XLA of the primitive.

    JAX compilation works by compiling each primitive into a graph of XLA
    operations.

    This is the biggest hurdle to adding new functionality to JAX, because the
    set of XLA operations is limited, and JAX already has pre-defined primitives
    for most of them. However, XLA includes a CustomCall operation that can be
    used to encapsulate arbitrary functionality defined using C++.

    For more details see the tutorial https://github.com/dfm/extending-jax.
    """
    op_name = platform.encode() + b"_integrate_image"

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return xops.CustomCallWithLayout(
            c,
            op_name,
            # The inputs:
            operands=(
                rmin,
                theta_min,
                dr,
                dtheta,
                nr,
                ntheta,
                rho,
                a1,
                a,
                e1,
                source_center,
            ),
            # The input shapes:
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.complex128), (), ()),
            ),
            # The output shapes:
            shape_with_layout=xla_client.Shape.array_shape(
                np.dtype(np.float64), (), ()
            ),
        )

    elif platform == "gpu":
        if integrate_image_gpu_op is None:
            raise ValueError(
                "The 'integrate_image' module was not compiled with CUDA support"
            )

        # opaque = integrate_image_gpu_op.build_integrate_image_descriptor(0)

        return xops.CustomCallWithLayout(
            c,
            op_name,
            operands=(
                rmin,
                theta_min,
                dr,
                dtheta,
                nr,
                ntheta,
                rho,
                a1,
                a,
                e1,
                source_center,
            ),
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.float64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.complex128), (), ()),
            ),
            # The output shapes:
            shape_with_layout=xla_client.Shape.array_shape(
                np.dtype(np.float64), (), ()
            ),
            opaque=b"",
        )

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************
# For each primitive numerical operation that the original function would have
# applied, the jvp-transformed function executes a “JVP rule” for that primitive
# that both evaluates the primitive on the primals and applies the primitive’s
# JVP at those primal values.
# @jit
# def _integrate_image_jvp(args, tangents):
#    """
#    Compute the Jacobian-vector product of the integration routine.
#
#
#    Args:
#        args (tuple): The arguments to the function `integrate_image`.
#        tangents (tuple): Small perturbation to the arguments.
#
#    Returns:
#        tuple: (z, dz) where z are the roots and dz is JVP.
#    """


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************
# def _integrate_image_batch(args, axes, **kwargs):
#    """
#    Computes the batched version of the primitive. This must be a JAX-traceable
#    function.
#
#    Args:
#        args: a tuple of two arguments, each being a tensor of matching
#            shape.
#        axes: the axes that are being batched. See vmap documentation.
#    Returns:
#        a tuple of the result, and the result axis that was batched.
#    """


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_integrate_image_prim = core.Primitive("integrate_image")
_integrate_image_prim.def_impl(partial(xla.apply_primitive, _integrate_image_prim))
_integrate_image_prim.def_abstract_eval(_integrate_image_abstract)

# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][_integrate_image_prim] = partial(
    _integrate_image_translation, platform="cpu"
)
xla.backend_specific_translations["gpu"][_integrate_image_prim] = partial(
    _integrate_image_translation, platform="gpu"
)

# Connect the JVP and batching rules
# ad.primitive_jvps[_integrate_image_prim] = _integrate_image_jvp
# batching.primitive_batchers[_integrate_image_prim] = _integrate_image_batch
