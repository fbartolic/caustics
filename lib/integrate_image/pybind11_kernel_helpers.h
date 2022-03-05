// This header extends kernel_helpers.h with the pybind11 specific interface to
// serializing descriptors. It also adds a pybind11 function for wrapping our
// custom calls in a Python capsule. This is separate from kernel_helpers so that
// the CUDA code itself doesn't include pybind11. I don't think that this is
// strictly necessary, but they do it in jaxlib, so let's do it here too.

#ifndef _INTEGRATE_POLAR_PYBIND11_KERNEL_HELPERS_H_
#define _INTEGRATE_POLAR_PYBIND11_KERNEL_HELPERS_H_

#include <pybind11/pybind11.h>

#include "kernel_helpers.h"

namespace integrate_polar
{

  template <typename T>
  pybind11::bytes PackDescriptor(const T &descriptor)
  {
    return pybind11::bytes(PackDescriptorAsString(descriptor));
  }

  template <typename T>
  pybind11::capsule EncapsulateFunction(T *fn)
  {
    return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
  }

} // namespace integrate_polar

#endif
